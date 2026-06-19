"""
proactive/drift_turn.py —— Drift 四段式执行链路。

DriftTurnPipeline.run() 是 Drift 的唯一入口：

  tick trigger (Judge skip)
    └─ DriftTurnPipeline.run()
       ├─ 1. Scan    扫描可用 skills
       ├─ 2. Prepare 构建 tool registry + system prompt + context frame
       ├─ 3. Execute LLM 工具调用循环
       └─ 4. Finish  记录退出状态

段之间通过 DriftAgentTickContext 传递状态，每段各司其职，
不跨段直接访问对方内部实现。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raven_agent.llm import LLMProvider
from raven_agent.messages import (
    system_message,
    user_message,
    assistant_message,
    tool_message,
    ToolCall,
)
from raven_agent.tools.executor import ToolExecutor
from raven_agent.tools.hooks import ToolExecutionRequest
from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore, SkillMeta
from raven_agent.proactive.drift_tools import build_drift_tool_registry

logger = logging.getLogger(__name__)


class DriftTurnPipeline:
    """Drift 空闲任务执行管线。

    参数:
        store: DriftStateStore 实例。
            store.source_dir — 技能源文件目录（注入 context frame）
            store.state_dir  — 运行时状态目录（传给 build_drift_tool_registry）
            两个路径在构造 store 时一次性确定，下游不再重复声明。
        provider: LLMProvider 实例。
        model: 使用的模型名。
        max_steps: 最大工具调用步数，默认 20。
        max_web_fetch_chars: web_fetch 返回内容最大字符数。
        memory: MarkdownMemoryStore 实例。
        shared_tools: 主 ToolRegistry。
        send_message_fn: 实际发送消息的异步函数。
        sessions: SessionManager 实例。
        tool_hooks: ToolHook 列表。
    """

    def __init__(
        self,
        *,
        store: DriftStateStore,
        provider: LLMProvider,
        model: str,
        max_steps: int = 20,
        max_web_fetch_chars: int = 8_000,
        memory: Any = None,
        shared_tools: Any = None,
        send_message_fn: Any = None,
        sessions: Any = None,
        tool_hooks: list | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._model = model
        self._max_steps = max_steps
        self._max_web_fetch_chars = max_web_fetch_chars
        self._memory = memory
        self._shared_tools = shared_tools
        self._send_message_fn = send_message_fn
        self._sessions = sessions
        self._tool_executor = ToolExecutor(tool_hooks or [])

    # ── 入口 ────────────────────────────────────────────────────────

    async def run(self, ctx: DriftAgentTickContext) -> bool:
        """执行一次 Drift。

        输入:
            ctx: DriftAgentTickContext 实例。

        输出:
            True 表示成功进入并执行了 Drift；False 表示没有可用的 skill。
        """
        # 1. Scan —— 扫描可用 skills。
        skills = self._scan_skills()
        if not skills:
            return False

        # 2. Prepare —— 构建 tool registry + system prompt + context frame。
        tools, messages, mounted_tool_names = self._prepare(ctx, skills)

        # 3. Execute —— LLM 工具调用循环。
        await self._execute_loop(ctx, tools, messages, mounted_tool_names)

        # 4. Finish —— 记录退出状态。
        self._finish(ctx)
        return True

    # ── 1. Scan ────────────────────────────────────────────────────

    def _scan_skills(self) -> list[SkillMeta]:
        """扫描可用 skills。

        调用 store.scan_skills()，如果结果为空则记录日志。

        输出:
            SkillMeta 列表。空列表表示无 skill 可用。
        """
        skills = self._store.scan_skills()
        if not skills:
            logger.info("[drift] skip: 无可用的 drift skill")
            return []

        # 过滤 requires_mcp 不满足的 skill
        if self._shared_tools is not None:
            connected = self._shared_tools.get_mcp_server_names()
            skills = [
                s for s in skills
                if not s.requires_mcp or set(s.requires_mcp) <= connected
            ]
            if not skills:
                logger.info("[drift] skip: 所有 skill 需要的 MCP server 都未连接")
                return []
        
        logger.info(
            "[drift] 进入 drift 模式: skills=%d max_steps=%d state_dir=%s",
            len(skills),
            self._max_steps,
            self._store.state_dir,
        )
        return skills

    # ── 2. Prepare ─────────────────────────────────────────────────

    def _prepare(
        self,
        ctx: DriftAgentTickContext,
        skills: list[SkillMeta],
    ) -> tuple[Any, list[dict[str, Any]], set[str]]:
        """设置 drift 标志、构建 tool registry 和初始 messages。

        输入:
            ctx: DriftAgentTickContext 实例。
            skills: scan_skills() 返回的 skill 列表。

        输出:
            (tools, messages, mounted_tool_names) 三元组。
        """
        # 2.1 设置 drift 标志。
        ctx.drift_entered = True
        ctx.drift_finished = False
        ctx.drift_message_sent = False

        # 2.2 构建 drift tool registry。
        mounted_tool_names: set[str] = set()
        tools = build_drift_tool_registry(
            ctx=ctx,
            store=self._store,
            state_dir=self._store.state_dir,
            shared_tools=self._shared_tools,
            send_message_fn=self._send_message_fn,
            max_web_fetch_chars=self._max_web_fetch_chars,
            mounted_tool_names=mounted_tool_names,
            sessions=self._sessions,
        )

        # 2.3 构建初始 messages。
        messages = [
            system_message(self._build_system_prompt()),
            user_message(self._build_context_message(skills)["content"]),
        ]

        return tools, messages, mounted_tool_names

    # ── 3. Execute ──────────────────────────────────────────────────

    async def _execute_loop(
        self,
        ctx: DriftAgentTickContext,
        tools: Any,
        messages: list,
        mounted_tool_names: set[str],
    ) -> None:
        """LLM 工具调用循环。

        输入:
            ctx: DriftAgentTickContext 实例。
            tools: drift ToolRegistry。
            messages: 初始消息列表（会被原地追加 tool 消息）。
            mounted_tool_names: 已挂载工具的名称集合。

        循环逻辑：
        1. 获取当前可用工具 schema
        2. 调 LLM 获取 tool_call
        3. 执行工具
        4. 追加 tool call + tool result 到 messages
        5. 检查 finish 标志，未完成且未达上限则继续
        """

        # 获取当前注册表里所有可用的工具描述 (JSON Schema)
        base_schemas = tools.get_schemas()
        # 初始化当前已执行的工具步数
        steps = 0

        # 核心循环条件：未达到最大步数硬限制，并且模型未显式宣布任务结束
        while steps < self._max_steps and not ctx.drift_finished:
            schemas = list(base_schemas)
            
            if mounted_tool_names and self._shared_tools:
                schemas += self._shared_tools.get_schemas(names=mounted_tool_names)

            # 如果在之前的步骤中，Agent 已经调用了发消息工具 (ctx.drift_message_sent == True)
            if ctx.drift_message_sent:
                # 只保留文件写入和结束工具，剥夺读取、搜索、Shell 等高危权限
                allowed_after_send = {"write_text_file", "edit_file", "finish_drift"}
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_after_send
                ]
                logger.info(
                    "[drift] message_push 已使用，"
                    "限制 schema 为 write_text_file/edit_file/finish_drift"
                )

            # 1. 向 LLM 询问下一步动作
            try:
                response = await self._provider.chat(
                    messages=messages,
                    tools=schemas,
                    model=self._model,
                    max_tokens=2048,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.warning("[drift] LLM 调用失败 step=%d: %s", steps, exc)
                break

            if not response.tool_calls:
                logger.warning(
                    "[drift] LLM 未返回 tool call step=%d text=%r",
                    steps,
                    (response.content or "")[:200],
                )
                break

            # 一次只执行一个工具（后台任务，哪怕漏掉其他的工具也可以慢慢来，不需要一次性做完更稳定）
            tc = response.tool_calls[0]
            tool_name = tc.name
            tool_args = tc.arguments
            tool_call_id = tc.id or f"drift_{steps}"

            logger.info(
                "[drift] step=%d tool=%s args=%s",
                steps,
                tool_name,
                json.dumps(tool_args, ensure_ascii=False)[:200],
            )

            exec_fn = tools.execute
            if not tools.has_tool(tool_name) and tool_name in mounted_tool_names:
                exec_fn = self._shared_tools.execute

            # 更新执行步数
            steps += 1
            ctx.steps_taken += 1

            # 2. 执行工具
            try:
                result = await self._tool_executor.execute(
                    ToolExecutionRequest(
                        call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=tool_args,
                        session_key=ctx.session_key,
                    ),
                    exec_fn,
                )
            except Exception as exc:
                logger.warning(
                    "[drift] 工具执行异常 step=%d tool=%s: %s",
                    steps, tool_name, exc,
                )
                result_output = json.dumps(
                    {"error": str(exc)}, ensure_ascii=False
                )
            else:
                result_output = str(result.output)

            logger.info(
                "[drift] step=%d tool=%s result=%s",
                steps,
                tool_name,
                str(result_output)[:300],
            )

            # 3. 追加 assistant + tool 消息
            messages.append(
                assistant_message(
                    content=response.content or f"调用工具 {tool_name}",
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id,
                            name=tool_name,
                            arguments=tool_args,
                        )
                    ],
                    reasoning_content=response.reasoning_content,
                )
            )
            messages.append(
                tool_message(
                    tool_call_id=tool_call_id,
                    content=result_output,
                )
            )

    # ── 4. Finish ──────────────────────────────────────────────────

    def _finish(self, ctx: DriftAgentTickContext) -> None:
        """记录 drift 退出状态。

        输入:
            ctx: DriftAgentTickContext 实例。
        """
        logger.info(
            "[drift] 退出: finished=%s message_sent=%s steps=%d",
            ctx.drift_finished,
            ctx.drift_message_sent,
            ctx.steps_taken,
        )

    # ── Prompt 构建 ──────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """构建 Drift 的 system prompt。

        输出:
            system prompt 字符串。
        """
        return (
            "你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，"
            "你可以自主决定做一件有意义的事。"
            "本轮记忆、skill 列表和工作区信息会在后续 context frame 里提供。\n\n"
            "【执行规则】\n"
            "1. 每次进入 Drift 都先重新比较所有可用 skill，"
            "不要因为某个 skill 最近刚运行过，或它的 next 很明确，就默认继续它。\n"
            "   只有当它仍然是当前最值得做的事时，才继续它；"
            "如果别的 skill 更久没运行、更有价值、或更适合当前空档时间，"
            "优先选别的 skill。\n"
            "2. 自主选择一个 skill，read_file 读它的 SKILL.md 了解细节。\n"
            "   路径格式是 skills/<skill_name>/SKILL.md，"
            "例如 skills/audit-memory/SKILL.md。\n"
            "3. read_file 读该 skill 的 working files 了解当前进度。\n"
            "4. 读完 skill 和 working files 后，要执行这个 skill "
            "当前最直接的下一步动作，不要只因为看到了 queue、next 或等待描述，"
            "就立刻 finish_drift。\n"
            "   如果这个 skill 当前明显处于'等待用户回复/等待外部条件'的状态，"
            "就不要选它，改选别的 skill。\n"
            "5. 只有在本轮已经完成了一个明确动作后，或确认该 skill 当前确实"
            "无事可做时，才允许 finish_drift。\n"
            "6. 有价值的发现必须立即 write_text_file 或 edit_file，"
            "不要积累到最后再写。\n"
            "7. 如果你决定 message_push，对用户的表达要像此刻自然想到的"
            "一句聊天，而不是像在执行队列、候选列表、记忆检索或内部流程。\n"
            "   先把内部依据转写成自然联想，再说出口：像突然想到、"
            "顺着刚才的感觉延伸、隐约记得用户会偏好什么、或此刻真的有点好奇。\n"
            "8. 单次 run 最多只能 message_push 一次。\n"
            "9. message_push 成功后不要再调用 recall_memory / web_fetch / "
            "web_search / fetch_messages / search_messages / shell，"
            "后续只允许 write_text_file、edit_file 和 finish_drift 收尾。\n"
            "10. 执行结束前必须调用 finish_drift 保存状态，"
            "并用 message_result 标注本轮是 sent 还是 silent。\n"
        )

    def _build_context_message(
        self,
        skills: list[SkillMeta],
    ) -> dict[str, str]:
        """构建 runtime context frame 消息。

        包含：
        - Drift 工作区路径
        - 长期记忆文本（MEMORY.md）
        - 近期上下文（RECENT_CONTEXT.md）
        - Skill 列表（名称 / 运行次数 / next / requires_mcp）
        - 近期 Drift 运行记录

        输入:
            skills: scan_skills() 返回的 skill 列表。

        输出:
            {"role": "user", "content": "..."} 格式的 context frame 消息。
        """
        # 长期记忆
        memory_text = ""
        if self._memory is not None:
            try:
                raw = str(self._memory.read_long_term() or "").strip()
                if raw:
                    memory_text = raw
            except Exception:
                pass

        # 近期上下文
        recent_context_text = ""
        if self._memory is not None:
            try:
                rc = str(self._memory.read_recent_context() or "").strip()
                if rc:
                    recent_context_text = rc
            except Exception:
                pass

        # Skill 列表
        lines: list[str] = []
        for skill in skills[:8]:
            next_text = skill.next[:80] if skill.next else ""
            line = f"- {skill.name}   {skill.run_count}次运行"
            if next_text:
                line += f'   next: "{next_text}"'
            if skill.requires_mcp:
                line += f"   [需要: {', '.join(skill.requires_mcp)}]"
            lines.append(line)
        skill_block = "\n".join(lines) if lines else "- (none)"

        # 近期运行记录
        recent_rows: list[str] = []
        for row in self._store.load_drift().get("recent_runs", [])[-5:][::-1]:
            run_at = str(row.get("run_at") or "")
            try:
                dt = datetime.fromisoformat(run_at).astimezone(timezone.utc)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_text = run_at[:16]
            recent_rows.append(
                f"- {time_text}  {row.get('skill', '')}   "
                f"[{row.get('message_result', 'silent')}] "
                f"{str(row.get('one_line', ''))[:150]}"
            )
        recent_block = "\n".join(recent_rows) if recent_rows else "- (none)"

        drift_note = str(
            self._store.load_drift().get("note") or ""
        )[:150]

        content = (
            f"【Drift 工作区（write_file/edit_file/read_file 相对路径解析到此）】\n"
            f"绝对路径：{self._store.state_dir.resolve()}\n"
            f"read_file / write_file / edit_file 的相对路径以此为根。\n"
            f"⚠️  相对路径格式：skills/<skill_name>/<文件>，不要带任何前缀。\n"
            f"   正确示例：skills/health-log/log.md\n"
            f"   错误示例：.raven/drift/skills/health-log/log.md\n\n"
            f"【Drift Skill 源文件目录（用此绝对路径读取 SKILL.md）】\n"
            f"{self._store.source_dir}\n\n"
            f"【长期记忆】\n{memory_text or '（空）'}\n\n"
            f"【近期上下文】\n{recent_context_text or '（空）'}\n\n"
            f"【可用 Skill 列表】\n{skill_block}\n\n"
            f"【近期 Drift 运行记录】\n{recent_block}\n\n"
            f"【Drift 笔记】\n{drift_note or '（空）'}"
        )

        return {"role": "user", "content": content}