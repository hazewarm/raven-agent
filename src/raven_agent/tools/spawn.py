from __future__ import annotations

import json
import logging
from typing import Any

from raven_agent.background.delegation import DelegationPolicy
from raven_agent.background.subagent_profiles import PROFILE_RESEARCH
from raven_agent.background.subagents import SubagentManager
from raven_agent.tools.base import Tool
from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome

logger = logging.getLogger(__name__)

_SPAWN_CONTEXT_KEYS = ("channel", "chat_id")


class SpawnToolContextHook(ToolHook):
    """向 spawn 工具注入当前 turn 的 channel/chat_id。

    输入:
        无。

    输出:
        SpawnToolContextHook 实例。ToolExecutor 在 pre_tool_use 阶段调用。
    """

    name = "spawn_tool_context"
    event = "pre_tool_use"

    def matches(self, context: ToolHookContext) -> bool:
        """判断是否匹配 spawn 工具调用。

        输入:
            context: 当前 ToolHookContext。

        输出:
            True 表示当前工具是 spawn。
        """
        return context.request.tool_name == "spawn"

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """把 metadata 中的 channel/chat_id 合并到工具参数。

        输入:
            context: 当前 ToolHookContext。request.metadata 来自 turn_pipeline 的 tool_context。

        输出:
            ToolHookOutcome；有字段注入时返回 updated_arguments。
        """
        merged = dict(context.current_arguments)
        changed = False
        for key in _SPAWN_CONTEXT_KEYS:
            value = context.request.metadata.get(key)
            if value and not merged.get(key):
                merged[key] = value
                changed = True
        if not changed:
            return ToolHookOutcome()
        return ToolHookOutcome(updated_arguments=merged)


class SpawnTool(Tool):
    """创建本地 SubAgent 子任务的工具。

    输入:
        manager: SubagentManager。
        policy: DelegationPolicy；不传时使用默认并发限制策略。

    输出:
        SpawnTool 实例，可注册到 ToolRegistry。
    """

    name = "spawn"
    description = """\
把一个有界的多步任务交给独立 subagent 执行，主 agent 专注决策和用户沟通。

何时使用 spawn（同时满足）：
- 预计需要 4 步以上工具调用
- 可以完全独立完成，中途不需要用户确认
- 产出是报告 / 文件 / 分析结论，而非立刻执行的行动

何时不用 spawn：
- 只需 1–3 次工具调用 → 直接调用工具
- 直接回答问题 → 直接回答
- 任务需要和用户来回确认才能推进
- 用户要求立即发送/立即执行的外部动作

执行模式：
- run_in_background=false（默认）：同步执行，当前 turn 等待结果，适合较短调研
- run_in_background=true：后台执行，当前 turn 只向用户确认，完成后系统会带回原会话

profile：
- research（默认）：只读调研，可读文件、列目录、搜索网页、抓网页
- scripting：执行型，可运行 shell、在任务目录写文件，工具集中不含 web_fetch/web_search（工具层无网络，但 shell 命令本身不受限制）
- general：研究与执行兼有，仅在明确需要时使用

task 必须像交接文档：任务目标、关键约束、关键上下文、期望输出格式都要写清楚。\
"""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "交给 subagent 的完整任务描述。必须包含目标、约束、上下文、期望输出格式。",
            },
            "label": {
                "type": "string",
                "description": "3–5 个字的任务短标签，用于状态显示。",
            },
            "profile": {
                "type": "string",
                "enum": ["research", "scripting", "general"],
                "description": "subagent 工具权限 profile，默认 research。",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "false 同步等待结果；true 后台执行，完成后系统回流。",
            },
            "retry_count": {
                "type": "integer",
                "minimum": 0,
                "description": "当前后台任务已重试次数。首次调用为 0，重试时传 1。",
            },
            "channel": {
                "type": "string",
                "description": "当前 channel，由运行时注入，模型通常不要手写。",
            },
            "chat_id": {
                "type": "string",
                "description": "当前 chat_id，由运行时注入，模型通常不要手写。",
            },
        },
        "required": ["task"],
    }

    def __init__(
        self,
        manager: SubagentManager,
        policy: DelegationPolicy | None = None,
    ) -> None:
        self._manager = manager
        self._policy = policy or DelegationPolicy()

    async def execute(
        self,
        task: str,
        label: str | None = None,
        profile: str = PROFILE_RESEARCH,
        run_in_background: bool = False,
        retry_count: int = 0,
        channel: str = "",
        chat_id: str = "",
        **_: Any,
    ) -> str:
        """执行 spawn 工具。

        输入:
            task: 子任务完整描述。
            label: 可选任务短标签。
            profile: research / scripting / general。
            run_in_background: 是否后台执行。
            retry_count: 已重试次数。
            channel: 当前 channel，由 SpawnToolContextHook 注入。
            chat_id: 当前 chat_id，由 SpawnToolContextHook 注入。

        输出:
            同步模式返回子任务结果；后台模式返回创建成功/失败说明。
        """
        retry_count = max(0, int(retry_count))
        running_count = self._manager.get_running_count() if run_in_background else 0
        decision = self._policy.decide(
            task=task,
            label=label,
            running_count=running_count,
        )
        logger.info(
            "[spawn] decision should_spawn=%s reason=%s label=%r profile=%s background=%s",
            decision.should_spawn,
            decision.meta.reason_code,
            decision.label,
            profile,
            run_in_background,
        )

        if not decision.should_spawn:
            return f"任务被拦截：{decision.block_reason}"

        if run_in_background:
            clean_channel = str(channel or "").strip()
            clean_chat_id = str(chat_id or "").strip()
            if not clean_channel or not clean_chat_id:
                return "错误：当前会话上下文缺失，无法创建后台任务"
            return await self._manager.spawn(
                task=task,
                label=label,
                origin_channel=clean_channel,
                origin_chat_id=clean_chat_id,
                decision=decision,
                profile=profile,
                retry_count=retry_count,
            )

        return await self._manager.spawn_sync(
            task=task,
            label=label,
            profile=profile,
        )


class SpawnManageTool(Tool):
    """管理后台 spawn 任务的工具。

    输入:
        manager: SubagentManager。

    输出:
        SpawnManageTool 实例。
    """

    name = "spawn_manage"
    description = """\
管理当前运行中的后台 subagent。

可用 action：
- list：列出正在运行/排队的后台任务，包含 job_id、label、profile、task_dir 和状态
- cancel：按 job_id 取消后台任务

只在用户询问后台任务状态、要求查看 job_id、或明确要求停止某个后台任务时使用。\
"""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "cancel"],
                "description": "list 查看任务；cancel 取消指定 job_id。",
            },
            "job_id": {
                "type": "string",
                "description": "action=cancel 时要取消的后台任务 job_id。",
            },
        },
        "required": ["action"],
    }

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    async def execute(
        self,
        action: str,
        job_id: str | None = None,
        **_: Any,
    ) -> str:
        """执行后台任务管理操作。

        输入:
            action: list 或 cancel。
            job_id: cancel 时要取消的任务 ID。

        输出:
            JSON 字符串，描述任务列表或取消状态。
        """
        if action == "list":
            return json.dumps(
                {
                    "running_count": self._manager.get_running_count(),
                    "jobs": self._manager.list_running_jobs(),
                },
                ensure_ascii=False,
            )
        if action == "cancel":
            target = (job_id or "").strip()
            if not target:
                return json.dumps({"error": "缺少 job_id"}, ensure_ascii=False)
            cancelled = await self._manager.cancel(target)
            return json.dumps(
                {
                    "job_id": target,
                    "status": "cancel_requested" if cancelled else "not_found",
                },
                ensure_ascii=False,
            )
        return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)