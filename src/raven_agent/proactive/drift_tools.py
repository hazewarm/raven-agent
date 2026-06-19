"""
proactive/drift_tools.py —— Drift 模式专属工具集。

Drift 工具体系：
  1. DriftReadFileTool   — 双路路径解析（相对 drift_dir + 绝对路径）
  2. SendMessageTool     — drift 版 message_push，单次限制
  3. FinishDriftTool     — 终止工具，保存状态 + 校验 message_result 一致性
  4. build_drift_tool_registry() — 工厂函数，组装完整注册表

与主 ToolRegistry 的关系：
  主 ToolRegistry 包含所有被动对话工具（schedule / spawn / push 等）。
  Drift 只需要一个子集：read_file / write_file / edit_file / recall_memory /
  web_fetch / web_search / shell / message_push / finish_drift。
  因此 build_drift_tool_registry() 创建独立的 ToolRegistry，
  从 shared_tools 中复用可以共享的工具，包装需要限制的工具。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from raven_agent.tools.base import Tool, ToolResult
from raven_agent.tools.filesystem import EditFileTool, WriteTextFileTool
from raven_agent.tools.readonly import ReadTextFileTool
from raven_agent.tools.registry import ToolRegistry
from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore

logger = logging.getLogger(__name__)


# ── DriftReadFileTool ───────────────────────────────────────────────

class DriftReadFileTool(Tool):
    """Drift 模式的文件读取工具。

    路径解析规则：
    1. 相对路径 → 相对于 state_dir 解析
    2. 绝对路径 → 直接使用（允许读取 workspace 内的任意文件）
    3. 路径 -> 无特殊处理

    参数:
        state_dir: drift 状态目录的绝对路径。
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._relative_reader = ReadTextFileTool(allowed_dir=state_dir)
        self._absolute_reader = ReadTextFileTool()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return self._absolute_reader.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._absolute_reader.parameters

    async def execute(self, path: str, **kwargs: Any) -> Any:
        """读取文件内容。

        输入:
            path: 文件路径。相对路径相对于 drift_dir 解析。

        输出:
            文件内容字符串或 ToolResult。
        """
        raw = str(path or "").strip()
        if not raw:
            return await self._absolute_reader.execute(path=path, **kwargs)

        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            return await self._absolute_reader.execute(path=path, **kwargs)

        return await self._relative_reader.execute(path=path, **kwargs)


# ── DriftWebFetchTool ───────────────────────────────────────────────

class DriftWebFetchTool(Tool):
    """Drift 模式的 web_fetch 包装器，限制返回内容长度。

    参数:
        wrapped: 被包装的 web_fetch Tool 实例。
        max_chars: 最大字符数，超出则截断。
    """

    def __init__(self, wrapped: Tool, max_chars: int) -> None:
        self._wrapped = wrapped
        self._max_chars = max(1, int(max_chars))

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    async def execute(self, **kwargs: Any) -> str | ToolResult:
        """执行 web_fetch 并截断过长结果。

        输出:
            JSON 字符串。如果 text 字段超过 max_chars，截断并添加 truncated 标记。
        """
        result = await self._wrapped.execute(**kwargs)
        if not isinstance(result, str):
            return result
        try:
            payload = json.loads(result)
        except Exception:
            return result
        text = payload.get("text")
        if not isinstance(text, str) or len(text) <= self._max_chars:
            return result
        payload["text"] = text[: self._max_chars]
        payload["length"] = len(payload["text"])
        payload["truncated"] = True
        payload["note"] = (
            f"内容已截断至 {self._max_chars} 字符，"
            "如需更多内容请缩小范围或改用更精确的读取方式"
        )
        return json.dumps(payload, ensure_ascii=False)


# ── SendMessageTool (Drift 版) ──────────────────────────────────────

class DriftSendMessageTool(Tool):
    """Drift 模式的消息推送工具。

    与主 message_push 的区别：
    1. 单次 Drift run 最多只能调用一次（ctx.drift_message_sent 标志）
    2. channel 和 chat_id 由 drift 上下文预设，可省略

    参数:
        ctx: DriftAgentTickContext 实例。
        send_message_fn: 实际执行推送的异步函数。
            签名: async def(text: str, media_paths: list[str]) -> bool
    """

    def __init__(
        self,
        ctx: DriftAgentTickContext,
        send_message_fn: Any,
    ) -> None:
        self._ctx = ctx
        self._send_message_fn = send_message_fn

    @property
    def name(self) -> str:
        return "message_push"

    @property
    def description(self) -> str:
        return (
            "向用户发送一条消息，可附带图片。单次 Drift run 最多只能调用一次。\n"
            "channel 和 chat_id 在 Drift 上下文中已由配置预设，可省略不填。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要发送的消息内容",
                },
                "image": {
                    "type": "string",
                    "description": "要发送的一张图片本地路径或 URL",
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要随消息发送的图片路径或 URL 列表",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        message: str = "",
        content: str = "",
        image: str = "",
        media: list[str] | str | None = None,
        **_: Any,
    ) -> str:
        """发送一条消息给用户。

        输入:
            message / content: 消息文本（两个参数等效，message 优先）。
            image: 单张图片路径或 URL。
            media: 图片路径/URL 列表。

        输出:
            JSON 字符串 `{"ok": true}` 或 `{"error": "..."}`。
        """
        text = str(message or content or "").strip()
        media_paths = self._normalize_media(image=image, media=media)

        if self._send_message_fn is None:
            logger.info("[drift_tools] message_push 不可用")
            return json.dumps(
                {"error": "message_push not configured"}, ensure_ascii=False
            )

        if self._ctx.drift_message_sent:
            logger.info("[drift_tools] message_push 拒绝: 本 tick 已使用")
            return json.dumps(
                {"error": "message_push 在本次 drift run 中已使用过"},
                ensure_ascii=False,
            )

        if not text and not media_paths:
            logger.info("[drift_tools] message_push 拒绝: 消息和媒体均为空")
            return json.dumps(
                {"error": "message 或 media 至少需要一个"}, ensure_ascii=False
            )

        ok = await self._send_message_fn(text, media_paths)
        if not ok:
            logger.warning("[drift_tools] message_push 失败")
            return json.dumps({"error": "message_push 发送失败"}, ensure_ascii=False)

        self._ctx.drift_message_sent = True
        logger.info("[drift_tools] message_push 成功")
        return json.dumps({"ok": True}, ensure_ascii=False)

    @staticmethod
    def _normalize_media(
        *,
        image: str = "",
        media: list[str] | str | None = None,
    ) -> list[str]:
        """规范化 media 参数。

        输入:
            image: 单张图片路径。
            media: 图片路径列表或单个字符串。

        输出:
            去重后的路径列表。
        """
        paths: list[str] = []
        if image:
            paths.append(str(image).strip())
        if isinstance(media, str):
            paths.append(media.strip())
        elif media:
            paths.extend(str(item).strip() for item in media)
        return [p for p in paths if p]


# ── FinishDriftTool ─────────────────────────────────────────────────

class FinishDriftTool(Tool):
    """Drift 终止工具。

    调用后立即结束本次 Drift 执行循环。保存 skill 状态到 state.json
    和 drift.json，并校验 message_result 与实际 message_push 调用的一致性。

    参数:
        ctx: DriftAgentTickContext 实例。
        store: DriftStateStore 实例。
    """

    def __init__(
        self,
        ctx: DriftAgentTickContext,
        store: DriftStateStore,
    ) -> None:
        self._ctx = ctx
        self._store = store

    @property
    def name(self) -> str:
        return "finish_drift"

    @property
    def description(self) -> str:
        return (
            "【终止工具】结束本次 Drift，保存进度状态。调用后 loop 立即结束。\n"
            "参数说明：\n"
            "- skill_used: 本次运行的 skill 名称\n"
            "- one_line: 一句话描述本轮做了什么\n"
            "- next: 下一步做什么（供下次 drift 参考）\n"
            "- message_result: \"sent\" 表示本轮已推送消息，"
            "\"silent\" 表示本轮静默结束\n"
            "- note: 可选的跨轮次笔记\n"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_used": {
                    "type": "string",
                    "description": "本次运行的 skill 名称",
                },
                "one_line": {
                    "type": "string",
                    "description": "一句话描述本轮做了什么",
                },
                "next": {
                    "type": "string",
                    "description": "下一步动作描述",
                },
                "message_result": {
                    "type": "string",
                    "enum": ["sent", "silent"],
                    "description": (
                        "sent 表示本轮已经成功调用 message_push；"
                        "silent 表示本轮确认不该打扰用户，静默结束。"
                    ),
                },
                "note": {
                    "type": "string",
                    "description": "可选的跨轮次笔记（写入 drift.json）",
                },
            },
            "required": ["skill_used", "one_line", "next", "message_result"],
        }

    async def execute(
        self,
        skill_used: str,
        one_line: str,
        next: str,
        message_result: str = "",
        note: str | None = None,
        **_: Any,
    ) -> str:
        """结束本次 Drift 并保存状态。

        输入:
            skill_used: skill 名称（必须存在于 scan_skills() 结果中）。
            one_line: 一句话运行摘要。
            next: 下一步动作描述。
            message_result: "sent" 或 "silent"。
            note: 可选笔记。

        输出:
            JSON 字符串 `{"ok": true}` 或 `{"error": "..."}`。

        校验规则:
            - skill_used 必须是已知 skill
            - one_line 和 next 不能为空
            - message_result 必须是 "sent" 或 "silent"
            - message_result="sent" 要求之前已成功调用 message_push
            - message_result="silent" 与 message_push 调用冲突
        """
        skill_name = str(skill_used or "").strip()

        # 校验 1: skill 必须已知
        if skill_name not in self._store.valid_skill_names():
            logger.info(
                "[drift_tools] finish_drift 拒绝: 未知 skill=%s", skill_name
            )
            return json.dumps(
                {"error": f"未知 skill: {skill_name}"}, ensure_ascii=False
            )

        # 校验 2: one_line 和 next 非空
        summary = str(one_line or "").strip()
        next_action = str(next or "").strip()
        if not summary:
            return json.dumps({"error": "one_line 不能为空"}, ensure_ascii=False)
        if not next_action:
            return json.dumps({"error": "next 不能为空"}, ensure_ascii=False)

        # 校验 3: message_result 合法性
        message_result_value = str(message_result or "").strip()
        if message_result_value not in {"sent", "silent"}:
            return json.dumps(
                {"error": "message_result 必须是 sent 或 silent"},
                ensure_ascii=False,
            )

        # 校验 4: sent 必须已推送
        if message_result_value == "sent" and not self._ctx.drift_message_sent:
            return json.dumps(
                {"error": "message_result=sent 要求先成功调用 message_push"},
                ensure_ascii=False,
            )

        # 校验 5: silent 不能已推送
        if message_result_value == "silent" and self._ctx.drift_message_sent:
            return json.dumps(
                {"error": "message_result=silent 与实际 message_push 调用冲突"},
                ensure_ascii=False,
            )

        # 保存状态
        note_text = str(note).strip() if note is not None else None
        self._store.save_finish(
            skill_used=skill_name,
            one_line=summary,
            next_action=next_action,
            message_result=message_result_value,
            note=note_text,
            now_utc=self._ctx.now_utc,
        )

        self._ctx.drift_finished = True
        logger.info(
            "[drift_tools] finish_drift ok: skill=%s one_line=%s next=%s",
            skill_name,
            summary[:120],
            next_action[:100],
        )
        return json.dumps({"ok": True}, ensure_ascii=False)

# ── MountServerTool ────────────────────────────────────────────────────

class MountServerTool(Tool):
    """挂载一个已连接 MCP server 的所有工具，使其在本次 Drift run 中可用。

    Drift 默认只能使用文件 + 记忆 + web + shell 等基础工具。
    如果 skill 需要调用 MCP server 工具，必须通过 mount_server 显式挂载。

    参数:
        shared_tools: 主 ToolRegistry（用于查询挂载的 MCP 工具）。
        mounted_tool_names: 当前 run 已挂载的 MCP tool 名称集合
            （同一个 set 对象在 DriftTurnPipeline._prepare() 中创建）。
    """

    def __init__(self, shared_tools: Any, mounted_tool_names: set[str]) -> None:
        self._shared = shared_tools
        self._mounted = mounted_tool_names

    @property
    def name(self) -> str:
        return "mount_server"

    @property
    def description(self) -> str:
        return (
            "挂载一个已连接 MCP server 的所有工具，使其在本次 Drift 中可用。"
            "挂载后即可直接调用该 server 的工具（如 list_events / create_issue）。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "要挂载的 MCP server 名称",
                },
            },
            "required": ["server"],
        }

    async def execute(self, server: str, **_: Any) -> str:
        """挂载指定 MCP server 的所有工具。

        输入:
            server: MCP server 名称。

        输出:
            JSON 字符串。
        """
        server = str(server or "").strip()
        if not server:
            return json.dumps({"error": "server 不能为空"}, ensure_ascii=False)

        names = self._shared.get_tool_names_by_source("mcp", server)
        if not names:
            return json.dumps(
                {"error": f"MCP server {server!r} 不存在或未连接"},
                ensure_ascii=False,
            )

        new = names - self._mounted
        if not new:
            return json.dumps(
                {
                    "ok": True,
                    "message": f"{server!r} 已挂载，无新增工具",
                    "tools": sorted(names),
                },
                ensure_ascii=False,
            )

        self._mounted |= new
        logger.info(
            "[drift_tools] mount_server ok: server=%s new=%s",
            server, sorted(new),
        )
        return json.dumps(
            {"ok": True, "tools": sorted(names), "new": sorted(new)},
            ensure_ascii=False,
        )




# ── FetchMessagesTool ────────────────────────────────────────────────


class FetchMessagesTool(Tool):
    """Drift 模式的对话历史读取工具。

    从目标 session 读取最近的被动对话消息，按时间排列。
    主动推送消息会被自动过滤。

    参数:
        sessions: SessionManager 实例。
        session_key: 目标会话 key（"channel:chat_id"）。
    """

    def __init__(
        self,
        sessions: Any,
        session_key: str,
    ) -> None:
        self._sessions = sessions
        self._session_key = session_key

    @property
    def name(self) -> str:
        return "fetch_messages"

    @property
    def description(self) -> str:
        return (
            "读取最近 N 条被动对话消息（user 和 assistant 的回复）。"
            "返回按时间排列的对话记录，每条含 role、content 和时间戳。"
            "用于回溯用户最近说了什么、之前的上下文等。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "最多返回多少条消息，默认 20",
                },
            },
            "required": [],
        }

    async def execute(self, n: int = 20, **_: Any) -> str:
        """读取最近 N 条被动对话消息。

        输入:
            n: 最多返回多少条消息，默认 20。

        输出:
            JSON 数组，每条含 role、content、timestamp。
        """
        if not self._session_key:
            return json.dumps(
                {"error": "session_key 未配置"}, ensure_ascii=False
            )
        try:
            session = self._sessions.get_or_create(self._session_key)
        except Exception:
            return json.dumps(
                {"error": "无法获取 session"}, ensure_ascii=False
            )

        max_n = max(1, min(int(n), 100))
        messages = session.messages[-max_n:]
        results: list[dict[str, str]] = []
        for msg in messages:
            role = getattr(msg, "role", "")
            content = str(getattr(msg, "content", ""))
            if role not in ("user", "assistant"):
                continue
            if not content:
                continue
            if content.startswith("[系统上下文]"):
                continue
            # 跳过主动推送消息
            if getattr(msg, "proactive", False):
                continue
            ts = getattr(msg, "timestamp", "")
            results.append({
                "role": role,
                "content": content[:500],
                "timestamp": str(ts) if ts else "",
            })

        return json.dumps(results, ensure_ascii=False)


# ── 工具注册表工厂 ──────────────────────────────────────────────────


def build_drift_tool_registry(
    *,
    ctx: DriftAgentTickContext,
    store: DriftStateStore,
    state_dir: Path,
    shared_tools: ToolRegistry | None = None,
    send_message_fn: Any = None,
    max_web_fetch_chars: int = 8_000,
    mounted_tool_names: set[str] | None = None,
    sessions: Any = None,
) -> ToolRegistry:
    """构建 Drift 模式的独立工具注册表。

    组装规则：
    1. 总是注册：DriftReadFileTool / WriteTextFileTool / EditFileTool
    2. 从 shared_tools 复用：recall_memory / web_fetch（包装截断）/
       web_search / shell / fetch_messages / search_messages
    3. 总是注册：DriftSendMessageTool / FinishDriftTool

    输入:
        ctx: DriftAgentTickContext 实例。
        store: DriftStateStore 实例。
        state_dir: drift 状态目录的绝对路径。
        shared_tools: 主 ToolRegistry（用于复用共享工具）。
        send_message_fn: 实际发送消息的异步函数。
        max_web_fetch_chars: web_fetch 最大字符数。

    输出:
        组装好的 ToolRegistry 实例。
    """
    tools = ToolRegistry()

    # 1. 文件系统工具
    tools.register(
        DriftReadFileTool(state_dir),
        risk="read-only",
    )
    tools.register(
        WriteTextFileTool(allowed_dir=state_dir),
        risk="write",
    )
    tools.register(
        EditFileTool(allowed_dir=state_dir),
        risk="write",
    )

    # 2. 对话历史（drift 专属，不依赖被动通道的 prompt 注入）
    tools.register(
        FetchMessagesTool(sessions, ctx.session_key),
        risk="read-only",
    )

    # 3. 从 shared_tools 复用
    if shared_tools is not None:
        for name in (
            "recall_memory",
            "web_fetch",
            "web_search",
            "search_messages",
            "shell",
        ):
            tool = shared_tools.get(name)
            if tool is not None:
                if name == "web_fetch":
                    tool = DriftWebFetchTool(tool, max_web_fetch_chars)
                risk = "external-side-effect" if name == "shell" else "read-only"
                tools.register(tool, risk=risk)

    # 3. Drift 专属工具
    # mount_server: 仅当 shared registry 中有 MCP server 时注册
    if shared_tools is not None and shared_tools.get_mcp_server_names():
        mounted = mounted_tool_names or set()
        tools.register(
            MountServerTool(shared_tools, mounted),
            risk="read-only",
        )
        
    tools.register(
        DriftSendMessageTool(ctx, send_message_fn),
        risk="external-side-effect",
    )
    tools.register(
        FinishDriftTool(ctx, store),
        risk="write",
    )

    return tools