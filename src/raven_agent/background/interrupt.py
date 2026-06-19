"""
Agent Turn 中断系统 —— 控制面。

组件:
  TurnInterruptState  — 被中断 turn 的内存快照（含 partial reply / thinking / 工具链）
  InterruptResult     — request_interrupt() 的返回值
  InterruptManager    — 中断状态管理、/stop 处理、恢复上下文拼接

中断流程:
  Channel 识别 /stop → InterruptManager.request_interrupt()
  → 取消活跃 asyncio Task → 保存 TurnInterruptState
  → 下次消息到达 → InterruptManager.build_resume_content()
  → 拼接进度上下文 → 作为新的 user message 发给 LLM
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TTL_S = 1800  # 30 分钟


# ── TurnInterruptState ──────────────────────────────────────────────


@dataclass
class TurnInterruptState:
    """一个被中断的 turn 的快照，纯内存态，不落库。

    字段:
        session_key: 被中断的会话 key。
        original_user_message: 用户原始输入。
        original_metadata: 原始消息的 metadata 字典。
        partial_reply: 中断时 LLM 已被 streaming 输出的部分回复（若有）。
        partial_thinking: 中断时 LLM 已被 streaming 输出的部分 thinking（若有）。
        tools_used: 中断前已完成执行的工具名称列表。
        tool_chain_partial: 中断前已完成执行的工具调用详情列表。
        interrupted_by: 中断命令，默认 "/stop"。
        interrupted_at: 中断发生时的 monotonic 时间戳。
        ttl_seconds: 中断态存活秒数，超时后自动抛弃。
    """

    session_key: str
    original_user_message: str
    original_metadata: dict[str, Any] = field(default_factory=dict)
    partial_reply: str = ""
    partial_thinking: str | None = None
    tools_used: list[str] = field(default_factory=list)
    tool_chain_partial: list[dict[str, Any]] = field(default_factory=list)
    interrupted_by: str = "/stop"
    interrupted_at: float = field(default_factory=time.monotonic)
    ttl_seconds: int = _DEFAULT_TTL_S

    @property
    def expired(self) -> bool:
        """检查中断状态是否已过期。

        输出:
            True 表示已超过 ttl_seconds，应丢弃。
        """
        return (time.monotonic() - self.interrupted_at) > self.ttl_seconds


# ── InterruptResult ─────────────────────────────────────────────────


@dataclass
class InterruptResult:
    """request_interrupt() 的返回值。

    字段:
        status: "interrupted" 表示成功中断；"idle" 表示该 session 没有活跃 turn。
        session_key: 被操作的会话 key。
        message: 给用户的即时反馈。
    """

    status: str
    session_key: str = ""
    message: str = ""


# ── InterruptManager ────────────────────────────────────────────────


class InterruptManager:
    """中断管理器（纯内存态）。

    负责:
    - 跟踪每个 session_key 的活跃 asyncio Task
    - 执行 /stop 中断（cancel task + 保存中断态）
    - 读取/消费中断态（恢复上下文拼接）
    - TTL 过期清理

    参数:
        ttl_seconds: 中断态存活秒数，默认 1800（30 分钟）。
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_S) -> None:
        self._ttl = ttl_seconds
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

    # ── Active task tracking ────────────────────────────────────

    def track_task(
        self,
        session_key: str,
        task: asyncio.Task[Any],
        state: TurnInterruptState | None = None,
    ) -> None:
        """开始追踪一个 session 的活跃 turn task。

        输入:
            session_key: 会话 key。
            task: 正在运行的 asyncio Task。
            state: 初始 TurnInterruptState；若不提供则创建一个空的。

        输出:
            None。
        """
        self._active_tasks[session_key] = task
        self._turn_states[session_key] = state or TurnInterruptState(
            session_key=session_key,
            original_user_message="",
        )

    def untrack_task(self, session_key: str) -> None:
        """停止追踪某个 session 的活跃 task。

        输入:
            session_key: 会话 key。

        输出:
            None。
        """
        self._active_tasks.pop(session_key, None)
        self._turn_states.pop(session_key, None)

    # ── TurnState update helpers ────────────────────────────────

    def get_turn_state(self, session_key: str) -> TurnInterruptState | None:
        """获取当前活跃 turn 的状态快照（不消费）。

        输入:
            session_key: 会话 key。

        输出:
            TurnInterruptState；无活跃 turn 时返回 None。
        """
        return self._turn_states.get(session_key)

    def append_partial_reply(self, session_key: str, delta: str) -> None:
        """追加 streaming 回复片段到活跃 turn 状态。

        ⚠️ 当 LLMProvider 为非流式时（llm.py 中 stream=False），
        一次 await 等完整响应，不存在 streaming delta。因此当前阶段
        没有代码调用 append_partial_reply——partial_reply 始终为空字符串。
        
        此方法是预留给后续流式输出升级的基础设施。届时每个 streaming
        chunk 到达时调用本方法，/stop 时 partial_reply 即包含已输出内容。

        输入:
            session_key: 会话 key。
            delta: streaming 输出的增量文本。

        输出:
            None。
        """
        state = self._turn_states.get(session_key)
        if state is not None and delta:
            state.partial_reply += delta

    def append_partial_thinking(self, session_key: str, delta: str) -> None:
        """追加 streaming thinking 片段到活跃 turn 状态。

        输入:
            session_key: 会话 key。
            delta: streaming thinking 的增量文本。

        输出:
            None。
        """
        state = self._turn_states.get(session_key)
        if state is not None and delta:
            state.partial_thinking = (state.partial_thinking or "") + delta

    def record_tool_call(
        self,
        session_key: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """记录已完成执行的工具调用。

        输入:
            session_key: 会话 key。
            tool_name: 工具名称。
            arguments: 工具调用参数。

        输出:
            None。
        """
        state = self._turn_states.get(session_key)
        if state is not None:
            state.tools_used.append(tool_name)
            state.tool_chain_partial.append({
                "tool": tool_name,
                "arguments": arguments,
            })

    # ── Interrupt ───────────────────────────────────────────────

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """Channel 层调用的中断入口。

        取消该 session_key 的活跃 asyncio Task，
        并把当前的 TurnInterruptState 移入中断态。

        输入:
            session_key: 要中断的会话 key。
            sender: 发起中断的用户标识（仅用于日志）。
            command: 中断命令名称。

        输出:
            InterruptResult。status="interrupted" 表示成功；"idle" 表示没有活跃任务。
        """
        task = self._active_tasks.get(session_key)
        if task is None or task.done():
            return InterruptResult(
                status="idle",
                session_key=session_key,
                message="当前没有正在执行的任务。",
            )

        # 把当前的 active turn state 写入 interrupt_states
        active_state = self._turn_states.get(session_key)
        if active_state is None:
            active_state = TurnInterruptState(
                session_key=session_key,
                original_user_message="",
            )
        self._interrupt_states[session_key] = TurnInterruptState(
            session_key=session_key,
            original_user_message=active_state.original_user_message,
            original_metadata=dict(active_state.original_metadata),
            partial_reply=active_state.partial_reply,
            partial_thinking=active_state.partial_thinking,
            tools_used=list(active_state.tools_used),
            tool_chain_partial=list(active_state.tool_chain_partial),
            interrupted_by=command,
            interrupted_at=time.monotonic(),
            ttl_seconds=self._ttl,
        )

        # 取消 asyncio Task（CancelledError 在 pipeline 的 await 点抛出）
        task.cancel()
        logger.info(
            "Turn interrupted  session_key=%s  sender=%s  command=%s",
            session_key, sender, command,
        )
        return InterruptResult(
            status="interrupted",
            session_key=session_key,
            message="本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    # ── Resume ──────────────────────────────────────────────────

    def get_interrupt_state(
        self,
        session_key: str,
    ) -> TurnInterruptState | None:
        """读取中断态（含 TTL 过期检查），不消费。

        输入:
            session_key: 会话 key。

        输出:
            TurnInterruptState；无中断态或已过期时返回 None。
        """
        state = self._interrupt_states.get(session_key)
        if state is None:
            return None
        if state.expired:
            logger.info("Interrupt state expired for %s, discarding", session_key)
            self._interrupt_states.pop(session_key, None)
            return None
        return state

    def pop_interrupt_state(
        self,
        session_key: str,
    ) -> TurnInterruptState | None:
        """消费中断态（读取后移除）。

        输入:
            session_key: 会话 key。

        输出:
            TurnInterruptState；无中断态或已过期时返回 None。
        """
        state = self._interrupt_states.get(session_key)
        if state is None:
            return None
        if state.expired:
            self._interrupt_states.pop(session_key, None)
            return None
        self._interrupt_states.pop(session_key, None)
        return state

    def build_resume_content(
        self,
        state: TurnInterruptState,
        new_message: str,
    ) -> str:
        """把中断现场拼接为恢复上下文字符串。

        这不是"从断点继续执行"——而是把中断前的进度编成文本，
        作为新的用户消息的一部分发给 LLM。LLM 看到上下文后自然知道
        自己在继续做这件事。

        输入:
            state: 中断态快照。
            new_message: 用户的新消息（如"继续"）。

        输出:
            拼接后的用户消息字符串。
        """
        parts: list[str] = []
        parts.append("[之前的任务已通过 /stop 中断，以下是中断前的进度，请继续完成]")
        parts.append(f"原始请求: {state.original_user_message}")

        if state.tools_used:
            parts.append(
                f"已完成的工具调用: {', '.join(state.tools_used)}"
            )
            if state.tool_chain_partial:
                for entry in state.tool_chain_partial:
                    tool_name = entry.get("tool", "?")
                    args_str = str(entry.get("arguments", {}))
                    if len(args_str) > 120:
                        args_str = args_str[:120] + "..."
                    parts.append(f"  - {tool_name}({args_str})")

        if state.partial_reply:
            parts.append(f"已生成的部分回复: {state.partial_reply}")

        if state.partial_thinking:
            thinking_preview = state.partial_thinking[:200]
            parts.append(f"已生成的部分思考: {thinking_preview}")

        parts.append(f"新指令: {new_message}")
        return "\n\n".join(parts)