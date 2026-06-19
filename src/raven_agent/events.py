from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _metadata() -> dict[str, Any]:
    """创建空 metadata 字典。

    返回:
        新的空字典。
    """

    return {}


@dataclass(frozen=True)
class InboundMessage:
    """运行时内部的入站消息。

    参数:
        channel: 消息来源渠道，例如 cli、telegram、api。
        sender: 发送者标识。
        chat_id: 渠道内会话 ID。
        content: 用户输入内容。
        timestamp: 消息进入系统的时间。
        metadata: 附加信息，供 Channel 或后续 Runtime 使用。
        media: 附件文件路径列表，例如图片、文件。
    """

    channel: str
    sender: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=_metadata)
    media: list[str] = field(default_factory=list)

    @property
    def session_key(self) -> str:
        """返回全局唯一会话 key。

        返回:
            由 channel 和 chat_id 拼出的会话标识。
        """

        return f"{self.channel}:{self.chat_id}"

    @property
    def request_time(self) -> str:
        """返回 ISO 格式的消息到达时间，供 schedule 等工具使用。

        返回:
            self.timestamp 的 ISO 格式字符串。
        """

        return self.timestamp.isoformat()


@dataclass(frozen=True)
class OutboundMessage:
    """运行时内部的出站消息。

    参数:
        channel: 目标渠道。
        chat_id: 目标会话 ID。
        content: 要发送给用户的文本。
        metadata: 附加信息，供 Channel 或后续 Runtime 使用。
        media: 附件文件路径列表（图片、文件等）。
    """

    channel: str
    chat_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=_metadata)
    media: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class SpawnCompletionEvent:
    """本地 SubAgent 后台任务完成事件。

    字段:
        job_id: 后台 job ID。
        label: 用户可读的任务标签。
        task: 交给 SubAgent 的原始任务描述。
        status: 子任务语义状态，通常为 "completed" | "incomplete" | "error" | "cancelled"。
        exit_reason: 退出原因，例如 "completed"、"forced_summary"、"error"、"cancelled"。
        result: SubAgent 返回的结果摘要，可能已被裁剪。
        retry_count: 当前任务已重试次数；首次为 0。
        profile: 使用的工具 profile：research / scripting / general。
    """

    job_id: str
    label: str
    task: str
    status: str
    exit_reason: str
    result: str
    retry_count: int = 0
    profile: str = ""


@dataclass(frozen=True)
class TurnStarted:
    """一轮对话开始事件。

    参数:
        session_key: 当前会话 key。
        inbound: 本轮入站消息。
    """

    session_key: str
    inbound: InboundMessage


@dataclass(frozen=True)
class TurnCompleted:
    """一轮对话完成事件。

    参数:
        session_key: 当前会话 key。
        inbound: 本轮入站消息。
        outbound: 本轮出站消息。
        tools_used: 本轮使用过的工具名称列表。
    """

    session_key: str
    inbound: InboundMessage
    outbound: OutboundMessage
    tools_used: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamStart:
    """流式回复开始事件。

    参数:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
    """

    session_key: str
    channel: str
    chat_id: str


@dataclass(frozen=True)
class StreamToken:
    """流式回复单个 token 事件。

    参数:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        token: 本次增量文本（通常 1-10 个字符）。
    """

    session_key: str
    channel: str
    chat_id: str
    token: str


@dataclass(frozen=True)
class StreamEnd:
    """流式回复结束事件。

    参数:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
    """

    session_key: str
    channel: str
    chat_id: str


@dataclass(frozen=True)
class ToolCallStarted:
    """工具调用开始事件（供 Telegram Live 消息使用）。

    参数:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        iteration: 当前 ReAct 步数。
        call_id: 工具调用 ID。
        tool_name: 工具名称。
        arguments: 工具参数字典。
    """

    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallCompleted:
    """工具调用完成事件（供 Telegram Live 消息使用）。

    参数:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        iteration: 当前 ReAct 步数。
        call_id: 工具调用 ID。
        tool_name: 工具名称。
        arguments: 原始参数字典。
        final_arguments: hook 改写后的最终参数。
        status: 执行结果——"success" | "error" | "blocked"。
        result_preview: 结果摘要（最多 200 字符）。
    """

    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    final_arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    result_preview: str = ""


@dataclass(frozen=True)
class RetrievalCompleted:
    """一次 memory 检索完成事件。

    Observe 插件监听此事件，将检索详情写入 rag_queries 表。

    参数:
        caller: 调用来源——"passive" | "proactive" | "explicit"。
        session_key: 当前会话 key。
        query: 实际检索 query（rewrite 后）。
        orig_query: 改写前原始查询；None = 未改写。
        aux_queries: HyDE 生成的假想条目列表。
        hits: 命中的记忆条目摘要列表。
            每个元素为 dict: {item_id, memory_type, score, summary, injected, forced}。
        injected_count: 最终注入 Prompt 的条目数。
        route_decision: "RETRIEVE" | "NO_RETRIEVE" | None。
        error: 检索错误；正常为 None。
    """

    caller: str
    session_key: str
    query: str
    orig_query: str | None = None
    aux_queries: list[str] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    injected_count: int = 0
    route_decision: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MemoryWritten:
    """一次记忆写入或替换完成事件。

    Observe 插件监听此事件，将操作详情写入 memory_writes 表。

    参数:
        session_key: 当前会话 key。
        source_ref: 来源引用。
        action: 操作类型——"write" | "supersede"。
        memory_type: write 时的记忆类型（event/profile/preference/procedure）。
        item_id: write 时的条目 id（"new:xxx" 或 "reinforced:xxx"）。
        summary: write 时的摘要文本。
        superseded_ids: supersede 时的被取代 id 列表。
        error: 操作错误；正常为 None。
    """

    session_key: str
    source_ref: str = ""
    action: str = ""
    memory_type: str | None = None
    item_id: str | None = None
    summary: str | None = None
    superseded_ids: list[str] = field(default_factory=list)
    error: str | None = None