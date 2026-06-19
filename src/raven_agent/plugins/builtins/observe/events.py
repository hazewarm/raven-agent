from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnTrace:
    """一轮对话的 observe 记录。

    输入:
        session_key: 当前会话 key。
        channel: 出站渠道。
        chat_id: 出站聊天标识。
        user_msg: 用户原文。
        reply: 最终回复文本。
        tools_used: 本轮使用过的工具名列表。
        iterations: 本轮 LLM 调用次数。
        cited_memory_ids: citation 插件解析出的引用 id 列表。
        error: 错误文本；正常为 None。

    输出:
        TurnTrace 实例。
    """

    session_key: str
    channel: str
    chat_id: str
    user_msg: str
    reply: str
    tools_used: list[str] = field(default_factory=list)
    iterations: int | None = None
    cited_memory_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ToolCallTrace:
    """一次工具调用的 observe 记录。

    输入:
        session_key: 当前会话 key。
        tool_name: 工具名。
        arguments: 工具参数。
        status: 工具执行状态，success / error / denied。
        plugin_source: 触发该 trace 的插件来源说明。
        error: 错误文本；正常为 None。

    输出:
        ToolCallTrace 实例。
    """

    session_key: str
    tool_name: str
    arguments: dict[str, object] = field(default_factory=dict)
    status: str = "success"
    plugin_source: str = ""
    error: str | None = None

@dataclass
class RagHitLog:
    """一次检索中命中的单条记忆条目。

    输入:
        item_id: 命中的记忆条目 id。
        memory_type: 记忆类型（event / profile / preference / procedure）。
        score: 检索相似度分数。
        summary: 记忆摘要文本（截断至 120 字符）。
        injected: 是否最终注入到 Prompt 上下文。
        confidence_label: 置信度标签，空串表示正常。
        forced: 是否因 tool_requirement 强制注入（非 score 过阈值）。

    输出:
        RagHitLog 实例。
    """

    item_id: str
    memory_type: str
    score: float
    summary: str
    injected: bool
    confidence_label: str = ""
    forced: bool = False


@dataclass
class RagQueryLog:
    """一次 memory 检索事件的完整记录。

    输入:
        caller: 调用来源——
               "passive"（被动对话的 before_reasoning 检索）、
               "proactive"（主动推送的检索）、
               "explicit"（recall_memory 工具触发的手动检索）。
        session_key: 当前会话 key。
        query: 实际检索用的 query（query rewriter 改写后的版本）。
        orig_query: 改写前的原始查询文本；None 表示未改写。
        aux_queries: HyDE 增强器生成的假想条目列表，用于向量检索。
        hits: 本次检索命中的所有条目（RagHitLog 列表）。
        injected_count: 最终注入到 Prompt 的条目数。
        route_decision: 检索 gate 的路由决策——
                        "RETRIEVE"（执行检索）、"NO_RETRIEVE"（跳过）、
                        None（未启用 gate）。
        error: 检索过程中发生的错误；正常为 None。

    输出:
        RagQueryLog 实例。
    """

    caller: str
    session_key: str
    query: str
    orig_query: str | None = None
    aux_queries: list[str] = field(default_factory=list)
    hits: list[RagHitLog] = field(default_factory=list)
    injected_count: int = 0
    route_decision: str | None = None
    error: str | None = None


@dataclass
class MemoryWriteTrace:
    """记忆写入或替换操作的 observe 记录。

    输入:
        session_key: 当前会话 key。
        source_ref: 来源引用（例如 "session:cli:default:turn:42"）。
        action: 操作类型——"write"（新写入记忆）或 "supersede"（替换旧记忆）。
        memory_type: write 操作时：写入的记忆类型
                     （event / profile / preference / procedure）。
        item_id: write 操作时：新条目 id（格式 "new:xxx" 或 "reinforced:xxx"）；
                 supersede 操作时为 None。
        summary: write 操作时：写入条目的摘要文本。
        superseded_ids: supersede 操作时：被取代的旧条目 id 列表。
        error: 写入过程中发生的错误；正常为 None。

    输出:
        MemoryWriteTrace 实例。
    """

    session_key: str
    source_ref: str = ""
    action: str = ""
    memory_type: str | None = None
    item_id: str | None = None
    summary: str | None = None
    superseded_ids: list[str] = field(default_factory=list)
    error: str | None = None

