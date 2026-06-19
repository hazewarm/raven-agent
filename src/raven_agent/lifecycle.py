from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from raven_agent.events import InboundMessage
from raven_agent.messages import ChatMessage


def _empty_str_list() -> list[str]:
    """创建空字符串列表。

    输入:
        无。

    输出:
        新的空列表。
    """

    return []

def _empty_metadata() -> dict[str, Any]:
    """创建空 metadata 字典。

    输入:
        无。

    输出:
        新的空字典。
    """

    return {}

@dataclass
class BeforeTurnCtx:
    """before_turn 阶段的可写上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        content: 当前用户输入内容。
        metadata: 入站消息 metadata 副本。

    输出:
        BeforeTurnCtx 实例。插件可改写 content，或设置 abort / abort_reply 提前结束本轮。
    """

    session_key: str
    channel: str
    chat_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    abort: bool = False
    abort_reply: str = ""
    outbound_metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass
class BeforeReasoningCtx:
    """before_reasoning 阶段的可写上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        content: 当前用户输入。

    输出:
        BeforeReasoningCtx 实例。插件可写 extra_hints / disabled_sections / abort。
    """

    session_key: str
    channel: str
    chat_id: str
    content: str
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    disabled_sections: set[str] = field(default_factory=set)
    abort: bool = False
    abort_reply: str = ""
    outbound_metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass
class PromptSection:
    """注入 prompt 的一段 system 内容。

    输入:
        name: section 名称，便于去重和调试。
        content: section 文本内容。

    输出:
        PromptSection 实例。
    """

    name: str
    content: str


@dataclass
class PromptRenderCtx:
    """prompt_render 阶段的可写上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        content: 当前用户输入。
        disabled_sections: before_reasoning 传下来的禁用 section 名集合。

    输出:
        PromptRenderCtx 实例。插件可向 system_sections_top / system_sections_bottom / extra_hints 追加内容。
    """

    session_key: str
    channel: str
    chat_id: str
    content: str
    disabled_sections: set[str] = field(default_factory=set)
    system_sections_top: list[PromptSection] = field(default_factory=list)
    system_sections_bottom: list[PromptSection] = field(default_factory=list)
    extra_hints: list[str] = field(default_factory=_empty_str_list)


@dataclass
class BeforeStepCtx:
    """每次 ReAct step 调模型前的可写上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        iteration: 当前 step 序号，从 1 开始。
        visible_tool_names: 本轮对模型可见的工具名集合。
        context_tokens_estimate: 本 step 输入消息的粗略 token 估算。

    输出:
        BeforeStepCtx 实例。插件可写 extra_hints；置 early_stop 提前结束工具循环。
    """

    session_key: str
    channel: str
    chat_id: str
    iteration: int
    visible_tool_names: frozenset[str]
    context_tokens_estimate: int = 0
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass
class AfterStepCtx:
    """每次 ReAct step 工具执行后的观察上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        iteration: 当前 step 序号。
        tools_called: 本 step 实际调用的工具名元组。
        partial_reply: 本 step 模型给出的文本（可能为空）。
        has_more: 本 step 后是否还会继续工具循环。
        context_tokens_estimate: 本 step 结束时上下文的粗略 token 估算。

    输出:
        AfterStepCtx 实例。本阶段为观察快照，插件读取为主。
    """

    session_key: str
    channel: str
    chat_id: str
    iteration: int
    tools_called: tuple[str, ...]
    partial_reply: str
    has_more: bool
    context_tokens_estimate: int = 0
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AfterReasoningCtx:
    """after_reasoning 阶段的可写上下文。

    输入:
        session_key: 当前会话 key。
        channel: 渠道名。
        chat_id: 聊天标识。
        tools_used: 本轮使用过的工具名元组。
        reply: 模型最终回复；插件可改写（例如清理协议标签）。

    输出:
        AfterReasoningCtx 实例。插件可改写 reply，并向 outbound_metadata 追加字段。
    """

    session_key: str
    channel: str
    chat_id: str
    tools_used: tuple[str, ...]
    reply: str
    outbound_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AfterTurnCtx:
    """after_turn 阶段的观察上下文。

    输入:
        session_key: 当前会话 key。
        channel: 出站渠道。
        chat_id: 出站聊天标识。
        reply: 最终回复文本。
        tools_used: 本轮使用过的工具名元组。
        outbound_metadata: 出站消息 metadata 副本。

    输出:
        AfterTurnCtx 实例。插件通常读取它；也可向 extra_metadata 追加观察数据。
    """

    session_key: str
    channel: str
    chat_id: str
    reply: str
    tools_used: tuple[str, ...]
    outbound_metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    extra_metadata: dict[str, Any] = field(default_factory=_empty_metadata)



class ReasoningLifecycle(Protocol):
    """ReAct step 生命周期协议。

    输入:
        无。实现者负责运行 before_step / after_step 模块链。

    输出:
        结构化协议类型，自身不实例化。
    """

    async def before_step(self, ctx: BeforeStepCtx) -> BeforeStepCtx:
        """在每次调模型前运行 before_step 模块链。

        输入:
            ctx: 当前 BeforeStepCtx。

        输出:
            可能被插件改写后的 BeforeStepCtx。
        """

        ...

    async def after_step(self, ctx: AfterStepCtx) -> AfterStepCtx:
        """在每次工具执行后运行 after_step 模块链。

        输入:
            ctx: 当前 AfterStepCtx。

        输出:
            可能被插件补充 metadata 后的 AfterStepCtx。
        """

        ...


@dataclass
class LifecycleModules:
    """插件向七段式生命周期注入的模块集合。

    输入:
        before_turn / before_reasoning / prompt_render / before_step /
        after_step / after_reasoning / after_turn: 各阶段的 PhaseModule 列表。

    输出:
        LifecycleModules 实例。默认全部为空列表，等价于无插件。
    """

    before_turn: list[object] = field(default_factory=list)
    before_reasoning: list[object] = field(default_factory=list)
    prompt_render: list[object] = field(default_factory=list)
    before_step: list[object] = field(default_factory=list)
    after_step: list[object] = field(default_factory=list)
    after_reasoning: list[object] = field(default_factory=list)
    after_turn: list[object] = field(default_factory=list)