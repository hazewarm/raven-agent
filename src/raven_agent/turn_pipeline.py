from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import cast

from raven_agent.agent import AgentRunResult, ReactAgent
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, OutboundMessage, TurnCompleted, TurnStarted, StreamStart, StreamEnd
from raven_agent.messages import ChatMessage
from raven_agent.phase import Phase, PhaseFrame, append_string_exports, collect_prefixed_slots
from raven_agent.prompt import PromptBuilder
from raven_agent.session import Session, SessionManager

from raven_agent.lifecycle import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
    LifecycleModules,
    PromptRenderCtx,
    PromptSection,
    ReasoningLifecycle,
)

from raven_agent.media import encode_image_to_data_uri
from raven_agent.messages import MediaItem

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from raven_agent.background.interrupt import InterruptManager


@dataclass(frozen=True)
class PassiveTurnPipelineDeps:
    """被动 Turn Pipeline 的外部依赖。

    输入:
        sessions: 会话管理器。
        prompt_builder: 构造模型输入消息的 PromptBuilder。
        agent: 执行 ReAct 推理的 ReactAgent。
        event_bus: 触发 TurnStarted / TurnCompleted 的 EventBus。
        lifecycle_modules: 插件向七段式生命周期注入的模块集合。
        interrupt_manager: 中断管理器。
        multimodal: 主模型是否原生支持多模态（来自 config.vl.multimodal）。
            True 时 Path A 生效——图片编码后注入主模型消息。
            False 时跳过注入，图片由 Path B（read_image_vision 工具）处理。

    输出:
        不可变依赖容器。
    """

    sessions: SessionManager
    prompt_builder: PromptBuilder
    agent: ReactAgent
    event_bus: EventBus
    lifecycle_modules: LifecycleModules = field(default_factory=LifecycleModules)
    interrupt_manager: "InterruptManager | None" = None
    multimodal: bool = True


# ——定义 pipeline 内部数据对象————————————————————
# PassiveTurnState 可变dataclass，其他均不可变
# 因为 BeforeTurnPhase 会允许事件 handler 改写 inbound：
@dataclass
class PassiveTurnState:
    """被动 turn 执行期间流转的状态。

    输入:
        inbound: 当前轮入站消息。
        session: 当前 session_key 对应的 Session。
        extra_hints: before_turn / before_reasoning 阶段插件追加的提示。
        disabled_sections: before_reasoning 阶段插件禁用的 section 名集合。
        abort: 是否跳过 LLM 推理。
        abort_reply: 跳过推理时直接返回的文本。
        outbound_metadata: 插件提前写入的出站 metadata。

    输出:
        PassiveTurnState 实例。
    """

    inbound: InboundMessage
    session: Session | None = None
    extra_hints: list[str] = field(default_factory=list)
    disabled_sections: set[str] = field(default_factory=set)
    abort: bool = False
    abort_reply: str = ""
    outbound_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptRenderResult:
    """PromptRenderPhase 的输出。

    参数:
        messages: 已构造完成、准备发送给 ReactAgent 的消息列表。
        session_key: 当前会话 key，用于 ReactAgent 隔离工具发现状态。
        tool_context: 每轮工具上下文，例如 current_user_source_ref / channel / chat_id。
    """

    messages: list[ChatMessage]
    session_key: str
    tool_context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningResult:
    """ReasoningPhase 的输出。

    参数:
        result: ReactAgent.run() 返回的推理结果。
    """

    result: AgentRunResult


@dataclass(frozen=True)
class AfterTurnInput:
    """AfterTurnPhase 的输入。

    输入:
        inbound: 当前轮最终采用的入站消息。
        session: 当前轮使用的 Session。
        reasoning: ReasoningPhase 得到的 AgentRunResult。
        reply: after_reasoning 清理后的回复文本。
        outbound_metadata: after_reasoning 追加的 outbound metadata。

    输出:
        AfterTurnInput 实例。
    """

    inbound: InboundMessage
    session: Session
    reasoning: AgentRunResult
    reply: str
    outbound_metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class AfterReasoningResult:
    """AfterReasoningPhase 的输出。

    输入:
        reply: 经插件清理后的最终回复文本。
        outbound_metadata: 插件追加的 outbound metadata。
        reasoning: 原始 AgentRunResult。

    输出:
        AfterReasoningResult 实例。
    """

    reply: str
    outbound_metadata: dict[str, object]
    reasoning: AgentRunResult



# ——定义四个 PhaseFrame——————————————————————

# 新增 BeforeReasoning 的 frame 与模块
@dataclass
class BeforeReasoningFrame(PhaseFrame[PassiveTurnState, PassiveTurnState]):
    """BeforeReasoningPhase 使用的数据帧。

    输入:
        input: 当前被动 turn 状态。
        slots: 模块之间共享的临时数据。
        output: 写入 hints / disabled_sections 后的被动 turn 状态。

    输出:
        BeforeReasoningFrame 实例。
    """


_BEFORE_REASONING_CTX_SLOT = "before_reasoning:ctx"
_BEFORE_REASONING_HINT_PREFIX = "before_reasoning:extra_hint:"
_BEFORE_REASONING_METADATA_PREFIX = "before_reasoning:metadata:"
_BEFORE_REASONING_ABORT_REPLY_SLOT = "before_reasoning:abort_reply"

class _EmitBeforeReasoningCtxModule:
    """通过 EventBus 触发 BeforeReasoningCtx GATE 事件。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _EmitBeforeReasoningCtxModule 实例。
    """

    slot = "before_reasoning.emit"
    requires = ("before_reasoning.build_ctx", _BEFORE_REASONING_CTX_SLOT)
    # produces = (_BEFORE_REASONING_CTX_SLOT,)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        """运行所有 @on_before_reasoning handler。

        输入:
            frame: 当前 BeforeReasoningFrame。

        输出:
            写回可能被 handler 改写后的 BeforeReasoningFrame。
        """

        ctx = cast(BeforeReasoningCtx, frame.slots[_BEFORE_REASONING_CTX_SLOT])
        frame.slots[_BEFORE_REASONING_CTX_SLOT] = await self._event_bus.emit(ctx)
        return frame


class _BuildBeforeReasoningCtxModule:
    """构造 BeforeReasoningCtx 的内置模块。

    输入:
        无。

    输出:
        _BuildBeforeReasoningCtxModule 实例。
    """

    slot = "before_reasoning.build_ctx"
    produces = (_BEFORE_REASONING_CTX_SLOT,)

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        """把当前 turn 信息打包成 BeforeReasoningCtx。

        输入:
            frame: 当前 BeforeReasoningFrame。

        输出:
            写入 ctx slot 后的 BeforeReasoningFrame。
        """

        inbound = frame.input.inbound
        frame.slots[_BEFORE_REASONING_CTX_SLOT] = BeforeReasoningCtx(
            session_key=inbound.session_key,
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            content=inbound.content,
            extra_hints=list(frame.input.extra_hints),
        )
        return frame


class _CollectBeforeReasoningModule:
    """收集插件导出的 hint slot 并写回 turn 状态的内置模块。

    输入:
        无。

    输出:
        _CollectBeforeReasoningModule 实例。
    """

    slot = "before_reasoning.collect"
    requires = ("before_reasoning.build_ctx", _BEFORE_REASONING_CTX_SLOT)

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        """把 ctx 与 prefix slot 中的 hints / disabled_sections / abort 写回状态。

        输入:
            frame: 当前 BeforeReasoningFrame。

        输出:
            写入 output 的 BeforeReasoningFrame。
        """

        ctx = cast(BeforeReasoningCtx, frame.slots[_BEFORE_REASONING_CTX_SLOT])
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _BEFORE_REASONING_HINT_PREFIX),
        )
        metadata = dict(ctx.outbound_metadata)
        metadata.update(collect_prefixed_slots(frame.slots, _BEFORE_REASONING_METADATA_PREFIX))
        abort_reply = frame.slots.get(_BEFORE_REASONING_ABORT_REPLY_SLOT)
        if isinstance(abort_reply, str) and abort_reply.strip():
            ctx.abort = True
            ctx.abort_reply = abort_reply.strip()

        frame.input.extra_hints = list(ctx.extra_hints)
        frame.input.disabled_sections = set(ctx.disabled_sections)
        frame.input.abort = ctx.abort
        frame.input.abort_reply = ctx.abort_reply
        frame.input.outbound_metadata.update(metadata)
        frame.output = frame.input
        return frame






@dataclass
class BeforeTurnFrame(PhaseFrame[PassiveTurnState, PassiveTurnState]):
    """BeforeTurnPhase 使用的数据帧。

    参数:
        input: 当前被动 turn 状态。
        slots: 模块之间共享的临时数据。
        output: BeforeTurnPhase 输出的被动 turn 状态。
    """


@dataclass
class PromptRenderFrame(PhaseFrame[PassiveTurnState, PromptRenderResult]):
    """PromptRenderPhase 使用的数据帧。

    参数:
        input: 当前被动 turn 状态。
        slots: 模块之间共享的临时数据。
        output: PromptRenderPhase 输出的 PromptRenderResult。
    """


@dataclass
class ReasoningFrame(PhaseFrame[PromptRenderResult, ReasoningResult]):
    """ReasoningPhase 使用的数据帧。

    参数:
        input: PromptRenderPhase 输出的 PromptRenderResult。
        slots: 模块之间共享的临时数据。
        output: ReasoningPhase 输出的 ReasoningResult。
    """


@dataclass
class AfterTurnFrame(PhaseFrame[AfterTurnInput, OutboundMessage]):
    """AfterTurnPhase 使用的数据帧。

    参数:
        input: AfterTurnPhase 所需的入站消息和推理结果。
        slots: 模块之间共享的临时数据。
        output: 当前 turn 最终产生的 OutboundMessage。
    """

# 定义内部 slot key常量
_PROMPT_MESSAGES_SLOT = "prompt_render:messages"
_AGENT_RESULT_SLOT = "reasoning:result"
_OUTBOUND_SLOT = "after_turn:outbound"
_TOOL_CONTEXT_SLOT = "prompt_render:tool_context"


# before_step / after_step
_BEFORE_STEP_CTX_SLOT = "before_step:ctx"
_BEFORE_STEP_HINT_PREFIX = "before_step:extra_hint:"
_BEFORE_STEP_ABORT_REPLY_SLOT = "before_step:abort_reply"

_AFTER_STEP_CTX_SLOT = "after_step:ctx"
_AFTER_STEP_TELEMETRY_PREFIX = "after_step:telemetry:"
_AFTER_STEP_COLLECTED_TELEMETRY_SLOT = "after_step:telemetry_collected"
_AFTER_STEP_EARLY_STOP_REASON_SLOT = "after_step:early_stop_reason"


@dataclass
class BeforeStepFrame(PhaseFrame[BeforeStepCtx, BeforeStepCtx]):
    """BeforeStepPhase 使用的数据帧。

    输入:
        input: 当前 BeforeStepCtx。
        slots: 模块之间共享的临时数据。
        output: 可能被插件改写后的 BeforeStepCtx。

    输出:
        BeforeStepFrame 实例。
    """

class _CopyBeforeStepCtxModule:
    """把 BeforeStepCtx 输入复制到 before_step:ctx slot。

    输入:
        无。

    输出:
        _CopyBeforeStepCtxModule 实例。
    """

    slot = "before_step.copy_ctx"
    produces = (_BEFORE_STEP_CTX_SLOT,)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        """写入 before_step:ctx。

        输入:
            frame: 当前 BeforeStepFrame。

        输出:
            写入 ctx slot 后的 BeforeStepFrame。
        """

        frame.slots[_BEFORE_STEP_CTX_SLOT] = frame.input
        return frame


class _EmitBeforeStepCtxModule:
    """通过 EventBus 触发 BeforeStepCtx GATE 事件。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _EmitBeforeStepCtxModule 实例。
    """

    slot = "before_step.emit"
    requires = ("before_step.copy_ctx", _BEFORE_STEP_CTX_SLOT)
    # produces = (_BEFORE_STEP_CTX_SLOT,)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        """运行所有 @on_before_step handler。

        输入:
            frame: 当前 BeforeStepFrame。

        输出:
            写回可能被 handler 改写后的 BeforeStepFrame。
        """

        ctx = cast(BeforeStepCtx, frame.slots[_BEFORE_STEP_CTX_SLOT])
        frame.slots[_BEFORE_STEP_CTX_SLOT] = await self._event_bus.emit(ctx)
        return frame


class _CollectBeforeStepModule:
    """收集 before_step prefix slots。

    输入:
        无。

    输出:
        _CollectBeforeStepModule 实例。
    """

    slot = "before_step.collect"
    requires = ("before_step.emit", _BEFORE_STEP_CTX_SLOT)
    # produces = (_BEFORE_STEP_CTX_SLOT,)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        """把 extra_hint / abort_reply slot 合并进 BeforeStepCtx。

        输入:
            frame: 当前 BeforeStepFrame。

        输出:
            更新 ctx slot 后的 BeforeStepFrame。
        """

        ctx = cast(BeforeStepCtx, frame.slots[_BEFORE_STEP_CTX_SLOT])
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _BEFORE_STEP_HINT_PREFIX),
        )
        early_stop_reply = frame.slots.get(_BEFORE_STEP_ABORT_REPLY_SLOT)
        if isinstance(early_stop_reply, str) and early_stop_reply.strip():
            ctx.early_stop = True
            ctx.early_stop_reply = early_stop_reply.strip()
        return frame


class _ReturnBeforeStepCtxModule:
    """返回 BeforeStepCtx。

    输入:
        无。

    输出:
        _ReturnBeforeStepCtxModule 实例。
    """

    slot = "before_step.return"
    requires = ("before_step.collect", _BEFORE_STEP_CTX_SLOT)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        """把 before_step:ctx 写入 output。

        输入:
            frame: 当前 BeforeStepFrame。

        输出:
            写入 output 后的 BeforeStepFrame。
        """

        frame.output = cast(BeforeStepCtx, frame.slots[_BEFORE_STEP_CTX_SLOT])
        return frame





@dataclass
class AfterStepFrame(PhaseFrame[AfterStepCtx, AfterStepCtx]):
    """AfterStepPhase 使用的数据帧。

    输入:
        input: 当前 AfterStepCtx。
        slots: 模块之间共享的临时数据。
        output: 可能被插件补充后的 AfterStepCtx。

    输出:
        AfterStepFrame 实例。
    """

class _CopyAfterStepCtxModule:
    """把 AfterStepCtx 输入复制到 after_step:ctx slot。

    输入:
        无。

    输出:
        _CopyAfterStepCtxModule 实例。
    """

    slot = "after_step.copy_ctx"
    produces = (_AFTER_STEP_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        """写入 after_step:ctx。

        输入:
            frame: 当前 AfterStepFrame。

        输出:
            写入 ctx slot 后的 AfterStepFrame。
        """

        frame.slots[_AFTER_STEP_CTX_SLOT] = frame.input
        return frame


class _CollectAfterStepTelemetryModule:
    """收集 after_step telemetry slots。

    输入:
        slot: 当前 collect 模块的 slot 名。
        requires: 当前 collect 模块的依赖。

    输出:
        _CollectAfterStepTelemetryModule 实例。
    """

    # produces = (_AFTER_STEP_CTX_SLOT,)

    def __init__(self, *, slot: str, requires: tuple[str, ...]) -> None:
        self.slot = slot
        self.requires = requires

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        """把 after_step:telemetry:* 合并进 ctx.extra_metadata。

        输入:
            frame: 当前 AfterStepFrame。

        输出:
            更新 ctx slot 后的 AfterStepFrame。
        """

        ctx = cast(AfterStepCtx, frame.slots[_AFTER_STEP_CTX_SLOT])
        collected = set(cast(set[str], frame.slots.get(_AFTER_STEP_COLLECTED_TELEMETRY_SLOT, set())))
        exports = collect_prefixed_slots(frame.slots, _AFTER_STEP_TELEMETRY_PREFIX)
        fresh = {key: value for key, value in exports.items() if key not in collected}
        ctx.extra_metadata.update(fresh)
        reason = frame.slots.get(_AFTER_STEP_EARLY_STOP_REASON_SLOT)
        if isinstance(reason, str) and reason.strip():
            ctx.extra_metadata["early_stop_reason"] = reason.strip()
        frame.slots[_AFTER_STEP_COLLECTED_TELEMETRY_SLOT] = collected | set(fresh)
        return frame


class _ObserveAfterStepCtxModule:
    """通过 EventBus 观察 AfterStepCtx。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _ObserveAfterStepCtxModule 实例。
    """

    slot = "after_step.observe"
    requires = ("after_step.collect_pre", _AFTER_STEP_CTX_SLOT)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        """运行所有 @on_after_step handler。

        输入:
            frame: 当前 AfterStepFrame。

        输出:
            原 frame。observe 模式忽略 handler 返回值。
        """

        ctx = cast(AfterStepCtx, frame.slots[_AFTER_STEP_CTX_SLOT])
        await self._event_bus.observe(ctx)
        return frame


class _ReturnAfterStepCtxModule:
    """返回 AfterStepCtx。

    输入:
        无。

    输出:
        _ReturnAfterStepCtxModule 实例。
    """

    slot = "after_step.return"
    requires = ("after_step.collect_post", _AFTER_STEP_CTX_SLOT)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        """把 after_step:ctx 写入 output。

        输入:
            frame: 当前 AfterStepFrame。

        输出:
            写入 output 后的 AfterStepFrame。
        """

        frame.output = cast(AfterStepCtx, frame.slots[_AFTER_STEP_CTX_SLOT])
        return frame



# ——实现 BeforeTurnPhase 的模块————————————————

_BEFORE_TURN_CTX_SLOT = "before_turn:ctx"
_BEFORE_TURN_HINT_PREFIX = "before_turn:extra_hint:"
_BEFORE_TURN_METADATA_PREFIX = "before_turn:metadata:"
_BEFORE_TURN_ABORT_REPLY_SLOT = "before_turn:abort_reply"


class _EmitTurnStartedModule:
    """触发 TurnStarted 事件的模块。

    参数:
        event_bus: 用于触发 TurnStarted 的 EventBus。
    """

    slot = "before_turn.emit_started"

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """触发 TurnStarted，并允许 handler 改写 inbound。

        参数:
            frame: 当前 BeforeTurnFrame。

        返回:
            更新 inbound 后的 BeforeTurnFrame。
        """

        inbound = frame.input.inbound
        started = await self._event_bus.emit(
            TurnStarted(
                session_key=inbound.session_key,
                inbound=inbound,
            )
        )
        frame.input.inbound = started.inbound
        return frame

class _LoadSessionModule:
    """根据入站消息加载 Session 的模块。

    参数:
        sessions: SessionManager。
    """

    slot = "before_turn.load_session"
    requires = ("before_turn.emit_started",)

    def __init__(self, sessions: SessionManager) -> None:
        self._sessions = sessions

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """根据 inbound.session_key 获取 Session。

        参数:
            frame: 当前 BeforeTurnFrame。

        返回:
            写入 session 后的 BeforeTurnFrame。
        """

        frame.input.session = self._sessions.get_or_create(frame.input.inbound.session_key)
        return frame

class _BuildBeforeTurnCtxModule:
    """构造 BeforeTurnCtx 的内置模块。

    输入:
        无。

    输出:
        _BuildBeforeTurnCtxModule 实例。
    """

    slot = "before_turn.build_ctx"
    requires = ("before_turn.load_session",)
    produces = (_BEFORE_TURN_CTX_SLOT,)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """把当前入站消息打包成 BeforeTurnCtx。

        输入:
            frame: 当前 BeforeTurnFrame。

        输出:
            写入 before_turn:ctx slot 后的 BeforeTurnFrame。
        """

        inbound = frame.input.inbound
        frame.slots[_BEFORE_TURN_CTX_SLOT] = BeforeTurnCtx(
            session_key=inbound.session_key,
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            content=inbound.content,
            metadata=dict(inbound.metadata),
        )
        return frame

class _EmitBeforeTurnCtxModule:
    """通过 EventBus 触发 BeforeTurnCtx GATE 事件。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _EmitBeforeTurnCtxModule 实例。
    """

    slot = "before_turn.emit"
    requires = ("before_turn.build_ctx", _BEFORE_TURN_CTX_SLOT)
    # produces = (_BEFORE_TURN_CTX_SLOT,)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """运行所有 @on_before_turn handler。

        输入:
            frame: 当前 BeforeTurnFrame。

        输出:
            写回可能被 handler 改写后的 BeforeTurnFrame。
        """

        ctx = cast(BeforeTurnCtx, frame.slots[_BEFORE_TURN_CTX_SLOT])
        frame.slots[_BEFORE_TURN_CTX_SLOT] = await self._event_bus.emit(ctx)
        return frame

class _CollectBeforeTurnModule:
    """收集 before_turn prefix slots 并写回 PassiveTurnState。

    输入:
        无。

    输出:
        _CollectBeforeTurnModule 实例。
    """

    slot = "before_turn.collect"
    requires = ("before_turn.emit", _BEFORE_TURN_CTX_SLOT)
    # produces = (_BEFORE_TURN_CTX_SLOT,)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """把 ctx / prefix slots 合并回 turn state。

        输入:
            frame: 当前 BeforeTurnFrame。

        输出:
            更新 PassiveTurnState 后的 BeforeTurnFrame。
        """

        ctx = cast(BeforeTurnCtx, frame.slots[_BEFORE_TURN_CTX_SLOT])
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _BEFORE_TURN_HINT_PREFIX),
        )
        metadata = dict(ctx.outbound_metadata)
        metadata.update(collect_prefixed_slots(frame.slots, _BEFORE_TURN_METADATA_PREFIX))
        abort_reply = frame.slots.get(_BEFORE_TURN_ABORT_REPLY_SLOT)
        if isinstance(abort_reply, str) and abort_reply.strip():
            ctx.abort = True
            ctx.abort_reply = abort_reply.strip()

        inbound = frame.input.inbound
        if ctx.content != inbound.content or ctx.metadata != inbound.metadata:
            frame.input.inbound = InboundMessage(
                channel=inbound.channel,
                sender=inbound.sender,
                chat_id=inbound.chat_id,
                content=ctx.content,
                timestamp=inbound.timestamp,
                metadata=dict(ctx.metadata),
            )

        frame.input.extra_hints.extend(ctx.extra_hints)
        frame.input.abort = ctx.abort
        frame.input.abort_reply = ctx.abort_reply
        frame.input.outbound_metadata.update(metadata)
        return frame

class _ReturnBeforeTurnModule:
    """返回 BeforeTurnPhase 输出的模块。

    输入:
        无。

    输出:
        _ReturnBeforeTurnModule 实例。
    """

    slot = "before_turn.return"
    requires = ("before_turn.collect", _BEFORE_TURN_CTX_SLOT)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        """把当前 PassiveTurnState 写入 output。

        输入:
            frame: 当前 BeforeTurnFrame。

        输出:
            写入 output 后的 BeforeTurnFrame。
        """

        frame.output = frame.input
        return frame


# ——实现 PromptRenderPhase 的模块——————————————————
_PROMPT_RENDER_CTX_SLOT = "prompt_render:ctx"
_PROMPT_SECTION_TOP_PREFIX = "prompt_render:section_top:"
_PROMPT_SECTION_BOTTOM_PREFIX = "prompt_render:section_bottom:"
_PROMPT_HINT_PREFIX = "prompt_render:extra_hint:"

class _EmitPromptRenderCtxModule:
    """通过 EventBus 触发 PromptRenderCtx GATE 事件。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _EmitPromptRenderCtxModule 实例。
    """

    slot = "prompt_render.emit"
    requires = ("prompt_render.build_ctx", _PROMPT_RENDER_CTX_SLOT)
    # produces = (_PROMPT_RENDER_CTX_SLOT,)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        """运行所有 @on_prompt_render handler。

        输入:
            frame: 当前 PromptRenderFrame。

        输出:
            写回可能被 handler 改写后的 PromptRenderFrame。
        """

        ctx = cast(PromptRenderCtx, frame.slots[_PROMPT_RENDER_CTX_SLOT])
        frame.slots[_PROMPT_RENDER_CTX_SLOT] = await self._event_bus.emit(ctx)
        return frame

class _BuildPromptRenderCtxModule:
    """构造 PromptRenderCtx 的内置模块。

    输入:
        无。

    输出:
        _BuildPromptRenderCtxModule 实例。
    """

    slot = "prompt_render.build_ctx"
    produces = (_PROMPT_RENDER_CTX_SLOT,)

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        """把当前 turn 信息打包成 PromptRenderCtx。

        输入:
            frame: 当前 PromptRenderFrame。

        输出:
            写入 ctx slot 后的 PromptRenderFrame。
        """

        state = frame.input
        if state.session is None:
            raise RuntimeError("PromptRender requires loaded session")
        frame.slots[_PROMPT_RENDER_CTX_SLOT] = PromptRenderCtx(
            session_key=state.inbound.session_key,
            channel=state.inbound.channel,
            chat_id=state.inbound.chat_id,
            content=state.inbound.content,
            disabled_sections=set(state.disabled_sections),
        )
        return frame



class _BuildPromptMessagesModule:
    """构造 LLM 输入消息并计算每轮工具上下文的模块。

    输入:
        prompt_builder: 用于构造 ChatMessage 列表的 PromptBuilder。
        sessions: 用于预测当前用户消息 message id 的 SessionManager。

    输出:
        _BuildPromptMessagesModule 实例。
    """

    slot = "prompt_render.build_messages"
    requires = ("prompt_render.emit", _PROMPT_RENDER_CTX_SLOT)
    produces = (_PROMPT_MESSAGES_SLOT, _TOOL_CONTEXT_SLOT)

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        sessions: SessionManager,
        multimodal: bool = True,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._sessions = sessions
        self._multimodal = multimodal

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        """收集插件 section 后构造消息列表并计算 tool_context。

        输入:
            frame: 当前 PromptRenderFrame。

        输出:
            写入 prompt messages 与 tool_context slot 后的 PromptRenderFrame。
        """

        state = frame.input
        session = state.session
        if session is None:
            raise RuntimeError("PromptRender requires loaded session")
        ctx = cast(PromptRenderCtx, frame.slots[_PROMPT_RENDER_CTX_SLOT])
        _append_sections(
            ctx.system_sections_top,
            collect_prefixed_slots(frame.slots, _PROMPT_SECTION_TOP_PREFIX),
        )
        _append_sections(
            ctx.system_sections_bottom,
            collect_prefixed_slots(frame.slots, _PROMPT_SECTION_BOTTOM_PREFIX),
        )
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _PROMPT_HINT_PREFIX),
        )
        ctx.extra_hints.extend(state.extra_hints)
        inbound = state.inbound

        # ═══════════════════════════════════════════════════════════════
        # Path A（条件开关）：仅当 multimodal=True 时编码图片注入主模型
        # ═══════════════════════════════════════════════════════════════
        from raven_agent.messages import MediaItem
        media_items: list[MediaItem] = []
        if self._multimodal and inbound.media:
            from raven_agent.media import encode_image_to_data_uri
            for media_path in inbound.media:
                try:
                    uri = encode_image_to_data_uri(media_path)
                    media_items.append(MediaItem(type="image", uri=uri))
                except ValueError:
                    pass  # 跳过无法编码的附件

        # ═══════════════════════════════════════════════════════════════
        # Path B 路径注入：multimodal=false 时告知 LLM 文件保存位置
        # ═══════════════════════════════════════════════════════════════
        user_input_text = inbound.content or ""
        if not self._multimodal and inbound.media:
            media_lines: list[str] = []
            for mp in inbound.media:
                media_lines.append(f"  - {mp}")
            if media_lines:
                media_hint = (
                    "\n\n[附件] 用户发送了以下文件，已保存到本地。"
                    "请根据文件类型选择合适的工具处理，将 path 参数设为以下路径：\n"
                    + "\n".join(media_lines)
                )
                user_input_text = (
                    f"{user_input_text}{media_hint}" if user_input_text.strip()
                    else f"用户发送了附件。{media_hint}"
                )

        frame.slots[_PROMPT_MESSAGES_SLOT] = self._prompt_builder.build(
            session=session,
            current_user_input=user_input_text,    # ← 改为拼接后的文本
            system_sections_top=ctx.system_sections_top,
            system_sections_bottom=ctx.system_sections_bottom,
            extra_hints=ctx.extra_hints,
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            request_time=inbound.request_time,
            media_items=media_items or None,        # ← Path A 产物
        )
        frame.slots[_TOOL_CONTEXT_SLOT] = self._build_tool_context(
            session.key, inbound,
        )
        return frame

    def _build_tool_context(
        self, session_key: str, inbound: InboundMessage
    ) -> dict[str, object]:
        next_message_id = self._sessions.peek_next_message_id(session_key)
        ctx: dict[str, object] = {
            "current_user_source_ref": json.dumps(
                [next_message_id], ensure_ascii=False
            ),
            "channel": inbound.channel,
            "chat_id": inbound.chat_id,
        }
        disabled = inbound.metadata.get("disabled_tools")
        if isinstance(disabled, list):
            ctx["disabled_tools"] = disabled
        return ctx

    def _build_tool_context(self, session_key: str, inbound: InboundMessage) -> dict[str, object]:
        """计算当前用户消息的来源引用等每轮工具上下文。

        输入:
            session_key: 当前会话 key。
            inbound: 当前轮入站消息。

        输出:
            包含 current_user_source_ref / channel / chat_id 的字典。
        """

        next_message_id = self._sessions.peek_next_message_id(session_key)
        ctx: dict[str, object] = {
            "current_user_source_ref": json.dumps([next_message_id], ensure_ascii=False),
            "channel": inbound.channel,
            "chat_id": inbound.chat_id,
        }
        disabled = inbound.metadata.get("disabled_tools")
        if isinstance(disabled, list):
            ctx["disabled_tools"] = disabled
        return ctx


class _ReturnPromptRenderModule:
    """返回 PromptRenderPhase 输出的模块。

    参数:
        无。
    """

    slot = "prompt_render.return"
    requires = ("prompt_render.build_messages",)

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        """把 prompt messages 包装为 PromptRenderResult。

        参数:
            frame: 当前 PromptRenderFrame。

        返回:
            写入 output 后的 PromptRenderFrame。
        """

        session = frame.input.session
        if session is None:
            raise RuntimeError("PromptRender requires loaded session")
        frame.output = PromptRenderResult(
            messages=cast(list[ChatMessage], frame.slots[_PROMPT_MESSAGES_SLOT]),
            session_key=session.key,
            tool_context=cast(dict[str, object], frame.slots[_TOOL_CONTEXT_SLOT]),
        )
        return frame

# ——实现 ReasoningPhase 的模块————————————————
class _RunAgentModule:
    """调用 ReactAgent 执行推理的模块。

    输入:
        agent: 当前用于执行 ReAct 推理的 ReactAgent。
        lifecycle: step 生命周期实现（通常是 PassiveTurnPipeline 自身）。

    输出:
        _RunAgentModule 实例。
    """

    slot = "reasoning.run_agent"
    produces = (_AGENT_RESULT_SLOT,)

    def __init__(self, agent: ReactAgent, lifecycle: ReasoningLifecycle, event_bus: EventBus | None = None,) -> None:
        self._agent = agent
        self._lifecycle = lifecycle
        self._event_bus = event_bus

    async def run(self, frame: ReasoningFrame) -> ReasoningFrame:
        """调用 ReactAgent.run() 并透传 step 生命周期。

        输入:
            frame: 当前 ReasoningFrame。

        输出:
            写入 agent result slot 后的 ReasoningFrame。
        """

        prompt_result = frame.input
        tool_context = prompt_result.tool_context
        channel = str(tool_context.get("channel", ""))
        chat_id = str(tool_context.get("chat_id", ""))
        session_key = prompt_result.session_key
        # —— 发射 StreamStart ——
        if self._event_bus is not None:
            await self._event_bus.observe(
                StreamStart(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                )
            )

        frame.slots[_AGENT_RESULT_SLOT] = await self._agent.run(
            prompt_result.messages,
            session_key=prompt_result.session_key,
            tool_context=tool_context,
            lifecycle=self._lifecycle,
            channel=str(tool_context.get("channel", "")),
            chat_id=str(tool_context.get("chat_id", "")),
        )

        # —— 发射 StreamEnd ——
        if self._event_bus is not None:
            await self._event_bus.observe(
                StreamEnd(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                )
            )
        return frame


class _ReturnReasoningModule:
    """返回 ReasoningPhase 输出的模块。

    参数:
        无。
    """

    slot = "reasoning.return"
    requires = ("reasoning.run_agent",)

    async def run(self, frame: ReasoningFrame) -> ReasoningFrame:
        """把 AgentRunResult 包装为 ReasoningResult。

        参数:
            frame: 当前 ReasoningFrame。

        返回:
            写入 output 后的 ReasoningFrame。
        """

        frame.output = ReasoningResult(
            result=cast(AgentRunResult, frame.slots[_AGENT_RESULT_SLOT])
        )
        return frame


# ——实现 AfterReasoningPhase 的模块————————————————
@dataclass
class AfterReasoningFrame(PhaseFrame[AfterTurnInput, AfterReasoningResult]):
    """AfterReasoningPhase 使用的数据帧。

    输入:
        input: AfterTurnInput（inbound / session / reasoning）。
        slots: 模块之间共享的临时数据。
        output: AfterReasoningResult。

    输出:
        AfterReasoningFrame 实例。
    """


_AFTER_REASONING_CTX_SLOT = "after_reasoning:ctx"
_OUTBOUND_METADATA_PREFIX = "after_reasoning:outbound_metadata:"


class _BuildAfterReasoningCtxModule:
    """构造 AfterReasoningCtx 的内置模块。

    输入:
        无。

    输出:
        _BuildAfterReasoningCtxModule 实例。
    """

    slot = "after_reasoning.build_ctx"
    produces = (_AFTER_REASONING_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        """把推理结果打包成 AfterReasoningCtx。

        输入:
            frame: 当前 AfterReasoningFrame。

        输出:
            写入 ctx slot 后的 AfterReasoningFrame。
        """

        data = frame.input
        result = data.reasoning
        frame.slots[_AFTER_REASONING_CTX_SLOT] = AfterReasoningCtx(
            session_key=data.inbound.session_key,
            channel=data.inbound.channel,
            chat_id=data.inbound.chat_id,
            tools_used=tuple(result.tools_used),
            reply=result.content,
        )
        return frame


class _EmitAfterReasoningCtxModule:
    """通过 EventBus 触发 AfterReasoningCtx GATE 事件。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _EmitAfterReasoningCtxModule 实例。
    """

    slot = "after_reasoning.emit"
    requires = ("after_reasoning.build_ctx", _AFTER_REASONING_CTX_SLOT)
    # produces = (_AFTER_REASONING_CTX_SLOT,)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        """运行所有 @on_after_reasoning handler。

        输入:
            frame: 当前 AfterReasoningFrame。

        输出:
            写回可能被 handler 改写后的 AfterReasoningFrame。
        """

        ctx = cast(AfterReasoningCtx, frame.slots[_AFTER_REASONING_CTX_SLOT])
        frame.slots[_AFTER_REASONING_CTX_SLOT] = await self._event_bus.emit(ctx)
        return frame



class _ReturnAfterReasoningModule:
    """收集插件改写并返回 AfterReasoningResult 的内置模块。

    输入:
        无。

    输出:
        _ReturnAfterReasoningModule 实例。
    """

    slot = "after_reasoning.return"
    requires = ("after_reasoning.build_ctx", _AFTER_REASONING_CTX_SLOT)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        """把 ctx.reply 与导出 metadata 包装成 AfterReasoningResult。

        输入:
            frame: 当前 AfterReasoningFrame。

        输出:
            写入 output 后的 AfterReasoningFrame。
        """

        ctx = cast(AfterReasoningCtx, frame.slots[_AFTER_REASONING_CTX_SLOT])
        metadata = dict(ctx.outbound_metadata)
        metadata.update(collect_prefixed_slots(frame.slots, _OUTBOUND_METADATA_PREFIX))
        frame.output = AfterReasoningResult(
            reply=ctx.reply,
            outbound_metadata=metadata,
            reasoning=frame.input.reasoning,
        )
        return frame


# ——实现 AfterTurnPhase 的模块————————————————
_AFTER_TURN_CTX_SLOT = "after_turn:ctx"
_AFTER_TURN_TELEMETRY_PREFIX = "after_turn:telemetry:"

class _BuildOutboundModule:
    """构造 OutboundMessage 的模块。

    参数:
        无。
    """

    slot = "after_turn.build_outbound"
    produces = (_OUTBOUND_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """根据入站消息和推理结果构造 OutboundMessage。

        参数:
            frame: 当前 AfterTurnFrame。

        返回:
            写入 outbound slot 后的 AfterTurnFrame。
        """

        inbound = frame.input.inbound
        result = frame.input.reasoning

        # 从 outbound_metadata 提取 media 文件路径
        outbound_meta = dict(frame.input.outbound_metadata)
        media_files: list[str] = []
        raw_media = outbound_meta.pop("media", None)
        if isinstance(raw_media, list):
            for p in raw_media:
                if isinstance(p, str) and p.strip():
                    media_files.append(p.strip())

        frame.slots[_OUTBOUND_SLOT] = OutboundMessage(
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            content=frame.input.reply,
            media=media_files,
            metadata={
                "iterations": result.iterations,
                "tools_used": list(result.tools_used),
                **outbound_meta,
            },
        )
        return frame


class _PersistSessionModule:
    """把本轮 user / assistant 消息写入 Session 并保存。

    参数:
        sessions: SessionManager。
    """

    slot = "after_turn.persist_session"
    requires = ("after_turn.build_outbound",)

    def __init__(self, sessions: SessionManager) -> None:
        self._sessions = sessions

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """向 Session 追加消息并保存。

        参数:
            frame: 当前 AfterTurnFrame。

        返回:
            写入并保存 Session 后的 AfterTurnFrame。
        """

        inbound = frame.input.inbound
        if inbound.metadata.get("omit_user_turn"):
            return frame  # 后台任务不持久化到 session
        session = frame.input.session
        media = inbound.media or None
        persist_user_content = inbound.metadata.get("persist_user_content")
        if isinstance(persist_user_content, str) and persist_user_content.strip():
            session.add_user_message(persist_user_content.strip(), media=media)
        else:
            session.add_user_message(inbound.content, media=media)
        session.add_assistant_message(frame.input.reply)
        self._sessions.save(session)
        return frame


class _ObserveTurnCompletedModule:
    """触发 TurnCompleted 观察事件的模块。

    参数:
        event_bus: 用于触发 TurnCompleted 的 EventBus。
    """

    slot = "after_turn.observe_completed"
    requires = ("after_turn.persist_session",)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """触发 TurnCompleted 观察事件。

        参数:
            frame: 当前 AfterTurnFrame。

        返回:
            观察事件触发后的 AfterTurnFrame。
        """

        inbound = frame.input.inbound
        if inbound.metadata.get("skip_post_memory"):
            return frame  # 后台任务不触发 TurnCompleted（memory_maintenance 等）
        result = frame.input.reasoning
        outbound = cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT])
        await self._event_bus.observe(
            TurnCompleted(
                session_key=inbound.session_key,
                inbound=inbound,
                outbound=outbound,
                tools_used=list(result.tools_used),
            )
        )
        return frame

class _BuildAfterTurnCtxModule:
    """构造 AfterTurnCtx。

    输入:
        无。

    输出:
        _BuildAfterTurnCtxModule 实例。
    """

    slot = "after_turn.build_ctx"
    requires = ("after_turn.observe_completed", _OUTBOUND_SLOT)
    produces = (_AFTER_TURN_CTX_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """根据 AfterTurnInput 和 OutboundMessage 构造 AfterTurnCtx。

        输入:
            frame: 当前 AfterTurnFrame。

        输出:
            写入 after_turn:ctx slot 后的 AfterTurnFrame。
        """

        outbound = cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT])
        frame.slots[_AFTER_TURN_CTX_SLOT] = AfterTurnCtx(
            session_key=frame.input.inbound.session_key,
            channel=outbound.channel,
            chat_id=outbound.chat_id,
            reply=outbound.content,
            tools_used=tuple(frame.input.reasoning.tools_used),
            outbound_metadata=dict(outbound.metadata),
        )
        return frame


class _CollectAfterTurnTelemetryModule:
    """收集 after_turn telemetry slots。

    输入:
        无。

    输出:
        _CollectAfterTurnTelemetryModule 实例。
    """

    slot = "after_turn.collect_telemetry"
    requires = ("after_turn.build_ctx", _AFTER_TURN_CTX_SLOT)
    # produces = (_AFTER_TURN_CTX_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """把 after_turn:telemetry:* 合并进 AfterTurnCtx.extra_metadata。

        输入:
            frame: 当前 AfterTurnFrame。

        输出:
            更新 ctx slot 后的 AfterTurnFrame。
        """

        ctx = cast(AfterTurnCtx, frame.slots[_AFTER_TURN_CTX_SLOT])
        ctx.extra_metadata.update(
            collect_prefixed_slots(frame.slots, _AFTER_TURN_TELEMETRY_PREFIX)
        )
        return frame


class _ObserveAfterTurnCtxModule:
    """通过 EventBus 观察 AfterTurnCtx。

    输入:
        event_bus: 当前 EventBus。

    输出:
        _ObserveAfterTurnCtxModule 实例。
    """

    slot = "after_turn.observe_ctx"
    requires = ("after_turn.collect_telemetry", _AFTER_TURN_CTX_SLOT)

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """运行所有 @on_after_turn handler。

        输入:
            frame: 当前 AfterTurnFrame。

        输出:
            原 frame。observe 模式忽略 handler 返回值。
        """

        await self._event_bus.observe(cast(AfterTurnCtx, frame.slots[_AFTER_TURN_CTX_SLOT]))
        return frame

class _ReturnAfterTurnModule:
    """返回 AfterTurnPhase 输出的模块。

    输入:
        无。

    输出:
        _ReturnAfterTurnModule 实例。
    """

    slot = "after_turn.return"
    requires = ("after_turn.observe_ctx",)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        """把 OutboundMessage 写入 output。

        输入:
            frame: 当前 AfterTurnFrame。

        输出:
            写入 output 后的 AfterTurnFrame。
        """

        frame.output = cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT])
        return frame


# ——实现 PassiveTurnPipeline
class PassiveTurnPipeline:
    """处理普通被动入站消息的七段式 Phase Pipeline。

    输入:
        deps: Pipeline 运行所需依赖与插件 lifecycle modules。

    输出:
        PassiveTurnPipeline 实例。它同时实现 ReasoningLifecycle 协议。
    """

    def __init__(self, deps: PassiveTurnPipelineDeps) -> None:
        self._deps = deps
        self._interrupt_manager = deps.interrupt_manager
        modules = deps.lifecycle_modules
        self._before_turn = self._build_before_turn_phase(modules.before_turn)
        self._before_reasoning = self._build_before_reasoning_phase(modules.before_reasoning)
        self._prompt_render = self._build_prompt_render_phase(modules.prompt_render)
        self._reasoning = self._build_reasoning_phase()
        self._after_reasoning = self._build_after_reasoning_phase(modules.after_reasoning)
        self._after_turn = self._build_after_turn_phase(modules.after_turn)
        self._before_step = self._build_before_step_phase(modules.before_step)
        self._after_step = self._build_after_step_phase(modules.after_step)

    def _build_before_turn_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[PassiveTurnState, PassiveTurnState, BeforeTurnFrame]:
        """构建 BeforeTurnPhase。

        输入:
            plugin_modules: 插件 before_turn modules。

        输出:
            处理 PassiveTurnState 的 BeforeTurn Phase。
        """

        return Phase(
            [
                _EmitTurnStartedModule(self._deps.event_bus),
                _LoadSessionModule(self._deps.sessions),
                _BuildBeforeTurnCtxModule(),
                *plugin_modules,
                _EmitBeforeTurnCtxModule(self._deps.event_bus),
                _CollectBeforeTurnModule(),
                _ReturnBeforeTurnModule(),
            ],
            frame_factory=BeforeTurnFrame,
        )

    def _build_after_turn_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[AfterTurnInput, OutboundMessage, AfterTurnFrame]:
        """构建 AfterTurnPhase。

        输入:
            plugin_modules: 插件 after_turn modules。

        输出:
            把 AfterTurnInput 转换为 OutboundMessage 的 Phase。
        """

        return Phase(
            [
                _BuildOutboundModule(),
                _PersistSessionModule(self._deps.sessions),
                _ObserveTurnCompletedModule(self._deps.event_bus),
                _BuildAfterTurnCtxModule(),
                *plugin_modules,
                _CollectAfterTurnTelemetryModule(),
                _ObserveAfterTurnCtxModule(self._deps.event_bus),
                _ReturnAfterTurnModule(),
            ],
            frame_factory=AfterTurnFrame,
        )
    

    def _build_before_reasoning_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[PassiveTurnState, PassiveTurnState, BeforeReasoningFrame]:
        """构建 BeforeReasoningPhase。

        输入:
            plugin_modules: 插件 before_reasoning modules。

        输出:
            处理 PassiveTurnState 的 BeforeReasoning Phase。
        """

        return Phase(
            [
                _BuildBeforeReasoningCtxModule(),
                *plugin_modules,
                _EmitBeforeReasoningCtxModule(self._deps.event_bus),
                _CollectBeforeReasoningModule(),
            ],
            frame_factory=BeforeReasoningFrame,
        )
    
    def _build_prompt_render_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[PassiveTurnState, PromptRenderResult, PromptRenderFrame]:
        """构建 PromptRenderPhase。

        输入:
            plugin_modules: 插件 prompt_render modules。

        输出:
            把 PassiveTurnState 转换为 PromptRenderResult 的 Phase。
        """

        return Phase(
        [
            _BuildPromptRenderCtxModule(),
            *plugin_modules,
            _EmitPromptRenderCtxModule(self._deps.event_bus),
            _BuildPromptMessagesModule(
                prompt_builder=self._deps.prompt_builder,
                sessions=self._deps.sessions,
                multimodal=self._deps.multimodal,
            ),
            _ReturnPromptRenderModule(),
        ],
        frame_factory=PromptRenderFrame,
    )
    
    def _build_reasoning_phase(
        self,
    ) -> Phase[PromptRenderResult, ReasoningResult, ReasoningFrame]:
        """构建 ReasoningPhase。

        输入:
            无。

        输出:
            把 PromptRenderResult 转换为 ReasoningResult 的 Phase。
        """

        return Phase(
            [
                _RunAgentModule(self._deps.agent, self, self._deps.event_bus),
                _ReturnReasoningModule(),
            ],
            frame_factory=ReasoningFrame,
        )
    
    def _build_after_reasoning_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[AfterTurnInput, AfterReasoningResult, AfterReasoningFrame]:
        """构建 AfterReasoningPhase。

        输入:
            plugin_modules: 插件 after_reasoning modules。

        输出:
            把 AfterTurnInput 转换为 AfterReasoningResult 的 Phase。
        """

        return Phase(
            [
                _BuildAfterReasoningCtxModule(),
                *plugin_modules,
                _EmitAfterReasoningCtxModule(self._deps.event_bus),
                _ReturnAfterReasoningModule(),
            ],
            frame_factory=AfterReasoningFrame,
        )
    
    def _build_before_step_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[BeforeStepCtx, BeforeStepCtx, BeforeStepFrame]:
        """构建 BeforeStepPhase。

        输入:
            plugin_modules: 插件 before_step modules。

        输出:
            处理 BeforeStepCtx 的 Phase。
        """

        return Phase(
            [
                _CopyBeforeStepCtxModule(),
                *plugin_modules,
                _EmitBeforeStepCtxModule(self._deps.event_bus),
                _CollectBeforeStepModule(),
                _ReturnBeforeStepCtxModule(),
            ],
            frame_factory=BeforeStepFrame,
        )

    def _build_after_step_phase(
        self,
        plugin_modules: list[object],
    ) -> Phase[AfterStepCtx, AfterStepCtx, AfterStepFrame]:
        """构建 AfterStepPhase。

        输入:
            plugin_modules: 插件 after_step modules。

        输出:
            处理 AfterStepCtx 的 Phase。
        """

        return Phase(
            [
                _CopyAfterStepCtxModule(),
                *plugin_modules,
                _CollectAfterStepTelemetryModule(
                    slot="after_step.collect_pre",
                    requires=("after_step.copy_ctx", _AFTER_STEP_CTX_SLOT),
                ),
                _ObserveAfterStepCtxModule(self._deps.event_bus),
                _CollectAfterStepTelemetryModule(
                    slot="after_step.collect_post",
                    requires=("after_step.observe", _AFTER_STEP_CTX_SLOT),
                ),
                _ReturnAfterStepCtxModule(),
            ],
            frame_factory=AfterStepFrame,
        )
    
    async def _complete_aborted_turn(self, state: PassiveTurnState) -> OutboundMessage:
        """用 abort_reply 直接完成本轮，不进入 LLM 推理。

        输入:
            state: 已设置 abort=True 的 PassiveTurnState。

        输出:
            当前轮最终产生的 OutboundMessage。
        """

        if state.session is None:
            raise RuntimeError("Aborted turn requires loaded session")
        reply = state.abort_reply or "（本轮已由插件提前结束）"
        result = AgentRunResult(content=reply, iterations=0, tools_used=[])
        return await self._after_turn.run(
            AfterTurnInput(
                inbound=state.inbound,
                session=state.session,
                reasoning=result,
                reply=reply,
                outbound_metadata={"aborted": True, **state.outbound_metadata},
            )
        )

    async def before_step(self, ctx: BeforeStepCtx) -> BeforeStepCtx:
        """运行 before_step 模块链。

        输入:
            ctx: 当前 BeforeStepCtx。

        输出:
            可能被插件改写后的 BeforeStepCtx。
        """

        return await self._before_step.run(ctx)

    async def after_step(self, ctx: AfterStepCtx) -> AfterStepCtx:
        """运行 after_step 模块链。

        输入:
            ctx: 当前 AfterStepCtx。

        输出:
            可能被插件补充后的 AfterStepCtx。
        """

        return await self._after_step.run(ctx)
    
    async def process_direct(
        self,
        content: str,
        channel: str,
        chat_id: str,
        session_key: str,
        omit_user_turn: bool = True,
        skip_post_memory: bool = True,
        disabled_tools: list[str] | None = None,
    ) -> str:
        """处理一条系统发起的直接消息（供 Scheduler SOFT 模式使用）。

        与普通被动 turn 的区别：
        - 使用指定的 session_key（可隔离 scheduler 会话）
        - omit_user_turn: 不将用户消息持久化到 session
        - skip_post_memory: 跳过后记忆维护
        - disabled_tools: 禁用指定工具（如 message_push 防止递归）

        输入:
            content: 要处理的提示词内容。
            channel: 渠道名。
            chat_id: 会话 ID。
            session_key: 使用的会话 key。
            omit_user_turn: 是否跳过用户消息持久化。
            skip_post_memory: 是否跳过后记忆维护。
            disabled_tools: 禁用的工具名列表。

        输出:
            Agent 的文本回复内容。
        """

        metadata: dict[str, object] = {}
        if omit_user_turn:
            metadata["omit_user_turn"] = True
        if skip_post_memory:
            metadata["skip_post_memory"] = True
        if disabled_tools:
            metadata["disabled_tools"] = list(disabled_tools)

        inbound = InboundMessage(
            channel=channel,
            sender="scheduler",
            chat_id=chat_id,
            content=content,
            session_key=session_key,
            metadata=metadata,
        )
        outbound = await self.run(inbound)
        return outbound.content

    async def run(self, inbound: InboundMessage) -> OutboundMessage:
        """处理一条入站消息。

        输入:
            inbound: 当前轮入站消息。

        输出:
            当前轮最终产生的 OutboundMessage。

        异常:
            asyncio.CancelledError: 被 InterruptManager 中断时重新抛出，
                抛出前会将中断态写入 InterruptManager。
        """
        try:
            state = await self._before_turn.run(PassiveTurnState(inbound=inbound))
            if state.session is None:
                raise RuntimeError("PassiveTurnPipeline requires loaded session")
            if state.abort:
                return await self._complete_aborted_turn(state)

            state = await self._before_reasoning.run(state)
            if state.abort:
                return await self._complete_aborted_turn(state)
            prompt_render = await self._prompt_render.run(state)
            reasoning = await self._reasoning.run(prompt_render)
            after_reasoning = await self._after_reasoning.run(
                AfterTurnInput(
                    inbound=state.inbound,
                    session=state.session,
                    reasoning=reasoning.result,
                    reply=reasoning.result.content,
                )
            )
            return await self._after_turn.run(
                AfterTurnInput(
                    inbound=state.inbound,
                    session=state.session,
                    reasoning=reasoning.result,
                    reply=after_reasoning.reply,
                    outbound_metadata=after_reasoning.outbound_metadata,
                )
            )
        except asyncio.CancelledError:
            # 中断态已由 InterruptManager.request_interrupt() 在取消 task 前保存。
            # 这里不做额外处理，让 CancelledError 继续向上传播到 AppRuntime。
            raise


def _append_sections(target: list[PromptSection], exports: dict[str, object]) -> None:
    """把 export 字典里的 PromptSection 或字符串追加到目标列表。

    输入:
        target: 要追加 section 的列表，例如 ctx.system_sections_bottom。
        exports: collect_prefixed_slots 的返回值。

    输出:
        None。会就地修改 target。
    """

    for name, value in exports.items():
        if isinstance(value, PromptSection):
            target.append(value)
        elif isinstance(value, str) and value.strip():
            target.append(PromptSection(name=name, content=value))