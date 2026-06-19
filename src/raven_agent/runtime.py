from __future__ import annotations

from raven_agent.agent import ReactAgent
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, OutboundMessage
from raven_agent.prompt import PromptBuilder
from raven_agent.session import SessionManager
from raven_agent.turn_pipeline import PassiveTurnPipeline, PassiveTurnPipelineDeps


async def handle_inbound_message(
    inbound: InboundMessage,
    sessions: SessionManager,
    prompt_builder: PromptBuilder,
    agent: ReactAgent,
    event_bus: EventBus | None = None,
) -> OutboundMessage:
    """处理一条入站消息并返回出站消息。

    参数:
        inbound: 入站消息。
        sessions: SessionManager，用于按 session_key 获取和保存会话。
        prompt_builder: 用于构造模型输入。
        agent: 负责执行 ReAct 推理的 ReactAgent。
        event_bus: 可选事件总线，用于发布 TurnStarted / TurnCompleted。

    返回:
        OutboundMessage，包含要发送给用户的回复。
    """

    bus = event_bus or EventBus()
    pipeline = PassiveTurnPipeline(
        PassiveTurnPipelineDeps(
            sessions=sessions,
            prompt_builder=prompt_builder,
            agent=agent,
            event_bus=bus,
        )
    )
    return await pipeline.run(inbound)