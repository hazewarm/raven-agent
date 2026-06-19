from __future__ import annotations

import asyncio

from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, OutboundMessage


def test_event_bus_emit_can_replace_event() -> None:
    """测试 EventBus.emit 支持 handler 替换事件。

    返回:
        None。
    """

    async def run() -> None:
        bus = EventBus()

        def add_suffix(event: InboundMessage) -> InboundMessage:
            return InboundMessage(
                channel=event.channel,
                sender=event.sender,
                chat_id=event.chat_id,
                content=event.content + " world",
                timestamp=event.timestamp,
                metadata=event.metadata,
            )

        bus.on(InboundMessage, add_suffix)
        event = InboundMessage(
            channel="cli",
            sender="local",
            chat_id="default",
            content="hello",
        )

        emitted = await bus.emit(event)

        assert emitted.content == "hello world"

    asyncio.run(run())


def test_event_bus_observe_ignores_return_value() -> None:
    """测试 EventBus.observe 会调用观察者但忽略返回值。

    返回:
        None。
    """

    async def run() -> None:
        bus = EventBus()
        seen: list[str] = []

        def observer(event: OutboundMessage) -> OutboundMessage:
            seen.append(event.content)
            return OutboundMessage(channel="cli", chat_id="default", content="changed")

        bus.on(OutboundMessage, observer)
        event = OutboundMessage(channel="cli", chat_id="default", content="original")

        await bus.observe(event)

        assert seen == ["original"]

    asyncio.run(run())