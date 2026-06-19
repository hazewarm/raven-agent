from __future__ import annotations

import asyncio

from raven_agent.events import InboundMessage, OutboundMessage
from raven_agent.message_bus import MessageBus


def test_message_bus_moves_inbound_messages() -> None:
    """测试 MessageBus 可以发布和消费入站消息。

    返回:
        None。
    """

    async def run() -> None:
        bus = MessageBus()
        message = InboundMessage(
            channel="cli",
            sender="local",
            chat_id="default",
            content="hello",
        )

        await bus.publish_inbound(message)
        consumed = await bus.consume_inbound()

        assert consumed == message
        assert consumed.session_key == "cli:default"
        assert bus.inbound_size == 0

    asyncio.run(run())


def test_message_bus_dispatches_outbound_to_subscriber() -> None:
    """测试 MessageBus 可以把出站消息分发给 channel 订阅者。

    返回:
        None。
    """

    async def run() -> None:
        bus = MessageBus()
        received: list[OutboundMessage] = []

        async def handler(message: OutboundMessage) -> None:
            received.append(message)

        outbound = OutboundMessage(channel="cli", chat_id="default", content="hi")
        bus.subscribe_outbound("cli", handler)
        await bus.publish_outbound(outbound)
        dispatched = await bus.dispatch_outbound_once()

        assert dispatched == outbound
        assert received == [outbound]
        assert bus.outbound_size == 0

    asyncio.run(run())