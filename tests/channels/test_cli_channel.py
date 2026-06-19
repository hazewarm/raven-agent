from __future__ import annotations

import asyncio

from raven_agent.channels.cli_channel import CLIChannel, clean_cli_input
from raven_agent.events import OutboundMessage
from raven_agent.message_bus import MessageBus


def test_clean_cli_input_removes_control_chars() -> None:
    assert clean_cli_input(" hello world\x00 ") == "hello world"


def test_cli_channel_subscribes_and_prints(capsys) -> None:
    async def run() -> None:
        bus = MessageBus()
        channel = CLIChannel(bus, chat_id="default")
        await channel.start()

        await bus.publish_outbound(
            OutboundMessage(channel="cli", chat_id="default", content="pong")
        )
        await bus.dispatch_outbound_once()

    asyncio.run(run())

    captured = capsys.readouterr()
    assert "Raven> pong" in captured.out
