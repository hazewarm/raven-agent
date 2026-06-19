from __future__ import annotations

import asyncio
import json

from raven_agent.channels.ipc_server import IPCServerChannel, parse_tcp_endpoint
from raven_agent.events import OutboundMessage
from raven_agent.message_bus import MessageBus


def test_parse_tcp_endpoint() -> None:
    assert parse_tcp_endpoint("127.0.0.1:8765") == ("127.0.0.1", 8765)
    assert parse_tcp_endpoint("/tmp/raven.sock") is None
    assert parse_tcp_endpoint("127.0.0.1:not-port") is None


async def _read_until_type(reader: asyncio.StreamReader, msg_type: str) -> dict:
    """读取直到匹配指定 type 的消息，跳过其它（如 session.bound）控制回执。"""
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=2)
        data = json.loads(line)
        if data.get("type") == msg_type:
            return data


def test_ipc_server_receives_and_sends() -> None:
    async def run() -> None:
        bus = MessageBus()
        server = IPCServerChannel(bus, "127.0.0.1:0")
        await server.start()
        assert server._server is not None
        host, port = server._server.sockets[0].getsockname()[:2]

        reader, writer = await asyncio.open_connection(host, port)
        # 未握手直接发消息：服务端会兜底自动新建 session（回一条 session.bound），再处理消息
        writer.write((json.dumps({"content": "ping"}) + "\n").encode("utf-8"))
        await writer.drain()

        inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=2)
        assert inbound.channel == "cli"
        assert inbound.content == "ping"

        await bus.publish_outbound(
            OutboundMessage(channel="cli", chat_id=inbound.chat_id, content="pong")
        )
        await bus.dispatch_outbound_once()

        # 跳过兜底产生的 session.bound 回执，读取真正的 assistant 回复
        payload = await _read_until_type(reader, "assistant")
        assert payload["content"] == "pong"

        writer.close()
        await writer.wait_closed()
        await server.stop()

    asyncio.run(run())
