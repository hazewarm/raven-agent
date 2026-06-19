from __future__ import annotations

import asyncio
import json

from raven_agent.channels.ipc_server import IPCServerChannel
from raven_agent.message_bus import MessageBus
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore


async def _read_until(reader: asyncio.StreamReader, command: str) -> dict:
    """读取直到匹配指定 command 的 command_result。"""
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=2)
        data = json.loads(line)
        if data.get("type") == "command_result" and data.get("command") == command:
            return data


def test_session_new_binds_fresh_chat_id(tmp_path) -> None:
    async def run() -> None:
        bus = MessageBus()
        sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
        server = IPCServerChannel(bus, "127.0.0.1:0", sessions=sessions)
        await server.start()
        host, port = server._server.sockets[0].getsockname()[:2]

        reader, writer = await asyncio.open_connection(host, port)
        writer.write((json.dumps({"type": "command", "command": "session.new"}) + "\n").encode())
        await writer.drain()

        bound = await _read_until(reader, "session.bound")
        assert bound["ok"] is True
        assert bound["created"] is True
        # chat_id 不带 channel 前缀
        assert bound["chat_id"]
        assert not bound["chat_id"].startswith("cli")

        writer.close()
        await writer.wait_closed()
        await server.stop()
        sessions.close()

    asyncio.run(run())


def test_session_continue_latest_reuses_recent(tmp_path) -> None:
    async def run() -> None:
        bus = MessageBus()
        sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
        # 预置一个历史 cli session：session_key = "cli:oldsession"，chat_id = "oldsession"
        existing = sessions.get_or_create("cli:oldsession")
        existing.add_user_message("上一轮对话")
        existing.add_assistant_message("收到")
        sessions.save(existing)

        server = IPCServerChannel(bus, "127.0.0.1:0", sessions=sessions)
        await server.start()
        host, port = server._server.sockets[0].getsockname()[:2]

        reader, writer = await asyncio.open_connection(host, port)
        writer.write(
            (json.dumps({"type": "command", "command": "session.continue_latest"}) + "\n").encode()
        )
        await writer.drain()

        bound = await _read_until(reader, "session.bound")
        assert bound["ok"] is True
        assert bound["created"] is False
        assert bound["chat_id"] == "oldsession"

        writer.close()
        await writer.wait_closed()
        await server.stop()
        sessions.close()

    asyncio.run(run())


def test_session_list_then_attach(tmp_path) -> None:
    async def run() -> None:
        bus = MessageBus()
        sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
        session = sessions.get_or_create("cli:pickme")
        session.add_user_message("帮我查 Memory2 检索")
        sessions.save(session)

        server = IPCServerChannel(bus, "127.0.0.1:0", sessions=sessions)
        await server.start()
        host, port = server._server.sockets[0].getsockname()[:2]

        reader, writer = await asyncio.open_connection(host, port)
        writer.write((json.dumps({"type": "command", "command": "session.list"}) + "\n").encode())
        await writer.drain()

        listing = await _read_until(reader, "session.list")
        assert listing["ok"] is True
        assert listing["sessions"]
        first = listing["sessions"][0]
        assert first["chat_id"] == "pickme"
        assert "Memory2" in first["preview"]

        writer.write(
            (json.dumps({"type": "command", "command": "session.attach", "chat_id": "pickme"}) + "\n").encode()
        )
        await writer.drain()
        bound = await _read_until(reader, "session.bound")
        assert bound["chat_id"] == "pickme"

        # 控制命令不应进入 MessageBus
        assert bus.inbound_size == 0

        writer.close()
        await writer.wait_closed()
        await server.stop()
        sessions.close()

    asyncio.run(run())


def test_content_after_attach_uses_bound_chat_id(tmp_path) -> None:
    async def run() -> None:
        bus = MessageBus()
        sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
        server = IPCServerChannel(bus, "127.0.0.1:0", sessions=sessions)
        await server.start()
        host, port = server._server.sockets[0].getsockname()[:2]

        reader, writer = await asyncio.open_connection(host, port)
        writer.write(
            (json.dumps({"type": "command", "command": "session.attach", "chat_id": "fixed"}) + "\n").encode()
        )
        await writer.drain()
        await _read_until(reader, "session.bound")

        writer.write((json.dumps({"content": "你好"}) + "\n").encode())
        await writer.drain()

        inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=2)
        assert inbound.chat_id == "fixed"
        assert inbound.session_key == "cli:fixed"

        writer.close()
        await writer.wait_closed()
        await server.stop()
        sessions.close()

    asyncio.run(run())
