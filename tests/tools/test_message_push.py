from __future__ import annotations

import asyncio

from raven_agent.tools.message_push import MessagePushTool


def test_push_tool_rejects_missing_channel() -> None:
    """验证未注册 channel 时返回错误提示。"""
    async def run() -> None:
        tool = MessagePushTool()
        result = await tool.execute(channel="unknown", chat_id="123", message="hi")
        assert "未注册" in result

    asyncio.run(run())


async def _noop_send(chat_id: str, message: str) -> None:
    pass


def test_push_tool_rejects_empty_payload() -> None:
    """验证 message/file/image 全为空时返回错误。"""
    async def run() -> None:
        tool = MessagePushTool()
        tool.register_channel("test", text=_noop_send)
        result = await tool.execute(channel="test", chat_id="123")
        assert "至少提供一个" in result

    asyncio.run(run())


def test_push_tool_sends_text() -> None:
    """验证文本消息被正确发送。"""
    async def run() -> None:
        tool = MessagePushTool()
        sent: list[tuple[str, str]] = []

        async def fake_send(chat_id: str, message: str) -> None:
            sent.append((chat_id, message))

        tool.register_channel("telegram", text=fake_send)
        result = await tool.execute(
            channel="telegram", chat_id="alice", message="你好",
        )
        assert "文本已发送" in result
        assert sent == [("alice", "你好")]

    asyncio.run(run())


def test_push_tool_handles_send_error() -> None:
    """验证 sender 抛异常时工具返回错误信息。"""
    async def run() -> None:
        tool = MessagePushTool()

        async def broken_send(chat_id: str, message: str) -> None:
            raise RuntimeError("network down")

        tool.register_channel("telegram", text=broken_send)
        result = await tool.execute(
            channel="telegram", chat_id="alice", message="ping",
        )
        assert "发送失败" in result or "network down" in result

    asyncio.run(run())