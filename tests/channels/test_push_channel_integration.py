from __future__ import annotations

import asyncio

from raven_agent.channels.base import ChannelAdapter
from raven_agent.channels.manager import ChannelManager
from raven_agent.tools.message_push import MessagePushTool


class _FakePushChannel(ChannelAdapter):
    """模拟一个可推送的 Channel。"""
    def __init__(self) -> None:
        self._sent: list[tuple[str, str]] = []

    @property
    def channel_name(self) -> str:
        return "fake"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, chat_id: str, message: str) -> None:
        self._sent.append((chat_id, message))

    def register_push_senders(self, push_tool: MessagePushTool) -> None:
        push_tool.register_channel("fake", text=self.send)


def test_channel_registers_sender_to_push_tool() -> None:
    """验证 Channel 的 sender 注册后 Push Tool 可路由消息。"""
    async def run() -> None:
        manager = ChannelManager()
        channel = _FakePushChannel()
        manager.register(channel)

        push_tool = MessagePushTool()
        for ch in manager.list_channels():
            register_fn = getattr(ch, "register_push_senders", None)
            if callable(register_fn):
                register_fn(push_tool)

        result = await push_tool.execute(
            channel="fake", chat_id="test", message="hello",
        )
        assert "文本已发送" in result
        assert channel._sent == [("test", "hello")]

    asyncio.run(run())