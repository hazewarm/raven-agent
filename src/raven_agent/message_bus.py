from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from raven_agent.events import InboundMessage, OutboundMessage


OutboundHandler = Callable[[OutboundMessage], None | Awaitable[None]]


class MessageBus:
    """Agent 核心与外部 Channel 之间的异步消息总线。

    参数:
        无。内部维护 inbound 和 outbound 两个异步队列。
    """

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._outbound_subscribers: dict[str, list[OutboundHandler]] = {}

    async def publish_inbound(self, message: InboundMessage) -> None:
        """发布入站消息。

        参数:
            message: 来自 Channel 的 InboundMessage。

        返回:
            None。
        """

        await self._inbound.put(message)

    async def consume_inbound(self) -> InboundMessage:
        """消费一条入站消息。

        返回:
            队列中的下一条 InboundMessage；如果队列为空则等待。
        """

        return await self._inbound.get()

    async def publish_outbound(self, message: OutboundMessage) -> None:
        """发布出站消息。

        参数:
            message: Agent 生成的 OutboundMessage。

        返回:
            None。
        """

        await self._outbound.put(message)

    def subscribe_outbound(self, channel: str, handler: OutboundHandler) -> None:
        """订阅某个 channel 的出站消息。

        参数:
            channel: 要订阅的渠道名称。
            handler: 收到该渠道 OutboundMessage 时调用的处理函数。

        返回:
            None。
        """

        self._outbound_subscribers.setdefault(channel, []).append(handler)

    async def dispatch_outbound_once(self) -> OutboundMessage:
        """分发一条出站消息给对应 channel 的订阅者。

        返回:
            被分发的 OutboundMessage。
        """

        message = await self._outbound.get()
        for handler in self._outbound_subscribers.get(message.channel, []):
            result = handler(message)
            if inspect.isawaitable(result):
                await result
        return message

    @property
    def inbound_size(self) -> int:
        """返回 inbound 队列长度。

        返回:
            当前等待处理的入站消息数量。
        """

        return self._inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """返回 outbound 队列长度。

        返回:
            当前等待发送的出站消息数量。
        """

        return self._outbound.qsize()