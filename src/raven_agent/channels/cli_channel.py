from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from raven_agent.channels.base import ChannelAdapter
from raven_agent.events import InboundMessage, OutboundMessage, StreamToken
from raven_agent.event_bus import EventBus

if TYPE_CHECKING:
    from raven_agent.message_bus import MessageBus

_EXIT_CMDS = {"exit", "quit", "q"}


def clean_cli_input(text: str) -> str:
    """清理终端输入中的异常字符。

    输入:
        text: 原始输入字符串。

    输出:
        清理后的字符串。会去除控制字符，并把全角空格/不间断空格替换为空格。
    """
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace(" ", " ").replace("　", " ")
    text = "".join(
        char for char in text if char in {"\t", "\n"} or ord(char) >= 32
    )
    return text.strip()


class CLIChannel(ChannelAdapter):
    """嵌入式命令行 Channel。

    输入:
        bus: MessageBus，用于发布入站消息和订阅出站消息。
        chat_id: CLI 会话 ID，默认 default。
        sender: 发送者标识。
        prompt: 输入提示符。

    输出:
        CLIChannel 实例。
    """

    def __init__(
        self,
        bus: "MessageBus",
        *,
        chat_id: str = "default",
        sender: str = "local",
        prompt: str = "You> ",
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._bus = bus
        self._chat_id = chat_id
        self._sender = sender
        self._prompt = prompt
        self._event_bus = event_bus
        self._running = False
        self._streamed = False

    @property
    def channel_name(self) -> str:
        """返回 Channel 名称。

        输入:
            无。

        输出:
            固定字符串 "cli"。
        """
        return "cli"

    @property
    def chat_id(self) -> str:
        """返回当前 CLI 会话 chat_id。

        输入:
            无。

        输出:
            chat_id 字符串。
        """
        return self._chat_id

    @property
    def sender(self) -> str:
        """返回发送者标识。

        输入:
            无。

        输出:
            sender 字符串。
        """
        return self._sender

    async def start(self) -> None:
        """启动 CLIChannel。

        输入:
            无。

        输出:
            None。启动后会订阅 cli 出站消息。
        """
        self._bus.subscribe_outbound(self.channel_name, self._on_outbound)
        # —— 订阅流式事件 ——
        if self._event_bus is not None:
            self._event_bus.on(StreamToken, self._on_stream_token)
        self._running = True
    
    async def stop(self) -> None:
        """停止 CLIChannel。

        输入:
            无。

        输出:
            None。
        """
        self._running = False

    def print_banner(self) -> None:
        """打印 CLI 欢迎横幅。

        输入:
            无。

        输出:
            None。
        """
        print("Raven Agent is ready. Type 'exit' or 'quit' to stop.")
        print("Type '/clear' to clear the current conversation history.")

    
    async def _on_stream_token(self, event: StreamToken) -> None:
        """逐 token 打印流式回复。

        输入:
            event: StreamToken 事件。

        输出:
            None。首 token 时打印 Raven> 前缀。
        """
        if event.chat_id != self._chat_id:
            return
        if not self._streamed:
            print("Raven> ", end="", flush=True)
            self._streamed = True
        print(event.token, end="", flush=True)
    
    
    
    async def read_input(self) -> str | None:
        """读取并清理一行用户输入。

        输入:
            无。

        输出:
            清理后的字符串；遇到 EOF / 中断时返回 None。
        """
        try:
            raw = await self._read_line()
        except (KeyboardInterrupt, EOFError):
            return None
        if raw == "":  # readline 在 EOF 时返回空字符串
            return None
        return clean_cli_input(raw)

    async def _on_outbound(self, message: OutboundMessage) -> None:
        """打印出站消息。如果本轮已流式输出，跳过（内容已逐 token 打印）。

        输入:
            message: Agent 产生的 OutboundMessage。

        输出:
            None。
        """
        if message.chat_id != self._chat_id:
            return
        if self._streamed:
            self._streamed = False
            return
        print(f"Raven> {message.content}")

    async def _read_line(self) -> str:
        """异步读取一行 stdin。

        输入:
            无。

        输出:
            用户输入的一行字符串。
        """
        loop = asyncio.get_event_loop()
        sys.stdout.write(f"\n{self._prompt}")
        sys.stdout.flush()
        return await loop.run_in_executor(None, sys.stdin.readline)