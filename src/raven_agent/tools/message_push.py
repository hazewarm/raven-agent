"""
统一消息推送工具，agent 通过 channel + chat_id 向任意已注册渠道发送消息、文件或图片。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from raven_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class MessagePushTool(Tool):
    """跨渠道消息推送工具。

    各 Channel 在启动时将自己的 send/send_file/send_image 注册进来。
    模型调用此工具时，按 channel 参数路由到对应 sender。

    输入:
        无。通过 register_channel() 注册各渠道 sender。

    输出:
        MessagePushTool 实例。
    """

    name = "message_push"
    description = (
        "向指定渠道的用户主动发送消息、文件或图片。"
        "需要提供渠道名（如 telegram）和目标 chat_id。"
        "message/file/image 三者至少提供一个。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "目标渠道名，如 telegram、cli",
            },
            "chat_id": {
                "type": "string",
                "description": "目标会话 ID（可用 @username 或数字 id）",
            },
            "message": {
                "type": "string",
                "description": "要发送的文本内容（可与 file/image 同时提供）",
            },
            "file": {
                "type": "string",
                "description": "要发送的文件本地路径，例如 /tmp/report.pdf",
            },
            "image": {
                "type": "string",
                "description": "要发送的图片本地路径或 URL",
            },
        },
        "required": ["channel", "chat_id"],
    }

    def __init__(self) -> None:
        # channel → { "text": fn, "file": fn, "image": fn }
        self._senders: dict[str, dict[str, Callable[..., Awaitable[None]]]] = {}

    def register_channel(
        self,
        channel: str,
        *,
        text: Callable[[str, str], Awaitable[None]] | None = None,
        file: Callable[[str, str, str | None], Awaitable[None]] | None = None,
        image: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        """注册渠道的各类 sender。

        输入:
            channel: 渠道名称，如 "telegram"。
            text: async (chat_id, message) -> None。
            file: async (chat_id, file_path, name=None) -> None。
            image: async (chat_id, image_path_or_url) -> None。

        输出:
            None。
        """
        entry: dict[str, Callable[..., Awaitable[None]]] = {}
        if text:
            entry["text"] = text
        if file:
            entry["file"] = file
        if image:
            entry["image"] = image
        self._senders[channel] = entry
        logger.debug(
            "[message_push] 注册渠道 %r  支持: %s",
            channel, list(entry.keys()),
        )

    async def execute(self, **kwargs: Any) -> str:
        """执行消息推送。

        输入:
            channel: 目标渠道名。
            chat_id: 目标会话 ID。
            message: 可选文本内容。
            file: 可选文件路径。
            image: 可选图片路径或 URL。

        输出:
            描述本次推送结果的字符串。
        """
        channel: str = kwargs["channel"]
        chat_id: str = str(kwargs["chat_id"])
        message: str | None = kwargs.get("message")
        file: str | None = kwargs.get("file")
        image: str | None = kwargs.get("image")

        if not message and not file and not image:
            return "错误：message、file、image 至少提供一个"

        senders = self._senders.get(channel)
        if senders is None:
            available = list(self._senders.keys()) or ["（无）"]
            return f"渠道 {channel!r} 未注册，可用渠道：{available}"

        results: list[str] = []
        try:
            if message and "text" in senders:
                await senders["text"](chat_id, message)
                preview = message[:60] + "..." if len(message) > 60 else message
                logger.info(
                    "[message_push] %s:%s ← text: %r", channel, chat_id, preview,
                )
                results.append("文本已发送")

            if file:
                if "file" not in senders:
                    results.append(f"渠道 {channel!r} 不支持发送文件")
                else:
                    import os

                    name = os.path.basename(file)
                    await senders["file"](chat_id, file, name)
                    logger.info(
                        "[message_push] %s:%s ← file: %r", channel, chat_id, file,
                    )
                    results.append(f"文件 {name!r} 已发送")

            if image:
                if "image" not in senders:
                    results.append(f"渠道 {channel!r} 不支持发送图片")
                else:
                    await senders["image"](chat_id, image)
                    logger.info(
                        "[message_push] %s:%s ← image: %r", channel, chat_id, image,
                    )
                    results.append("图片已发送")

        except Exception as e:
            logger.error("[message_push] 发送失败 %s:%s: %s", channel, chat_id, e)
            return f"发送失败：{e}"

        return "；".join(results) if results else f"渠道 {channel!r} 没有可用的 sender"