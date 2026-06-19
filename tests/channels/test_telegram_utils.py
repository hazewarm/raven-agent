from __future__ import annotations

import asyncio

from raven_agent.channels.telegram.utils import (
    TelegramOutboundLimiter,
    _split_text,
    _strip_chunk,
)
from telegramify_markdown.entity import MessageEntity


def test_split_text_respects_limit() -> None:
    """验证文本按限制切分。"""
    text = "hello world\\n" * 500
    chunks = _split_text(text, 100)
    for chunk in chunks:
        assert len(chunk) <= 100


def test_split_text_single_long_line() -> None:
    """验证超长单行能被强制切断。"""
    text = "a" * 5000
    chunks = _split_text(text, 1000)
    assert len(chunks) >= 5
    for chunk in chunks:
        assert len(chunk) <= 1000


def test_strip_chunk_noop() -> None:
    """验证无首尾换行时 entity 不变。"""
    entities = [
        MessageEntity(type="bold", offset=0, length=5, url=None, language=None, custom_emoji_id=None)
    ]
    text, result = _strip_chunk("hello", entities)
    assert text == "hello"
    assert result[0].offset == 0
    assert result[0].length == 5


def test_limiter_sends_successfully() -> None:
    """验证限流器正常执行操作。"""
    async def run() -> None:
        limiter = TelegramOutboundLimiter()
        calls: list[int] = []

        async def fake_action() -> str:
            calls.append(1)
            return "ok"

        result = await limiter.run(123, kind="send", label="test", action=fake_action)
        assert result == "ok"
        assert len(calls) == 1

    asyncio.run(run())