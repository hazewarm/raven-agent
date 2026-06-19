from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raven_agent.memory2.procedure_tagger import ProcedureTagger, validate_trigger_tags


@dataclass(frozen=True)
class FakeResponse:
    """测试用 LLM 响应。

    参数:
        content: 响应文本。

    返回:
        FakeResponse 实例。
    """

    content: str


class FakeProvider:
    """测试用 provider。

    参数:
        content: chat 返回文本。
        raise_error: 是否抛出异常。

    返回:
        FakeProvider 实例。
    """

    def __init__(self, content: str, *, raise_error: bool = False) -> None:
        self.content = content
        self.raise_error = raise_error
        self.messages = []

    async def chat(self, **kwargs: object) -> FakeResponse:
        """返回固定响应。

        参数:
            kwargs: provider 调用参数。

        返回:
            FakeResponse。
        """

        self.messages.append(kwargs.get("messages"))
        if self.raise_error:
            raise RuntimeError("boom")
        return FakeResponse(self.content)


def test_validate_trigger_tags_filters_unknown_tools_and_short_keywords() -> None:
    """测试 trigger_tags 校验会过滤非法工具和过短关键词。"""

    result = validate_trigger_tags(
        {"tools": ["shell", "bad"], "skills": ["rss"], "keywords": ["x", "pacman"], "scope": "global"},
        valid_tools={"shell"},
        valid_skills={"rss"},
    )

    assert result == {
        "tools": ["shell"],
        "skills": ["rss"],
        "keywords": ["pacman"],
        "scope": "tool_triggered",
    }


def test_procedure_tagger_uses_loose_json_parser() -> None:
    """测试 ProcedureTagger 能解析 fenced JSON。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(
            '```json\n{"tools":["web_fetch","bad"],"skills":[],"keywords":["B站"],"scope":"tool_triggered"}\n```'
        )
        tagger = ProcedureTagger(
            provider=provider,  # type: ignore[arg-type]
            tools_fn=lambda: ["web_fetch", "shell"],
        )

        result = await tagger.tag("用户发送 B 站链接时先抓网页")

        assert result == {
            "tools": ["web_fetch"],
            "skills": [],
            "keywords": ["B站"],
            "scope": "tool_triggered",
        }

    asyncio.run(run())


def test_procedure_tagger_returns_none_on_provider_error() -> None:
    """测试 provider 异常时返回 None。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        tagger = ProcedureTagger(
            provider=FakeProvider("", raise_error=True),  # type: ignore[arg-type]
            tools_fn=lambda: ["shell"],
        )

        assert await tagger.tag("测试") is None

    asyncio.run(run())