from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raven_agent.memory2.profile_extractor import ProfileFact, ProfileFactExtractor


@dataclass(frozen=True)
class FakeResponse:
    """测试用 LLM 响应。"""

    content: str


class FakeProvider:
    """测试用 provider。"""

    def __init__(self, content: str, *, raise_error: bool = False) -> None:
        self.content = content
        self.raise_error = raise_error
        self.prompts: list[str] = []

    async def chat(self, *, messages, **kwargs: object) -> FakeResponse:
        """返回固定响应并记录 prompt。

        参数:
            messages: provider messages。
            kwargs: 其他参数。

        返回:
            FakeResponse。
        """

        self.prompts.append(messages[0].content)
        if self.raise_error:
            raise RuntimeError("boom")
        return FakeResponse(self.content)


def test_profile_fact_fields() -> None:
    """测试 ProfileFact 字段。"""

    fact = ProfileFact(summary="用户有一块 Fitbit 手表", category="personal_fact", happened_at=None)

    assert fact.summary == "用户有一块 Fitbit 手表"
    assert fact.category == "personal_fact"
    assert fact.happened_at is None


def test_profile_extractor_parses_purchase_fact() -> None:
    """测试 profile extractor 解析 purchase fact。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(
            """
<facts>
<fact><summary>用户购买了罗技 MX Master 3 鼠标</summary><category>purchase</category><happened_at>2026-05-30</happened_at></fact>
</facts>
"""
        )
        extractor = ProfileFactExtractor(provider=provider)  # type: ignore[arg-type]

        facts = await extractor.extract("USER: 我买了罗技 MX Master 3 鼠标")

        assert facts == [ProfileFact("用户购买了罗技 MX Master 3 鼠标", "purchase", "2026-05-30")]

    asyncio.run(run())


def test_profile_extractor_discards_preference_category() -> None:
    """测试 preference 不会被 profile extractor 接收。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(
            """
<facts>
<fact><summary>用户希望先给结论</summary><category>preference</category><happened_at></happened_at></fact>
</facts>
"""
        )
        extractor = ProfileFactExtractor(provider=provider)  # type: ignore[arg-type]

        facts = await extractor.extract("USER: 回答时先给结论")

        assert facts == []

    asyncio.run(run())


def test_profile_extractor_prompt_contains_user_first_rules() -> None:
    """测试 prompt 包含 USER-first 证据源规则。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider("<facts></facts>")
        extractor = ProfileFactExtractor(provider=provider)  # type: ignore[arg-type]

        await extractor.extract("USER: 测试")

        assert provider.prompts
        prompt = provider.prompts[0]
        assert "ASSISTANT 的回复只作为背景参考" in prompt
        assert "用户提问" in prompt
        assert "假设" in prompt
        assert "preference，不是 profile" in prompt

    asyncio.run(run())


def test_profile_extractor_returns_empty_on_provider_error() -> None:
    """测试 provider 异常时返回空列表。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        extractor = ProfileFactExtractor(provider=FakeProvider("", raise_error=True))  # type: ignore[arg-type]

        assert await extractor.extract("USER: 我买了鼠标") == []

    asyncio.run(run())