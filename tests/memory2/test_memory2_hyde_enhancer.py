from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raven_agent.memory2.hyde_enhancer import HyDEEnhancer, _union_dedup


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
        delay_s: 可选延迟秒数。

    返回:
        FakeProvider 实例。
    """

    def __init__(self, content: str, delay_s: float = 0.0) -> None:
        self._content = content
        self._delay_s = delay_s

    async def chat(self, **kwargs: object) -> FakeResponse:
        """返回固定响应。

        参数:
            kwargs: provider 参数，本测试忽略。

        返回:
            FakeResponse。
        """

        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return FakeResponse(self._content)


def test_union_dedup_preserves_raw_scores() -> None:
    """测试 _union_dedup 保留 raw 条目和 raw 分数。"""

    raw = [{"id": "a", "score": 0.7}, {"id": "b", "score": 0.6}]
    hyde = [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.8}]

    result = _union_dedup(raw, hyde)

    assert [item["id"] for item in result] == ["a", "b", "c"]
    assert result[1]["score"] == 0.6


def test_hyde_timeout_falls_back_to_raw() -> None:
    """测试 HyDE hypothesis 超时时降级为 raw 结果。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        enhancer = HyDEEnhancer(provider=FakeProvider("假想条目", delay_s=0.2), timeout_s=0.05)  # type: ignore[arg-type]

        async def retrieve_fn(query: str, **kwargs: object) -> list[dict[str, object]]:
            """返回 raw hit。

            参数:
                query: 查询文本。
                kwargs: 额外参数。

            返回:
                hit 列表。
            """

            return [{"id": "raw", "score": 0.7}]

        result = await enhancer.augment(raw_query="问题", context="", retrieve_fn=retrieve_fn, top_k=5)

        assert result.used_hyde is False
        assert result.items == [{"id": "raw", "score": 0.7}]

    asyncio.run(run())


def test_hyde_appends_new_items() -> None:
    """测试 HyDE 追加 raw 中没有的新条目。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        enhancer = HyDEEnhancer(provider=FakeProvider("假想条目"), timeout_s=1.0)  # type: ignore[arg-type]

        async def retrieve_fn(query: str, **kwargs: object) -> list[dict[str, object]]:
            """根据 query 返回不同 hit。

            参数:
                query: 查询文本。
                kwargs: 额外参数。

            返回:
                hit 列表。
            """

            if query == "假想条目":
                return [{"id": "hyde", "score": 0.8}]
            return [{"id": "raw", "score": 0.7}]

        result = await enhancer.augment(raw_query="问题", context="", retrieve_fn=retrieve_fn, top_k=5)

        assert result.used_hyde is True
        assert [item["id"] for item in result.items] == ["raw", "hyde"]
        assert result.hypothesis == "假想条目"

    asyncio.run(run())