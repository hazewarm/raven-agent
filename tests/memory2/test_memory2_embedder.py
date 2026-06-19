from __future__ import annotations

import asyncio

from raven_agent.memory2 import DisabledEmbeddingProvider, EmbeddingProvider
from raven_agent.memory2.embedder import _clean_texts


def test_disabled_embedding_provider_matches_protocol() -> None:
    """测试 DisabledEmbeddingProvider 满足 EmbeddingProvider 协议。"""

    provider = DisabledEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)


def test_disabled_embedding_provider_returns_empty_vectors() -> None:
    """测试 DisabledEmbeddingProvider 返回空向量。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = DisabledEmbeddingProvider()

        assert await provider.embed_text("hello") == []
        assert await provider.embed_batch(["a", "b"]) == [[], []]

    asyncio.run(run())


def test_clean_texts_trims_truncates_and_replaces_empty_text() -> None:
    """测试 _clean_texts 会清理空白、截断并替换空文本。"""

    cleaned = _clean_texts(["  abc  ", "", "abcdef"], max_chars=3)

    assert cleaned == ["abc", " ", "abc"]