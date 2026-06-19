from __future__ import annotations

import asyncio

from raven_agent.memory2 import EmbeddingProvider, MemoryStore2, Retriever
from raven_agent.memory2.retriever import _rrf_merge


class DeterministicEmbeddingProvider:
    """测试用 deterministic embedding provider。

    参数:
        mapping: 文本到向量的映射。

    返回:
        DeterministicEmbeddingProvider 实例。
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed_text(self, text: str) -> list[float]:
        """返回测试向量。

        参数:
            text: 输入文本。

        返回:
            测试向量；未命中返回 [0.0, 0.0]。
        """

        return list(self._mapping.get(text, [0.0, 0.0]))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量返回测试向量。

        参数:
            texts: 输入文本列表。

        返回:
            测试向量列表。
        """

        return [await self.embed_text(text) for text in texts]

    async def close(self) -> None:
        """关闭测试 provider。

        返回:
            None。
        """

        return None


def test_deterministic_provider_matches_protocol() -> None:
    """测试测试 provider 满足 EmbeddingProvider 协议。"""

    assert isinstance(DeterministicEmbeddingProvider({}), EmbeddingProvider)


def test_rrf_merge_promotes_item_seen_in_both_lanes() -> None:
    """测试 RRF 会提升同时出现在两路结果中的条目。"""

    vector_items = [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.8},
    ]
    keyword_items = [
        {"id": "b", "keyword_score": 1.0},
        {"id": "c", "keyword_score": 0.9},
    ]

    merged = _rrf_merge(vector_items, keyword_items, top_n=3)

    assert merged[0]["id"] == "b"
    assert {item["id"] for item in merged} == {"a", "b", "c"}
    assert "rrf_score" in merged[0]


def test_retriever_combines_vector_and_keyword_lanes(tmp_path) -> None:
    """测试 Retriever 同时使用 vector lane 和 keyword lane。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        retriever = Retriever(
            store=store,
            embedder=DeterministicEmbeddingProvider({"回答风格": [1.0, 0.0]}),
            score_threshold=0.0,
        )

        try:
            store.upsert_item(
                memory_type="preference",
                summary="用户喜欢简洁回答。",
                embedding=[1.0, 0.0],
            )
            store.upsert_item(
                memory_type="profile",
                summary="用户有一块 Fitbit Charge 6 手环。",
                embedding=[0.0, 1.0],
            )

            results = await retriever.retrieve("回答风格 Fitbit", top_k=5)
            summaries = [item["summary"] for item in results]

            assert "用户喜欢简洁回答。" in summaries
            assert "用户有一块 Fitbit Charge 6 手环。" in summaries
        finally:
            store.close()

    asyncio.run(run())


def test_retriever_builds_injection_block(tmp_path) -> None:
    """测试 Retriever 可以生成 prompt 注入块。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        retriever = Retriever(
            store=store,
            embedder=DeterministicEmbeddingProvider({"回答风格": [1.0, 0.0]}),
            score_threshold=0.0,
        )

        try:
            store.upsert_item(
                memory_type="preference",
                summary="用户喜欢先给结论再解释。",
                embedding=[1.0, 0.0],
                source_ref='["cli:default:0"]',
            )
            hits = await retriever.retrieve("回答风格", top_k=3)
            block, injected_ids = retriever.build_injection_block(hits)

            assert "用户喜欢先给结论再解释" in block
            assert injected_ids
        finally:
            store.close()

    asyncio.run(run())


def test_retriever_keyword_lane_with_chinese_terms(tmp_path) -> None:
    """测试中文 query 经过 jieba 分词后可以命中 keyword lane。"""

    async def run() -> None:
        store = MemoryStore2(tmp_path / "memory2.db")
        retriever = Retriever(
            store=store,
            embedder=DeterministicEmbeddingProvider({"长江大桥": [1.0, 0.0]}),
            score_threshold=0.99,
        )

        try:
            store.upsert_item(
                memory_type="event",
                summary="用户今天路过了上海市长江大桥。",
                embedding=[0.0, 1.0],
            )

            results = await retriever.retrieve("长江大桥", top_k=5)
            summaries = [item["summary"] for item in results]

            assert "用户今天路过了上海市长江大桥。" in summaries
        finally:
            store.close()

    asyncio.run(run())