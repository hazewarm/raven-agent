from __future__ import annotations

import asyncio
from datetime import datetime

from raven_agent.memory import MemoryIngestRequest, MemoryMutation, MemoryQuery, MemoryQueryFilters
from raven_agent.memory2 import EmbeddingProvider, Memory2Engine, MemoryStore2


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
        """返回文本对应的测试向量。

        参数:
            text: 要编码的文本。

        返回:
            测试向量；未命中时返回 [0.0, 0.0]。
        """

        return list(self._mapping.get(text, [0.0, 0.0]))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量返回文本对应的测试向量。

        参数:
            texts: 要编码的文本列表。

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


def test_memory2_engine_matches_embedding_provider_protocol() -> None:
    """测试测试 provider 满足 EmbeddingProvider 协议。"""

    provider = DeterministicEmbeddingProvider({})

    assert isinstance(provider, EmbeddingProvider)


def test_memory2_engine_remember_and_query_records(tmp_path) -> None:
    """测试 Memory2Engine 可以 remember 并 query 记忆。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider(
                {
                    "用户喜欢简洁回答。": [1.0, 0.0],
                    "回答风格": [1.0, 0.0],
                }
            ),
        )

        try:
            mutation = await engine.mutate(
                MemoryMutation(
                    kind="remember",
                    summary="用户喜欢简洁回答。",
                    memory_kind="preference",
                    source_ref="test@1",
                )
            )
            result = await engine.query(
                MemoryQuery(
                    text="回答风格",
                    filters=MemoryQueryFilters(kinds=("preference",)),
                    limit=3,
                )
            )

            assert mutation.accepted is True
            assert mutation.item_id
            assert len(result.records) == 1
            assert result.records[0].summary == "用户喜欢简洁回答。"
            assert result.records[0].kind == "preference"
            assert result.trace["engine"] == "memory2"
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_forget_hides_item_from_query(tmp_path) -> None:
    """测试 forget 会把条目标记为 superseded 并隐藏检索结果。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider(
                {
                    "旧偏好": [1.0, 0.0],
                    "查询偏好": [1.0, 0.0],
                }
            ),
        )

        try:
            remembered = await engine.mutate(
                MemoryMutation(kind="remember", summary="旧偏好", memory_kind="preference")
            )
            forgotten = await engine.mutate(
                MemoryMutation(kind="forget", ids=(remembered.item_id,))
            )
            result = await engine.query(MemoryQuery(text="查询偏好"))

            assert forgotten.accepted is True
            assert forgotten.affected_ids == [remembered.item_id]
            assert result.records == []
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_update_records_replacement(tmp_path) -> None:
    """测试 update 会写入新条目、失效旧条目并记录 replacement。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider(
                {
                    "旧偏好": [1.0, 0.0],
                    "新偏好": [0.9, 0.0],
                }
            ),
        )

        try:
            remembered = await engine.mutate(
                MemoryMutation(kind="remember", summary="旧偏好", memory_kind="preference")
            )
            updated = await engine.mutate(
                MemoryMutation(
                    kind="update",
                    ids=(remembered.item_id,),
                    summary="新偏好",
                    memory_kind="preference",
                )
            )
            replacements = store.list_replacements()
            old_item = store.get_item(remembered.item_id)

            assert updated.accepted is True
            assert updated.status == "updated"
            assert old_item is not None
            assert old_item.status == "superseded"
            assert len(replacements) == 1
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_query_parses_json_message_source_ref(tmp_path) -> None:
    """测试 Memory2Engine 会把 JSON message-id source_ref 转成 message_range evidence。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider(
                {
                    "用户有 Fitbit Charge 6。": [1.0, 0.0],
                    "Fitbit 型号": [1.0, 0.0],
                }
            ),
        )

        try:
            await engine.mutate(
                MemoryMutation(
                    kind="remember",
                    summary="用户有 Fitbit Charge 6。",
                    memory_kind="profile",
                    source_ref='["cli:default:0", "cli:default:1"]#h:abc',
                )
            )
            result = await engine.query(MemoryQuery(text="Fitbit 型号", limit=3))

            assert result.records
            evidence = result.records[0].evidence[0]
            assert evidence.kind == "message_range"
            assert evidence.refs == ["cli:default:0", "cli:default:1"]
        finally:
            await engine.close()

    asyncio.run(run())

def test_memory2_engine_context_query_returns_text_block_and_injected_records(tmp_path) -> None:
    """测试 context 查询返回注入块并标记 injected records。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider(
                {
                    "用户喜欢先给结论。": [1.0, 0.0],
                    "回答风格": [1.0, 0.0],
                }
            ),
        )

        try:
            await engine.mutate(
                MemoryMutation(
                    kind="remember",
                    summary="用户喜欢先给结论。",
                    memory_kind="preference",
                )
            )
            result = await engine.query(MemoryQuery(text="回答风格", intent="context", limit=3))

            assert "用户喜欢先给结论" in result.text_block
            assert result.records
            assert any(record.injected for record in result.records)
        finally:
            await engine.close()

    asyncio.run(run())

def test_memory2_engine_timeline_query_returns_events_in_range(tmp_path) -> None:
    """测试 timeline 查询按时间范围返回事件。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider({}),
        )

        try:
            store.upsert_item(
                memory_type="event",
                summary="用户讨论 Memory2 检索设计。",
                embedding=[1.0, 0.0],
                happened_at="2026-05-20T10:00:00+00:00",
            )
            result = await engine.query(
                MemoryQuery(
                    text="Memory2",
                    intent="timeline",
                    filters=MemoryQueryFilters(
                        time_start=datetime.fromisoformat("2026-05-01T00:00:00+00:00"),
                        time_end=datetime.fromisoformat("2026-06-01T00:00:00+00:00"),
                    ),
                )
            )

            assert len(result.records) == 1
            assert result.records[0].summary == "用户讨论 Memory2 检索设计。"
            assert result.trace["intent"] == "timeline"
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_remember_procedure_enriches_rule_schema(tmp_path) -> None:
    """测试 Memory2Engine._remember 会通过 Memorizer 补齐 procedure metadata。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        summary = "运行 shell 命令时必须先说明风险。"
        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider({summary: [1.0, 0.0]}),
        )
        try:
            result = await engine.mutate(
                MemoryMutation(
                    kind="remember",
                    memory_kind="procedure",
                    summary=summary,
                    metadata={"tool_requirement": "shell"},
                )
            )
            item = store.get_item(result.item_id)

            assert result.accepted is True
            assert item is not None
            assert item.extra_json["rule_schema"]["required_tools"] == ["shell"]
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_ingest_consolidation_event_semantic_dedup(tmp_path) -> None:
    """测试 Memory2Engine.ingest 会对 consolidation event 做 semantic dedup。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        first = "用户把仓库脱敏后公开发布"
        second = "用户公开了脱敏后的仓库"
        store = MemoryStore2(tmp_path / "memory2.db")
        engine = Memory2Engine(
            store=store,
            embedder=DeterministicEmbeddingProvider({first: [1.0, 0.0], second: [0.99, 0.01]}),
        )
        try:
            await engine.ingest(
                MemoryIngestRequest(
                    content={"summary": first},
                    source_kind="consolidation_event",
                    metadata={"source_ref": '["cli:default:0"]'},
                )
            )
            result = await engine.ingest(
                MemoryIngestRequest(
                    content={"summary": second},
                    source_kind="consolidation_event",
                    metadata={"source_ref": '["cli:default:1"]'},
                )
            )

            assert result.accepted is True
            assert result.summary.startswith("semantic_dedup:")
            assert len(store.list_by_type("event")) == 1
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory2_engine_tool_profile_exposes_three_specs(tmp_path) -> None:
    """测试 Memory2Engine.tool_profile 暴露 recall / memorize / forget 三件套。"""

    store = MemoryStore2(tmp_path / "memory2.db")
    engine = Memory2Engine(store=store, embedder=DeterministicEmbeddingProvider({}))

    profile = engine.tool_profile()

    assert profile.recall is not None
    assert profile.memorize is not None
    assert profile.forget is not None
    assert profile.recall.risk == "read-only"
    assert profile.memorize.risk == "write"
    assert profile.forget.risk == "write"
    assert "query" in profile.recall.parameters["properties"]
    assert profile.memorize.parameters["required"] == ["summary"]

    store.close()