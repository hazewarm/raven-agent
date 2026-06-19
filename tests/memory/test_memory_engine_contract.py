from __future__ import annotations

import asyncio
from datetime import datetime

from raven_agent.memory import (
    DisabledMemoryEngine,
    EngineProfile,
    EvidenceRef,
    MemoryCapability,
    MemoryEngine,
    MemoryIngestRequest,
    MemoryMutation,
    MemoryQuery,
    MemoryQueryFilters,
    MemoryRecord,
    MemoryScope,
)


def test_memory_scope_keeps_session_and_channel_identity() -> None:
    """测试 MemoryScope 保存 session/channel/chat 作用域。"""

    scope = MemoryScope(session_key="cli:default", channel="cli", chat_id="default")

    assert scope.session_key == "cli:default"
    assert scope.channel == "cli"
    assert scope.chat_id == "default"


def test_memory_query_filters_normalize_kinds_and_freeze_hints() -> None:
    """测试 MemoryQueryFilters 会清理 kinds 并冻结 hints。"""

    filters = MemoryQueryFilters(
        kinds=("event", "", " preference "),
        hints={"require_scope_match": True},
    )

    assert filters.kinds == ("event", "preference")
    assert filters.hints["require_scope_match"] is True
    try:
        filters.hints["x"] = 1  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("hints should be read-only")


def test_memory_mutation_normalizes_ids_and_freezes_metadata() -> None:
    """测试 MemoryMutation 会清理 ids 并冻结 metadata。"""

    mutation = MemoryMutation(
        kind="forget",
        ids=("mem-1", "", " mem-2 "),
        metadata={"reason": "wrong"},
    )

    assert mutation.ids == ("mem-1", "mem-2")
    assert mutation.metadata["reason"] == "wrong"
    try:
        mutation.metadata["x"] = 1  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("metadata should be read-only")


def test_memory_record_can_carry_evidence_refs() -> None:
    """测试 MemoryRecord 可以携带 EvidenceRef。"""

    evidence = EvidenceRef(
        kind="message_range",
        refs=["m1", "m2"],
        resolver="session",
        source_ref="cli:default@0-2",
    )
    record = MemoryRecord(
        id="mem-1",
        kind="preference",
        summary="用户喜欢简洁回答。",
        score=0.9,
        engine_kind="test",
        evidence=[evidence],
        signals={"source": "unit-test"},
        injected=True,
    )

    assert record.evidence[0].refs == ["m1", "m2"]
    assert record.signals["source"] == "unit-test"
    assert record.injected is True


def test_disabled_memory_engine_describes_empty_capabilities() -> None:
    """测试 DisabledMemoryEngine 会声明空能力。"""

    engine = DisabledMemoryEngine()
    descriptor = engine.describe()

    assert isinstance(engine, MemoryEngine)
    assert descriptor.name == "disabled"
    assert descriptor.profile == EngineProfile.CONTEXT_RESOURCE_ENGINE
    assert descriptor.capabilities == frozenset()
    assert MemoryCapability.RETRIEVE_SEMANTIC not in descriptor.capabilities


def test_disabled_memory_engine_query_returns_empty_trace() -> None:
    """测试 DisabledMemoryEngine 查询返回空结果和 disabled trace。"""

    async def run() -> None:
        engine = DisabledMemoryEngine()

        result = await engine.query(
            MemoryQuery(
                text="用户喜欢什么回答风格？",
                intent="answer",
                scope=MemoryScope(session_key="cli:default"),
            )
        )

        assert result.text_block == ""
        assert result.records == []
        assert result.trace["engine"] == "disabled"
        assert result.trace["intent"] == "answer"
        assert result.trace["reason"] == "disabled"

    asyncio.run(run())


def test_disabled_memory_engine_ingest_rejects_request() -> None:
    """测试 DisabledMemoryEngine 会拒绝 ingest。"""

    async def run() -> None:
        engine = DisabledMemoryEngine()

        result = await engine.ingest(
            MemoryIngestRequest(
                content={"user_message": "以后用中文"},
                source_kind="conversation_turn",
                scope=MemoryScope(session_key="cli:default"),
            )
        )

        assert result.accepted is False
        assert result.summary == "semantic memory disabled"
        assert result.raw["reason"] == "disabled"
        assert result.raw["source_kind"] == "conversation_turn"

    asyncio.run(run())


def test_disabled_memory_engine_mutate_rejects_semantic_mutations() -> None:
    """测试 DisabledMemoryEngine 会拒绝 remember / update / forget / restore。"""

    async def run() -> None:
        engine = DisabledMemoryEngine()

        remember = await engine.mutate(
            MemoryMutation(
                kind="remember",
                summary="用户喜欢简洁回答。",
                memory_kind="preference",
            )
        )
        update = await engine.mutate(
            MemoryMutation(
                kind="update",
                ids=("mem-1",),
                summary="用户喜欢直接但保留关键解释的回答。",
                memory_kind="preference",
            )
        )
        forget = await engine.mutate(
            MemoryMutation(
                kind="forget",
                ids=("mem-1", "mem-2"),
            )
        )
        restore = await engine.mutate(
            MemoryMutation(
                kind="restore",
                ids=("mem-3",),
            )
        )

        assert remember.accepted is False
        assert remember.status == "disabled"
        assert update.accepted is False
        assert update.status == "disabled"
        assert update.missing_ids == ["mem-1"]
        assert forget.accepted is False
        assert forget.status == "disabled"
        assert forget.missing_ids == ["mem-1", "mem-2"]
        assert restore.accepted is False
        assert restore.status == "disabled"
        assert restore.missing_ids == ["mem-3"]

    asyncio.run(run())


def test_disabled_memory_engine_admin_methods_return_empty_values() -> None:
    """测试 DisabledMemoryEngine 的 admin 方法返回空值。"""

    engine = DisabledMemoryEngine()

    assert engine.tool_profile().recall is None
    assert engine.keyword_match_procedures(["search"]) == []
    assert engine.list_events_by_time_range(
        datetime(2026, 5, 1),
        datetime(2026, 5, 29),
    ) == []
    assert engine.list_items_for_dashboard() == ([], 0)
    assert engine.get_item_for_dashboard("mem-1") is None
    assert engine.update_item_for_dashboard("mem-1", status="active") is None
    assert engine.delete_item("mem-1") is False
    assert engine.delete_items_batch(["mem-1"]) == 0
    assert engine.find_similar_items_for_dashboard("mem-1") == []