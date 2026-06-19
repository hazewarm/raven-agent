from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from raven_agent.memory2 import MemoryStore2
from raven_agent.memory2.memorizer import Memorizer, parse_history_entry_happened_at


class FakeEmbedder:
    """测试用 embedding provider。

    参数:
        mapping: 文本到向量的映射。

    返回:
        FakeEmbedder 实例。
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed_text(self, text: str) -> list[float]:
        """返回测试向量。

        参数:
            text: 输入文本。

        返回:
            测试向量；未命中时返回 [0.0, 0.0]。
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


class FakeTagger:
    """测试用 ProcedureTagger。"""

    async def tag(self, summary: str) -> dict[str, object]:
        """返回固定 trigger_tags。

        参数:
            summary: procedure 摘要。

        返回:
            trigger_tags 字典。
        """

        return {"tools": ["web_fetch"], "skills": [], "keywords": ["B站"], "scope": "tool_triggered"}


def test_parse_history_entry_happened_at() -> None:
    """测试从 history entry 解析 happened_at。"""

    assert parse_history_entry_happened_at("[2026-05-30 09:10] 用户测试 Memory2") == "2026-05-30T09:10:00"
    assert parse_history_entry_happened_at("无时间前缀") is None


def test_memorizer_exact_hash_reinforces_duplicate(tmp_path) -> None:
    """测试完全相同 summary 写两次只保留一条并强化。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        memorizer = Memorizer(store=store, embedder=FakeEmbedder({"用户喜欢先给结论": [1.0, 0.0]}))
        try:
            await memorizer.save_item(
                summary="用户喜欢先给结论",
                memory_type="preference",
                extra_json={},
                source_ref="turn1",
            )
            await memorizer.save_item(
                summary="用户喜欢先给结论",
                memory_type="preference",
                extra_json={},
                source_ref="turn2",
            )

            items = store.list_by_type("preference")
            assert len(items) == 1
            assert items[0].reinforcement == 2
        finally:
            store.close()

    asyncio.run(run())


def test_memorizer_semantic_dedup_recent_event(tmp_path) -> None:
    """测试近期近似 event 不重复写入。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        embedder = FakeEmbedder(
            {
                "用户把仓库脱敏后公开发布": [1.0, 0.0],
                "用户公开了脱敏后的仓库": [0.99, 0.01],
            }
        )
        memorizer = Memorizer(store=store, embedder=embedder)
        try:
            await memorizer.save_from_consolidation(
                history_entry="用户把仓库脱敏后公开发布",
                source_ref='["cli:default:0"]',
                scope_channel="cli",
                scope_chat_id="default",
            )
            result = await memorizer.save_from_consolidation(
                history_entry="用户公开了脱敏后的仓库",
                source_ref='["cli:default:1"]',
                scope_channel="cli",
                scope_chat_id="default",
                emotional_weight=8,
            )

            items = store.list_by_type("event")
            assert result.startswith("semantic_dedup:")
            assert len(items) == 1
            assert items[0].reinforcement == 2
            assert items[0].emotional_weight == 8
        finally:
            store.close()

    asyncio.run(run())


def test_memorizer_event_dedup_window_is_seven_days(tmp_path) -> None:
    """测试超过 7 天的近似 event 不会被 semantic dedup。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        store = MemoryStore2(tmp_path / "memory2.db")
        embedder = FakeEmbedder(
            {
                "用户把仓库脱敏后公开发布": [1.0, 0.0],
                "用户公开了脱敏后的仓库": [0.99, 0.01],
            }
        )
        memorizer = Memorizer(store=store, embedder=embedder)
        try:
            await memorizer.save_from_consolidation(
                history_entry="用户把仓库脱敏后公开发布",
                source_ref='["cli:default:0"]',
                scope_channel="cli",
                scope_chat_id="default",
            )
            old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            store._db.execute(
                "UPDATE memory_items SET created_at=?, updated_at=? WHERE memory_type='event'",
                (old_time, old_time),
            )
            store._db.commit()

            await memorizer.save_from_consolidation(
                history_entry="用户公开了脱敏后的仓库",
                source_ref='["cli:default:1"]',
                scope_channel="cli",
                scope_chat_id="default",
            )

            assert len(store.list_by_type("event")) == 2
        finally:
            store.close()

    asyncio.run(run())


def test_memorizer_procedure_enriches_rule_schema_and_trigger_tags(tmp_path) -> None:
    """测试 procedure 写入会补 rule_schema 和 trigger_tags。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        summary = "用户发送 B 站视频链接时必须先使用 web_fetch。"
        store = MemoryStore2(tmp_path / "memory2.db")
        memorizer = Memorizer(
            store=store,
            embedder=FakeEmbedder({summary: [1.0, 0.0]}),
            procedure_tagger=FakeTagger(),  # type: ignore[arg-type]
        )
        try:
            result = await memorizer.save_item_with_supersede(
                summary=summary,
                memory_type="procedure",
                extra_json={"tool_requirement": "web_fetch"},
                source_ref="manual:test",
            )
            item = store.get_item(result.split(":", 1)[1])

            assert item is not None
            assert item.extra_json["rule_schema"]["required_tools"] == ["web_fetch"]
            assert item.extra_json["trigger_tags"]["tools"] == ["web_fetch"]
        finally:
            store.close()

    asyncio.run(run())


def test_memorizer_merges_same_tool_procedure(tmp_path) -> None:
    """测试同 tool_requirement 的近似 procedure 会 merge。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        first = "用户发送 B 站视频链接时必须先使用 web_fetch。"
        second = "用户发送 B 站链接时先使用 web_fetch，然后总结标题。"
        store = MemoryStore2(tmp_path / "memory2.db")
        memorizer = Memorizer(
            store=store,
            embedder=FakeEmbedder({first: [1.0, 0.0], second: [0.99, 0.01]}),
        )
        try:
            await memorizer.save_item_with_supersede(
                summary=first,
                memory_type="procedure",
                extra_json={"tool_requirement": "web_fetch"},
                source_ref="turn1",
            )
            result = await memorizer.save_item_with_supersede(
                summary=second,
                memory_type="procedure",
                extra_json={"tool_requirement": "web_fetch"},
                source_ref="turn2",
            )

            items = store.list_by_type("procedure")
            assert result.startswith("merged:")
            assert len(items) == 1
            assert "总结标题" in items[0].summary
        finally:
            store.close()

    asyncio.run(run())