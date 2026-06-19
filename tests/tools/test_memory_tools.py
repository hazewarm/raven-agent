from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from raven_agent.memory import MemoryMutation
from raven_agent.memory2 import Memory2Engine, MemoryStore2
from raven_agent.tools import ForgetMemoryTool, MemorizeTool, RecallMemoryTool
import raven_agent.tools.memory_tools as memory_tools_module


class _DeterministicEmbedder:
    """测试用 deterministic embedding provider。

    参数:
        mapping: 文本到向量的映射。

    返回:
        _DeterministicEmbedder 实例。
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed_text(self, text: str) -> list[float]:
        """返回文本对应测试向量。

        参数:
            text: 输入文本。

        返回:
            命中映射的向量；未命中返回 [0.0, 0.0]。
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


def _engine(tmp_path, mapping: dict[str, list[float]]) -> Memory2Engine:
    """构造测试用 Memory2Engine。

    参数:
        tmp_path: pytest 临时目录。
        mapping: 文本到向量映射。

    返回:
        Memory2Engine 实例。
    """

    store = MemoryStore2(tmp_path / "memory2.db")
    return Memory2Engine(store=store, embedder=_DeterministicEmbedder(mapping))


def test_memorize_tool_writes_with_bound_source_ref(tmp_path) -> None:
    """测试 memorize 工具写入记忆并绑定 source_ref。"""

    async def run() -> None:
        summary = "用户希望回答先给结论。"
        engine = _engine(tmp_path, {summary: [1.0, 0.0]})
        spec = engine.tool_profile().memorize
        assert spec is not None
        tool = MemorizeTool(engine, spec)
        try:
            text = await tool.execute(
                summary=summary,
                memory_kind="preference",
                current_user_source_ref='["cli:default:0"]',
                channel="cli",
                chat_id="default",
            )

            assert "已记住" in text
            items = engine._store.list_by_type("preference")  # type: ignore[attr-defined]
            assert len(items) == 1
            assert items[0].summary == summary
            assert items[0].source_ref == '["cli:default:0"]'
        finally:
            await engine.close()

    asyncio.run(run())


def test_memorize_tool_collects_tool_requirement_into_metadata(tmp_path) -> None:
    """测试 memorize 把 tool_requirement 收进 metadata 并生成 procedure rule_schema。"""

    async def run() -> None:
        summary = "查 Steam 信息时必须先使用 steam_mcp。"
        engine = _engine(tmp_path, {summary: [1.0, 0.0]})
        spec = engine.tool_profile().memorize
        assert spec is not None
        tool = MemorizeTool(engine, spec)
        try:
            await tool.execute(
                summary=summary,
                memory_kind="procedure",
                tool_requirement="steam_mcp",
            )

            items = engine._store.list_by_type("procedure")  # type: ignore[attr-defined]
            assert len(items) == 1
            assert items[0].extra_json["rule_schema"]["required_tools"] == ["steam_mcp"]
        finally:
            await engine.close()

    asyncio.run(run())


def test_recall_tool_returns_records_with_citation_protocol(tmp_path) -> None:
    """测试 recall 工具返回记忆条目并附带 citation 协议。"""

    async def run() -> None:
        summary = "用户希望回答先给结论。"
        engine = _engine(tmp_path, {summary: [1.0, 0.0], "回答方式": [1.0, 0.0]})
        try:
            await engine.mutate(
                MemoryMutation(
                    kind="remember",
                    summary=summary,
                    memory_kind="preference",
                    source_ref='["cli:default:0"]',
                )
            )
            spec = engine.tool_profile().recall
            assert spec is not None
            tool = RecallMemoryTool(engine, spec)

            raw = await tool.execute(query="回答方式", intent="answer", memory_kind="preference")
            payload = json.loads(raw)

            assert payload["count"] == 1
            assert payload["items"][0]["summary"] == summary
            assert payload["citation_required"] is True
            assert payload["cited_item_ids"] == [payload["items"][0]["id"]]
            evidence = payload["items"][0]["evidence"][0]
            assert evidence["kind"] == "message_range"
            assert evidence["refs"] == ["cli:default:0"]
        finally:
            await engine.close()

    asyncio.run(run())


def test_recall_tool_rejects_invalid_time_filter(tmp_path) -> None:
    """测试 recall 工具对非法 time_filter 返回错误。"""

    async def run() -> None:
        engine = _engine(tmp_path, {})
        spec = engine.tool_profile().recall
        assert spec is not None
        tool = RecallMemoryTool(engine, spec)
        try:
            raw = await tool.execute(query="任何主题", time_filter="not-a-date")
            payload = json.loads(raw)

            assert payload["error"] == "invalid_time_filter"
        finally:
            await engine.close()

    asyncio.run(run())


def test_forget_tool_supersedes_and_reports_missing(tmp_path) -> None:
    """测试 forget 工具 supersede 已有条目并报告缺失 id。"""

    async def run() -> None:
        summary = "用户喜欢用 emoji。"
        engine = _engine(tmp_path, {summary: [1.0, 0.0]})
        try:
            remembered = await engine.mutate(
                MemoryMutation(kind="remember", summary=summary, memory_kind="preference")
            )
            spec = engine.tool_profile().forget
            assert spec is not None
            tool = ForgetMemoryTool(engine, spec)

            raw = await tool.execute(ids=[remembered.item_id, "missing", remembered.item_id])
            payload = json.loads(raw)

            assert payload["requested_ids"] == [remembered.item_id, "missing"]
            assert payload["superseded_ids"] == [remembered.item_id]
            assert payload["missing_ids"] == ["missing"]
            assert engine._store.get_item(remembered.item_id).status == "superseded"  # type: ignore[attr-defined,union-attr]
        finally:
            await engine.close()

    asyncio.run(run())


def test_parse_time_filter_supports_presets_and_ranges(monkeypatch) -> None:
    """测试 time_filter 解析支持预设与日期区间。"""

    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(
        memory_tools_module,
        "_now_local",
        lambda: datetime(2026, 5, 30, 15, 30, tzinfo=tz),
    )

    today = memory_tools_module._parse_time_filter("today")
    assert today == (
        datetime(2026, 5, 30, 0, 0, tzinfo=tz),
        datetime(2026, 5, 31, 0, 0, tzinfo=tz),
    )

    recent = memory_tools_module._parse_time_filter("recent_3d")
    assert recent == (
        datetime(2026, 5, 27, 15, 30, tzinfo=tz),
        datetime(2026, 5, 30, 15, 30, tzinfo=tz),
    )

    date_range = memory_tools_module._parse_time_filter("2026-05-01~2026-05-10")
    assert date_range == (
        datetime(2026, 5, 1, 0, 0, tzinfo=tz),
        datetime(2026, 5, 11, 0, 0, tzinfo=tz),
    )

    assert memory_tools_module._parse_time_filter("") is None
    assert memory_tools_module._parse_time_filter("garbage") is None