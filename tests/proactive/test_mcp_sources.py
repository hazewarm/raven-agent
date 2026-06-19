"""Tests for SourceStore and McpSourceFetcher."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from raven_agent.proactive.mcp_sources import SourceStore


class TestSourceStore:
    """SourceStore 配置管理测试。"""

    def test_load_empty(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        assert store.load_sources() == []

    def test_add_and_load(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        store.add_source({
            "name": "test-source", "server": "test-server",
            "channel": "alert", "enabled": True,
        })
        sources = store.load_sources()
        assert len(sources) == 1
        assert sources[0]["name"] == "test-source"

    def test_remove_source(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        store.add_source({"name": "s1", "server": "github", "channel": "alert"})
        store.add_source({"name": "s2", "server": "rss", "channel": "content"})
        result = store.remove_source("s1")
        assert result == "ok"
        sources = store.load_sources()
        assert len(sources) == 1

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        result = store.remove_source("nonexistent")
        assert "未找到" in result

    def test_add_updates_existing(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        store.add_source({"name": "s1", "server": "old", "channel": "alert"})
        store.add_source({"name": "s1", "server": "new", "channel": "content"})
        sources = store.load_sources()
        assert len(sources) == 1
        assert sources[0]["server"] == "new"

    def test_list_sources(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        store.add_source({"name": "s1", "server": "github", "channel": "alert"})
        result = store.list_sources()
        assert "s1" in result

    def test_load_filters_disabled(self, tmp_path: Path) -> None:
        store = SourceStore(tmp_path / "proactive_sources.json")
        store.add_source(
            {"name": "enabled", "server": "srv", "channel": "alert", "enabled": True}
        )
        store.add_source(
            {"name": "disabled", "server": "srv2", "channel": "content", "enabled": False}
        )
        sources = store.load_sources()
        assert len(sources) == 1
        assert sources[0]["name"] == "enabled"


class TestMcpSourceFetcher:
    """McpSourceFetcher 拉取测试。"""

    @pytest.mark.asyncio
    async def test_fetch_alerts(self) -> None:
        from raven_agent.proactive.mcp_sources import McpSourceFetcher
        from raven_agent.proactive.contracts import AlertContract

        mock_registry = MagicMock()
        mock_registry.connected_server_names.return_value = {"github"}
        mock_registry.call_tool = AsyncMock(return_value=[{
            "kind": "alert", "event_id": "ev-1", "title": "CI failed",
            "content": "pytest failed", "severity": "high",
            "suggested_tone": "direct",
        }])

        source_store = MagicMock()
        source_store.load_sources.return_value = [
            {"name": "gh", "server": "github", "channel": "alert", "get_tool": "get_events"},
        ]

        fetcher = McpSourceFetcher(mock_registry, source_store)
        alerts = await fetcher.fetch_alerts()

        assert len(alerts) == 1
        assert isinstance(alerts[0], AlertContract)
        assert alerts[0].item_id == "github:ev-1"

    @pytest.mark.asyncio
    async def test_fetch_skips_unconnected_server(self) -> None:
        from raven_agent.proactive.mcp_sources import McpSourceFetcher

        mock_registry = MagicMock()
        mock_registry.connected_server_names.return_value = set()
        mock_registry.call_tool = AsyncMock()

        source_store = MagicMock()
        source_store.load_sources.return_value = [
            {"name": "gh", "server": "github", "channel": "alert", "get_tool": "get_events"},
        ]

        fetcher = McpSourceFetcher(mock_registry, source_store)
        alerts = await fetcher.fetch_alerts()

        assert alerts == []
        mock_registry.call_tool.assert_not_called()