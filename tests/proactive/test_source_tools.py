"""Tests for proactive source management tools."""

from pathlib import Path

import pytest

from raven_agent.proactive.mcp_sources import SourceStore
from raven_agent.proactive.source_tools import (
    ProactiveSourceAddTool,
    ProactiveSourceListTool,
    ProactiveSourceRemoveTool,
)


@pytest.fixture
def store(tmp_path: Path) -> SourceStore:
    return SourceStore(tmp_path / "proactive_sources.json")


@pytest.mark.asyncio
async def test_add_tool(store: SourceStore) -> None:
    tool = ProactiveSourceAddTool(store)
    result = await tool.execute(name="test-source", server="github", channel="alert")
    assert "已保存" in result
    assert len(store.load_sources()) == 1


@pytest.mark.asyncio
async def test_add_tool_rejects_empty(store: SourceStore) -> None:
    tool = ProactiveSourceAddTool(store)
    result = await tool.execute(name="", server="", channel="alert")
    assert "错误" in result


@pytest.mark.asyncio
async def test_remove_tool(store: SourceStore) -> None:
    store.add_source({"name": "s1", "server": "srv", "channel": "alert"})
    tool = ProactiveSourceRemoveTool(store)
    result = await tool.execute(name="s1")
    assert result == "ok"


@pytest.mark.asyncio
async def test_list_tool(store: SourceStore) -> None:
    store.add_source({"name": "s1", "server": "srv", "channel": "alert"})
    tool = ProactiveSourceListTool(store)
    result = await tool.execute()
    assert "s1" in result