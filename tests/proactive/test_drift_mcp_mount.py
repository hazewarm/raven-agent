"""Tests for Drift MountServerTool and requires_mcp filtering."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore
from raven_agent.proactive.drift_tools import MountServerTool, build_drift_tool_registry


def _make_skill(source_dir: Path, state_dir: Path, name: str) -> DriftStateStore:
    """创建包含单个 skill 的 DriftStateStore。SKILL.md 写到 source_dir，状态写入 state_dir。"""
    skill_dir = source_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {name} 的测试描述\n---\n\n# 目标"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return DriftStateStore(source_dir=source_dir, state_dir=state_dir)


class TestMountServerTool:
    """MountServerTool 行为测试。"""

    @pytest.mark.asyncio
    async def test_mount_ok(self) -> None:
        shared = MagicMock()
        shared.get_tool_names_by_source.return_value = {"tool_a", "tool_b"}
        mounted: set[str] = set()

        tool = MountServerTool(shared, mounted)
        result = json.loads(await tool.execute(server="calendar"))

        assert result["ok"] is True
        assert "tool_a" in result["tools"]
        assert mounted == {"tool_a", "tool_b"}

    @pytest.mark.asyncio
    async def test_mount_unknown_server(self) -> None:
        shared = MagicMock()
        shared.get_tool_names_by_source.return_value = set()
        mounted: set[str] = set()

        tool = MountServerTool(shared, mounted)
        result = json.loads(await tool.execute(server="unknown"))

        assert "error" in result


class TestBuildDriftRegistryWithMount:
    """build_drift_tool_registry 含 mount_server 组装测试。"""

    def test_registry_includes_mount_when_mcp_available(self) -> None:
        from raven_agent.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            shared = ToolRegistry()
            shared.get_mcp_server_names = lambda: {"calendar"}

            registry = build_drift_tool_registry(
                ctx=ctx, store=store, state_dir=state_dir,
                shared_tools=shared,
            )
            assert "mount_server" in set(registry.list_names())

    def test_registry_excludes_mount_when_no_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            registry = build_drift_tool_registry(
                ctx=ctx, store=store, state_dir=state_dir,
            )
            assert "mount_server" not in set(registry.list_names())