from __future__ import annotations

import asyncio

import pytest

from raven_agent.lifecycle import BeforeTurnCtx
from raven_agent.plugins import plugin_registry
from raven_agent.plugins.builtins.memory_rollup import MemoryRollupPlugin
from raven_agent.plugins.context import PluginContext, PluginKVStore


@pytest.fixture(autouse=True)
def _reset_plugin_handlers():
    """只清理已注册插件实例。"""

    yield
    for import_path in list(getattr(plugin_registry, "_instances", {}).keys()):
        plugin_registry.remove_plugin(import_path)


class _FakeOptimizer:
    """记录是否被调用的 fake optimizer。

    输入:
        无。

    输出:
        _FakeOptimizer 实例。
    """

    def __init__(self) -> None:
        self.called = False

    async def optimize(self) -> None:
        """模拟一次归档。"""

        self.called = True


def test_memory_rollup_command_triggers_optimizer(tmp_path) -> None:
    """测试 /memory_rollup 触发 MemoryOptimizer.optimize。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        optimizer = _FakeOptimizer()
        plugin = MemoryRollupPlugin()
        plugin.context = PluginContext(
            event_bus=None,
            tool_registry=None,
            plugin_id="memory_rollup",
            plugin_dir=tmp_path / "memory_rollup",
            kv_store=PluginKVStore(tmp_path / "memory_rollup" / ".kv.json"),
            memory_optimizer=optimizer,
        )
        ctx = BeforeTurnCtx(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            content="/memory_rollup",
        )

        result = await plugin.handle_command(ctx)

        assert optimizer.called is True
        assert result.abort is True
        assert "归档" in result.abort_reply

    asyncio.run(run())