from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from raven_agent.event_bus import EventBus
from raven_agent.plugins import BuiltinPluginSpec, PluginManager, plugin_registry
from raven_agent.plugins.builtins import tool_loop_guard as _builtin_module
from raven_agent.plugins.builtins.tool_loop_guard import ToolLoopGuardPlugin
from raven_agent.tools import ToolExecutionRequest, ToolExecutor, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _reset_plugin_handlers():
    """重新注册内置插件 handler，并在结束时清理已注册插件实例。

    其它测试文件可能调用 plugin_registry.clear() 清掉内置插件 handler metadata，
    模块缓存导致装饰器不会重跑。这里先移除该模块残留 metadata，再 reload，确保恰好
    注册一份（tool_loop_guard 的 @on_tool_pre 没有去重，重复注册会让计数翻倍）。

    输入:
        无。

    输出:
        None。
    """

    plugin_registry.remove_plugin(_builtin_module.__name__)
    importlib.reload(_builtin_module)
    yield
    for import_path in list(getattr(plugin_registry, "_instances", {}).keys()):
        plugin_registry.remove_plugin(import_path)


def test_tool_loop_guard_denies_after_repeat_limit(tmp_path) -> None:
    """测试连续相同工具调用达到阈值后被拦截。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        manager = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=ToolRegistry(),
            workspace=tmp_path,
            builtin_specs=[BuiltinPluginSpec(name="tool_loop_guard", plugin_class=ToolLoopGuardPlugin)],
        )
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, _arguments: dict[str, Any]) -> ToolResult:
            """固定回显。"""

            return ToolResult(text="ok", metadata={"ok": True})

        statuses = []
        for _ in range(3):
            result = await executor.execute(
                ToolExecutionRequest(
                    call_id="c",
                    tool_name="echo",
                    arguments={"text": "same"},
                    session_key="cli:default",
                ),
                invoke,
            )
            statuses.append(result.status)

        # 前两次放行，第三次（达到 repeat_limit=3）被拦截。
        assert statuses == ["success", "success", "denied"]

    asyncio.run(run())


def test_tool_loop_guard_resets_on_different_call(tmp_path) -> None:
    """测试切换工具或参数后重置计数。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        manager = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=ToolRegistry(),
            workspace=tmp_path,
            builtin_specs=[BuiltinPluginSpec(name="tool_loop_guard", plugin_class=ToolLoopGuardPlugin)],
        )
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, _arguments: dict[str, Any]) -> ToolResult:
            """固定回显。"""

            return ToolResult(text="ok", metadata={"ok": True})

        async def call(text: str) -> str:
            """执行一次 echo 并返回状态。"""

            result = await executor.execute(
                ToolExecutionRequest(
                    call_id="c",
                    tool_name="echo",
                    arguments={"text": text},
                    session_key="cli:default",
                ),
                invoke,
            )
            return result.status

        assert await call("a") == "success"
        assert await call("a") == "success"
        # 切换参数，计数重置。
        assert await call("b") == "success"
        assert await call("b") == "success"

    asyncio.run(run())