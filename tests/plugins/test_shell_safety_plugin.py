from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from raven_agent.event_bus import EventBus
from raven_agent.plugins import BuiltinPluginSpec, PluginManager, plugin_registry
from raven_agent.plugins.builtins import shell_safety as _builtin_module
from raven_agent.plugins.builtins.shell_safety import ShellSafetyPlugin
from raven_agent.tools import ToolExecutionRequest, ToolExecutor, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _reset_plugin_handlers():
    """重新注册内置插件 handler，并在结束时清理已注册插件实例。

    其它测试文件的 _clean_plugin_registry 可能调用 plugin_registry.clear()，
    清掉内置插件在 import 时注册的 handler metadata；由于模块已缓存，装饰器不会
    自动重跑。这里先移除该模块残留 metadata，再 reload，确保恰好注册一份。

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


def test_shell_safety_denies_interactive_command(tmp_path) -> None:
    """测试 shell_safety 拦截交互式 shell 命令。

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
            builtin_specs=[BuiltinPluginSpec(name="shell_safety", plugin_class=ShellSafetyPlugin)],
        )
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, _arguments: dict[str, Any]) -> ToolResult:
            """正常情况下应被拦截，不会调用到这里。"""

            return ToolResult(text="should not run", metadata={"ok": True})

        result = await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="shell", arguments={"command": "vim a.txt"}),
            invoke,
        )

        assert result.status == "denied"
        assert "vim" in result.output

    asyncio.run(run())


def test_shell_safety_allows_safe_command(tmp_path) -> None:
    """测试 shell_safety 放行安全命令。

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
            builtin_specs=[BuiltinPluginSpec(name="shell_safety", plugin_class=ShellSafetyPlugin)],
        )
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, arguments: dict[str, Any]) -> ToolResult:
            """回显命令。"""

            return ToolResult(text=f"ran:{arguments['command']}", metadata={"ok": True})

        result = await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="shell", arguments={"command": "ls -la"}),
            invoke,
        )

        assert result.status == "success"
        assert result.output == "ran:ls -la"

    asyncio.run(run())