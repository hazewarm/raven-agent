from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from raven_agent.event_bus import EventBus
from raven_agent.plugins import PluginManager, plugin_registry
from raven_agent.tools import ToolExecutionRequest, ToolExecutor, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _clean_plugin_registry():
    """清理全局 plugin registry，避免测试互相污染。

    输入:
        无。

    输出:
        None。
    """

    plugin_registry.clear()
    yield
    plugin_registry.clear()


def _write_plugin(root: Path, name: str, code: str) -> Path:
    """写入测试插件。

    输入:
        root: 插件根目录。
        name: 插件目录名。
        code: plugin.py 源码。

    输出:
        插件目录路径。
    """

    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(code.strip(), encoding="utf-8")
    return plugin_dir


def test_plugin_post_tool_hook_records_trace_metadata(tmp_path) -> None:
    """测试 @on_tool_post 能观察成功工具调用，并在 trace 中带 plugin metadata。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        root = tmp_path / "plugins"
        _write_plugin(
            root,
            "post_observer",
            '''
from raven_agent.plugins import Plugin, on_tool_post


class PostObserver(Plugin):
    """测试用 post tool hook 插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        PostObserver 实例。
    """

    name = "post_observer"

    @on_tool_post(tool_name="echo")
    async def observe(self, event):
        """记录成功工具调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            None。
        """

        self.context.kv_store.set("tool", event.tool_name)
        self.context.kv_store.set("result_text", event.result.text)
''',
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=ToolRegistry())
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, arguments: dict[str, Any]) -> ToolResult:
            """测试用工具执行函数。

            输入:
                _name: 工具名。
                arguments: 工具参数。

            输出:
                ToolResult。
            """

            return ToolResult(text=f"echo:{arguments['text']}", metadata={"ok": True})

        result = await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="echo", arguments={"text": "hi"}),
            invoke,
        )

        trace = result.post_hook_trace[0]
        assert trace.metadata["source_type"] == "plugin"
        assert trace.metadata["plugin_id"] == "post_observer"
        assert trace.metadata["handler"] == "observe"
        assert (root / "post_observer" / ".kv.json").exists()

    asyncio.run(run())


def test_plugin_error_tool_hook_observes_failed_tool(tmp_path) -> None:
    """测试 @on_tool_error 能观察失败工具调用。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        root = tmp_path / "plugins"
        _write_plugin(
            root,
            "error_observer",
            '''
from raven_agent.plugins import Plugin, on_tool_error


class ErrorObserver(Plugin):
    """测试用 error tool hook 插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        ErrorObserver 实例。
    """

    name = "error_observer"

    @on_tool_error(tool_name="boom")
    async def observe(self, event):
        """记录失败工具调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            None。
        """

        self.context.kv_store.set("error", event.error)
''',
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=ToolRegistry())
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        async def invoke(_name: str, _arguments: dict[str, Any]) -> ToolResult:
            """测试用失败工具执行函数。

            输入:
                _name: 工具名。
                _arguments: 工具参数。

            输出:
                metadata.ok=False 的 ToolResult。
            """

            return ToolResult(text="boom failed", metadata={"ok": False})

        result = await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="boom", arguments={}),
            invoke,
        )

        assert result.status == "error"
        assert result.post_hook_trace[0].metadata["plugin_id"] == "error_observer"
        assert "boom failed" in (root / "error_observer" / ".kv.json").read_text(encoding="utf-8")

    asyncio.run(run())


def test_tool_executor_can_remove_plugin_hooks(tmp_path) -> None:
    """测试 ToolExecutor.remove_hooks_by_prefix 可以移除插件 hook。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        root = tmp_path / "plugins"
        _write_plugin(
            root,
            "rewriter",
            '''
from raven_agent.plugins import Plugin, on_tool_pre


class Rewriter(Plugin):
    """测试用 pre tool hook 改参插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        Rewriter 实例。
    """

    name = "rewriter"

    @on_tool_pre(tool_name="echo")
    async def rewrite(self, event):
        """把 echo.text 改写为固定值。

        输入:
            event: PluginToolHookEvent。

        输出:
            新的工具参数字典。
        """

        return {"text": "rewritten"}
''',
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=ToolRegistry())
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)
        executor.remove_hooks_by_prefix("plugin:")
        captured: dict[str, Any] = {}

        async def invoke(_name: str, arguments: dict[str, Any]) -> ToolResult:
            """测试用工具执行函数。

            输入:
                _name: 工具名。
                arguments: 工具参数。

            输出:
                ToolResult。
            """

            captured.update(arguments)
            return ToolResult(text="ok", metadata={"ok": True})

        await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="echo", arguments={"text": "raw"}),
            invoke,
        )

        assert captured["text"] == "raw"

    asyncio.run(run())