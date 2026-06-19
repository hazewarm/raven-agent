from __future__ import annotations

import asyncio
import importlib
import sqlite3

import pytest

from raven_agent.event_bus import EventBus
from raven_agent.lifecycle import AfterTurnCtx
from raven_agent.plugins import BuiltinPluginSpec, PluginManager, plugin_registry
from raven_agent.plugins.builtins.observe import plugin as _builtin_module
from raven_agent.plugins.builtins.observe.plugin import ObservePlugin
from raven_agent.tools import ToolExecutionRequest, ToolExecutor, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _reset_plugin_handlers():
    """重新注册内置插件 handler，并在结束时清理已注册插件实例。

    其它测试文件可能调用 plugin_registry.clear() 清掉内置插件 handler metadata，
    模块缓存导致装饰器不会重跑。这里先移除该模块残留 metadata，再 reload，确保恰好注册一份。

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


def test_observe_records_turn_and_tool_call(tmp_path) -> None:
    """测试 observe 把 turn 与 tool_call 写入 observe.db。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        bus = EventBus()
        manager = PluginManager(
            plugin_dirs=[],
            event_bus=bus,
            tool_registry=ToolRegistry(),
            workspace=tmp_path,
            builtin_specs=[BuiltinPluginSpec(name="observe", plugin_class=ObservePlugin)],
        )
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)

        # 触发一次 after_turn 事件（observe 监听 AfterTurnCtx）。
        await bus.observe(
            AfterTurnCtx(
                session_key="cli:default",
                channel="cli",
                chat_id="default",
                reply="hello",
                tools_used=("echo",),
                outbound_metadata={"cited_memory_ids": ["m1"]},
            )
        )

        async def invoke(_name: str, arguments: dict[str, object]) -> ToolResult:
            """回显工具结果。"""

            return ToolResult(text="ok", metadata={"ok": True})

        await executor.execute(
            ToolExecutionRequest(
                call_id="c1",
                tool_name="echo",
                arguments={"text": "hi"},
                session_key="cli:default",
            ),
            invoke,
        )

        # 等待异步写库 flush。
        writer = manager  # 仅占位，下面通过 drain 等待
        instance = plugin_registry.get_instance("builtin:observe")
        await instance._writer.drain()  # type: ignore[attr-defined]
        await manager.terminate_all()

        db_path = tmp_path / "observe" / "observe.db"
        conn = sqlite3.connect(str(db_path))
        try:
            turn_rows = conn.execute("SELECT reply, cited_memory_ids FROM turns").fetchall()
            tool_rows = conn.execute("SELECT tool_name, status FROM tool_calls").fetchall()
        finally:
            conn.close()

        assert turn_rows and turn_rows[0][0] == "hello"
        assert tool_rows and tool_rows[0] == ("echo", "success")

    asyncio.run(run())