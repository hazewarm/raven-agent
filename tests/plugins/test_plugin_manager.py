from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from raven_agent.config import (
    AgentConfig,
    Config,
    LLMConfig,
    PluginsConfig,
    ToolsConfig,
    WebSearchConfig,
)
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, TurnStarted
from raven_agent.plugins import PluginManager, plugin_registry
from raven_agent.tools import ToolExecutor, ToolExecutionRequest, ToolRegistry


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


def test_plugin_manager_loads_event_handler(tmp_path) -> None:
    """测试 PluginManager 加载插件并绑定 TurnStarted handler。

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
            "hello",
            """
from raven_agent.plugins import Plugin, on_turn_started


class Hello(Plugin):
    name = "hello"

    @on_turn_started()
    async def touch(self, event):
        event.inbound.metadata["hello_touched"] = True
        return event
""",
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)

        await manager.load_all()
        started = await bus.emit(
            TurnStarted(
                session_key="cli:default",
                inbound=InboundMessage(channel="cli", sender="local", chat_id="default", content="hi"),
            )
        )

        assert manager.loaded_count == 1
        assert started.inbound.metadata["hello_touched"] is True

    asyncio.run(run())


def test_manifest_overrides_metadata(tmp_path) -> None:
    """测试 manifest.yaml 覆盖插件类属性。

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
        plugin_dir = _write_plugin(
            root,
            "manifested",
            """
from raven_agent.plugins import Plugin


class Manifested(Plugin):
    name = "class_name"
""",
        )
        (plugin_dir / "manifest.yaml").write_text(
            "name: manifest_name\nversion: 0.2.0\ndesc: from manifest\nauthor: tester\n",
            encoding="utf-8",
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus())
        await manager.load_all()
        instance = next(iter(plugin_registry._instances.values()))

        assert instance.name == "manifest_name"
        assert instance.version == "0.2.0"
        assert instance.context.plugin_id == "manifest_name"

    asyncio.run(run())


def test_plugin_config_defaults_and_override(tmp_path) -> None:
    """测试 _conf_schema.json 默认值与 plugin_config.json 覆盖。

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
        plugin_dir = _write_plugin(
            root,
            "configured",
            """
from raven_agent.plugins import Plugin


class Configured(Plugin):
    name = "configured"
""",
        )
        (plugin_dir / "_conf_schema.json").write_text(
            json.dumps(
                {
                    "api_key": {"default": "test-key"},
                    "enabled": {"default": True},
                    "max_results": {"default": 10},
                }
            ),
            encoding="utf-8",
        )
        (plugin_dir / "plugin_config.json").write_text(
            json.dumps({"api_key": "override-key", "enabled": False}),
            encoding="utf-8",
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus())
        await manager.load_all()
        instance = next(iter(plugin_registry._instances.values()))

        assert instance.context.config.api_key == "override-key"
        assert instance.context.config.max_results == 10
        assert instance.context.config.enabled is False

    asyncio.run(run())


def test_plugin_disabled_marker_skips_plugin(tmp_path) -> None:
    """测试 plugin.disabled 跳过插件。

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
        plugin_dir = _write_plugin(
            root,
            "disabled",
            """
from raven_agent.plugins import Plugin


class Disabled(Plugin):
    name = "disabled"
""",
        )
        (plugin_dir / "plugin.disabled").write_text("", encoding="utf-8")
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus())
        await manager.load_all()

        assert manager.loaded_count == 0
        assert plugin_registry._instances == {}

    asyncio.run(run())


def test_plugin_tool_registration_and_execution(tmp_path) -> None:
    """测试 @tool 注册 ToolRegistry 工具并可执行。

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
            "weather",
            """
from raven_agent.plugins import Plugin, tool


class Weather(Plugin):
    name = "weather"

    @tool("get_weather", risk="read-only", always_on=True, search_hint="weather city")
    async def get_weather(self, event, city: str) -> str:
        \"\"\"Get current weather.

        Args:
            city: The city name.
        \"\"\"
        return f"{city}: sunny"
""",
        )
        tools = ToolRegistry()
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=tools)
        await manager.load_all()

        result = await tools.execute("get_weather", {"city": "Paris"})
        document = tools.get_document("get_weather")

        assert result.text == "Paris: sunny"
        assert document.source_type == "plugin"
        assert "get_weather" in tools.get_always_on_names()

    asyncio.run(run())


def test_plugin_tool_pre_hook_rewrites_arguments(tmp_path) -> None:
    """测试 @on_tool_pre 改写工具参数。

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
            """
from raven_agent.plugins import Plugin, on_tool_pre


class Rewriter(Plugin):
    name = "rewriter"

    @on_tool_pre(tool_name="echo")
    async def rewrite(self, event):
        args = dict(event.arguments)
        args["text"] = "rewritten"
        return args
""",
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=ToolRegistry())
        await manager.load_all()
        executor = ToolExecutor(manager.tool_hooks)
        captured: dict[str, Any] = {}

        async def invoke(_name: str, arguments: dict[str, Any]):
            """测试用工具执行函数。

            输入:
                _name: 工具名。
                arguments: 最终参数。

            输出:
                ToolResult。
            """

            captured.update(arguments)
            from raven_agent.tools import ToolResult

            return ToolResult(text="ok", metadata={"ok": True})

        result = await executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="echo", arguments={"text": "raw"}),
            invoke,
        )

        assert result.status == "success"
        assert captured["text"] == "rewritten"

    asyncio.run(run())


def test_plugin_manager_collects_lifecycle_modules(tmp_path) -> None:
    """测试 PluginManager 收集 phase modules 进 LifecycleModules。

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
            "phaser",
            """
from raven_agent.plugins import Plugin


class PromptModule:
    slot = "phaser.prompt"
    async def run(self, frame):
        return frame


class AfterReasoningModule:
    slot = "phaser.after_reasoning"
    async def run(self, frame):
        return frame


class Phaser(Plugin):
    name = "phaser"

    def prompt_render_modules(self):
        return [PromptModule()]

    def after_reasoning_modules(self):
        return [AfterReasoningModule()]
""",
        )
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus())
        await manager.load_all()
        modules = manager.lifecycle_modules()

        assert [m.slot for m in modules.prompt_render] == ["phaser.prompt"]
        assert [m.slot for m in modules.after_reasoning] == ["phaser.after_reasoning"]
        assert modules.before_turn == []

    asyncio.run(run())


def test_plugin_manager_rollback_on_init_failure(tmp_path) -> None:
    """测试 initialize 失败时回滚工具、hook 和 lifecycle modules。

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
            "broken",
            """
from raven_agent.plugins import Plugin, tool


class Broken(Plugin):
    name = "broken"

    @tool("broken_tool", always_on=True)
    async def broken_tool(self, event) -> str:
        return "x"

    async def initialize(self):
        raise RuntimeError("boom")
""",
        )
        tools = ToolRegistry()
        manager = PluginManager(plugin_dirs=[root], event_bus=EventBus(), tool_registry=tools)
        await manager.load_all()

        assert manager.loaded_count == 0
        assert "broken_tool" not in tools.list_names()

    asyncio.run(run())


def test_plugin_manager_binds_lifecycle_decorator(tmp_path) -> None:
    """测试 PluginManager 能绑定 @on_before_reasoning 到 EventBus。

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

        from raven_agent.lifecycle import BeforeReasoningCtx

        root = tmp_path / "plugins"
        _write_plugin(
            root,
            "reasoning_hint",
            '''
from raven_agent.plugins import Plugin, on_before_reasoning


class ReasoningHint(Plugin):
    """测试用 before_reasoning hint 插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        ReasoningHint 实例。
    """

    name = "reasoning_hint"

    @on_before_reasoning()
    async def hint(self, ctx):
        """向 BeforeReasoningCtx 追加 hint。

        输入:
            ctx: 当前 BeforeReasoningCtx。

        输出:
            修改后的 BeforeReasoningCtx。
        """

        ctx.extra_hints.append("reasoning hint")
        return ctx
''',
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)
        await manager.load_all()

        ctx = await bus.emit(
            BeforeReasoningCtx(
                session_key="cli:default",
                channel="cli",
                chat_id="default",
                content="hi",
            )
        )

        assert ctx.extra_hints == ["reasoning hint"]

    asyncio.run(run())