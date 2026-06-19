from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from raven_agent.agent import AgentRunResult
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage
from raven_agent.lifecycle import LifecycleModules
from raven_agent.messages import ChatMessage
from raven_agent.plugins import PluginManager, plugin_registry
from raven_agent.prompt import PromptBuilder
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore
from raven_agent.turn_pipeline import PassiveTurnPipeline, PassiveTurnPipelineDeps


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


class _FakeAgent:
    """测试用假 Agent。

    输入:
        content: 固定回复文本。

    输出:
        _FakeAgent 实例。
    """

    def __init__(self, content: str = "pong") -> None:
        self.content = content
        self.messages: list[ChatMessage] = []
        self.session_key = ""
        self.tool_context: dict[str, object] = {}

    async def run(
        self,
        messages: list[ChatMessage],
        session_key: str = "__default__",
        tool_context: dict[str, object] | None = None,
        lifecycle: object | None = None,
        channel: str = "",
        chat_id: str = "",
    ) -> AgentRunResult:
        """模拟 Agent 单轮运行。

        输入:
            messages: Pipeline 构造出的模型输入。
            session_key: 当前会话 key。
            tool_context: 每轮工具上下文。
            lifecycle: step 生命周期；本 fake 不调用。
            channel: 当前渠道。
            chat_id: 当前聊天标识。

        输出:
            固定 AgentRunResult。
        """

        self.messages = messages
        self.session_key = session_key
        self.tool_context = dict(tool_context or {})
        return AgentRunResult(content=self.content, iterations=1, tools_used=[])


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


def _build_pipeline(
    tmp_path: Path,
    modules: LifecycleModules,
    event_bus: EventBus,
    agent: _FakeAgent | None = None,
) -> tuple[PassiveTurnPipeline, SessionManager, _FakeAgent]:
    """构建测试用 Pipeline。

    输入:
        tmp_path: pytest 临时目录。
        modules: 插件 lifecycle modules。
        event_bus: 当前 EventBus。
        agent: 可选 fake agent。

    输出:
        (pipeline, sessions, fake_agent) 三元组。
    """

    sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
    fake_agent = agent or _FakeAgent()
    pipeline = PassiveTurnPipeline(
        PassiveTurnPipelineDeps(
            sessions=sessions,
            prompt_builder=PromptBuilder(system_prompt="sys"),
            agent=fake_agent,  # type: ignore[arg-type]
            event_bus=event_bus,
            lifecycle_modules=modules,
        )
    )
    return pipeline, sessions, fake_agent

def test_on_prompt_render_decorator_injects_hint(tmp_path) -> None:
    """测试 @on_prompt_render 可以向 PromptRenderCtx 写入 hint。

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
            "prompt_hint",
            '''
from raven_agent.plugins import Plugin, on_prompt_render


class PromptHint(Plugin):
    """测试用 prompt hint 插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        PromptHint 实例。
    """

    name = "prompt_hint"

    @on_prompt_render()
    async def add_hint(self, ctx):
        """向 PromptRenderCtx 追加 hint。

        输入:
            ctx: 当前 PromptRenderCtx。

        输出:
            修改后的 PromptRenderCtx。
        """

        ctx.extra_hints.append("plugin hint")
        return ctx
''',
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)
        await manager.load_all()
        pipeline, sessions, agent = _build_pipeline(tmp_path, manager.lifecycle_modules(), bus)
        try:
            await pipeline.run(InboundMessage(channel="cli", sender="local", chat_id="default", content="hi"))
        finally:
            sessions.close()

        msgs = [message.content for message in agent.messages]
        assert "sys" in msgs[0]
        assert msgs[1:] == ["hi", "[hint] plugin hint"]

    asyncio.run(run())

def test_before_turn_abort_skips_agent_and_persists_reply(tmp_path) -> None:
    """测试 before_turn 插件可以提前结束本轮。

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
            "aborter",
            '''
from raven_agent.plugins import Plugin, on_before_turn


class Aborter(Plugin):
    """测试用 before_turn abort 插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        Aborter 实例。
    """

    name = "aborter"

    @on_before_turn()
    async def abort(self, ctx):
        """在 /status 命令时提前结束本轮。

        输入:
            ctx: 当前 BeforeTurnCtx。

        输出:
            修改后的 BeforeTurnCtx。
        """

        if ctx.content.strip() == "/status":
            ctx.abort = True
            ctx.abort_reply = "status ok"
            ctx.outbound_metadata["aborted_by"] = "aborter"
        return ctx
''',
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)
        await manager.load_all()
        fake_agent = _FakeAgent(content="should not run")
        pipeline, sessions, agent = _build_pipeline(tmp_path, manager.lifecycle_modules(), bus, fake_agent)
        try:
            outbound = await pipeline.run(
                InboundMessage(channel="cli", sender="local", chat_id="default", content="/status")
            )
            session = sessions.get_or_create("cli:default")
        finally:
            sessions.close()

        assert agent.messages == []
        assert outbound.content == "status ok"
        assert outbound.metadata["aborted"] is True
        assert outbound.metadata["aborted_by"] == "aborter"
        assert [message.content for message in session.messages] == ["/status", "status ok"]

    asyncio.run(run())

def test_after_reasoning_decorator_rewrites_reply(tmp_path) -> None:
    """测试 @on_after_reasoning 可以改写最终回复并追加 metadata。

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
            "cleaner",
            '''
from raven_agent.plugins import Plugin, on_after_reasoning


class Cleaner(Plugin):
    """测试用 after_reasoning 清理插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        Cleaner 实例。
    """

    name = "cleaner"

    @on_after_reasoning()
    async def clean(self, ctx):
        """清理回复中的内部标签。

        输入:
            ctx: 当前 AfterReasoningCtx。

        输出:
            修改后的 AfterReasoningCtx。
        """

        ctx.reply = ctx.reply.replace("[internal]", "").strip()
        ctx.outbound_metadata["cleaned"] = True
        return ctx
''',
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)
        await manager.load_all()
        pipeline, sessions, _agent = _build_pipeline(
            tmp_path,
            manager.lifecycle_modules(),
            bus,
            _FakeAgent(content="hello [internal]"),
        )
        try:
            outbound = await pipeline.run(InboundMessage(channel="cli", sender="local", chat_id="default", content="hi"))
            session = sessions.get_or_create("cli:default")
        finally:
            sessions.close()

        assert outbound.content == "hello"
        assert outbound.metadata["cleaned"] is True
        assert session.messages[-1].content == "hello"

    asyncio.run(run())

def test_after_turn_module_and_decorator_observe_ctx(tmp_path) -> None:
    """测试 after_turn_modules 和 @on_after_turn 都会运行。

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
            "after_turn_observer",
            '''
from raven_agent.plugins import Plugin, on_after_turn


class TelemetryModule:
    """测试用 after_turn telemetry 模块。

    输入:
        无。模块依赖 frame.slots["after_turn:ctx"]。

    输出:
        TelemetryModule 实例。
    """

    slot = "after_turn_observer.telemetry"
    requires = ("after_turn.build_ctx", "after_turn:ctx")

    async def run(self, frame):
        """写入 after_turn telemetry slot。

        输入:
            frame: 当前 AfterTurnFrame。

        输出:
            修改后的 AfterTurnFrame。
        """

        frame.slots["after_turn:telemetry:module_seen"] = True
        return frame


class AfterTurnObserver(Plugin):
    """测试用 after_turn 观察插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        AfterTurnObserver 实例。
    """

    name = "after_turn_observer"

    def after_turn_modules(self):
        """返回 after_turn 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return [TelemetryModule()]

    @on_after_turn()
    async def observe(self, ctx):
        """把 after_turn ctx 写入插件 KV。

        输入:
            ctx: 当前 AfterTurnCtx。

        输出:
            None。
        """

        self.context.kv_store.set("reply", ctx.reply)
        self.context.kv_store.set("module_seen", ctx.extra_metadata.get("module_seen"))
''',
        )
        bus = EventBus()
        manager = PluginManager(plugin_dirs=[root], event_bus=bus)
        await manager.load_all()
        pipeline, sessions, _agent = _build_pipeline(tmp_path, manager.lifecycle_modules(), bus)
        try:
            await pipeline.run(InboundMessage(channel="cli", sender="local", chat_id="default", content="hi"))
            kv_text = (root / "after_turn_observer" / ".kv.json").read_text(encoding="utf-8")
        finally:
            sessions.close()
            await manager.terminate_all()

        assert '"reply": "pong"' in kv_text
        assert '"module_seen": true' in kv_text

    asyncio.run(run())

