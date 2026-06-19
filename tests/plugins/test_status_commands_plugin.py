from __future__ import annotations

import asyncio
import importlib

import pytest

from raven_agent.agent import AgentRunResult
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage
from raven_agent.messages import ChatMessage
from raven_agent.plugins import BuiltinPluginSpec, PluginManager, plugin_registry
from raven_agent.plugins.builtins import status_commands as _builtin_module
from raven_agent.plugins.builtins.status_commands import StatusCommandsPlugin
from raven_agent.prompt import PromptBuilder
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore
from raven_agent.tools import ToolRegistry
from raven_agent.turn_pipeline import PassiveTurnPipeline, PassiveTurnPipelineDeps


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


class _FakeAgent:
    """测试用假 Agent。

    输入:
        无。

    输出:
        _FakeAgent 实例。
    """

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def run(self, messages, session_key="__default__", tool_context=None,
                  lifecycle=None, channel="", chat_id="") -> AgentRunResult:
        """记录调用并返回固定结果。"""

        self.messages = messages
        return AgentRunResult(content="should not run", iterations=1, tools_used=[])


def test_status_command_aborts_and_reports(tmp_path) -> None:
    """测试 /status 命令直接返回运行状态，不进入 LLM。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        bus = EventBus()
        sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
        tools = ToolRegistry()
        manager = PluginManager(
            plugin_dirs=[],
            event_bus=bus,
            tool_registry=tools,
            workspace=tmp_path,
            session_manager=sessions,
            builtin_specs=[BuiltinPluginSpec(name="status_commands", plugin_class=StatusCommandsPlugin)],
        )
        await manager.load_all()
        agent = _FakeAgent()
        pipeline = PassiveTurnPipeline(
            PassiveTurnPipelineDeps(
                sessions=sessions,
                prompt_builder=PromptBuilder(system_prompt="sys"),
                agent=agent,  # type: ignore[arg-type]
                event_bus=bus,
                lifecycle_modules=manager.lifecycle_modules(),
            )
        )
        try:
            outbound = await pipeline.run(
                InboundMessage(channel="cli", sender="local", chat_id="default", content="/status")
            )
        finally:
            sessions.close()

        assert agent.messages == []
        assert "运行状态" in outbound.content
        assert outbound.metadata["status_command"] == "/status"

    asyncio.run(run())