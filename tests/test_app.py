from __future__ import annotations

import asyncio

from pathlib import Path

from raven_agent.app import AppRuntime
from raven_agent.config import (
    AgentConfig,
    Config,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    PluginsConfig,
    ToolsConfig,
    WebSearchConfig,
)
from raven_agent.events import InboundMessage, OutboundMessage


def _test_config() -> Config:
    """创建测试用 Config。

    返回:
        测试 Config。
    """

    return Config(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        agent=AgentConfig(system_prompt="You are Raven."),
        tools=ToolsConfig(
            web_search=WebSearchConfig(
                api_key="test-web-search-key",
                gl="cn",
                hl="zh-cn",
            )
        ),
    )


def test_app_runtime_create_wires_core_objects(tmp_path) -> None:
    """测试 AppRuntime.create 会组装核心对象。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    app = AppRuntime.create(
        _test_config(),
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )

    assert app.workspace == tmp_path / ".raven"
    assert app.core.workspace.root == tmp_path / ".raven"
    assert app.core.config.agent.system_prompt == "You are Raven."
    assert app.core.prompt_builder.system_prompt == "You are Raven."
    assert app.core.prompt_builder.memory_store is app.core.memory
    assert app.core.memory.memory_dir == tmp_path / ".raven" / "memory"
    assert app.core.memory.memory_file.exists()
    assert app.core.memory.self_file.exists()
    assert app.core.memory.pending_file.exists()
    assert app.core.memory.history_file.exists()
    assert app.core.memory.recent_context_file.exists()
    assert app.core.memory.journal_dir.exists()
    assert app.core.sessions.get_or_create("cli:default").key == "cli:default"
    assert app.core.tool_executor is not None
    assert app.core.memory_maintenance is not None
    assert app.core.memory_optimizer is not None
    assert app.core.memory_engine.describe().name == "disabled"


def test_app_runtime_start_creates_workspace(tmp_path) -> None:
    """测试 start 会创建 workspace。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        workspace = tmp_path / ".raven"
        app = AppRuntime.create(
            _test_config(),
            workspace=workspace,
            allowed_dir=tmp_path,
        )

        await app.start()

        assert workspace.exists()
        assert app.core.workspace.sessions_dir.exists()
        assert app.core.workspace.memory_dir.exists()

    asyncio.run(run())


def test_app_runtime_message_bus_helpers_dispatch_outbound(tmp_path) -> None:
    """测试 AppRuntime 的 MessageBus 辅助方法可以发布和分发 outbound。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        app = AppRuntime.create(
            _test_config(),
            workspace=tmp_path / ".raven",
            allowed_dir=tmp_path,
        )
        seen: list[str] = []

        def on_cli(message: OutboundMessage) -> None:
            """记录 CLI outbound。

            参数:
                message: 出站消息。

            返回:
                None。
            """

            seen.append(message.content)

        app.subscribe_outbound("cli", on_cli)
        await app.publish_outbound(
            OutboundMessage(channel="cli", chat_id="default", content="hello")
        )
        dispatched = await app.dispatch_outbound_once()

        assert dispatched.content == "hello"
        assert seen == ["hello"]

    asyncio.run(run())


def test_app_runtime_clear_session_resets_session(tmp_path) -> None:
    """测试 clear_session 会清空指定会话。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    app = AppRuntime.create(
        _test_config(),
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )
    session = app.core.sessions.get_or_create("cli:default")
    session.add_user_message("hello")
    app.core.sessions.save(session)

    app.clear_session("cli:default")

    assert app.core.sessions.get_or_create("cli:default").messages == []

def test_app_runtime_optimize_memory_delegates_to_optimizer(tmp_path) -> None:
    """测试 AppRuntime.optimize_memory 会委托给 MemoryOptimizer。"""

    async def run() -> None:
        app = AppRuntime.create(
            _test_config(),
            workspace=tmp_path / ".raven",
            allowed_dir=tmp_path,
        )
        called = False

        async def fake_optimize() -> None:
            nonlocal called
            called = True

        app.core.memory_optimizer.optimize = fake_optimize  # type: ignore[method-assign]

        await app.optimize_memory()

        assert called is True

    asyncio.run(run())


def test_app_runtime_create_wires_memory2_when_enabled(tmp_path) -> None:
    """测试启用 memory.enabled 后 AppRuntime 会装配 Memory2Engine。"""

    config = Config(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        agent=AgentConfig(system_prompt="You are Raven."),
        tools=ToolsConfig(web_search=WebSearchConfig()),
        memory=MemoryConfig(
            enabled=True,
            embedding=EmbeddingConfig(enabled=False),
        ),
    )

    app = AppRuntime.create(
        config,
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )

    assert app.core.memory_engine.describe().name == "memory2"
    assert (tmp_path / ".raven" / "memory2" / "memory2.db").exists()


def test_app_runtime_create_wires_memory2_retriever_when_enabled(tmp_path) -> None:
    """测试启用 Memory2 后 AppRuntime 装配带 Retriever 的 Memory2Engine。"""

    config = Config(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        agent=AgentConfig(system_prompt="You are Raven."),
        tools=ToolsConfig(web_search=WebSearchConfig()),
        memory=MemoryConfig(
            enabled=True,
            embedding=EmbeddingConfig(enabled=False, dimensions=2),
        ),
    )

    app = AppRuntime.create(
        config,
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )

    assert app.core.memory_engine.describe().name == "memory2"
    assert app.core.memory_engine.describe().notes["retrieval"] == "sqlite_vec_numpy_fts_rrf"


def test_app_runtime_wires_memory2_write_components_when_enabled(tmp_path) -> None:
    """测试启用 Memory2 后 AppRuntime 装配写入相关组件。"""

    config = Config(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        agent=AgentConfig(system_prompt="You are Raven."),
        tools=ToolsConfig(web_search=WebSearchConfig()),
        memory=MemoryConfig(
            enabled=True,
            embedding=EmbeddingConfig(enabled=False, dimensions=2),
        ),
    )

    app = AppRuntime.create(
        config,
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )

    assert app.core.memory_engine.describe().name == "memory2"
    assert app.core.procedure_tagger is not None
    assert app.core.profile_extractor is not None


def test_app_runtime_registers_memory_tools_when_enabled(tmp_path) -> None:
    """测试启用 Memory2 后 AppRuntime 注册三个记忆工具。"""

    config = Config(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        agent=AgentConfig(system_prompt="You are Raven."),
        tools=ToolsConfig(web_search=WebSearchConfig()),
        memory=MemoryConfig(enabled=True, embedding=EmbeddingConfig(enabled=False, dimensions=2)),
    )

    app = AppRuntime.create(config, workspace=tmp_path / ".raven", allowed_dir=tmp_path)

    names = app.core.tools.list_names()
    assert "memorize" in names
    assert "recall_memory" in names
    assert "forget_memory" in names
    assert {"memorize", "recall_memory", "forget_memory"} <= app.core.tools.get_always_on_names()


def test_app_runtime_does_not_register_memory_tools_when_disabled(tmp_path) -> None:
    """测试未启用 Memory2 时 AppRuntime 不注册记忆工具。"""

    app = AppRuntime.create(
        _test_config(),
        workspace=tmp_path / ".raven",
        allowed_dir=tmp_path,
    )

    names = app.core.tools.list_names()
    assert "memorize" not in names
    assert "recall_memory" not in names
    assert "forget_memory" not in names

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

def test_app_runtime_plugin_tool_through_pipeline(tmp_path) -> None:
    """测试启用插件后 AppRuntime 加载工具，stop 后注销。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    from raven_agent.app import AppRuntime

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
            "hello_tool",
            """
from raven_agent.plugins import Plugin, tool


class HelloTool(Plugin):
    name = "hello_tool"

    @tool("hello_tool", risk="read-only", always_on=True)
    async def hello_tool(self, event, text: str) -> str:
        return "hello " + text
""",
        )
        config = Config(
            llm=LLMConfig(
                provider="test",
                model="test-model",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            agent=AgentConfig(system_prompt="You are Raven."),
            tools=ToolsConfig(web_search=WebSearchConfig()),
            plugins=PluginsConfig(enabled=True, dirs=(str(root),)),
        )
        app = AppRuntime.create(config, workspace=tmp_path / ".raven", allowed_dir=tmp_path)

        assert app.core.plugin_manager is not None
        assert "hello_tool" not in app.core.tools.list_names()

        await app.start()
        assert "hello_tool" in app.core.tools.list_names()

        await app.stop()
        assert "hello_tool" not in app.core.tools.list_names()

    asyncio.run(run())


def test_app_runtime_loads_builtin_plugins_by_default(tmp_path) -> None:
    """测试默认配置下 AppRuntime 加载内置安全插件。

    输入:
        tmp_path: pytest 临时目录。

    输出:
        None。
    """

    import asyncio

    async def run() -> None:
        """执行异步测试主体。"""

        app = AppRuntime.create(
            _test_config(),
            workspace=tmp_path / ".raven",
            allowed_dir=tmp_path,
        )
        assert app.core.plugin_manager is not None

        await app.start()
        # shell_safety 是 tool hook，会拦截 shell 命令。
        hook_names = [hook.name for hook in app.core.tool_executor._hooks]  # type: ignore[attr-defined]
        assert any("shell_safety" in name for name in hook_names)
        await app.stop()
        # stop 后 plugin hooks 已移除。
        hook_names_after = [hook.name for hook in app.core.tool_executor._hooks]  # type: ignore[attr-defined]
        assert all(not name.startswith("plugin:") for name in hook_names_after)

    asyncio.run(run())