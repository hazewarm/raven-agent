from __future__ import annotations

import asyncio

from raven_agent.memory import DisabledMemoryEngine
from raven_agent.memory2 import Memory2Engine, MemoryStore2
from raven_agent.tools import (
    MemoryToolContextHook,
    ToolRegistry,
    register_memory_tools,
)
from raven_agent.tools.hooks import ToolExecutionRequest, ToolHookContext


class _NoEmbedder:
    """测试用空 embedding provider。"""

    async def embed_text(self, text: str) -> list[float]:
        """返回空向量。

        参数:
            text: 输入文本。

        返回:
            空列表。
        """

        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量返回空向量。

        参数:
            texts: 输入文本列表。

        返回:
            等长空向量列表。
        """

        return [[] for _ in texts]

    async def close(self) -> None:
        """关闭测试 provider。

        返回:
            None。
        """

        return None


def test_register_memory_tools_registers_three_tools_for_memory2(tmp_path) -> None:
    """测试启用的 Memory2Engine 会注册三个记忆工具且 always-on。"""

    async def run() -> None:
        engine = Memory2Engine(
            store=MemoryStore2(tmp_path / "memory2.db"),
            embedder=_NoEmbedder(),
        )
        try:
            registry = ToolRegistry()
            registered = register_memory_tools(registry, engine)

            assert set(registered) == {"recall_memory", "memorize", "forget_memory"}
            assert registry.has_tool("recall_memory")
            assert registry.has_tool("memorize")
            assert registry.has_tool("forget_memory")
            always_on = registry.get_always_on_names()
            assert {"recall_memory", "memorize", "forget_memory"} <= always_on
            # risk 来自 spec：recall 只读，memorize/forget 写
            assert registry.get_document("recall_memory").risk == "read-only"
            assert registry.get_document("memorize").risk == "write"
        finally:
            await engine.close()

    asyncio.run(run())


def test_register_memory_tools_registers_nothing_for_disabled_engine() -> None:
    """测试 DisabledMemoryEngine 不注册任何记忆工具。"""

    registry = ToolRegistry()
    registered = register_memory_tools(registry, DisabledMemoryEngine())

    assert registered == []
    assert registry.list_names() == []


def test_register_memory_tools_is_idempotent(tmp_path) -> None:
    """测试重复注册不会抛错也不会重复。"""

    async def run() -> None:
        engine = Memory2Engine(
            store=MemoryStore2(tmp_path / "memory2.db"),
            embedder=_NoEmbedder(),
        )
        try:
            registry = ToolRegistry()
            register_memory_tools(registry, engine)
            second = register_memory_tools(registry, engine)

            assert second == []
            assert registry.list_names().count("memorize") == 1
        finally:
            await engine.close()

    asyncio.run(run())


def test_memory_tool_context_hook_injects_turn_context() -> None:
    """测试 MemoryToolContextHook 把每轮上下文注入记忆工具参数。"""

    async def run() -> None:
        hook = MemoryToolContextHook()
        request = ToolExecutionRequest(
            call_id="c1",
            tool_name="memorize",
            arguments={"summary": "x"},
            session_key="cli:default",
            metadata={
                "current_user_source_ref": '["cli:default:3"]',
                "channel": "cli",
                "chat_id": "default",
            },
        )
        context = ToolHookContext(
            event="pre_tool_use",
            request=request,
            current_arguments={"summary": "x"},
        )

        assert hook.matches(context) is True
        outcome = await hook.run(context)

        assert outcome.updated_arguments is not None
        assert outcome.updated_arguments["current_user_source_ref"] == '["cli:default:3"]'
        assert outcome.updated_arguments["channel"] == "cli"

    asyncio.run(run())


def test_memory_tool_context_hook_skips_non_memory_tools() -> None:
    """测试 MemoryToolContextHook 不匹配非记忆工具。"""

    hook = MemoryToolContextHook()
    request = ToolExecutionRequest(
        call_id="c1",
        tool_name="read_text_file",
        arguments={"path": "a.txt"},
        metadata={"current_user_source_ref": '["cli:default:3"]'},
    )
    context = ToolHookContext(
        event="pre_tool_use",
        request=request,
        current_arguments={"path": "a.txt"},
    )

    assert hook.matches(context) is False