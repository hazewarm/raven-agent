from __future__ import annotations

import asyncio

import pytest

from raven_agent.llm import LLMResponse
from raven_agent.memory import MemoryOptimizer, MemoryOptimizerBusy
from raven_agent.memory.markdown import MarkdownMemoryStore


class FakeProvider:
    """按顺序返回固定响应的 provider。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    async def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        """返回下一条固定响应。"""

        self.calls.append(messages)
        return LLMResponse(content=self.responses.pop(0))


def test_memory_optimizer_merges_pending_into_memory_and_self(tmp_path) -> None:
    """测试 optimizer 会把 pending 合并进 MEMORY 和 SELF。"""

    async def run() -> None:
        store = MarkdownMemoryStore(tmp_path / "memory")
        store.write_long_term("# User Memory\n\n- old")
        store.write_self("# Raven 的自我认知\n\n## 协作方式\n- old self")
        store.append_pending_once("- [preference] 用户喜欢简洁回答。", source_ref="batch-1")
        provider = FakeProvider(
            [
                "# User Memory\n\n- 用户喜欢简洁回答。",
                "# Raven 的自我认知\n\n## 协作方式\n- Raven 会保持简洁。",
            ]
        )
        optimizer = MemoryOptimizer(store=store, provider=provider)  # type: ignore[arg-type]

        await optimizer.optimize()

        assert "用户喜欢简洁回答" in store.read_long_term()
        assert "Raven 会保持简洁" in store.read_self()
        assert store.read_pending() == ""
        assert not store.pending_snapshot_file.exists()

    asyncio.run(run())

def test_memory_optimizer_rolls_back_when_memory_merge_returns_empty(tmp_path) -> None:
    """测试 MEMORY 合并失败时会回滚 pending snapshot。"""

    async def run() -> None:
        store = MarkdownMemoryStore(tmp_path / "memory")
        store.append_pending_once("- [preference] 用户喜欢简洁回答。", source_ref="batch-1")
        provider = FakeProvider([""])
        optimizer = MemoryOptimizer(store=store, provider=provider)  # type: ignore[arg-type]

        await optimizer.optimize()

        assert "用户喜欢简洁回答" in store.read_pending()
        assert not store.pending_snapshot_file.exists()

    asyncio.run(run())

def test_memory_optimizer_reports_busy(tmp_path) -> None:
    """测试 optimizer 正在运行时再次调用会报 busy。"""

    async def run() -> None:
        store = MarkdownMemoryStore(tmp_path / "memory")
        provider = FakeProvider([])
        optimizer = MemoryOptimizer(store=store, provider=provider)  # type: ignore[arg-type]
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked() -> None:
            started.set()
            await release.wait()

        optimizer._optimize_unlocked = blocked  # type: ignore[method-assign]
        task = asyncio.create_task(optimizer.optimize())
        await started.wait()

        with pytest.raises(MemoryOptimizerBusy):
            await optimizer.optimize()

        release.set()
        await task

    asyncio.run(run())