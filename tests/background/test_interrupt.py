"""Tests for InterruptManager: interrupt, resume, TTL expiry."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from raven_agent.background.interrupt import (
    InterruptManager,
    TurnInterruptState,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def manager() -> InterruptManager:
    """创建 InterruptManager 实例（短 TTL 便于测试过期）。"""
    return InterruptManager(ttl_seconds=1)


# ── Tracking ─────────────────────────────────────────────────────


async def test_track_and_untrack(manager: InterruptManager) -> None:
    """track_task 后可从 active 中查到；untrack 后移除。"""
    async def dummy_work() -> str:
        await asyncio.sleep(0.01)
        return "done"

    task = asyncio.create_task(dummy_work(), name="test_turn")
    manager.track_task("test:123", task)

    assert manager.get_turn_state("test:123") is not None

    await task  # 等待完成
    manager.untrack_task("test:123")
    assert manager.get_turn_state("test:123") is None


async def test_partial_reply_accumulates(manager: InterruptManager) -> None:
    """append_partial_reply 增量追加到 turn state。"""
    manager._turn_states["test:1"] = TurnInterruptState(
        session_key="test:1",
        original_user_message="hello",
    )

    manager.append_partial_reply("test:1", "北京")
    manager.append_partial_reply("test:1", "今天")
    manager.append_partial_reply("test:1", "天气晴")

    state = manager.get_turn_state("test:1")
    assert state is not None
    assert state.partial_reply == "北京今天天气晴"


async def test_partial_thinking_accumulates(manager: InterruptManager) -> None:
    """append_partial_thinking 增量追加到 turn state。"""
    manager._turn_states["test:2"] = TurnInterruptState(
        session_key="test:2",
        original_user_message="complex question",
    )

    manager.append_partial_thinking("test:2", "用户想知道")
    manager.append_partial_thinking("test:2", "北京的天气")

    state = manager.get_turn_state("test:2")
    assert state is not None
    assert state.partial_thinking == "用户想知道北京的天气"


async def test_record_tool_call(manager: InterruptManager) -> None:
    """record_tool_call 追加工具名和参数到工具链。"""
    manager._turn_states["test:3"] = TurnInterruptState(
        session_key="test:3",
        original_user_message="search weather",
    )

    manager.record_tool_call(
        "test:3", "web_search", {"query": "北京天气"}
    )
    manager.record_tool_call(
        "test:3", "read_text_file", {"path": "/tmp/weather.txt"}
    )

    state = manager.get_turn_state("test:3")
    assert state is not None
    assert state.tools_used == ["web_search", "read_text_file"]
    assert len(state.tool_chain_partial) == 2
    assert state.tool_chain_partial[0]["tool"] == "web_search"


# ── Interrupt ────────────────────────────────────────────────────


async def test_interrupt_idle_session(manager: InterruptManager) -> None:
    """空闲 session 中断返回 status="idle"。"""
    result = manager.request_interrupt("nonexistent:123")
    assert result.status == "idle"
    assert "没有" in result.message


async def test_interrupt_active_session(manager: InterruptManager) -> None:
    """活跃 session 中断成功返回 status="interrupted"。"""
    async def long_work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(long_work(), name="test_long")
    state = TurnInterruptState(
        session_key="test:abc",
        original_user_message="帮我写一篇长文章",
    )
    manager.track_task("test:abc", task, state)

    # 在 task 完成前中断
    result = manager.request_interrupt("test:abc", sender="user")

    assert result.status == "interrupted"
    assert "中断" in result.message

    # task 应被取消
    with pytest.raises(asyncio.CancelledError):
        await task

    manager.untrack_task("test:abc")


async def test_interrupt_preserves_state(manager: InterruptManager) -> None:
    """中断后 interrupt_states 包含 turn state 快照。"""
    async def long_work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(long_work(), name="test_preserve")
    state = TurnInterruptState(
        session_key="test:preserve",
        original_user_message="查询天气",
        partial_reply="正在查询",
        tools_used=["web_search"],
        tool_chain_partial=[{"tool": "web_search", "arguments": {"query": "北京天气"}}],
    )
    manager.track_task("test:preserve", task, state)

    manager.request_interrupt("test:preserve")

    saved = manager.get_interrupt_state("test:preserve")
    assert saved is not None
    assert saved.original_user_message == "查询天气"
    assert saved.partial_reply == "正在查询"
    assert saved.tools_used == ["web_search"]

    with pytest.raises(asyncio.CancelledError):
        await task
    manager.untrack_task("test:preserve")


# ── Resume ───────────────────────────────────────────────────────


def test_build_resume_content_basic(manager: InterruptManager) -> None:
    """build_resume_content 拼接原始消息、工具链、新消息。"""
    state = TurnInterruptState(
        session_key="cli:default",
        original_user_message="分析今年股市走势",
        tools_used=["web_search", "read_text_file"],
        tool_chain_partial=[
            {"tool": "web_search", "arguments": {"query": "2026 stock market"}},
            {"tool": "read_text_file", "arguments": {"path": "/tmp/report.txt"}},
        ],
        partial_reply="根据已获取的数据，今年股市呈现波动上升趋势",
    )

    content = manager.build_resume_content(state, "请继续写完整分析报告")

    assert "分析今年股市走势" in content
    assert "web_search" in content
    assert "read_text_file" in content
    assert "波动上升趋势" in content
    assert "请继续写完整分析报告" in content


def test_build_resume_minimal_state(manager: InterruptManager) -> None:
    """最小状态（只有原始消息）也能正确拼接。"""
    state = TurnInterruptState(
        session_key="cli:default",
        original_user_message="hello",
    )

    content = manager.build_resume_content(state, "继续")

    assert "hello" in content
    assert "继续" in content


def test_build_resume_with_thinking(manager: InterruptManager) -> None:
    """partial_thinking 也出现在恢复上下文中。"""
    state = TurnInterruptState(
        session_key="cli:default",
        original_user_message="复杂问题",
        partial_thinking="需要先查资料，再分析数据，最后总结",
    )

    content = manager.build_resume_content(state, "继续")

    assert "查资料" in content


# ── TTL ──────────────────────────────────────────────────────────


async def test_interrupt_state_expires(manager: InterruptManager) -> None:
    """超过 TTL 的中断态被 get_interrupt_state 判定为过期。"""
    async def long_work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(long_work())
    state = TurnInterruptState(
        session_key="test:expiry",
        original_user_message="test",
    )
    manager.track_task("test:expiry", task, state)
    manager.request_interrupt("test:expiry")

    # 模拟时间流逝——直接将 interrupted_at 设为过去
    saved = manager._interrupt_states.get("test:expiry")
    assert saved is not None
    saved.interrupted_at = time.monotonic() - 5  # 5 秒前，超过 1 秒 TTL

    assert manager.get_interrupt_state("test:expiry") is None

    with pytest.raises(asyncio.CancelledError):
        await task
    manager.untrack_task("test:expiry")


async def test_pop_interrupt_state_consumes(manager: InterruptManager) -> None:
    """pop 读取后状态被移除。"""
    async def long_work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(long_work())
    state = TurnInterruptState(
        session_key="test:pop",
        original_user_message="test pop",
    )
    manager.track_task("test:pop", task, state)
    manager.request_interrupt("test:pop")

    first = manager.pop_interrupt_state("test:pop")
    assert first is not None
    assert first.original_user_message == "test pop"

    second = manager.pop_interrupt_state("test:pop")
    assert second is None  # 已被消费

    with pytest.raises(asyncio.CancelledError):
        await task
    manager.untrack_task("test:pop")


# ── get_interrupt_state does not consume ─────────────────────────


async def test_get_does_not_consume(manager: InterruptManager) -> None:
    """get_interrupt_state 读取后不消费中断态。"""
    async def long_work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(long_work())
    state = TurnInterruptState(
        session_key="test:get",
        original_user_message="test",
    )
    manager.track_task("test:get", task, state)
    manager.request_interrupt("test:get")

    first = manager.get_interrupt_state("test:get")
    second = manager.get_interrupt_state("test:get")
    assert first is not None
    assert second is not None  # 未被消费

    with pytest.raises(asyncio.CancelledError):
        await task
    manager.untrack_task("test:get")