"""test_subagent_manager.py —— SubagentManager 后台提交/取消/回流测试。

覆盖：
  - spawn 后台提交到 BackgroundRuntime
  - 完成回流注入 InboundMessage，metadata 包含 spawn_completion/persist_user_content
  - cancel 后台任务
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from raven_agent.background.runtime import BackgroundRuntime, BackgroundJobRunner
from raven_agent.background.subagent_profiles import SubagentRuntime
from raven_agent.background.subagents import (
    SpawnAwareBackgroundJobRunner,
    SubagentJobRunner,
    SubagentManager,
)
from raven_agent.config import LLMConfig
from raven_agent.events import InboundMessage
from raven_agent.llm import LLMProvider
from raven_agent.message_bus import MessageBus


def _runtime() -> SubagentRuntime:
    """创建测试用 SubagentRuntime，provider 不会被实际调用。"""
    return SubagentRuntime(
        provider=LLMProvider(
            LLMConfig(provider="test", model="dummy", api_key="dummy", base_url="http://127.0.0.1:1")
        ),
        model="dummy",
    )


def _manager(tmp_path: Path, bus: MessageBus, rt: BackgroundRuntime) -> SubagentManager:
    return SubagentManager(
        runtime=_runtime(), workspace=tmp_path,
        task_root=tmp_path / "subagent-runs",
        bus=bus, background_runtime=rt,
    )


class FakeCompletedSubAgent:
    """模拟正常完成的 SubAgent。"""
    last_exit_reason = "completed"
    async def run(self, task: str) -> str:
        return f"调研完成: {task[:20]}"


@pytest.mark.asyncio
async def test_spawn_submits_to_background_runtime(tmp_path: Path) -> None:
    """后台 spawn 通过 BackgroundRuntime 提交，job metadata 正确标记。"""
    bus = MessageBus()
    rt = BackgroundRuntime(max_concurrent=1)
    manager = _manager(tmp_path, bus, rt)

    text = await manager.spawn(
        task="research this", label="research",
        origin_channel="telegram", origin_chat_id="42",
    )

    assert "已创建后台任务" in text
    jobs = rt.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].metadata["job_kind"] == "conversation_spawn"
    assert jobs[0].channel == "telegram"
    assert jobs[0].chat_id == "42"


@pytest.mark.asyncio
async def test_completion_injects_to_origin_session(tmp_path: Path) -> None:
    """后台完成后 announce_completion 注入 InboundMessage 到原 session。"""
    bus = MessageBus()
    rt = BackgroundRuntime(max_concurrent=1)
    manager = _manager(tmp_path, bus, rt)
    manager._build_subagent = lambda **kwargs: FakeCompletedSubAgent()  # type: ignore[method-assign]

    # 设置 runner
    runner = SpawnAwareBackgroundJobRunner(
        default_runner=SubagentJobRunner(manager),
        subagent_runner=SubagentJobRunner(manager),
    )
    rt.set_runner(runner)
    rt.on_complete(manager.announce_completion)
    await rt.start()

    await manager.spawn(
        task="research this", label="research",
        origin_channel="telegram", origin_chat_id="42",
    )

    item = await asyncio.wait_for(bus.consume_inbound(), timeout=2.0)
    await rt.stop()

    assert isinstance(item, InboundMessage)
    assert item.channel == "telegram"
    assert item.chat_id == "42"
    assert item.sender == "spawn"
    assert item.metadata["spawn_completion"] is True
    assert "调研完成" in item.content
    assert "persist_user_content" in item.metadata
    assert "spawn_event" in item.metadata


@pytest.mark.asyncio
async def test_cancel_background_spawn(tmp_path: Path) -> None:
    """取消后台 spawn 正确标记 status=cancelled。"""
    bus = MessageBus()
    rt = BackgroundRuntime(max_concurrent=1)
    manager = _manager(tmp_path, bus, rt)

    await manager.spawn(
        task="long task", label="long",
        origin_channel="telegram", origin_chat_id="42",
    )
    job_id = rt.list_jobs()[0].job_id
    assert await manager.cancel(job_id) is True
    assert rt.get(job_id).status == "cancelled"