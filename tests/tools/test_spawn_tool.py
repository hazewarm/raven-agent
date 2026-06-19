"""test_spawn_tool.py —— spawn 工具同步/后台/context hook 测试。

覆盖：
  - SpawnToolContextHook 从 metadata 注入 channel/chat_id
  - 后台模式调用 manager.spawn，上下文缺失时返回错误
  - 同步模式调用 manager.spawn_sync，不需要上下文
  - DelegationPolicy 并发上限拦截
  - spawn_manage list / cancel
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from raven_agent.background.delegation import SpawnDecision, SpawnDecisionMeta
from raven_agent.tools.hooks import ToolExecutionRequest, ToolHookContext
from raven_agent.tools.spawn import SpawnManageTool, SpawnTool, SpawnToolContextHook


def _make_manager(spawn_return="started", spawn_sync_return="sync-result") -> AsyncMock:
    """创建 mock SubagentManager。"""
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value=spawn_return)
    manager.spawn_sync = AsyncMock(return_value=spawn_sync_return)
    manager.get_running_count = Mock(return_value=0)
    manager.list_running_jobs = Mock(return_value=[])
    manager.cancel = AsyncMock(return_value=False)
    return manager


@pytest.mark.asyncio
async def test_context_hook_injects_channel_and_chat_id() -> None:
    """SpawnToolContextHook 从 metadata 注入 channel/chat_id。"""
    hook = SpawnToolContextHook()
    ctx = ToolHookContext(
        event="pre_tool_use",
        request=ToolExecutionRequest(
            call_id="call-1",
            tool_name="spawn",
            arguments={"task": "do work"},
            metadata={"channel": "telegram", "chat_id": "123"},
        ),
        current_arguments={"task": "do work"},
    )
    outcome = await hook.run(ctx)
    assert outcome.updated_arguments is not None
    assert outcome.updated_arguments["channel"] == "telegram"
    assert outcome.updated_arguments["chat_id"] == "123"


@pytest.mark.asyncio
async def test_background_mode_calls_manager_spawn() -> None:
    """后台模式传入 channel/chat_id，调用 manager.spawn。"""
    manager = _make_manager()
    tool = SpawnTool(manager)
    result = await tool.execute(
        task="do work", label="job",
        run_in_background=True, channel="telegram", chat_id="123",
    )
    assert result == "started"
    manager.spawn.assert_awaited_once()
    call_kwargs = manager.spawn.await_args.kwargs
    assert call_kwargs["origin_channel"] == "telegram"
    assert call_kwargs["origin_chat_id"] == "123"
    assert call_kwargs["decision"].should_spawn is True


@pytest.mark.asyncio
async def test_background_returns_error_when_context_missing() -> None:
    """后台模式缺失 channel/chat_id 返回错误，不调用 manager。"""
    manager = _make_manager()
    tool = SpawnTool(manager)
    result = await tool.execute(task="do work", run_in_background=True)
    assert "上下文缺失" in result
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_sync_mode_calls_manager_spawn_sync() -> None:
    """同步模式不需要上下文，直接调用 manager.spawn_sync。"""
    manager = _make_manager()
    tool = SpawnTool(manager)
    result = await tool.execute(task="do work", label="job")
    assert result == "sync-result"
    manager.spawn_sync.assert_awaited_once_with(
        task="do work", label="job", profile="research",
    )
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_blocks_at_concurrency_limit() -> None:
    """达到并发上限时 DelegationPolicy 拦截并返回原因。"""
    manager = _make_manager()
    manager.get_running_count = Mock(return_value=3)
    tool = SpawnTool(manager)
    result = await tool.execute(
        task="another task", run_in_background=True,
        channel="telegram", chat_id="123",
    )
    assert "任务被拦截" in result
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_manage_list() -> None:
    """spawn_manage list 返回运行中任务列表。"""
    manager = _make_manager()
    manager.get_running_count = Mock(return_value=1)
    manager.list_running_jobs = Mock(
        return_value=[{"job_id": "abc123", "label": "job"}]
    )
    tool = SpawnManageTool(manager)
    payload = json.loads(await tool.execute(action="list"))
    assert payload["running_count"] == 1
    assert payload["jobs"][0]["job_id"] == "abc123"


@pytest.mark.asyncio
async def test_manage_cancel() -> None:
    """spawn_manage cancel 调用 manager.cancel。"""
    manager = _make_manager()
    manager.cancel = AsyncMock(return_value=True)
    tool = SpawnManageTool(manager)
    payload = json.loads(await tool.execute(action="cancel", job_id="abc123"))
    assert payload["status"] == "cancel_requested"
    manager.cancel.assert_awaited_once_with("abc123")