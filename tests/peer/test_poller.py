"""
test_poller.py —— PeerAgentPoller 测试。

覆盖：
  - 任务注册与超时
  - 任务完成注入
  - 任务失败注入
  - tasks/get JSON-RPC 解析
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from raven_agent.peer.poller import PeerAgentPoller, _PendingTask, _TASK_TIMEOUT_S


class TestPeerAgentPoller:
    """PeerAgentPoller 核心行为测试。"""

    @pytest.fixture
    def mock_bus(self) -> MagicMock:
        """mock MessageBus。"""
        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        return bus

    @pytest.fixture
    def mock_pm(self) -> MagicMock:
        """mock PeerProcessManager。"""
        pm = MagicMock()
        pm.terminate = AsyncMock()
        return pm

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """mock httpx.AsyncClient。"""
        return MagicMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def poller(self, mock_bus, mock_pm, mock_client) -> PeerAgentPoller:
        """创建测试用 Poller。"""
        return PeerAgentPoller(
            mock_bus, mock_pm, mock_client,
            artifacts_dir=Path("/tmp/test_artifacts"),
        )

    def test_register_adds_task(self, poller):
        """register() 后 pending 中有一条任务。"""
        poller.register(
            task_id="task-1",
            agent_name="test-agent",
            agent_url="http://127.0.0.1:8090",
            channel="cli",
            chat_id="default",
            goal="调研",
        )
        assert len(poller._pending) == 1
        assert "task-1" in poller._pending

    @pytest.mark.asyncio
    async def test_check_timeout(
        self, poller, mock_bus, mock_pm,
    ):
        """超时任务注入失败通知并销毁进程。"""
        meta = _PendingTask(
            task_id="task-1",
            agent_name="test-agent",
            agent_url="http://127.0.0.1:8090",
            channel="cli",
            chat_id="default",
            goal="调研",
            submitted_at=time.monotonic() - _TASK_TIMEOUT_S - 1,
        )
        poller._pending["task-1"] = meta

        await poller._check("task-1", meta)

        # 任务应从 pending 中移除
        assert "task-1" not in poller._pending
        # 应注入失败通知
        mock_bus.publish_inbound.assert_called_once()
        call_args = mock_bus.publish_inbound.call_args[0][0]
        assert "超时" in call_args.content
        # 应销毁子进程
        mock_pm.terminate.assert_awaited_once_with("test-agent")

    @pytest.mark.asyncio
    async def test_check_completed(
        self, poller, mock_bus, mock_pm, mock_client,
    ):
        """完成任务注入完成通知并销毁进程。"""
        # mock tasks/get 返回 completed
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "name": "report.md",
                        "parts": [{"text": "# 调研报告\n\n..."}],
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        meta = _PendingTask(
            task_id="task-1",
            agent_name="test-agent",
            agent_url="http://127.0.0.1:8090",
            channel="cli",
            chat_id="default",
            goal="调研",
        )
        poller._pending["task-1"] = meta

        await poller._check("task-1", meta)

        assert "task-1" not in poller._pending
        mock_bus.publish_inbound.assert_called_once()
        call_args = mock_bus.publish_inbound.call_args[0][0]
        assert "已完成" in call_args.content
        assert "report.md" in call_args.content
        mock_pm.terminate.assert_awaited_once_with("test-agent")

    @pytest.mark.asyncio
    async def test_check_failed(
        self, poller, mock_bus, mock_pm, mock_client,
    ):
        """失败任务注入失败通知并销毁进程。"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "status": {
                    "state": "failed",
                    "message": {
                        "parts": [{"text": "模型 API 不可用"}],
                    },
                },
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        meta = _PendingTask(
            task_id="task-2",
            agent_name="test-agent",
            agent_url="http://127.0.0.1:8090",
            channel="cli",
            chat_id="default",
            goal="调研",
        )
        poller._pending["task-2"] = meta

        await poller._check("task-2", meta)

        assert "task-2" not in poller._pending
        mock_bus.publish_inbound.assert_called_once()
        call_args = mock_bus.publish_inbound.call_args[0][0]
        assert "模型 API 不可用" in call_args.content
        mock_pm.terminate.assert_awaited_once_with("test-agent")

    @pytest.mark.asyncio
    async def test_check_working_keeps_pending(
        self, poller, mock_client,
    ):
        """working 状态的任务保留在 pending 中。"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"status": {"state": "working"}}
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        meta = _PendingTask(
            task_id="task-3",
            agent_name="test-agent",
            agent_url="http://127.0.0.1:8090",
            channel="cli",
            chat_id="default",
            goal="调研",
        )
        poller._pending["task-3"] = meta

        await poller._check("task-3", meta)

        # working 状态不移除
        assert "task-3" in poller._pending