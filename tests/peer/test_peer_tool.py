"""
test_peer_tool.py —— PeerAgentTool 测试。

覆盖：
  - _slugify() 名称转换
  - 工具名生成
  - execute() 正常提交 / 冷启动失败 / 提交失败
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from raven_agent.peer.models import AgentCard, AgentSkill
from raven_agent.peer.tool import PeerAgentTool, _slugify


class TestSlugify:
    """_slugify 名称转换测试。"""

    def test_simple_name(self):
        assert _slugify("Deep Research") == "deep_research"

    def test_name_with_hyphens(self):
        assert _slugify("Code-Review Bot") == "code_review_bot"

    def test_single_word(self):
        assert _slugify("Research") == "research"

    def test_leading_trailing_spaces(self):
        assert _slugify("  My Agent  ") == "my_agent"


class TestPeerAgentTool:
    """PeerAgentTool 核心行为测试。"""

    @pytest.fixture
    def card(self) -> AgentCard:
        """创建测试用 AgentCard。"""
        return AgentCard(
            name="Deep Research",
            url="http://127.0.0.1:8090",
            description="深度调研",
            skills=[
                AgentSkill(
                    id="research",
                    name="Research",
                    description="执行深度调研，生成结构化报告",
                    tags=["research"],
                )
            ],
        )

    @pytest.fixture
    def mock_pm(self) -> MagicMock:
        """mock PeerProcessManager。"""
        pm = MagicMock()
        pm.ensure_ready = AsyncMock()
        return pm

    @pytest.fixture
    def mock_poller(self) -> MagicMock:
        """mock PeerAgentPoller。"""
        return MagicMock()

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """mock httpx.AsyncClient。"""
        return MagicMock(spec=httpx.AsyncClient)

    def test_tool_name(self, card, mock_pm, mock_poller, mock_client):
        """工具名格式为 delegate_<slug>。"""
        tool = PeerAgentTool(card, mock_pm, mock_poller, mock_client)
        assert tool.name == "delegate_deep_research"

    def test_tool_description(self, card, mock_pm, mock_poller, mock_client):
        """工具描述包含 skill description 和异步提示。"""
        tool = PeerAgentTool(card, mock_pm, mock_poller, mock_client)
        assert "深度调研" in tool.description
        assert "异步执行" in tool.description

    @pytest.mark.asyncio
    async def test_execute_submit_success(
        self, card, mock_pm, mock_poller, mock_client,
    ):
        """正常提交任务返回 submitted 状态。"""
        # mock A2A 响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"id": "task-abc123"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tool = PeerAgentTool(card, mock_pm, mock_poller, mock_client)
        result = json.loads(
            await tool.execute(
                goal="调研 Python 3.14",
                channel="cli",
                chat_id="default",
            )
        )

        assert result["status"] == "submitted"
        assert result["task_id"] == "task-abc123"
        assert result["agent"] == "Deep Research"
        mock_pm.ensure_ready.assert_awaited_once_with("Deep Research")
        mock_poller.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_startup_failure(
        self, card, mock_pm, mock_poller, mock_client,
    ):
        """冷启动失败返回 error。"""
        mock_pm.ensure_ready.side_effect = RuntimeError("启动超时")

        tool = PeerAgentTool(card, mock_pm, mock_poller, mock_client)
        result = json.loads(
            await tool.execute(goal="test", channel="cli", chat_id="default")
        )

        assert "error" in result
        assert "启动超时" in result["error"]
        mock_poller.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_submit_failure(
        self, card, mock_pm, mock_poller, mock_client,
    ):
        """A2A 提交失败返回 error。"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError(
            "Bad Gateway"
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        tool = PeerAgentTool(card, mock_pm, mock_poller, mock_client)
        result = json.loads(
            await tool.execute(goal="test", channel="cli", chat_id="default")
        )

        assert "error" in result
        assert "Bad Gateway" in result["error"]
        mock_poller.register.assert_not_called()