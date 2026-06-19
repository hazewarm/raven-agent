"""
test_peer_registry.py —— PeerAgentRegistry 测试。

覆盖：
  - discover_all(): 空配置 / 单个 agent（server 在线 + 离线）
  - 工具名称唯一性
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from raven_agent.config import PeerAgentConfig
from raven_agent.peer.registry import PeerAgentRegistry


class TestPeerAgentRegistry:
    """PeerAgentRegistry 核心行为测试。"""

    @pytest.fixture
    def mock_pm(self) -> MagicMock:
        """mock PeerProcessManager。"""
        return MagicMock()

    @pytest.fixture
    def mock_poller(self) -> MagicMock:
        """mock PeerAgentPoller。"""
        return MagicMock()

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """mock httpx.AsyncClient。"""
        return MagicMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def registry(self, mock_pm, mock_poller, mock_client):
        """创建测试用 PeerAgentRegistry。"""
        return PeerAgentRegistry(mock_pm, mock_poller, mock_client)

    @pytest.mark.asyncio
    async def test_discover_empty_config(self, registry):
        """空配置返回空工具列表。"""
        tools = await registry.discover_all([])
        assert tools == []

    @pytest.mark.asyncio
    async def test_discover_single(self, registry):
        """TOML 配置构建正确的工具。"""
        cfg = PeerAgentConfig(
            name="Test Agent",
            base_url="http://127.0.0.1:8090",
            launcher=("python", "server.py"),
            description="测试 Agent",
        )

        tools = await registry.discover_all([cfg])

        assert len(tools) == 1
        assert tools[0].name == "delegate_test_agent"
        assert "测试 Agent" in tools[0].description

    @pytest.mark.asyncio
    async def test_discover_multiple_agents(self, registry):
        """多个 agent 各生成一个工具，名称不冲突。"""
        cfgs = [
            PeerAgentConfig(
                name="Agent A",
                base_url="http://127.0.0.1:8091",
                launcher=("python", "a.py"),
            ),
            PeerAgentConfig(
                name="Agent B",
                base_url="http://127.0.0.1:8092",
                launcher=("python", "b.py"),
            ),
        ]

        tools = await registry.discover_all(cfgs)

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"delegate_agent_a", "delegate_agent_b"}