"""
peer/registry.py —— Peer Agent 发现与工具注册。

启动时扫描所有配置的 Peer Agent，生成 PeerAgentTool 并注册到 ToolRegistry。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from raven_agent.peer.models import AgentCard, AgentSkill
from raven_agent.peer.tool import PeerAgentTool

logger = logging.getLogger(__name__)


class PeerAgentRegistry:
    """启动时从 TOML 配置构建所有 Peer Agent 工具。

    冷启动 + 用完即销决定了：工具注册时 server 还没起，任务完成后就销毁了。
    AgentCard 的唯一来源就是 TOML。

    参数:
        process_manager: PeerProcessManager 实例。
        poller: PeerAgentPoller 实例。
        client: httpx.AsyncClient 实例。
    """

    def __init__(
        self,
        process_manager: Any,
        poller: Any,
        client: httpx.AsyncClient,
    ) -> None:
        self._pm = process_manager
        self._poller = poller
        self._client = client

    async def discover_all(
        self,
        peer_configs: list[Any],
    ) -> list[PeerAgentTool]:
        """从 TOML 配置构建所有 Peer Agent 工具。

        AgentCard 直接从 TOML 字段构建——没有第二个来源。

        输入:
            peer_configs: PeerAgentConfig 列表。

        输出:
            PeerAgentTool 列表。
        """
        tools: list[PeerAgentTool] = []
        for cfg in peer_configs:
            skill = AgentSkill(
                id=cfg.name,
                name=cfg.name,
                description=cfg.description or f"委托 {cfg.name} 执行深度任务",
            )
            card = AgentCard(
                name=cfg.name,
                url=cfg.base_url,
                description=cfg.description,
                skills=[skill],
            )

            tool = PeerAgentTool(
                card=card,
                process_manager=self._pm,
                poller=self._poller,
                client=self._client,
            )
            tools.append(tool)
            logger.info(
                "[PeerAgentRegistry] 工具已注册: %s  描述: %.60s...",
                tool.name, tool.description,
            )
        return tools