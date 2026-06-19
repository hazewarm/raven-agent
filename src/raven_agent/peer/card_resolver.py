"""
peer/card_resolver.py —— AgentCard 解析器。

通过 HTTP GET {base_url}/.well-known/agent.json 获取 AgentCard。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from raven_agent.peer.models import AgentCard, AgentSkill

logger = logging.getLogger(__name__)

_CARD_TIMEOUT_S = 3.0


async def fetch_agent_card(
    base_url: str,
    client: httpx.AsyncClient,
) -> AgentCard:
    """从 Peer Agent 的 /.well-known/agent.json 端点获取 AgentCard。

    输入:
        base_url: Peer Agent 的 base URL（如 "http://127.0.0.1:8090"）。
        client: httpx.AsyncClient 实例。

    输出:
        AgentCard 实例。

    异常:
        RuntimeError: HTTP 请求失败或 JSON 解析失败时抛出。
    """
    url = base_url.rstrip("/") + "/.well-known/agent.json"
    try:
        response = await client.get(url, timeout=_CARD_TIMEOUT_S)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except Exception as exc:
        raise RuntimeError(f"无法获取 AgentCard from {url}: {exc}") from exc

    skills = [
        AgentSkill(
            id=s.get("id", ""),
            name=s.get("name", ""),
            description=s.get("description", ""),
            tags=s.get("tags", []),
            examples=s.get("examples", []),
        )
        for s in data.get("skills", [])
    ]
    return AgentCard(
        name=data["name"],
        url=data.get("url", base_url),
        description=data.get("description", ""),
        skills=skills,
    )


def build_static_card(
    name: str,
    base_url: str,
    description: str = "",
) -> AgentCard:
    """从配置文件构建静态 AgentCard（不依赖 server 在线）。

    与 fetch_agent_card() 互斥使用——当 server 未启动时，
    PeerAgentRegistry 调用此函数创建最小可用的 AgentCard。

    输入:
        name: Agent 名称。
        base_url: A2A 端点 URL。
        description: 工具描述（来自 TOML 配置）。

    输出:
        AgentCard 实例。
    """
    skill = AgentSkill(
        id=name,
        name=name,
        description=description or f"委托 {name} 执行深度任务，生成结构化报告",
    )
    return AgentCard(
        name=name,
        url=base_url,
        description=description,
        skills=[skill],
    )