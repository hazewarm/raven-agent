"""
peer/builder.py —— Peer Agent 资源同步构建函数。

将 ProcessManager 和 Poller 的构建集中在一个函数中，
返回平铺的 (PeerProcessManager | None, PeerAgentPoller | None) 元组。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from raven_agent.peer.models import PeerProcessConfig
from raven_agent.peer.process_manager import PeerProcessManager
from raven_agent.peer.poller import PeerAgentPoller

if TYPE_CHECKING:
    from raven_agent.config import Config, PeerAgentConfig


def build_peer_agent_resources(
    config: Config,
    bus: Any,
    client: Any,
    log_dir: Path,
) -> tuple[PeerProcessManager | None, PeerAgentPoller | None]:
    """同步构建 PeerProcessManager 和 PeerAgentPoller。

    异步部分（工具发现 + 注册 + Poller 启动）在 CoreRuntime.start() 中完成。

    输入:
        config: 根 Config 实例。
        bus: MessageBus 实例。
        client: httpx.AsyncClient 实例。
        log_dir: Peer Agent 子进程日志目录。

    输出:
        (PeerProcessManager | None, PeerAgentPoller | None) 元组。
        未启用 peer agent 时返回 (None, None)。
    """
    if not config.peer_agents:
        return None, None

    proc_configs = [
        PeerProcessConfig(
            name=pa.name,
            base_url=pa.base_url,
            launcher=list(pa.launcher),
            cwd=pa.cwd,
            health_path=pa.health_path,
            startup_timeout_s=pa.startup_timeout_s,
            shutdown_timeout_s=pa.shutdown_timeout_s,
        )
        for pa in config.peer_agents
    ]
    pm = PeerProcessManager(
        configs=proc_configs,
        client=client,
        log_dir=log_dir,
    )
    poller = PeerAgentPoller(
        bus=bus,
        process_manager=pm,
        client=client,
        artifacts_dir=log_dir,
    )
    return pm, poller