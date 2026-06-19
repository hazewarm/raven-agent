"""raven-agent Peer Agent 委托体系。

包含:
  - AgentCard / AgentSkill / PeerProcessConfig: 数据模型
  - PeerProcessManager: 子进程生命周期管理
  - PeerAgentPoller: A2A 任务轮询
  - PeerAgentTool: Fire-and-forget 委托工具
  - PeerAgentRegistry: 启动时发现与注册（AgentCard 从 TOML 直接构建）
  - build_peer_agent_resources: 同步构建 ProcessManager + Poller
"""

from raven_agent.peer.models import AgentCard, AgentSkill, PeerProcessConfig
from raven_agent.peer.process_manager import PeerProcessManager
from raven_agent.peer.poller import PeerAgentPoller
from raven_agent.peer.tool import PeerAgentTool
from raven_agent.peer.registry import PeerAgentRegistry
from raven_agent.peer.builder import build_peer_agent_resources

__all__ = [
    "AgentCard",
    "AgentSkill",
    "PeerProcessConfig",
    "PeerAgentPoller",
    "PeerAgentRegistry",
    "PeerAgentTool",
    "PeerProcessManager",
    "build_peer_agent_resources",
]