"""raven-agent Proactive 主动推送系统。

包含:
  - PresenceStore: 用户在线心跳管理
  - Energy Model: 多时间尺度指数衰减电量 + 三维复合打分
  - ProactiveLoop: 自适应 tick 循环骨架
"""

from raven_agent.proactive.energy import (
    composite_score,
    compute_energy,
    d_content,
    d_energy,
    d_recent,
    next_tick_from_score,
    random_weight,
)
from raven_agent.proactive.loop import ProactiveLoop
from raven_agent.proactive.presence import PresenceStore

from raven_agent.proactive.sensor import Sensor

from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore, SkillMeta
from raven_agent.proactive.drift_tools import (
    DriftReadFileTool,
    DriftSendMessageTool,
    DriftWebFetchTool,
    FinishDriftTool,
    build_drift_tool_registry,
)
from raven_agent.proactive.drift_turn import DriftTurnPipeline

from raven_agent.proactive.contracts import (
    AlertContract,
    ContentContract,
    ContextContract,
    normalize_alert,
    normalize_content,
    normalize_context,
)
from raven_agent.proactive.mcp_sources import McpSourceFetcher, SourceStore
from raven_agent.proactive.source_tools import (
    ProactiveSourceAddTool,
    ProactiveSourceListTool,
    ProactiveSourceRemoveTool,
)
from raven_agent.proactive.state import ProactiveStateStore

__all__ = [
    "PresenceStore",
    "ProactiveLoop",
    "composite_score",
    "compute_energy",
    "d_content",
    "d_energy",
    "d_recent",
    "next_tick_from_score",
    "random_weight",
    "Sensor",
    "DriftAgentTickContext",
    "DriftReadFileTool",
    "DriftSendMessageTool",
    "DriftTurnPipeline",
    "DriftWebFetchTool",
    "DriftStateStore",
    "FinishDriftTool",
    "SkillMeta",
    "build_drift_tool_registry",
    "AlertContract",
    "ContentContract",
    "ContextContract",
    "McpSourceFetcher",
    "ProactiveSourceAddTool",
    "ProactiveSourceListTool",
    "ProactiveSourceRemoveTool",
    "ProactiveStateStore",
    "SourceStore",
    "normalize_alert",
    "normalize_content",
    "normalize_context",
]