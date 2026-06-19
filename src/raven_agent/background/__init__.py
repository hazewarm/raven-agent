"""raven-agent 后台控制面模块。

包含:
- InterruptManager / TurnInterruptState / InterruptResult: 中断系统
- BackgroundRuntime / BackgroundJob / BackgroundJobRunner: 后台任务运行时
"""

from raven_agent.background.interrupt import (
    InterruptManager,
    InterruptResult,
    TurnInterruptState,
)
from raven_agent.background.runtime import (
    BackgroundJob,
    BackgroundJobRunner,
    BackgroundRuntime,
)

from raven_agent.background.delegation import (
    DelegationPolicy,
    SpawnDecision,
    SpawnDecisionMeta,
)
from raven_agent.background.subagent import SubAgent
from raven_agent.background.subagent_profiles import (
    PROFILE_GENERAL,
    PROFILE_RESEARCH,
    PROFILE_SCRIPTING,
    SubagentRuntime,
    SubagentSpec,
    build_spawn_spec,
    build_spawn_subagent_prompt,
)
from raven_agent.background.subagents import (
    RunningSubagentJob,
    SpawnAwareBackgroundJobRunner,
    SubagentJobRunner,
    SubagentManager,
)

__all__ = [
    "BackgroundJob",
    "BackgroundJobRunner",
    "BackgroundRuntime",
    "InterruptManager",
    "InterruptResult",
    "TurnInterruptState",
    "DelegationPolicy",
    "SpawnDecision",
    "SpawnDecisionMeta",
    "SubAgent",
    "SubagentRuntime",
    "SubagentSpec",
    "build_spawn_spec",
    "build_spawn_subagent_prompt",
    "RunningSubagentJob",
    "SpawnAwareBackgroundJobRunner",
    "SubagentJobRunner",
    "SubagentManager",
]