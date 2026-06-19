from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpawnDecisionSource = Literal["heuristic", "llm", "manual_rule"]
SpawnDecisionConfidence = Literal["high", "medium", "low"]
SpawnDecisionReasonCode = Literal[
    "long_running",
    "context_isolation_needed",
    "tool_chain_heavy",
    "stay_inline",
    "fallback_inline",
]

_MAX_CONCURRENT_SPAWNS = 3


@dataclass(frozen=True)
class SpawnDecisionMeta:
    """一次 spawn 决策的元信息。

    输入:
        source: 决策来源。当前主要是 "llm" 或 "heuristic"。
        confidence: 决策置信度。
        reason_code: 决策原因代码。

    输出:
        SpawnDecisionMeta 实例。
    """

    source: SpawnDecisionSource
    confidence: SpawnDecisionConfidence
    reason_code: SpawnDecisionReasonCode


@dataclass(frozen=True)
class SpawnDecision:
    """是否允许创建 spawn 子任务的决策。

    输入:
        should_spawn: True 表示允许创建任务。
        label: 标准化后的任务标签。
        meta: 决策元信息。
        block_reason: 被拦截时给模型看的原因。

    输出:
        SpawnDecision 实例。
    """

    should_spawn: bool
    label: str
    meta: SpawnDecisionMeta
    block_reason: str = ""


class DelegationPolicy:
    """本地 spawn 委托策略。

    当前策略：
    - 后台模式下，如果运行中/排队中的 spawn 任务数达到上限，则拒绝。
    - 其他情况允许，由 LLM 根据 SpawnTool.description 的说明决定是否调用。

    输入:
        max_concurrent_spawns: 允许的最大并发后台 spawn 数，默认 3。
    """

    def __init__(self, max_concurrent_spawns: int = _MAX_CONCURRENT_SPAWNS) -> None:
        self._max_concurrent_spawns = max(1, int(max_concurrent_spawns))

    def decide(
        self,
        *,
        task: str,
        label: str | None = None,
        running_count: int = 0,
    ) -> SpawnDecision:
        """判断是否允许创建 spawn 任务。

        输入:
            task: 子任务完整描述。
            label: 可选短标签。
            running_count: 当前运行中/排队中的后台 spawn 数。

        输出:
            SpawnDecision。should_spawn=False 时包含 block_reason。
        """
        normalized_label = (label or (task or "")[:30] or "").strip()
        if running_count >= self._max_concurrent_spawns:
            return SpawnDecision(
                should_spawn=False,
                label=normalized_label,
                block_reason=(
                    f"已有 {running_count} 个并发子任务在运行，"
                    f"上限 {self._max_concurrent_spawns}，请等待当前任务完成后再试"
                ),
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="high",
                    reason_code="stay_inline",
                ),
            )
        return SpawnDecision(
            should_spawn=True,
            label=normalized_label,
            meta=SpawnDecisionMeta(
                source="llm",
                confidence="high",
                reason_code="tool_chain_heavy",
            ),
        )