"""
proactive/drift_context.py —— Drift tick 级上下文。

每次 Drift 执行时，ProactiveLoop 创建一个 DriftAgentTickContext 实例，
跟踪整条 drift 链路的状态标志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class DriftAgentTickContext:
    """单次 Drift tick 的运行时上下文。

    字段:
        tick_id: 本次 tick 的唯一 ID（8 位 hex）。
        now_utc: tick 开始时的 UTC 时间（整条链路共享同一时间戳）。
        session_key: 目标会话 key（"channel:chat_id"）。
        drift_entered: 是否已进入 Drift 模式（Prepare 阶段设为 True）。
        drift_finished: 是否已正常结束（FinishDriftTool 调用后设为 True）。
        drift_message_sent: 是否已发送过消息推送（SendMessageTool 调用后设为 True）。
        steps_taken: 已执行的工具步数。
    """

    tick_id: str = field(default_factory=lambda: uuid4().hex[:8])
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    drift_entered: bool = False
    drift_finished: bool = False
    drift_message_sent: bool = False
    steps_taken: int = 0