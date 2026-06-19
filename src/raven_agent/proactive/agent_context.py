"""
proactive/agent_context.py —— Agent Loop 单次 tick 的运行时状态。

AgentTickContext 在 Phase 2（Fetch）构建，在 Phase 3（Agent Loop）中被
各 tool dispatch 函数修改，在 Phase 4（Resolve）读取决策结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentTickContext:
    """一次 proactive tick 的完整运行时上下文。

    字段:
        session_key: 目标会话 key（channel:chat_id）。
        now_utc: tick 触发时的 UTC 时间。
        max_steps: Agent Loop 最大工具调用步数，默认 20。

        # ── Fetch 阶段填充 ──
        alerts: 本轮 alert 事件列表（AlertContract 实例）。
        contents: 本轮 content 事件列表（ContentContract 实例）。
        contexts: 本轮 context 数据列表（ContextContract 实例）。
        recent_chat: 近期对话文本（已格式化，带时间戳）。
        memory_text: 长期记忆文本。
        context_rules: PROACTIVE_CONTEXT.md 规则文本。
        recent_proactive_text: 近期主动消息摘要。

        # ── Agent Loop 阶段填充 ──
        interesting_ids: LLM 标记为 interesting 的条目 ID 集合。
        discarded_ids: LLM 标记为不感兴趣的条目 ID 集合。
        cited_ids: 最终消息引用的条目 ID 列表（evidence，用于去重和 ACK）。
        final_message: LLM 撰写的最终推送消息文本。
        terminal_action: "reply" 或 "skip"（由 finish_turn 设置）。
        skip_reason: skip 时的原因。
        steps_taken: Agent Loop 已执行步数。

        # ── 辅助 ──
        recent_chat_raw: 近期对话原始数据列表，供 get_recent_chat 工具返回。
    """

    # ── 标识 ──
    session_key: str = ""
    now_utc: datetime | None = None
    max_steps: int = 20

    # ── Fetch 阶段填充 ──
    alerts: list = field(default_factory=list)
    contents: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    recent_chat: str = ""
    memory_text: str = ""
    context_rules: str = ""
    recent_proactive_text: str = ""

    # ── 预取正文缓存（item_id → body text）──
    content_store: dict[str, str] = field(default_factory=dict)

    # ── Agent Loop 阶段填充 ──
    interesting_ids: set[str] = field(default_factory=set)
    discarded_ids: set[str] = field(default_factory=set)
    cited_ids: list[str] = field(default_factory=list)
    final_message: str = ""
    terminal_action: str | None = None
    skip_reason: str = ""
    steps_taken: int = 0

    # ── 辅助 ──
    recent_chat_raw: list[dict] = field(default_factory=list)

    @property
    def all_content_ids(self) -> set[str]:
        """本轮所有 content 候选条目的 ID 集合。"""
        return {c.item_id for c in self.contents}

    @property
    def all_alert_ids(self) -> set[str]:
        """本轮所有 alert 的 ID 集合。"""
        return {a.item_id for a in self.alerts}

    @property
    def unclassified_ids(self) -> set[str]:
        """尚未被分类（interesting/discarded）的 content ID 集合。"""
        return self.all_content_ids - self.interesting_ids - self.discarded_ids