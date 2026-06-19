from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from raven_agent.events import TurnCompleted
from raven_agent.llm import LLMProvider
from raven_agent.llm_json import load_json_object_loose
from raven_agent.memory.markdown import MarkdownMemoryStore
from raven_agent.messages import ChatMessage, system_message, user_message
from raven_agent.session import Session, SessionManager


@dataclass(frozen=True)
class ConsolidationWindow:
    """一次 consolidation 选中的消息窗口。

    参数:
        messages: 本次要整理的 ChatMessage 列表。
        start: 起始列表下标，包含；这是 session.messages 的 index，不是 message.seq。
        end: 结束列表下标，不包含；这是 session.messages 的 index，不是 message.seq。

    返回:
        ConsolidationWindow 实例。
    """

    messages: list[ChatMessage]
    start: int
    end: int


@dataclass(frozen=True)
class ConsolidationDraft:
    """LLM 提取出的 Markdown 记忆草稿。

    参数:
        history_entries: 要写入 HISTORY.md 的事件条目。
        pending_items: 要写入 PENDING.md 的候选事实 bullet。
        recent_context: 要写入 RECENT_CONTEXT.md 的完整内容。
    """

    history_entries: list[str] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    recent_context: str = ""


@dataclass(frozen=True)
class ConsolidationResult:
    """consolidation 执行结果。

    参数:
        consolidated_count: 实际整理的消息数量。
        source_ref: 本次写入使用的幂等来源标识。
        skipped: 是否因为窗口不足或提取失败而跳过。
    """

    consolidated_count: int = 0
    source_ref: str = ""
    skipped: bool = False


# 选择 consolidation 窗口的逻辑
def select_consolidation_window(
    session: Session,
    *,
    keep_count: int,
    min_new_messages: int,
    force: bool = False,
) -> ConsolidationWindow | None:
    total = len(session.messages)
    start = session.last_consolidated
    if start >= total:
        return None

    end = total if force else max(start, total - keep_count)
    if end <= start:
        return None

    messages = session.messages[start:end]
    if not force and len(messages) < min_new_messages:
        return None
    return ConsolidationWindow(messages=messages, start=start, end=end)

# 格式化对话窗口
def _format_conversation(messages: list[ChatMessage]) -> str:
    """把 ChatMessage 窗口格式化成 LLM 可读文本。

    参数:
        messages: 要整理的消息列表。

    返回:
        包含 USER / ASSISTANT 行的文本。
    """

    lines: list[str] = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"{message.role.upper()}: {content}")
    return "\n".join(lines)

# 生成 source_ref
def _source_ref(window: ConsolidationWindow) -> str:
    """生成本次 consolidation 的幂等来源标识。

    参数:
        window: 本次整理窗口；窗口内消息必须已经由 SessionManager.save() 分配 id。

    返回:
        JSON 编码的 message id 列表，例如 ["cli:default:0", "cli:default:1"]。

    异常:
        ValueError: 当窗口内存在尚未持久化、没有 id 的消息时抛出。
    """

    missing = [message for message in window.messages if not message.id]
    if missing:
        raise ValueError("consolidation window contains messages without id; save session before consolidation")
    return json.dumps([message.id for message in window.messages], ensure_ascii=False)


# 归一化 pending items
_ALLOWED_PENDING_TAGS = {
    "identity",
    "preference",
    "key_info",
    "health_long_term",
    "requested_memory",
    "correction",
    "agent_context",
}


def _normalize_pending_items(raw_items: object) -> list[str]:
    """把 LLM pending_items 归一化为 Markdown bullet。

    参数:
        raw_items: LLM JSON 中的 pending_items 字段。

    返回:
        形如 - [tag] 内容 的列表。
    """

    if not isinstance(raw_items, list):
        return []
    lines: list[str] = []
    # 使用 set 去重而不是直接用 list，
    # 因为 list 的查找时间复杂度是 O(N)，而 set 基于哈希表实现查找的时间复杂度是 O(1)
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if tag not in _ALLOWED_PENDING_TAGS or not content:
            continue
        line = f"- [{tag}] {content}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines

# 归一化 history entries
def _normalize_history_entries(raw_entries: object) -> list[str]:
    """把 LLM history_entries 归一化为字符串列表。

    参数:
        raw_entries: LLM JSON 中的 history_entries 字段。

    返回:
        历史事件文本列表。
    """

    if not isinstance(raw_entries, list):
        return []
    entries: list[str] = []
    seen: set[str] = set()
    for item in raw_entries:
        if isinstance(item, str):
            summary = item.strip()
        elif isinstance(item, dict):
            summary = str(item.get("summary", "")).strip()
        else:
            continue
        if not summary or summary in seen:
            continue
        seen.add(summary)
        entries.append(summary)
    return entries


# 从 history entry 推导 journal 日期
_HISTORY_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})")


def _append_history_entries_to_journal(
    store: MarkdownMemoryStore,
    entries: list[str],
    source_ref: str,
) -> None:
    """把带日期前缀的 history entries 写入 journal。

    参数:
        store: MarkdownMemoryStore。
        entries: history entries。
        source_ref: 本次 consolidation source_ref。

    返回:
        None。
    """

    # === 阶段 1：按日期对离散数据进行分组 (Grouping) ===
    # 定义一个哈希表，键是日期字符串 (如 '2026-05-29')，值是属于该日期的所有记录列表。
    # 目的：为了后续批量写入，避免同一天的数据被反复多次进行 I/O 操作。
    by_date: dict[str, list[str]] = {}
    for entry in entries:
        match = _HISTORY_DATE_RE.match(entry)
        if match is None:
            continue
        by_date.setdefault(match.group(1), []).append(entry)
    
    # === 阶段 2：将归类好的数据批量写入底层存储 (Batch Writing) ===
    # 遍历刚才分好组的字典
    for date_str, date_entries in by_date.items():
        store.append_journal(
            date_str,
            "\n".join(date_entries),
            source_ref=source_ref,
            kind=f"journal:{date_str}",
        )



class MarkdownMemoryMaintenance:
    """对话完成后的 Markdown 记忆维护器。

    参数:
        store: MarkdownMemoryStore。
        provider: LLMProvider。
        sessions: SessionManager。
        keep_count: 保留最近多少条消息不整理。
        min_new_messages: 至少累计多少条旧消息才触发 consolidation。
    """

    def __init__(
        self,
        *,
        store: MarkdownMemoryStore,
        provider: LLMProvider,
        sessions: SessionManager,
        keep_count: int = 4,
        min_new_messages: int = 10,
    ) -> None:
        self._store = store
        self._provider = provider
        self._sessions = sessions
        self._keep_count = keep_count
        self._min_new_messages = min_new_messages

    async def on_turn_completed(self, event: TurnCompleted) -> None:
        """处理 TurnCompleted 观察事件。

        参数:
            event: 当前 turn 完成事件。

        返回:
            None。
        """

        session = self._sessions.get_or_create(event.session_key)
        result = await self.consolidate_session(session)
        if not result.skipped:
            self._sessions.save(session)
    
    async def consolidate_session(
        self,
        session: Session,
        *,
        force: bool = False,
    ) -> ConsolidationResult:
        """整理一个 session 的旧消息窗口。

        参数:
            session: 要整理的 Session。
            force: 是否强制整理所有未整理消息。

        返回:
            ConsolidationResult。
        """
        self._sessions.save(session)  # 确保 session.messages 中的消息都有 id

        window = select_consolidation_window(
            session,
            keep_count=self._keep_count,
            min_new_messages=self._min_new_messages,
            force=force,
        )
        if window is None:
            tail = session.history_for_prompt(self._keep_count)
            self._store.replace_recent_turns(tail)
            return ConsolidationResult(skipped=True)

        source_ref = _source_ref(window)
        draft = await self._extract_draft(window)
        if draft is None:
            return ConsolidationResult(skipped=True, source_ref=source_ref)

        if draft.history_entries:
            self._store.append_history_once(
                "\n".join(draft.history_entries),
                source_ref=source_ref,
                kind="history_entry",
            )
            _append_history_entries_to_journal(
                self._store,
                draft.history_entries,
                source_ref,
            )

        if draft.pending_items:
            self._store.append_pending_once(
                "\n".join(draft.pending_items),
                source_ref=source_ref,
                kind="pending_items",
            )

        if draft.recent_context:
            self._store.write_recent_context(draft.recent_context)
        else:
            tail = session.history_for_prompt(self._keep_count)
            self._store.replace_recent_turns(tail)

        session.last_consolidated = window.end
        self._sessions.save(session)  # 持久化游标位置

        return ConsolidationResult(
            consolidated_count=len(window.messages),
            source_ref=source_ref,
            skipped=False,
        )
    
    async def _extract_draft(self, window: ConsolidationWindow) -> ConsolidationDraft | None:
        """调用 LLM 从消息窗口提取 Markdown 记忆草稿。

        参数:
            window: 本次要整理的消息窗口。

        返回:
            ConsolidationDraft；解析失败返回 None。
        """

        conversation = _format_conversation(window.messages)
        if not conversation:
            return ConsolidationDraft()

        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = f"""你是 Raven 的记忆整理代理。请只根据【待整理对话】提取可沉淀的信息。

当前时间：{today}

输出 JSON，格式必须是：
{{
  "history_entries": [{{"summary": "[YYYY-MM-DD HH:MM] 事件摘要"}}],
  "pending_items": [{{"tag": "identity|preference|key_info|health_long_term|requested_memory|correction|agent_context", "content": "候选长期事实"}}],
  "recent_context": {{
    "active_topics": [],
    "user_preferences": [],
    "follow_ups": [],
    "avoidances": [],
    "ongoing_threads": []
  }}
}}

规则：
- history_entries 记录用户明确表达的事件、计划、状态或讨论主题，1-2 句即可。
- pending_items 只记录跨对话仍有长期价值的用户事实、偏好、明确要求记住的信息或助手操作上下文。
- 不要把 assistant 的建议当作用户事实。
- 不要记录短期情绪、一次性状态、普通寒暄、工具调用过程。
- 如果没有合适内容，返回空数组。
- 只返回合法 JSON，不要 markdown 代码块，不要解释。

【待整理对话】
{conversation}
"""
        response = await self._provider.chat(
            messages=[
                system_message("你只输出合法 JSON。"),
                user_message(prompt),
            ],
            tools=[],
            tool_choice="none",
        )
        payload = load_json_object_loose(response.content)
        if payload is None:
            return None

        history_entries = _normalize_history_entries(payload.get("history_entries"))
        pending_items = _normalize_pending_items(payload.get("pending_items"))
        recent_context = self._render_recent_context(payload.get("recent_context"), window.messages)
        return ConsolidationDraft(
            history_entries=history_entries,
            pending_items=pending_items,
            recent_context=recent_context,
        )
    
    
    def _render_recent_context(
        self,
        raw_context: object,
        messages: list[ChatMessage],
    ) -> str:
        """把 LLM recent_context JSON 渲染为 RECENT_CONTEXT.md。

        参数:
            raw_context: LLM 返回的 recent_context 字段。
            messages: 当前整理窗口消息，用于刷新 Recent Turns。

        返回:
            RECENT_CONTEXT.md 完整文本。
        """

        context = raw_context if isinstance(raw_context, dict) else {}
        sections = [
            ("最近持续关注", context.get("active_topics", [])),
            ("最近明确偏好", context.get("user_preferences", [])),
            ("最近待延续话题", context.get("follow_ups", [])),
            ("最近避免事项", context.get("avoidances", [])),
        ]
        lines = ["# Recent Context", "", "## Compression", f"until: {datetime.now().isoformat(timespec='minutes')}"]
        wrote_any = False
        for title, values in sections:
            cleaned = [str(value).strip() for value in values if str(value).strip()] if isinstance(values, list) else []
            if not cleaned:
                continue
            wrote_any = True
            lines.append(f"- {title}：{'；'.join(cleaned[:3])}")
        if not wrote_any:
            lines.append("- none")

        lines.extend(["", "## Ongoing Threads"])
        ongoing = context.get("ongoing_threads", [])
        ongoing_items = [str(value).strip() for value in ongoing if str(value).strip()] if isinstance(ongoing, list) else []
        if ongoing_items:
            for item in ongoing_items[:3]:
                lines.append(f"- {item}")
        else:
            lines.append("- none")

        recent_turns = self._store.format_recent_turns(messages)  # 当前章节内复用第 14 章实现
        lines.extend(["", "## Recent Turns", "<!-- a-preview = assistant reply preview only -->"])
        lines.append(recent_turns or "- none")
        return "\n".join(lines).rstrip() + "\n"