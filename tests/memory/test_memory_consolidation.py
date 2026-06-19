from __future__ import annotations

import asyncio
import json

from raven_agent.llm import LLMResponse
from raven_agent.memory import MarkdownMemoryMaintenance, select_consolidation_window
from raven_agent.memory.markdown import MarkdownMemoryStore
from raven_agent.session import Session, SessionManager
from raven_agent.session_store import SessionStore
from raven_agent.memory.consolidation import _source_ref


class FakeProvider:
    """测试用 LLMProvider。"""

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []

    async def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        """返回固定 LLMResponse。"""

        self.messages = messages
        return LLMResponse(content=self.response)


def test_select_consolidation_window_keeps_recent_messages() -> None:
    """测试 consolidation window 会保留最近消息。"""

    session = Session(key="cli:default")
    for index in range(6):
        session.add_user_message(f"u{index}")

    window = select_consolidation_window(
        session,
        keep_count=2,
        min_new_messages=2,
    )

    assert window is not None
    assert window.start == 0
    assert window.end == 4
    assert [message.content for message in window.messages] == ["u0", "u1", "u2", "u3"]

def test_memory_maintenance_refreshes_recent_turns_when_window_not_ready(tmp_path) -> None:
    """测试消息不足时只刷新 Recent Turns。"""

    async def run() -> None:
        store = MarkdownMemoryStore(tmp_path / "memory")
        sessions = SessionManager(SessionStore(tmp_path / "sessions"))
        session = sessions.get_or_create("cli:default")
        session.add_user_message("你好")
        session.add_assistant_message("你好，我是 Raven。")
        provider = FakeProvider("{}")
        maintenance = MarkdownMemoryMaintenance(
            store=store,
            provider=provider,  # type: ignore[arg-type]
            sessions=sessions,
            keep_count=4,
            min_new_messages=4,
        )

        result = await maintenance.consolidate_session(session)

        assert result.skipped is True
        assert "[user] 你好" in store.read_recent_context()
        assert session.last_consolidated == 0

    asyncio.run(run())


def test_memory_maintenance_consolidates_session_to_markdown_files(tmp_path) -> None:
    """测试 consolidation 会写入 HISTORY/PENDING/RECENT_CONTEXT/journal 并更新游标。"""

    async def run() -> None:
        response = """
{
  "history_entries": [
    {"summary": "[2026-05-28 10:00] 用户开始实现 Raven 的记忆 consolidation。"}
  ],
  "pending_items": [
    {"tag": "preference", "content": "用户喜欢平滑增量式教程。"}
  ],
  "recent_context": {
    "active_topics": ["Raven Markdown 记忆系统"],
    "user_preferences": ["平滑增量式教程"],
    "follow_ups": [],
    "avoidances": [],
    "ongoing_threads": ["raven-agent 教程编写"]
  }
}
"""
        store = MarkdownMemoryStore(tmp_path / "memory")
        sessions = SessionManager(SessionStore(tmp_path / "sessions"))
        session = sessions.get_or_create("cli:default")
        for index in range(6):
            session.add_user_message(f"用户消息 {index}")
            session.add_assistant_message(f"助手消息 {index}")
        provider = FakeProvider(response)
        maintenance = MarkdownMemoryMaintenance(
            store=store,
            provider=provider,  # type: ignore[arg-type]
            sessions=sessions,
            keep_count=2,
            min_new_messages=4,
        )

        result = await maintenance.consolidate_session(session)

        assert result.skipped is False
        assert result.consolidated_count == 10
        assert session.last_consolidated == 10
        assert "记忆 consolidation" in store.read_history()
        assert "平滑增量式教程" in store.read_pending()
        assert "Raven Markdown 记忆系统" in store.read_recent_context()
        assert (store.journal_dir / "2026-05-28.md").exists()

    asyncio.run(run())


def test_consolidation_source_ref_uses_message_ids(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)

    try:
        session = manager.get_or_create("cli:default")
        session.add_user_message("old")
        manager.save(session)

        window = select_consolidation_window(
            session,
            keep_count=0,
            min_new_messages=1,
            force=True,
        )

        assert window is not None
        assert json.loads(_source_ref(window)) == ["cli:default:0"]
    finally:
        store.close()

def test_source_ref_does_not_collide_after_clear(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)

    try:
        first = manager.get_or_create("cli:default")
        first.add_user_message("old")
        manager.save(first)
        first_window = select_consolidation_window(
            first,
            keep_count=0,
            min_new_messages=1,
            force=True,
        )
        assert first_window is not None
        first_ref = _source_ref(first_window)

        second = manager.clear("cli:default")
        second.add_user_message("new")
        manager.save(second)
        second_window = select_consolidation_window(
            second,
            keep_count=0,
            min_new_messages=1,
            force=True,
        )
        assert second_window is not None
        second_ref = _source_ref(second_window)

        assert json.loads(first_ref) == ["cli:default:0"]
        assert json.loads(second_ref) == ["cli:default:1"]
        assert first_ref != second_ref
    finally:
        store.close()

def test_last_consolidated_remains_list_cursor(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)

    try:
        session = manager.get_or_create("cli:default")
        for text in ["m0", "m1", "m2", "m3"]:
            session.add_user_message(text)
        manager.save(session)

        session.last_consolidated = 1
        window = select_consolidation_window(
            session,
            keep_count=1,
            min_new_messages=1,
            force=False,
        )

        assert window is not None
        assert [message.content for message in window.messages] == ["m1", "m2"]
        assert window.start == 1
        assert window.end == 3
    finally:
        store.close()