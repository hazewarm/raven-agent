"""Tests for PresenceStore: record, read, list."""

import time
from datetime import datetime, timezone

import pytest

from raven_agent.proactive.presence import PresenceStore, _parse_iso
from raven_agent.session_store import SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    """创建临时 SessionStore（含 presence 表）。"""
    db_path = tmp_path / "sessions.db"
    return SessionStore(db_path)


@pytest.fixture
def presence(store: SessionStore) -> PresenceStore:
    """创建使用临时 SessionStore 的 PresenceStore。"""
    return PresenceStore(store)


# ── record_user_message ────────────────────────────────────────────


def test_record_user_message_stores_timestamp(
    presence: PresenceStore, store: SessionStore
) -> None:
    """record_user_message 后 get_last_user_at 返回合理时间。"""
    now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    presence.record_user_message("tg:123", now)

    result = presence.get_last_user_at("tg:123")
    assert result is not None
    assert abs((result - now).total_seconds()) < 1


def test_record_user_message_updates_existing(
    presence: PresenceStore, store: SessionStore
) -> None:
    """多次调用 record_user_message 更新而非追加。"""
    t1 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 6, 1, 12, 30, 0, tzinfo=timezone.utc)

    presence.record_user_message("tg:123", t1)
    presence.record_user_message("tg:123", t2)

    result = presence.get_last_user_at("tg:123")
    assert result is not None
    # 应该是最新的 t2
    assert abs((result - t2).total_seconds()) < 1


# ── record_proactive_sent ──────────────────────────────────────────


def test_record_proactive_sent_stores_timestamp(
    presence: PresenceStore,
) -> None:
    """record_proactive_sent 后 get_last_proactive_at 返回合理时间。"""
    now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    presence.record_proactive_sent("tg:123", now)

    result = presence.get_last_proactive_at("tg:123")
    assert result is not None
    assert abs((result - now).total_seconds()) < 1


def test_user_and_proactive_independent(
    presence: PresenceStore,
) -> None:
    """user 和 proactive 时间戳独立存储，互不覆盖。"""
    t_user = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_proactive = datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc)

    presence.record_user_message("tg:456", t_user)
    presence.record_proactive_sent("tg:456", t_proactive)

    assert presence.get_last_user_at("tg:456") == t_user
    assert presence.get_last_proactive_at("tg:456") == t_proactive


# ── get_* returns None for unknown session ─────────────────────────


def test_get_last_user_at_unknown_session(
    presence: PresenceStore,
) -> None:
    """未知 session 返回 None。"""
    assert presence.get_last_user_at("unknown:999") is None


def test_get_last_proactive_at_unknown_session(
    presence: PresenceStore,
) -> None:
    """未知 session 返回 None。"""
    assert presence.get_last_proactive_at("unknown:999") is None


# ── most_recent_user_at ────────────────────────────────────────────


def test_most_recent_user_at_returns_max(
    presence: PresenceStore,
) -> None:
    """most_recent_user_at 返回所有 session 中最近的时间。"""
    t1 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    presence.record_user_message("tg:111", t1)
    presence.record_user_message("tg:222", t2)
    presence.record_user_message("cli:default", t3)

    result = presence.most_recent_user_at()
    assert result == t2  # 14:00 是最晚的


def test_most_recent_user_at_empty(presence: PresenceStore) -> None:
    """无任何记录时返回 None。"""
    assert presence.most_recent_user_at() is None


# ── get_all_sessions ───────────────────────────────────────────────


def test_get_all_sessions_lists_all(
    presence: PresenceStore,
) -> None:
    """get_all_sessions 列出所有有 presence 记录的 session。"""
    t1 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc)

    presence.record_user_message("tg:aaa", t1)
    presence.record_proactive_sent("tg:bbb", t2)

    all_sessions = presence.get_all_sessions()
    assert "tg:aaa" in all_sessions
    assert "tg:bbb" in all_sessions
    assert all_sessions["tg:aaa"]["last_user_at"] == t1
    assert all_sessions["tg:bbb"]["last_proactive_at"] == t2


# ── _parse_iso ─────────────────────────────────────────────────────


def test_parse_iso_valid() -> None:
    """正确解析 ISO 格式字符串。"""
    dt = _parse_iso("2025-06-01T12:00:00+00:00")
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 6
    assert dt.hour == 12


def test_parse_iso_none() -> None:
    """None 输入返回 None。"""
    assert _parse_iso(None) is None


def test_parse_iso_empty() -> None:
    """空字符串返回 None。"""
    assert _parse_iso("") is None


def test_parse_iso_invalid() -> None:
    """无效格式返回 None。"""
    assert _parse_iso("not-a-date") is None