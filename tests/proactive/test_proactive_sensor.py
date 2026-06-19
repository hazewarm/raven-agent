"""Tests for Sensor: recent chat collection, interruptibility, delivery tracking."""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raven_agent.messages import ChatMessage
from raven_agent.proactive.sensor import Sensor


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_sessions() -> MagicMock:
    """返回 mock SessionManager。"""
    m = MagicMock()
    session = MagicMock()
    session.messages = [
        ChatMessage(role="user", content="你好"),
        ChatMessage(role="assistant", content="你好！有什么可以帮你的？"),
        ChatMessage(role="user", content="今天天气怎么样？"),
    ]
    m.get_or_create.return_value = session
    return m


@pytest.fixture
def mock_presence() -> MagicMock:
    """返回 mock PresenceStore。"""
    m = MagicMock()
    m.get_last_user_at.return_value = datetime.now(timezone.utc) - timedelta(minutes=10)
    m.get_last_proactive_at.return_value = None
    m.most_recent_user_at.return_value = datetime.now(timezone.utc) - timedelta(minutes=5)
    return m


@pytest.fixture
def mock_memory() -> MagicMock:
    """返回 mock MarkdownMemoryStore。"""
    m = MagicMock()
    m.read_long_term.return_value = "- 用户喜欢科技新闻\n- 用户每天早上8点起床"
    return m


@pytest.fixture
def mock_cfg() -> MagicMock:
    """返回 mock ProactiveConfig。"""
    m = MagicMock()
    m.recent_chat_messages = 20
    m.interrupt_weight_reply = 0.35
    m.interrupt_weight_activity = 0.25
    m.interrupt_weight_fatigue = 0.15
    m.interrupt_activity_decay_minutes = 180.0
    m.interrupt_reply_decay_minutes = 120.0
    m.interrupt_no_reply_decay_minutes = 360.0
    m.interrupt_fatigue_window_hours = 24
    m.interrupt_fatigue_soft_cap = 6.0
    m.interrupt_random_strength = 0.12
    m.interrupt_min_floor = 0.08
    m.score_recent_scale = 10.0
    return m


@pytest.fixture
def sensor(
    mock_sessions: MagicMock,
    mock_presence: MagicMock,
    mock_memory: MagicMock,
    mock_cfg: MagicMock,
    tmp_path: Path,
) -> Sensor:
    """创建测试用 Sensor。"""
    return Sensor(
        sessions=mock_sessions,
        presence=mock_presence,
        memory=mock_memory,
        workspace_root=tmp_path,
        cfg=mock_cfg,
    )


# ── collect_recent_chat ───────────────────────────────────────────


def test_collect_recent_chat_returns_messages(sensor: Sensor) -> None:
    """collect_recent_chat 返回近期的 user + assistant 消息。"""
    result = sensor.collect_recent_chat("tg:123")
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "你好"


def test_collect_recent_chat_truncates_content(sensor: Sensor) -> None:
    """每条消息的 content 被截断到 200 字符。"""
    long_msg = "x" * 500
    session = MagicMock()
    session.messages = [
        ChatMessage(role="user", content=long_msg),
    ]
    sensor._sessions.get_or_create.return_value = session

    result = sensor.collect_recent_chat("tg:123")
    assert len(result[0]["content"]) <= 200


def test_collect_recent_chat_skips_system_context(sensor: Sensor) -> None:
    """系统上下文帧（以 [系统上下文] 开头）被跳过。"""
    session = MagicMock()
    session.messages = [
        ChatMessage(role="user", content="[系统上下文] channel=telegram"),
        ChatMessage(role="user", content="真实用户消息"),
    ]
    sensor._sessions.get_or_create.return_value = session

    result = sensor.collect_recent_chat("tg:123")
    assert len(result) == 1
    assert result[0]["content"] == "真实用户消息"


def test_collect_recent_chat_session_not_found(sensor: Sensor) -> None:
    """session 不存在时返回空列表。"""
    sensor._sessions.get_or_create.side_effect = Exception("not found")
    result = sensor.collect_recent_chat("nonexistent:999")
    assert result == []


# ── read_long_term_memory ─────────────────────────────────────────


def test_read_long_term_memory(sensor: Sensor) -> None:
    """read_long_term_memory 返回 MEMORY.md 内容。"""
    text = sensor.read_long_term_memory()
    assert "科技新闻" in text


def test_read_long_term_memory_none(sensor: Sensor) -> None:
    """无 memory store 时返回空字符串。"""
    sensor._memory = None
    assert sensor.read_long_term_memory() == ""


def test_has_memory_true(sensor: Sensor) -> None:
    """有记忆时 has_memory 返回 True。"""
    assert sensor.has_memory() is True


def test_has_memory_false(sensor: Sensor) -> None:
    """无记忆时 has_memory 返回 False。"""
    sensor._memory = None
    assert sensor.has_memory() is False


# ── read_proactive_context ───────────────────────────────────────


def test_read_proactive_context_file_exists(
    mock_sessions: MagicMock,
    mock_presence: MagicMock,
    mock_memory: MagicMock,
    mock_cfg: MagicMock,
    tmp_path: Path,
) -> None:
    """read_proactive_context 读取 PROACTIVE_CONTEXT.md 内容。"""
    (tmp_path / "PROACTIVE_CONTEXT.md").write_text(
        "- 白名单：科技、游戏\n- 黑名单：政治",
        encoding="utf-8",
    )
    s = Sensor(
        sessions=mock_sessions,
        presence=mock_presence,
        memory=mock_memory,
        workspace_root=tmp_path,
        cfg=mock_cfg,
    )
    text = s.read_proactive_context()
    assert "白名单" in text
    assert "黑名单" in text


def test_read_proactive_context_missing_file(sensor: Sensor) -> None:
    """规则文件不存在时返回空字符串。"""
    # tmp_path 下没有 PROACTIVE_CONTEXT.md
    assert sensor.read_proactive_context() == ""


# ── compute_interruptibility ──────────────────────────────────────


def test_compute_interruptibility_returns_range(
    sensor: Sensor, mock_presence: MagicMock
) -> None:
    """打断系数在 [min_floor, 1.0] 范围内。"""
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now - timedelta(minutes=10)

    score, detail = sensor.compute_interruptibility(
        "tg:123", now_utc=now, recent_msg_count=5
    )
    assert 0.0 <= score <= 1.0
    assert "f_reply" in detail
    assert "f_activity" in detail
    assert "f_fatigue" in detail
    assert "random_delta" in detail


def test_compute_interruptibility_no_presence(sensor: Sensor) -> None:
    """无 presence 时返回默认最大值 1.0。"""
    sensor._presence = None
    score, detail = sensor.compute_interruptibility("tg:123")
    assert score == 1.0


def test_reply_factor_no_proactive_yet(sensor: Sensor) -> None:
    """从未推送过时 f_reply 返回中性值 0.6。"""
    now = datetime.now(timezone.utc)
    f = sensor._reply_factor("tg:123", now)
    assert f == pytest.approx(0.6)


def test_reply_factor_user_replied_quickly(sensor: Sensor) -> None:
    """用户在上次推送后快速回复 → f_reply 较高。"""
    now = datetime.now(timezone.utc)
    sensor._presence.get_last_proactive_at.return_value = now - timedelta(minutes=10)
    sensor._presence.get_last_user_at.return_value = now - timedelta(minutes=9)
    # 1 分钟后回复 → f_reply = exp(-1/120) ≈ 0.992
    f = sensor._reply_factor("tg:123", now)
    assert f > 0.95


def test_reply_factor_no_reply_long_silence(sensor: Sensor) -> None:
    """推送后用户长时间未回复 → f_reply 很低。"""
    now = datetime.now(timezone.utc)
    sensor._presence.get_last_proactive_at.return_value = now - timedelta(hours=10)
    # 10h 无回复 → f_reply = 0.15 + 0.35 * exp(-600/360) ≈ 0.15 + 0.066 ≈ 0.216
    f = sensor._reply_factor("tg:123", now)
    assert f < 0.3


def test_activity_factor_active_user(sensor: Sensor) -> None:
    """活跃用户 f_activity 较高。"""
    now = datetime.now(timezone.utc)
    sensor._presence.most_recent_user_at.return_value = now - timedelta(minutes=1)
    f = sensor._activity_factor("tg:123", now, recent_msg_count=10)
    assert f > 0.5


def test_activity_factor_idle_user(sensor: Sensor) -> None:
    """长时间不活跃用户 f_activity 很低。"""
    now = datetime.now(timezone.utc)
    sensor._presence.most_recent_user_at.return_value = now - timedelta(hours=24)
    f = sensor._activity_factor("tg:123", now, recent_msg_count=0)
    assert f < 0.05


def test_fatigue_factor_no_deliveries(sensor: Sensor) -> None:
    """从未推送过 → f_fatigue = 1.0。"""
    now = datetime.now(timezone.utc)
    f = sensor._fatigue_factor("tg:123", now)
    assert f == pytest.approx(1.0)


def test_fatigue_factor_many_deliveries(sensor: Sensor) -> None:
    """大量推送后 f_fatigue 显著下降。"""
    now = datetime.now(timezone.utc)
    # 模拟 10 次推送
    for _ in range(10):
        sensor.record_delivery("tg:123", now - timedelta(minutes=30))
    f = sensor._fatigue_factor("tg:123", now)
    # f = 1 / (1 + 10/6) ≈ 0.375
    assert f < 0.5


# ── delivery tracking ────────────────────────────────────────────


def test_record_delivery_increments_count(sensor: Sensor) -> None:
    """record_delivery 后计数增加。"""
    now = datetime.now(timezone.utc)
    initial = sensor._count_deliveries_in_window("tg:abc", 24, now)
    sensor.record_delivery("tg:abc", now)
    after = sensor._count_deliveries_in_window("tg:abc", 24, now)
    assert after == initial + 1


def test_count_deliveries_respects_window(sensor: Sensor) -> None:
    """只统计窗口内的推送。"""
    now = datetime.now(timezone.utc)
    sensor.record_delivery("tg:xyz", now - timedelta(hours=25))
    count = sensor._count_deliveries_in_window("tg:xyz", 24, now)
    assert count == 0


# ── collect_recent_proactive 当前为桩，在第 33 章激活 ─────
# 届时测试基于 ProactiveStateStore 的 delivery 记录，而非 ChatMessage 历史。