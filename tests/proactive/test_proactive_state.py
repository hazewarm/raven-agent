"""Tests for ProactiveStateStore: seen items, deliveries, cleanup."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from raven_agent.proactive.state import ProactiveStateStore


@pytest.fixture
def store(tmp_path: Path) -> ProactiveStateStore:
    """创建临时 ProactiveStateStore。"""
    return ProactiveStateStore(tmp_path / "test_state.db")


def test_is_item_seen_false_initially(store: ProactiveStateStore) -> None:
    """新 store 中所有 item 都未见过。"""
    assert store.is_item_seen("github", "event-001", 72) is False


def test_mark_and_see_item(store: ProactiveStateStore) -> None:
    """标记后再查询返回 True。"""
    store.mark_items_seen([("github", "event-001")])
    assert store.is_item_seen("github", "event-001", 72) is True


def test_item_expires(store: ProactiveStateStore) -> None:
    """TTL 过期后 is_item_seen 返回 False。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=48)
    store.mark_items_seen([("github", "event-001")], now=old)
    assert store.is_item_seen("github", "event-001", 24, now=now) is False


def test_delivery_tracking(store: ProactiveStateStore) -> None:
    """mark_delivery 后 count_deliveries_in_window 增加。"""
    store.mark_delivery("tg:123", "key-001", content="测试消息")
    assert store.count_deliveries_in_window("tg:123", 24) == 1


def test_delivery_duplicate(store: ProactiveStateStore) -> None:
    """重复推送检测。"""
    store.mark_delivery("tg:123", "key-001", content="已推送过")
    assert store.is_delivery_duplicate("tg:123", "key-001", 24) is True
    assert store.is_delivery_duplicate("tg:123", "key-002", 24) is False


def test_recent_deliveries(store: ProactiveStateStore) -> None:
    """recent_deliveries 返回最近 N 条推送。"""
    for i in range(5):
        store.mark_delivery("tg:123", f"key-{i:03d}", content=f"消息 {i}")
    recent = store.recent_deliveries("tg:123", n=3)
    assert len(recent) == 3
    assert "消息 4" in recent[0]["content"]


def test_cleanup_removes_old_items(store: ProactiveStateStore) -> None:
    """cleanup 删除过期数据。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=200)
    store.mark_items_seen([("github", "old-event")], now=old)
    store.mark_delivery("tg:123", "old-key", content="旧消息")
    store._db.execute("UPDATE seen_items SET seen_at = ?", (old.isoformat(),))
    store._db.execute("UPDATE deliveries SET sent_at = ?", (old.isoformat(),))
    store._db.commit()

    store.cleanup(seen_ttl_hours=168, delivery_ttl_hours=168)
    assert store.is_item_seen("github", "old-event", 168) is False
    assert store.count_deliveries_in_window("tg:123", 300) == 0