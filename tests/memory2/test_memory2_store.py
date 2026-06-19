from __future__ import annotations

from datetime import datetime
from raven_agent.memory2 import MemoryStore2, now_iso, content_hash


def test_memory_store_creates_sqlite_database(tmp_path) -> None:
    """测试 MemoryStore2 会创建 SQLite 数据库文件。"""

    db_path = tmp_path / "memory2.db"
    store = MemoryStore2(db_path)

    try:
        assert db_path.exists()
    finally:
        store.close()


def test_memory_store_upsert_item_writes_and_reinforces_duplicate(tmp_path) -> None:
    """测试 upsert_item 写入新条目并强化重复条目。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        first = store.upsert_item(
            memory_type="preference",
            summary="用户喜欢简洁回答。",
            embedding=[1.0, 0.0],
            source_ref="test@1",
        )
        second = store.upsert_item(
            memory_type="preference",
            summary="用户喜欢简洁回答。",
            embedding=[1.0, 0.0],
            source_ref="test@2",
        )
        item_id = first.split(":", 1)[1]
        item = store.get_item(item_id)

        assert first.startswith("new:")
        assert second == f"reinforced:{item_id}"
        assert item is not None
        assert item.reinforcement == 2
        assert item.status == "active"
    finally:
        store.close()


def test_memory_store_vector_search_orders_by_cosine_similarity(tmp_path) -> None:
    """测试 vector_search 按 cosine similarity 降序返回。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        store.upsert_item(
            memory_type="preference",
            summary="高相似条目",
            embedding=[1.0, 0.0],
        )
        store.upsert_item(
            memory_type="preference",
            summary="低相似条目",
            embedding=[0.0, 1.0],
        )

        results = store.vector_search(
            query_embedding=[1.0, 0.0],
            top_k=2,
            score_threshold=0.0,
        )

        assert [item["summary"] for item in results] == ["高相似条目", "低相似条目"]
        assert results[0]["score"] >= results[1]["score"]
    finally:
        store.close()


def test_memory_store_superseded_items_are_hidden_from_vector_search(tmp_path) -> None:
    """测试 superseded 条目默认不会出现在向量检索结果中。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        result = store.upsert_item(
            memory_type="procedure",
            summary="旧规则",
            embedding=[1.0, 0.0],
        )
        old_id = result.split(":", 1)[1]
        store.mark_superseded_batch([old_id])
        store.upsert_item(
            memory_type="procedure",
            summary="新规则",
            embedding=[0.9, 0.0],
        )

        results = store.vector_search(query_embedding=[1.0, 0.0], top_k=5)

        assert "旧规则" not in [item["summary"] for item in results]
        assert "新规则" in [item["summary"] for item in results]
    finally:
        store.close()


def test_memory_store_consolidation_event_is_idempotent_by_source_ref(tmp_path) -> None:
    """测试 upsert_consolidation_event 按 source_ref 幂等。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        first = store.upsert_consolidation_event(
            source_ref="cli:default@0-6",
            summary="[2026-05-29 10:00] 用户讨论 Memory2。",
            embedding=[1.0, 0.0],
        )
        second = store.upsert_consolidation_event(
            source_ref="cli:default@0-6",
            summary="[2026-05-29 10:01] 重复写入不应生效。",
            embedding=[0.0, 1.0],
        )
        events = store.list_by_type("event")

        assert first.startswith("new:")
        assert second.startswith("skipped:")
        assert len(events) == 1
        assert "用户讨论 Memory2" in events[0].summary
    finally:
        store.close()


def test_memory_store_records_replacement_relation(tmp_path) -> None:
    """测试 record_replacement 会保存替换关系。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        old_result = store.upsert_item(
            memory_type="preference",
            summary="用户喜欢很短回答。",
            embedding=[1.0, 0.0],
        )
        new_result = store.upsert_item(
            memory_type="preference",
            summary="用户喜欢简洁但保留关键解释的回答。",
            embedding=[0.9, 0.0],
        )
        old_item = store.get_item(old_result.split(":", 1)[1])
        new_item = store.get_item(new_result.split(":", 1)[1])
        assert old_item is not None
        assert new_item is not None

        store.record_replacement(old_item=old_item, new_item=new_item)
        replacements = store.list_replacements()

        assert len(replacements) == 1
        assert replacements[0]["old_item_id"] == old_item.id
        assert replacements[0]["new_item_id"] == new_item.id
    finally:
        store.close()

def test_memory_store_keyword_search_summary_matches_specific_terms(tmp_path) -> None:
    """测试 keyword_search_summary 可以命中专有名词。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        store.upsert_item(
            memory_type="profile",
            summary="用户有一块 Fitbit Charge 6 手环。",
            embedding=[0.1, 0.2],
            source_ref='["cli:default:0"]',
        )
        store.upsert_item(
            memory_type="profile",
            summary="用户喜欢阅读长篇技术文章。",
            embedding=[0.2, 0.1],
        )

        results = store.keyword_search_summary(["Fitbit", "Charge"], limit=5)

        assert len(results) == 1
        assert results[0]["summary"] == "用户有一块 Fitbit Charge 6 手环。"
        assert results[0]["keyword_score"] > 0
    finally:
        store.close()

def test_memory_store_vector_search_returns_score_debug(tmp_path) -> None:
    """测试 vector_search 返回 semantic/hotness/final 调试分数。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        store.upsert_item(
            memory_type="preference",
            summary="用户喜欢先给结论。",
            embedding=[1.0, 0.0],
        )

        results = store.vector_search(
            query_embedding=[1.0, 0.0],
            top_k=1,
            score_threshold=0.0,
        )

        assert len(results) == 1
        debug = results[0]["_score_debug"]
        assert isinstance(debug, dict)
        assert "semantic" in debug
        assert "hotness" in debug
        assert "final" in debug
    finally:
        store.close()


def test_memory_store_hotness_can_boost_frequently_used_recent_item(tmp_path) -> None:
    """测试 hotness_alpha 可以让常用新鲜条目排到更前。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        first = store.upsert_item(
            memory_type="preference",
            summary="常用新鲜偏好",
            embedding=[0.90, 0.436],
        )
        second = store.upsert_item(
            memory_type="preference",
            summary="一次性陈旧偏好",
            embedding=[0.95, 0.312],
        )
        first_id = first.split(":", 1)[1]
        second_id = second.split(":", 1)[1]
        store._db.execute(
            "UPDATE memory_items SET reinforcement=10, updated_at=? WHERE id=?",
            (now_iso(), first_id),
        )
        store._db.execute(
            "UPDATE memory_items SET reinforcement=1, updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (second_id,),
        )
        store._db.commit()

        results = store.vector_search(
            query_embedding=[1.0, 0.0],
            top_k=2,
            score_threshold=0.0,
            hotness_alpha=0.2,
        )

        assert results[0]["summary"] == "常用新鲜偏好"
    finally:
        store.close()

def test_memory_store_list_events_by_time_range(tmp_path) -> None:
    """测试 list_events_by_time_range 按时间范围列出 event。"""

    store = MemoryStore2(tmp_path / "memory2.db")

    try:
        store.upsert_item(
            memory_type="event",
            summary="范围内事件",
            embedding=[1.0, 0.0],
            happened_at="2026-05-20T10:00:00+00:00",
        )
        store.upsert_item(
            memory_type="event",
            summary="范围外事件",
            embedding=[1.0, 0.0],
            happened_at="2026-04-20T10:00:00+00:00",
        )

        results = store.list_events_by_time_range(
            datetime.fromisoformat("2026-05-01T00:00:00+00:00"),
            datetime.fromisoformat("2026-06-01T00:00:00+00:00"),
        )

        assert [item["summary"] for item in results] == ["范围内事件"]
    finally:
        store.close()


def test_memory_store_find_similar_recent_events(tmp_path) -> None:
    """测试 find_similar_recent_events 只返回近期相似 event。"""

    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        result = store.upsert_item(
            memory_type="event",
            summary="用户把仓库脱敏后公开发布",
            embedding=[1.0, 0.0],
        )
        item_id = result.split(":", 1)[1]

        similar = store.find_similar_recent_events([0.99, 0.01], threshold=0.92)

        assert similar == [item_id]
    finally:
        store.close()

def test_memory_store_keyword_match_procedures_uses_trigger_tags(tmp_path) -> None:
    """测试 keyword_match_procedures 会匹配 trigger_tags。"""

    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        store.upsert_item(
            memory_type="procedure",
            summary="用户发送 B 站链接时先抓取页面。",
            embedding=[1.0, 0.0],
            extra_json={
                "trigger_tags": {
                    "tools": ["web_fetch"],
                    "skills": [],
                    "keywords": ["B站"],
                    "scope": "tool_triggered",
                }
            },
        )

        hits = store.keyword_match_procedures(["web_fetch", "B站视频"])

        assert len(hits) == 1
        assert hits[0]["memory_type"] == "procedure"
    finally:
        store.close()


def test_memory_store_merge_item_raw_updates_summary_and_reinforcement(tmp_path) -> None:
    """测试 merge_item_raw 原地更新 summary 并强化。"""

    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        result = store.upsert_item(
            memory_type="procedure",
            summary="旧规则",
            embedding=[1.0, 0.0],
        )
        item_id = result.split(":", 1)[1]

        store.merge_item_raw(
            item_id=item_id,
            new_summary="新规则",
            new_hash=content_hash("新规则", "procedure"),
            new_embedding=[0.0, 1.0],
            new_extra_json={"note": "merged"},
        )
        item = store.get_item(item_id)

        assert item is not None
        assert item.summary == "新规则"
        assert item.reinforcement == 2
        assert item.extra_json["note"] == "merged"
    finally:
        store.close()