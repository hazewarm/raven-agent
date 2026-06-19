from __future__ import annotations

from raven_agent.session import Session, SessionManager
from raven_agent.session_store import SessionStore


def test_session_store_saves_and_loads_session(tmp_path) -> None:
    """测试 SessionStore 可以保存并加载 Session。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = SessionStore(tmp_path / "sessions.db")
    try:
        store.upsert_session(
            "cli:default",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:00+00:00",
            last_consolidated=0,
            metadata={"source": "test"},
        )
        row = store.insert_message(
            "cli:default",
            role="user",
            content="hello",
            ts="2026-05-30T00:00:01+00:00",
            seq=0,
        )

        loaded = store.fetch_session_messages("cli:default")
        meta = store.get_session_meta("cli:default")

        assert row["id"] == "cli:default:0"
        assert meta is not None
        assert meta["metadata"] == {"source": "test"}
        assert [message.content for message in loaded] == ["hello"]
        assert [message.id for message in loaded] == ["cli:default:0"]
        assert store.next_seq("cli:default") == 1
    finally:
        store.close()


def test_session_store_uses_single_sqlite_file(tmp_path) -> None:
    """测试 SessionStore 使用单个 SQLite 文件，不再按 session key 生成文件名。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    db_path = tmp_path / ".raven" / "sessions.db"
    store = SessionStore(db_path)

    try:
        store.upsert_session(
            "telegram:chat/123",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:00+00:00",
            last_consolidated=0,
            metadata={},
        )
        store.insert_message(
            "telegram:chat/123",
            role="user",
            content="hello",
            ts="2026-05-30T00:00:01+00:00",
            seq=0,
        )

        assert store.db_path == db_path
        assert db_path.exists()
        assert store.fetch_session_messages("telegram:chat/123")[0].id == "telegram:chat/123:0"
    finally:
        store.close()


def test_session_store_returns_none_for_missing_session(tmp_path) -> None:
    """测试不存在的 Session 会返回 None。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = SessionStore(tmp_path / "sessions.db")
    try:
        assert store.get_session_meta("missing") is None
        assert store.fetch_session_messages("missing") == []
    finally:
        store.close()


def test_session_manager_clear_resets_runtime_history_without_removing_messages(tmp_path) -> None:
    """测试 clear 只清空运行时历史，不删除持久化消息。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)
    try:
        session = manager.get_or_create("cli:default")
        session.add_user_message("hello")
        manager.save(session)

        cleared = manager.clear("cli:default")

        assert cleared.messages == []
        persisted = store.fetch_session_messages("cli:default")
        assert [message.content for message in persisted] == ["hello"]
        assert [message.id for message in persisted] == ["cli:default:0"]
    finally:
        manager.close()

def test_clear_only_resets_runtime_history_and_next_message_id_keeps_growing(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)

    try:
        session = manager.get_or_create("cli:default")
        session.add_user_message("old")
        manager.save(session)

        cleared = manager.clear("cli:default")
        assert cleared.messages == []

        cleared.add_user_message("new")
        manager.save(cleared)

        persisted = store.fetch_session_messages("cli:default")
        assert [message.id for message in persisted] == [
            "cli:default:0",
            "cli:default:1",
        ]
        assert [message.content for message in persisted] == ["old", "new"]
    finally:
        store.close()