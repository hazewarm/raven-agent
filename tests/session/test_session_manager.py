from __future__ import annotations

from raven_agent.session import Session, SessionManager
from raven_agent.session_store import SessionStore


def test_session_manager_get_or_create_returns_cached_session(tmp_path) -> None:
    """测试 get_or_create 会复用缓存中的 Session。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    manager = SessionManager(SessionStore(tmp_path / "sessions.db"))
    try:
        first = manager.get_or_create("cli:default")
        second = manager.get_or_create("cli:default")

        assert first is second
    finally:
        manager.close()

def test_session_manager_loads_existing_session_from_store(tmp_path) -> None:
    """测试 SessionManager 会从 store 加载已有 Session。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)
    try:
        store.upsert_session(
            "cli:default",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:00+00:00",
            last_consolidated=0,
            metadata={},
        )
        store.insert_message(
            "cli:default",
            role="user",
            content="hello",
            ts="2026-05-30T00:00:01+00:00",
            seq=0,
        )

        loaded = manager.get_or_create("cli:default")

        assert [message.content for message in loaded.messages] == ["hello"]
        assert [message.id for message in loaded.messages] == ["cli:default:0"]
    finally:
        manager.close()


def test_session_manager_save_persists_session(tmp_path) -> None:
    """测试 save 会把 Session 写入磁盘。

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

        reloaded_manager = SessionManager(store)
        loaded = reloaded_manager.get_or_create("cli:default")

        assert [message.content for message in loaded.messages] == ["hello"]
        assert [message.id for message in loaded.messages] == ["cli:default:0"]
    finally:
        manager.close()


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

        assert cleared.key == "cli:default"
        assert cleared.messages == []
        persisted = store.fetch_session_messages("cli:default")
        assert [message.content for message in persisted] == ["hello"]
        assert [message.id for message in persisted] == ["cli:default:0"]
    finally:
        store.close()

def test_session_manager_save_assigns_message_ids(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(store)

    try:
        session = manager.get_or_create("cli:default")
        session.add_user_message("hello")
        session.add_assistant_message("hi")

        manager.save(session)

        assert [message.id for message in session.messages] == [
            "cli:default:0",
            "cli:default:1",
        ]
        assert store.next_seq("cli:default") == 2
    finally:
        store.close()