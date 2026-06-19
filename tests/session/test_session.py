from __future__ import annotations

from raven_agent.session import Session


def test_session_adds_user_and_assistant_messages() -> None:
    """测试 Session 可以追加 user 和 assistant 消息。

    返回:
        None。
    """

    session = Session(key="cli:default")

    session.add_user_message("hello")
    session.add_assistant_message("hi")

    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert [message.content for message in session.messages] == ["hello", "hi"]


def test_session_history_for_prompt_uses_recent_window() -> None:
    """测试 history_for_prompt 只返回最近 max_messages 条消息。

    返回:
        None。
    """

    session = Session(key="cli:default")
    session.add_user_message("u1")
    session.add_assistant_message("a1")
    session.add_user_message("u2")

    history = session.history_for_prompt(max_messages=2)

    assert [message.content for message in history] == ["a1", "u2"]


def test_session_clear_removes_messages_and_resets_consolidation_cursor() -> None:
    """测试 clear 会清空会话历史并重置 last_consolidated。

    返回:
        None。
    """

    session = Session(key="cli:default", last_consolidated=2)
    session.add_user_message("hello")

    session.clear()

    assert session.messages == []
    assert session.last_consolidated == 0


def test_session_round_trips_dict_payload() -> None:
    """测试 Session 可以序列化并恢复。

    返回:
        None。
    """

    session = Session(key="cli:default", last_consolidated=1)
    session.add_user_message("hello")
    session.add_assistant_message("hi")

    restored = Session.from_dict(session.to_dict())

    assert restored.key == "cli:default"
    assert restored.last_consolidated == 1
    assert [message.role for message in restored.messages] == ["user", "assistant"]
    assert [message.content for message in restored.messages] == ["hello", "hi"]

def test_session_adds_runtime_messages_without_allocating_global_seq() -> None:
    session = Session(key="cli:default")

    session.add_user_message("hello")
    session.add_assistant_message("hi")

    assert [message.id for message in session.messages] == ["", ""]
    assert [message.seq for message in session.messages] == [-1, -1]