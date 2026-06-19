from __future__ import annotations

from raven_agent.prompt import PromptBuilder
from raven_agent.session import Session

from raven_agent.memory import MarkdownMemoryStore


def test_prompt_builder_orders_system_history_and_current_user() -> None:
    """测试 PromptBuilder 按 system → history → current user 排列消息。

    返回:
        None。
    """

    session = Session(key="cli:default")
    session.add_user_message("hello")
    session.add_assistant_message("hi")
    builder = PromptBuilder(system_prompt="You are Raven.")

    messages = builder.build(session=session, current_user_input="What did I say?")

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    contents = [message.content for message in messages]
    assert "You are Raven." in contents[0]
    assert contents[1:] == [
        "hello",
        "hi",
        "What did I say?",
    ]


def test_prompt_builder_respects_history_window() -> None:
    """测试 PromptBuilder 会限制历史消息数量。

    返回:
        None。
    """

    session = Session(key="cli:default")
    session.add_user_message("u1")
    session.add_assistant_message("a1")
    session.add_user_message("u2")
    builder = PromptBuilder(system_prompt="sys", history_window=1)

    messages = builder.build(session=session, current_user_input="now")

    contents = [message.content for message in messages]
    assert "sys" in contents[0]
    assert contents[1:] == ["u2", "now"]


def test_prompt_builder_injects_markdown_memory_into_system_message(tmp_path) -> None:
    """测试 PromptBuilder 会把 Markdown memory 注入 system message。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    memory_store = MarkdownMemoryStore(tmp_path / "memory")
    memory_store.write_self("# Self")
    memory_store.write_long_term("- 用户喜欢简洁回答。")
    session = Session(key="cli:default")
    builder = PromptBuilder(
        system_prompt="You are Raven.",
        memory_store=memory_store,
    )

    messages = builder.build(session=session, current_user_input="hello")

    assert messages[0].role == "system"
    assert "You are Raven." in messages[0].content
    assert "## Raven Self Model" in messages[0].content
    assert "# Self" in messages[0].content
    assert "## Long-term Memory" in messages[0].content
    assert "- 用户喜欢简洁回答。" in messages[0].content
    assert messages[-1].content == "hello"


def test_prompt_builder_injects_recent_context_without_recent_turns(tmp_path) -> None:
    """测试 PromptBuilder 会注入 recent context 但不注入 Recent Turns。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    memory_store = MarkdownMemoryStore(tmp_path / "memory")
    memory_store.write_self("")
    memory_store.write_long_term("")
    memory_store.write_recent_context(
        "# Recent Context\n\n"
        "## Compression\n"
        "until: 2026-05-28T10:00:00\n"
        "- 最近持续关注：Markdown 记忆系统\n\n"
        "## Ongoing Threads\n"
        "- raven-agent 教程编写\n\n"
        "## Recent Turns\n"
        "<!-- a-preview = assistant reply preview only -->\n"
        "[user] 不应进入 system prompt\n"
    )
    session = Session(key="cli:default")
    builder = PromptBuilder(
        system_prompt="You are Raven.",
        memory_store=memory_store,
    )

    messages = builder.build(session=session, current_user_input="hello")

    assert "最近持续关注" in messages[0].content
    assert "raven-agent 教程编写" in messages[0].content
    assert "不应进入 system prompt" not in messages[0].content