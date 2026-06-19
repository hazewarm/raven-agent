from __future__ import annotations

from raven_agent.memory import DEFAULT_SELF_MD, MarkdownMemoryStore
from raven_agent.session import Session


def test_markdown_memory_store_creates_memory_files(tmp_path) -> None:
    """测试 MarkdownMemoryStore 会创建基础 Markdown memory 文件。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    assert store.memory_file.exists()
    assert store.self_file.exists()
    assert store.pending_file.exists()
    assert store.history_file.exists()
    assert store.recent_context_file.exists()
    assert store.journal_dir.exists()
    assert store.read_long_term() == ""
    assert store.read_self() == DEFAULT_SELF_MD
    assert store.read_pending() == ""
    assert store.read_history() == ""
    assert "# Recent Context" in store.read_recent_context()


def test_markdown_memory_store_preserves_existing_self_file(tmp_path) -> None:
    """测试已有 SELF.md 不会被默认内容覆盖。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    self_file = memory_dir / "SELF.md"
    self_file.write_text("custom self", encoding="utf-8")

    store = MarkdownMemoryStore(memory_dir)

    assert store.read_self() == "custom self"


def test_markdown_memory_store_reads_and_writes_long_term_memory(tmp_path) -> None:
    """测试 MEMORY.md 可以读写。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    store.write_long_term("- 用户喜欢简洁回答。")

    assert store.read_long_term() == "- 用户喜欢简洁回答。"
    assert store.has_long_term_memory() is True


def test_markdown_memory_store_reads_and_writes_self_memory(tmp_path) -> None:
    """测试 SELF.md 可以读写。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    store.write_self("# Custom Self")

    assert store.read_self() == "# Custom Self"


def test_markdown_memory_store_renders_prompt_block(tmp_path) -> None:
    """测试 render_prompt_block 会组合 SELF.md 和 MEMORY.md。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")
    store.write_self("# Self")
    store.write_long_term("- 用户喜欢简洁回答。")

    block = store.render_prompt_block()

    assert "## Raven Self Model" in block
    assert "# Self" in block
    assert "## Long-term Memory" in block
    assert "- 用户喜欢简洁回答。" in block


def test_markdown_memory_store_appends_history_once(tmp_path) -> None:
    """测试 HISTORY.md 会按 source_ref 幂等追加。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    first = store.append_history_once(
        "[2026-05-28 10:00] 用户开始实现 Markdown 记忆系统。",
        source_ref="batch-1",
    )
    second = store.append_history_once(
        "[2026-05-28 10:00] 用户开始实现 Markdown 记忆系统。",
        source_ref="batch-1",
    )

    assert first is True
    assert second is False
    assert store.read_history().count("用户开始实现 Markdown 记忆系统") == 1


def test_markdown_memory_store_appends_and_clears_pending(tmp_path) -> None:
    """测试 PENDING.md 可以幂等追加并清空。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    first = store.append_pending_once(
        "- [preference] 用户喜欢简洁回答。",
        source_ref="batch-1",
    )
    second = store.append_pending_once(
        "- [preference] 用户喜欢简洁回答。",
        source_ref="batch-1",
    )

    assert first is True
    assert second is False
    assert store.read_pending().count("用户喜欢简洁回答") == 1

    store.clear_pending()

    assert store.read_pending() == ""


def test_markdown_memory_store_appends_journal_by_date(tmp_path) -> None:
    """测试 journal 会按日期创建文件并追加事件。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")

    written = store.append_journal(
        "2026-05-28",
        "[2026-05-28 10:00] 用户讨论第 14 章记忆系统。",
        source_ref="batch-1",
    )
    duplicate = store.append_journal(
        "2026-05-28",
        "[2026-05-28 10:00] 用户讨论第 14 章记忆系统。",
        source_ref="batch-1",
    )
    invalid = store.append_journal("../bad", "bad")

    journal_file = store.journal_dir / "2026-05-28.md"

    assert written is True
    assert duplicate is False
    assert invalid is False
    assert journal_file.exists()
    assert journal_file.read_text(encoding="utf-8").count("第 14 章记忆系统") == 1


def test_markdown_memory_store_replaces_recent_turns(tmp_path) -> None:
    """测试 RECENT_CONTEXT.md 的 Recent Turns 块可以刷新。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    session = Session(key="cli:default")
    session.add_user_message("你好")
    session.add_assistant_message("你好，我是 Raven。")
    store = MarkdownMemoryStore(tmp_path / "memory")

    store.replace_recent_turns(session.messages)

    text = store.read_recent_context()
    assert "## Recent Turns" in text
    assert "[user] 你好" in text
    assert "[a-preview] 你好，我是 Raven。" in text


def test_markdown_memory_store_renders_recent_context_without_recent_turns(tmp_path) -> None:
    """测试 recent context 注入 prompt 时会排除 Recent Turns。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")
    store.write_recent_context(
        "# Recent Context\n\n"
        "## Compression\n"
        "until: 2026-05-28T10:00:00\n"
        "- 最近持续关注：Markdown 记忆系统\n\n"
        "## Ongoing Threads\n"
        "- raven-agent 教程编写\n\n"
        "## Recent Turns\n"
        "<!-- a-preview = assistant reply preview only -->\n"
        "[user] 不应进入 prompt 的滑动窗口重复内容\n"
    )

    block = store.render_recent_context_prompt_block()

    assert "最近持续关注" in block
    assert "raven-agent 教程编写" in block
    assert "Recent Turns" not in block
    assert "不应进入 prompt" not in block

def test_markdown_memory_store_snapshots_and_commits_pending(tmp_path) -> None:
    """测试 pending snapshot 可以提交。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")
    store.append_pending_once("- [preference] 用户喜欢简洁回答。", source_ref="batch-1")

    pending = store.snapshot_pending()

    assert "用户喜欢简洁回答" in pending
    assert store.read_pending() == ""
    assert store.pending_snapshot_file.exists()

    store.commit_pending_snapshot()

    assert not store.pending_snapshot_file.exists()
    assert store.read_pending() == ""

def test_markdown_memory_store_rolls_back_pending_snapshot(tmp_path) -> None:
    """测试 pending snapshot 可以回滚并保留运行期新增 pending。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    store = MarkdownMemoryStore(tmp_path / "memory")
    store.append_pending_once("- [preference] 旧 pending。", source_ref="batch-1")

    _ = store.snapshot_pending()
    store.append_pending_once("- [identity] 新 pending。", source_ref="batch-2")
    store.rollback_pending_snapshot()

    pending = store.read_pending()
    assert "旧 pending" in pending
    assert "新 pending" in pending
    assert not store.pending_snapshot_file.exists()