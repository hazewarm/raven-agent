from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from raven_agent.messages import ChatMessage

# 幂等追加时写入隐藏 marker。
_MEMORY_MARKER_PREFIX = "<!-- raven-memory:"
_MEMORY_MARKER_SUFFIX = " -->"

# 确保 journal 文件名只能是 YYYY-MM-DD.md
_JOURNAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 首次创建 RECENT_CONTEXT.md 的默认结构
_DEFAULT_RECENT_CONTEXT_MD = """# Recent Context

## Compression
until: none
- none

## Ongoing Threads
- none

## Recent Turns
<!-- a-preview = assistant reply preview only -->
- none
"""

DEFAULT_SELF_MD = """# Raven 的自我认知

## 人格与形象
- 我是 Raven，一个简洁、可靠、面向任务的 AI 助手。

## 我对当前用户的理解
- 我只依据长期记忆和当前对话理解用户，不在缺少证据时编造画像。

## 协作方式
- 我优先直接解决当前问题，并在必要时说明关键取舍。
"""


def _strip_markers(text: str) -> str:
    """移除 Markdown memory 隐藏 marker 行。

    参数:
        text: 原始 Markdown 文本。

    返回:
        去掉 marker 行后的文本。
    """

    lines = []
    for line in text.splitlines():
        if line.startswith(_MEMORY_MARKER_PREFIX) and line.endswith(_MEMORY_MARKER_SUFFIX):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


class MarkdownMemoryStore:
    """基于 Markdown 文件的长期记忆存储。

    参数:
        memory_dir: 保存 Markdown memory 文件的目录。
    """

    def __init__(self, memory_dir: str | Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.journal_dir = self.memory_dir / "journal"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.self_file = self.memory_dir / "SELF.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.recent_context_file = self.memory_dir / "RECENT_CONTEXT.md"
        self.ensure()

    def ensure(self) -> None:
        """确保 memory 目录和基础 Markdown 文件存在。

        返回:
            None。
        """

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("", encoding="utf-8")
        if not self.self_file.exists():
            self.self_file.write_text(DEFAULT_SELF_MD, encoding="utf-8")
        if not self.pending_file.exists():
            self.pending_file.write_text("", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("", encoding="utf-8")
        # 不会覆盖已有文件，只在首次创建时写入默认结构
        if not self.recent_context_file.exists():
            self.recent_context_file.write_text(
                _DEFAULT_RECENT_CONTEXT_MD,
                encoding="utf-8",
            )

    def _marker(self, source_ref: str, kind: str) -> str:
        """为一次幂等写入生成隐藏 marker。

        参数:
            source_ref: 写入来源标识，例如一批消息 id 或测试中的固定 key。
            kind: 写入类型，例如 history_entry、pending_items、journal。

        返回:
            可写入 Markdown 注释的 marker 字符串。
        """

        safe_source = source_ref.replace("\n", " ").strip()
        safe_kind = kind.replace("\n", " ").strip()
        return f"{_MEMORY_MARKER_PREFIX}{safe_source}:{safe_kind}{_MEMORY_MARKER_SUFFIX}"

    # 最小幂等机制：在追加文本前检查 marker 是否存在，存在则跳过写入；写入时在文本前添加 marker。
    def _append_once(
        self,
        path: Path,
        text: str,
        *,
        source_ref: str,
        kind: str,
        trailing_blank_line: bool,
    ) -> bool:
        """向文件幂等追加文本。

        参数:
            path: 目标文件路径。
            text: 要追加的正文。
            source_ref: 幂等来源标识。
            kind: 写入类型。
            trailing_blank_line: 是否在正文后额外添加空行。

        返回:
            实际写入返回 True；发现重复或正文为空返回 False。
        """

        normalized = text.strip()
        if not normalized:
            return False
        marker = self._marker(source_ref=source_ref, kind=kind)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker in existing:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(marker + "\n")
            file.write(normalized + "\n")
            if trailing_blank_line:
                file.write("\n")
        return True

    def read_long_term(self) -> str:
        """读取长期记忆 MEMORY.md。

        返回:
            MEMORY.md 的文本内容；文件不存在时返回空字符串。
        """

        if not self.memory_file.exists():
            return ""
        return self.memory_file.read_text(encoding="utf-8")

    def write_long_term(self, content: str) -> None:
        """覆盖写入长期记忆 MEMORY.md。

        参数:
            content: 新的长期记忆 Markdown 内容。

        返回:
            None。
        """

        self.ensure()
        self.memory_file.write_text(content, encoding="utf-8")

    def read_history(self, max_chars: int = 0) -> str:
        """读取 HISTORY.md。

        参数:
            max_chars: 最多返回末尾多少个字符；0 表示返回全部。

        返回:
            去掉隐藏 marker 后的 HISTORY.md 内容。
        """

        if not self.history_file.exists():
            return ""
        text = self.history_file.read_text(encoding="utf-8")
        cleaned = _strip_markers(text)
        if max_chars > 0 and len(cleaned) > max_chars:
            return cleaned[-max_chars:]
        return cleaned

    def append_history_once(
        self,
        entry: str,
        *,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        """向 HISTORY.md 幂等追加事件。

        参数:
            entry: 要追加的历史事件文本。
            source_ref: 幂等来源标识。
            kind: 写入类型，默认 history_entry。

        返回:
            实际写入返回 True；重复或空内容返回 False。
        """

        self.ensure()
        return self._append_once(
            self.history_file,
            entry,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=True,
        )

    def read_pending(self) -> str:
        """读取 PENDING.md。

        返回:
            去掉隐藏 marker 后的 pending 文本。
        """

        if not self.pending_file.exists():
            return ""
        return _strip_markers(self.pending_file.read_text(encoding="utf-8"))

    def append_pending_once(
        self,
        facts: str,
        *,
        source_ref: str,
        kind: str = "pending_items",
    ) -> bool:
        """向 PENDING.md 幂等追加待归档事实。

        参数:
            facts: 待归档事实，通常是若干 Markdown bullet。
            source_ref: 幂等来源标识。
            kind: 写入类型，默认 pending_items。

        返回:
            实际写入返回 True；重复或空内容返回 False。
        """

        self.ensure()
        return self._append_once(
            self.pending_file,
            facts,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=False,
        )

    def clear_pending(self) -> None:
        """清空 PENDING.md。

        返回:
            None。
        """

        self.ensure()
        self.pending_file.write_text("", encoding="utf-8")
    
    @property
    def pending_snapshot_file(self) -> Path:
        """返回 PENDING.md 的 snapshot 文件路径。

        返回:
            PENDING.snapshot.md 路径。
        """

        return self.pending_file.with_name("PENDING.snapshot.md")
    
    def snapshot_pending(self) -> str:
        """把 PENDING.md 原子移动为 snapshot，并返回待归档内容。

        返回:
            去掉隐藏 marker 后的 pending 内容；没有内容时返回空字符串。
        """

        # 先归档再snapshot，防止丢失未归档内容
        self.rollback_pending_snapshot()
        self.ensure()
        if not self.pending_file.exists() or self.pending_file.stat().st_size == 0:
            return ""
        self.pending_file.replace(self.pending_snapshot_file)
        # 移动后要重新创建空的 PENDING.md，这样 consolidation 仍然可以追加新的 pending facts。
        self.pending_file.write_text("", encoding="utf-8")
        return _strip_markers(self.pending_snapshot_file.read_text(encoding="utf-8"))
    
    def commit_pending_snapshot(self) -> None:
        """提交 pending snapshot，表示其中内容已成功归档。

        返回:
            None。
        """

        if self.pending_snapshot_file.exists():
            # 提交即删除 snapshot 文件，表示 pending 中的内容已成功归档，不再需要保留。
            self.pending_snapshot_file.unlink()
        self.ensure()

    def rollback_pending_snapshot(self) -> None:
        """回滚 pending snapshot，把未归档内容合回 PENDING.md。

        返回:
            None。
        """

        if not self.pending_snapshot_file.exists():
            return
        snapshot_text = self.pending_snapshot_file.read_text(encoding="utf-8").strip()
        current_text = self.pending_file.read_text(encoding="utf-8").strip() if self.pending_file.exists() else ""
        
        # snapshot 内容在前，运行期新增 pending 在后
        merged = "\n".join(part for part in [snapshot_text, current_text] if part)
        self.pending_file.write_text(merged + ("\n" if merged else ""), encoding="utf-8")
        self.pending_snapshot_file.unlink()


    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        """向某一天的 journal 文件追加事件。

        参数:
            date_str: 日期字符串，格式必须是 YYYY-MM-DD。
            entry: 要追加的事件文本。
            source_ref: 可选幂等来源标识。
            kind: 写入类型，默认 journal。

        返回:
            实际写入返回 True；日期非法、内容为空或重复时返回 False。
        """

        normalized_date = date_str.strip()
        normalized_entry = entry.strip()
        if not _JOURNAL_DATE_RE.fullmatch(normalized_date):
            return False
        if not normalized_entry:
            return False

        self.ensure()
        journal_file = self.journal_dir / f"{normalized_date}.md"
        # 【冷启动处理】如果这个文件今天还没被创建过（今天是第一次写入）
        # 创建文件，并打上标准的 Markdown 一级标题，例如: "# 2026-05-29"
        if not journal_file.exists():
            journal_file.write_text(f"# {normalized_date}\n\n", encoding="utf-8")

        if source_ref:
            return self._append_once(
                journal_file,
                normalized_entry,
                source_ref=source_ref,
                kind=kind,
                trailing_blank_line=True,
            )

        with journal_file.open("a", encoding="utf-8") as file:
            file.write(normalized_entry + "\n\n")
        return True

    def read_recent_context(self) -> str:
        """读取 RECENT_CONTEXT.md。

        返回:
            RECENT_CONTEXT.md 文本；文件不存在时返回空字符串。
        """

        if not self.recent_context_file.exists():
            return ""
        return self.recent_context_file.read_text(encoding="utf-8")

    def write_recent_context(self, content: str) -> None:
        """覆盖写入 RECENT_CONTEXT.md。

        参数:
            content: 新的近期上下文 Markdown 内容。

        返回:
            None。
        """

        self.ensure()
        self.recent_context_file.write_text(content, encoding="utf-8")

    def format_recent_turns(self, messages: Sequence[ChatMessage]) -> str:
        """把最近消息格式化为 RECENT_CONTEXT.md 的 Recent Turns 文本。

        参数:
            messages: 最近一段 ChatMessage。

        返回:
            Recent Turns 文本。
        """

        lines: list[str] = []
        for message in messages:
            content = message.content.strip()
            if not content:
                continue
            if message.role == "user":
                lines.append(f"[user] {content}")
            elif message.role == "assistant":
                lines.append(f"[a-preview] {content[:60]}")
        return "\n".join(lines).strip()

    def _replace_recent_turns_block(self, existing_text: str, recent_turns: str) -> str:
        """替换 RECENT_CONTEXT.md 中的 Recent Turns 块。

        参数:
            existing_text: 现有 RECENT_CONTEXT.md 内容。
            recent_turns: 新的 Recent Turns 文本。

        返回:
            替换后的 RECENT_CONTEXT.md 内容。
        """

        block = "\n".join(
            [
                "## Recent Turns",
                "<!-- a-preview = assistant reply preview only -->",
                recent_turns.strip() or "- none",
            ]
        ).rstrip() + "\n"
        marker = "\n## Recent Turns\n"
        text = existing_text.strip()
        if marker in text:
            prefix, _ = text.split(marker, 1)
            return prefix.rstrip() + "\n\n" + block
        if text:
            return text.rstrip() + "\n\n" + block
        return _DEFAULT_RECENT_CONTEXT_MD

    def replace_recent_turns(self, messages: Sequence[ChatMessage]) -> None:
        """刷新 RECENT_CONTEXT.md 的 Recent Turns 块。

        参数:
            messages: 最近一段 ChatMessage。

        返回:
            None。
        """

        self.ensure()
        recent_turns = self.format_recent_turns(messages)
        existing = self.read_recent_context()
        updated = self._replace_recent_turns_block(existing, recent_turns)
        self.write_recent_context(updated)

    def read_self(self) -> str:
        """读取自我认知 SELF.md。

        返回:
            SELF.md 的文本内容；文件不存在时返回空字符串。
        """

        if not self.self_file.exists():
            return ""
        return self.self_file.read_text(encoding="utf-8")

    def write_self(self, content: str) -> None:
        """覆盖写入自我认知 SELF.md。

        参数:
            content: 新的自我认知 Markdown 内容。

        返回:
            None。
        """

        self.ensure()
        self.self_file.write_text(content, encoding="utf-8")

    def has_long_term_memory(self) -> bool:
        """判断 MEMORY.md 是否包含非空长期记忆。

        返回:
            MEMORY.md 去掉空白后是否还有内容。
        """

        return bool(self.read_long_term().strip())

    def render_recent_context_prompt_block(self) -> str:
        """渲染可注入 system prompt 的近期上下文块。

        返回:
            不包含 Recent Turns 的 RECENT_CONTEXT.md 内容；没有有效内容时返回空字符串。
        """

        content = self.read_recent_context().strip()
        if not content:
            return ""
        marker = "\n## Recent Turns"
        if marker in content:
            content = content.split(marker, 1)[0].strip()
        empty_context = "# Recent Context\n\n## Compression\nuntil: none\n- none\n\n## Ongoing Threads\n- none"
        if not content or content == empty_context:
            return ""
        return content

    def render_prompt_block(self) -> str:
        """把 Markdown memory 渲染为 system prompt 片段。

        返回:
            可追加进 system message 的 Markdown 文本；没有内容时返回空字符串。
        """

        sections: list[str] = []
        self_text = self.read_self().strip()
        if self_text:
            sections.append(f"## Raven Self Model\n\n{self_text}")

        memory_text = self.read_long_term().strip()
        if memory_text:
            sections.append(f"## Long-term Memory\n\n{memory_text}")

        recent_context = self.render_recent_context_prompt_block()
        if recent_context:
            sections.append(recent_context)

        return "\n\n".join(sections)