from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass

from raven_agent.messages import ChatMessage, MessageRole, ToolCall


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    key               TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_consolidated INTEGER NOT NULL DEFAULT 0,
    metadata          TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT,
    tool_chain  TEXT,
    extra       TEXT,
    ts          TEXT NOT NULL,
    UNIQUE(session_key, seq)
);

CREATE TABLE IF NOT EXISTS presence (
    session_key     TEXT PRIMARY KEY,
    last_user_at    TEXT,
    last_proactive_at TEXT
);
"""

def _now_iso() -> str:
    """生成当前本地时区 ISO 时间字符串。

    参数:
        无。

    返回:
        ISO 格式时间字符串。
    """

    return datetime.now().astimezone().isoformat()


def _json_dumps(value: dict[str, Any] | None) -> str | None:
    """把字典序列化为 JSON 字符串。

    参数:
        value: 要序列化的字典；None 或空字典返回 None。

    返回:
        JSON 字符串或 None。
    """

    if not value:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_object(raw: object) -> dict[str, Any]:
    """把 SQLite 中的 JSON 字段解析为字典。

    参数:
        raw: SQLite row 中的 JSON 字符串。

    返回:
        解析后的字典；解析失败或不是对象时返回空字典。
    """

    if not raw:
        return {}
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}


def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    """把 ToolCall 转换为 SQLite JSON 字段可保存的字典。

    参数:
        tool_call: 要序列化的 ToolCall。

    返回:
        包含 id、name、arguments 的字典。
    """

    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _tool_call_from_dict(payload: dict[str, Any]) -> ToolCall:
    """从 SQLite JSON 字段恢复 ToolCall。

    参数:
        payload: 工具调用字典。

    返回:
        ToolCall 实例。
    """

    arguments = payload.get("arguments", {})
    return ToolCall(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        arguments=arguments if isinstance(arguments, dict) else {},
    )


def _tool_calls_dumps(tool_calls: object | None) -> str | None:
    """把工具调用列表序列化为 JSON 字符串。

    参数:
        tool_calls: ToolCall 字典列表、ToolCall 列表或 None。

    返回:
        JSON 字符串；空值返回 None。
    """

    if not tool_calls:
        return None
    return json.dumps(tool_calls, ensure_ascii=False)


def _tool_calls_loads(raw: object) -> list[ToolCall]:
    """把 SQLite 中的工具调用 JSON 字段恢复为 ToolCall 列表。

    参数:
        raw: messages.tool_chain 字段。

    返回:
        ToolCall 列表。
    """

    if not raw:
        return []
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [
        _tool_call_from_dict(cast(dict[str, Any], item))
        for item in loaded
        if isinstance(item, dict)
    ]

def _row_to_message(row: sqlite3.Row) -> ChatMessage:
    """把 messages 查询结果转换为 ChatMessage。

    参数:
        row: SELECT messages 得到的一行。

    返回:
        ChatMessage 实例。
    """

    role = str(row["role"])
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    extra = _json_object(row["extra"])
    return ChatMessage(
        role=cast(MessageRole, role),
        content=str(row["content"] or ""),
        tool_calls=_tool_calls_loads(row["tool_chain"]),
        tool_call_id=str(extra.get("tool_call_id") or ""),
        reasoning_content=str(extra.get("reasoning_content") or ""),
        id=str(row["id"]),
        seq=int(row["seq"]),
        timestamp=str(row["ts"]),
    )


def _safe_session_filename(key: str) -> str:
    """把 session key 转换为安全文件名。

    参数:
        key: 原始 session key，例如 cli:default。

    返回:
        可作为文件名使用的字符串。
    """

    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key).strip("_")
    return safe or "default"

@dataclass(frozen=True)
class SessionSummary:
    """单个 session 的摘要信息，供 CLI / Dashboard 列表展示。

    输入:
        session_key: 完整 session key，例如 cli:a8f3c1d2e4b5。
        chat_id: 去掉 channel 前缀后的 chat_id。
        updated_at: 最近更新时间 ISO 字符串。
        message_count: 该 session 的消息条数。
        preview: 最后一条用户消息的前若干字符，用于人类辨认。

    输出:
        SessionSummary 实例。
    """

    session_key: str
    chat_id: str
    updated_at: str
    message_count: int
    preview: str



class SessionStore:
    """SQLite-backed store for session metadata and messages。

    参数:
        db_path: SQLite 数据库文件路径，例如 .raven/sessions.db。

    返回:
        SessionStore 实例。
    """

    def __init__(self, db_path: str | Path) -> None:
        """初始化 SQLite 连接并确保 schema 存在。

        参数:
            db_path: SQLite 数据库文件路径。

        返回:
            None。
        """

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        self._has_fts = False
        self._init_schema()

    def _init_schema(self) -> None:
        """初始化 sessions/messages schema。

        参数:
            无。

        返回:
            None。
        """

        with self._lock:
            self._conn.executescript(SCHEMA)
            self._ensure_session_columns()
            self._ensure_next_seq_values()
            self._ensure_fts()
            self._conn.commit()

    def _ensure_session_columns(self) -> None:
        """补齐旧库缺失的 sessions 扩展列。

        参数:
            无。

        返回:
            None。
        """

        rows = self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "last_user_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_user_at TEXT")
        if "last_proactive_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_proactive_at TEXT")
        if "next_seq" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN next_seq INTEGER NOT NULL DEFAULT 0"
            )

    def _ensure_next_seq_values(self) -> None:
        """确保 sessions.next_seq 不小于 messages 中已有最大 seq + 1。

        参数:
            无。

        返回:
            None。
        """

        rows = self._conn.execute("SELECT key, next_seq FROM sessions").fetchall()
        for row in rows:
            session_key = str(row["key"])
            current = int(row["next_seq"] or 0)
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            required = int((seq_row["next_seq"] if seq_row else 0) or 0)
            if current < required:
                self._conn.execute(
                    "UPDATE sessions SET next_seq = ? WHERE key = ?",
                    (required, session_key),
                )

    def _ensure_fts(self) -> None:
        """尽力创建 messages FTS5 索引。

        参数:
            无。

        返回:
            None。SQLite 环境不支持 FTS5 时仅把 _has_fts 置为 False。
        """

        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='rowid',
                    tokenize='trigram'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """
            )
            self._has_fts = True
        except sqlite3.OperationalError:
            self._has_fts = False

    def close(self) -> None:
        """关闭 SQLite 连接。

        参数:
            无。

        返回:
            None。
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()
    
    def upsert_session(
        self,
        key: str,
        *,
        created_at: str,
        updated_at: str,
        last_consolidated: int,
        metadata: dict[str, Any],
    ) -> None:
        """创建或更新 sessions 元数据行。

        参数:
            key: session key。
            created_at: session 创建时间 ISO 字符串。
            updated_at: session 更新时间 ISO 字符串。
            last_consolidated: 当前 session.messages 的 consolidation 列表游标。
            metadata: session 扩展元数据。

        返回:
            None。
        """

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    last_consolidated = excluded.last_consolidated,
                    metadata = excluded.metadata
                """,
                (key, created_at, updated_at, int(last_consolidated), _json_dumps(metadata)),
            )
            self._conn.commit()


    def get_session_meta(self, key: str) -> dict[str, Any] | None:
        """读取 session 元数据。

        参数:
            key: session key。

        返回:
            metadata 字典；不存在时返回 None。
        """

        with self._lock:
            row = self._conn.execute(
                """
                SELECT key, created_at, updated_at, last_consolidated, metadata,
                       last_user_at, last_proactive_at, next_seq
                FROM sessions
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": str(row["key"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_consolidated": int(row["last_consolidated"] or 0),
            "metadata": _json_object(row["metadata"]),
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
            "next_seq": int(row["next_seq"] or 0),
        }
    
    def next_seq(self, session_key: str) -> int:
        """读取某个 session 下一条消息应该使用的 seq。

        参数:
            session_key: session key。

        返回:
            下一条消息 seq。
        """

        with self._lock:
            meta = self._conn.execute(
                "SELECT next_seq FROM sessions WHERE key = ?",
                (session_key,),
            ).fetchone()
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        from_messages = int((row["next_seq"] if row else 0) or 0)
        if meta is None:
            return from_messages
        return max(int(meta["next_seq"] or 0), from_messages)


    def insert_message(
        self,
        session_key: str,
        *,
        role: str,
        content: str,
        ts: str,
        seq: int,
        tool_chain: object | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """向 messages 表追加一条消息。

        参数:
            session_key: 消息所属 session key。
            role: 消息角色。
            content: 消息文本内容。
            ts: 消息时间 ISO 字符串。
            seq: session 内单调递增序号。
            tool_chain: assistant 消息携带的工具调用链；没有时传 None。
            extra: 额外字段，例如 tool_call_id / reasoning_content。

        返回:
            包含 id、session_key、seq、role、content、timestamp 的字典。
        """

        message_id = f"{session_key}:{seq}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages (id, session_key, seq, role, content, tool_chain, extra, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_key,
                    int(seq),
                    role,
                    content,
                    _tool_calls_dumps(tool_chain),
                    _json_dumps(extra or {}),
                    ts,
                ),
            )
            self._conn.execute(
                """
                UPDATE sessions
                SET next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                WHERE key = ?
                """,
                (int(seq) + 1, int(seq) + 1, session_key),
            )
            self._conn.commit()
        return {
            "id": message_id,
            "session_key": session_key,
            "seq": int(seq),
            "role": role,
            "content": content,
            "timestamp": ts,
        }
    
    def fetch_session_messages(self, session_key: str) -> list[ChatMessage]:
        """读取某个 session 的完整消息历史。

        参数:
            session_key: session key。

        返回:
            按 seq 升序排列的 ChatMessage 列表。
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                ORDER BY seq ASC
                """,
                (session_key,),
            ).fetchall()
        return [_row_to_message(row) for row in rows]
    
    
    def fetch_by_ids(self, ids: list[str]) -> list[ChatMessage]:
        """按 message ids 读取消息。

        参数:
            ids: message id 列表。

        返回:
            按输入 ids 顺序返回存在的 ChatMessage 列表。
        """

        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE id IN ({placeholders})
                """,
                clean_ids,
            ).fetchall()
        by_id = {str(row["id"]): _row_to_message(row) for row in rows}
        return [by_id[item_id] for item_id in clean_ids if item_id in by_id]
    
    def fetch_by_ids_with_context(self, ids: list[str], context: int) -> list[ChatMessage]:
        """按 message ids 读取消息，并扩展同 session 的前后文。

        参数:
            ids: message id 列表。
            context: 每条命中消息前后各扩展多少条。

        返回:
            按 session_key、seq 排列的 ChatMessage 列表。
        """

        if context <= 0:
            return self.fetch_by_ids(ids)
        session_seqs: dict[str, set[int]] = {}
        for message_id in ids:
            parts = message_id.rsplit(":", 1)
            if len(parts) != 2:
                continue
            session_key, seq_text = parts
            try:
                seq = int(seq_text)
            except ValueError:
                continue
            session_seqs.setdefault(session_key, set()).add(seq)

        results: list[ChatMessage] = []
        with self._lock:
            for session_key, seqs in session_seqs.items():
                expanded: set[int] = set()
                for seq in seqs:
                    for item in range(max(0, seq - context), seq + context + 1):
                        expanded.add(item)
                placeholders = ",".join("?" for _ in expanded)
                rows = self._conn.execute(
                    f"""
                    SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                    FROM messages
                    WHERE session_key = ? AND seq IN ({placeholders})
                    ORDER BY seq ASC
                    """,
                    [session_key, *expanded],
                ).fetchall()
                results.extend(_row_to_message(row) for row in rows)
        results.sort(key=lambda message: (message.id.rsplit(":", 1)[0], message.seq))
        return results
    
    # 暂时保留的后续接口，未来可能会删除或改动：
    def list_session_keys(self, prefix: str = "") -> list[str]:
        """列出 session key。

        输入:
            prefix: 可选前缀过滤，例如 "telegram:"。

        输出:
            session key 字符串列表。
        """
        if prefix:
            rows = self._conn.execute(
                "SELECT key FROM sessions WHERE key LIKE ? ORDER BY key",
                (f"{prefix}%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key FROM sessions ORDER BY key"
            ).fetchall()
        return [str(row["key"]) for row in rows]
    
    
    def list_session_summaries(
        self,
        *,
        channel: str,
        limit: int = 20,
    ) -> list[SessionSummary]:
        """按 Channel 列出 session 摘要，按最近更新时间倒序。

        输入:
            channel: Channel 名称，例如 "cli"。
            limit: 最多返回多少个 session。

        输出:
            SessionSummary 列表，最近更新的排在最前面。
        """
        prefix = f"{channel}:"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, updated_at
                FROM sessions
                WHERE key LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (f"{prefix}%", int(limit)),
            ).fetchall()

            summaries: list[SessionSummary] = []
            for row in rows:
                key = str(row["key"])
                count_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE session_key = ?",
                    (key,),
                ).fetchone()
                message_count = int((count_row["n"] if count_row else 0) or 0)
                preview_row = self._conn.execute(
                    """
                    SELECT content
                    FROM messages
                    WHERE session_key = ? AND role = 'user'
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    (key,),
                ).fetchone()
                preview_text = str(preview_row["content"] if preview_row else "" or "")
                preview = preview_text.strip().replace("\n", " ")[:80]
                summaries.append(
                    SessionSummary(
                        session_key=key,
                        chat_id=key.removeprefix(prefix),
                        updated_at=str(row["updated_at"] or ""),
                        message_count=message_count,
                        preview=preview,
                    )
                )
        return summaries

    def count_messages(self, session_key: str) -> int:
        """统计某个 session 的消息数量。

        输入:
            session_key: session key。

        输出:
            消息条数。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return int((row["n"] if row else 0) or 0)

    def session_exists(self, session_key: str) -> bool:
        """检查 session 是否存在。

        输入:
            session_key: session key。

        输出:
            True 表示该 session 在 sessions 表中存在。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE key = ?",
                (session_key,),
            ).fetchone()
        return row is not None
    
    
    # 4个心跳方法，用于 ProactiveLoop 计算 D_energy / D_content / D_recent 和全局心跳检测：
    def update_presence(
        self,
        session_key: str,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> None:
        """更新某个 session 的 presence 字段。

        只更新传入的非 None 字段——调用者可以只更新 last_user_at
        而保留 last_proactive_at 不变（反之亦然）。

        输入:
            session_key: 会话 key。
            last_user_at: 可选，用户最后活跃时间（ISO 格式）。
            last_proactive_at: 可选，Agent 最后主动推送时间（ISO 格式）。

        输出:
            None。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT last_user_at, last_proactive_at FROM presence WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            existing_user = row["last_user_at"] if row else None
            existing_proactive = row["last_proactive_at"] if row else None
            new_user = (
                last_user_at if last_user_at is not None else existing_user
            )
            new_proactive = (
                last_proactive_at
                if last_proactive_at is not None
                else existing_proactive
            )
            self._conn.execute(
                """
                INSERT INTO presence(session_key, last_user_at, last_proactive_at)
                VALUES(?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    last_user_at = excluded.last_user_at,
                    last_proactive_at = excluded.last_proactive_at
                """,
                (session_key, new_user, new_proactive),
            )
            self._conn.commit()


    def get_presence(
        self, session_key: str
    ) -> dict[str, str | None]:
        """读取某个 session 的 presence 数据。

        输入:
            session_key: 会话 key。

        输出:
            含 last_user_at 和 last_proactive_at 的字典；
            session 不存在时两个值均为 None。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT last_user_at, last_proactive_at FROM presence WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        if row is None:
            return {"last_user_at": None, "last_proactive_at": None}
        return {
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }


    def most_recent_user_at(self) -> str | None:
        """返回所有 session 中最近一次用户活跃时间。

        用于全局心跳检测——即使用户在某个特定 session 中静默，
        只要在其他 session 中有活动，Proactive 系统就能感知。

        输出:
            最近的 last_user_at ISO 字符串；没有任何记录时返回 None。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(last_user_at) AS val FROM presence"
            ).fetchone()
        return row["val"] if row else None


    def list_presence(
        self,
    ) -> dict[str, dict[str, str | None]]:
        """列出所有 session 的 presence 数据。

        输出:
            {session_key: {"last_user_at": ..., "last_proactive_at": ...}} 字典。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_key, last_user_at, last_proactive_at FROM presence"
            ).fetchall()
        return {
            row["session_key"]: {
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        }
    
    
    
    
    def list_sessions_for_dashboard(
        self,
        *,
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 Dashboard 使用的 session 摘要。

        输入:
            q: 可选文本搜索——匹配 session key 和 chat_id。
            channel: 可选 Channel 名称过滤。
            updated_from: 可选 updated_at 范围起点（ISO 字符串）。
            updated_to: 可选 updated_at 范围终点（ISO 字符串）。
            has_proactive: 可选——True 只返回有 proactive 数据的 session，
                           False 只返回没有的，None 不限。
            page: 页码，从 1 开始。
            page_size: 每页数量，上限 200。
            sort_by: 排序字段。可选：key、created_at、updated_at、message_count。
            sort_order: "asc" 或 "desc"，默认 "desc"。

        输出:
            二元组：(当前页 session 字典列表, 总 session 数)。
        """

        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        # 排序字段白名单
        allowed_sort = {"key", "created_at", "updated_at", "message_count"}
        safe_sort = sort_by if sort_by in allowed_sort else "updated_at"
        safe_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"

        # 构建 WHERE 子句
        clauses: list[str] = []
        params: list[Any] = []

        if q.strip():
            clauses.append("(s.key LIKE ? OR CAST(s.metadata AS TEXT) LIKE ?)")
            like_val = f"%{q.strip()}%"
            params.extend([like_val, like_val])

        if channel.strip():
            clauses.append("s.key LIKE ?")
            params.append(f"{channel.strip()}:%")

        if updated_from.strip():
            clauses.append("s.updated_at >= ?")
            params.append(updated_from.strip())

        if updated_to.strip():
            clauses.append("s.updated_at <= ?")
            params.append(updated_to.strip())

        if has_proactive is True:
            clauses.append(
                "EXISTS (SELECT 1 FROM presence p WHERE p.session_key = s.key "
                "AND p.last_proactive_at IS NOT NULL)"
            )
        elif has_proactive is False:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM presence p WHERE p.session_key = s.key "
                "AND p.last_proactive_at IS NOT NULL)"
            )

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        # message_count 需要子查询
        count_expr = (
            "(SELECT COUNT(*) FROM messages m WHERE m.session_key = s.key)"
        )

        with self._lock:
            # NOTE: COUNT(*) on large tables (1M+ rows) can be slow in SQLite.
            # For personal-scale usage (<100K sessions) this is negligible.
            # If scaling up, consider returning a has_next_page cursor
            # instead of an exact total count.
            total_row = self._conn.execute(
                f"SELECT COUNT(*) FROM sessions s{where}",
                tuple(params),
            ).fetchone()
            total = int(total_row[0]) if total_row else 0

            # 分页数据
            if safe_sort == "message_count":
                order_clause = f"ORDER BY {count_expr} {safe_order}"
            else:
                order_clause = f"ORDER BY s.{safe_sort} {safe_order}"

            rows = self._conn.execute(
                f"""
                SELECT s.key, s.created_at, s.updated_at, s.last_consolidated,
                       s.metadata, s.last_user_at, s.last_proactive_at,
                       {count_expr} AS message_count
                FROM sessions s{where}
                {order_clause}
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"])
            # 从 key 中提取 channel 和 chat_id
            if ":" in key:
                ch, chat_id = key.split(":", 1)
            else:
                ch, chat_id = "", key
            items.append({
                "key": key,
                "channel": ch,
                "chat_id": chat_id,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_consolidated": int(row["last_consolidated"] or 0),
                "metadata": _json_object(row["metadata"]),
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
                "message_count": int(row["message_count"] or 0),
            })

        return items, total


    def list_messages_for_dashboard(
        self,
        *,
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 Dashboard 使用的 message 摘要。

        输入:
            session_key: 可选 session key 过滤；为 None 时跨所有 session。
            q: 可选文本搜索（匹配 content）。
            role: 可选角色过滤（user / assistant / tool / system）。
            page: 页码，从 1 开始。
            page_size: 每页数量，上限 200。
            sort_by: 排序字段。可选：ts、seq、role。
            sort_order: "asc" 或 "desc"，默认 "desc"。

        输出:
            二元组：(当前页 message 字典列表, 总 message 数)。
        """

        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        allowed_sort = {"ts", "seq", "role", "id"}
        safe_sort = sort_by if sort_by in allowed_sort else "ts"
        safe_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"

        clauses: list[str] = []
        params: list[Any] = []

        if session_key:
            clauses.append("session_key = ?")
            params.append(session_key)

        if role.strip():
            clauses.append("role = ?")
            params.append(role.strip())

        if q.strip():
            if self._has_fts:
                # 使用 FTS5 进行文本搜索
                clauses.append("rowid IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?)")
                # 将搜索词用双引号包裹进行短语匹配
                params.append(f'"{q.strip()}"')
            else:
                clauses.append("content LIKE ?")
                params.append(f"%{q.strip()}%")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            # NOTE: COUNT(*) over messages can get slow at 1M+ rows.
            # Personal-scale usage (<100K messages) is negligible.
            total_row = self._conn.execute(
                f"SELECT COUNT(*) FROM messages{where}",
                tuple(params),
            ).fetchone()
            total = int(total_row[0]) if total_row else 0

            rows = self._conn.execute(
                f"""
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages{where}
                ORDER BY {safe_sort} {safe_order}
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            msg_id = str(row["id"])
            extra = _json_object(row["extra"])
            tool_chain_raw = row["tool_chain"]
            try:
                tool_chain = json.loads(tool_chain_raw) if tool_chain_raw else None
            except json.JSONDecodeError:
                tool_chain = None
            items.append({
                "id": msg_id,
                "session_key": str(row["session_key"]),
                "seq": int(row["seq"]),
                "role": str(row["role"]),
                "content": str(row["content"] or ""),
                "tool_chain": tool_chain,
                "extra": extra,
                "ts": row["ts"],
            })

        return items, total


    def search_messages(
        self,
        query: str,
        *,
        session_key: str | None = None,
        limit: int = 10,
    ) -> tuple[list[ChatMessage], int]:
        """搜索 session messages。

        输入:
            query: 搜索关键词。
            session_key: 可选 session key 过滤。
            limit: 最多返回多少条。

        输出:
            二元组：(命中的 ChatMessage 列表, 命中总数)。
        """

        return self.list_messages_for_dashboard(
            session_key=session_key,
            q=query,
            page=1,
            page_size=limit,
            sort_by="ts",
            sort_order="desc",
        )

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        """按 message id 读取单条消息。

        输入:
            message_id: 消息 id，例如 cli:default:3。

        输出:
            消息字典；不存在时返回 None。
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        extra = _json_object(row["extra"])
        tool_chain_raw = row["tool_chain"]
        try:
            tool_chain = json.loads(tool_chain_raw) if tool_chain_raw else None
        except json.JSONDecodeError:
            tool_chain = None
        return {
            "id": str(row["id"]),
            "session_key": str(row["session_key"]),
            "seq": int(row["seq"]),
            "role": str(row["role"]),
            "content": str(row["content"] or ""),
            "tool_chain": tool_chain,
            "extra": extra,
            "ts": row["ts"],
        }

    def update_message(
        self,
        message_id: str,
        *,
        role: str | None = None,
        content: str | None = None,
        tool_chain: object | None = None,
        extra: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> dict[str, Any] | None:
        """更新某条消息的字段。

        只更新传入的非 None 字段。tool_chain 为 None 时不清空原有值——
        需要显式传空列表才能清空。

        输入:
            message_id: 消息 id。
            role: 新角色。
            content: 新内容。
            tool_chain: 新工具调用链。
            extra: 新扩展字段。
            ts: 新时间戳。

        输出:
            更新后的消息字典；message 不存在时返回 None。
        """
        existing = self.get_message(message_id)
        if existing is None:
            return None

        new_role = role if role is not None else existing["role"]
        new_content = content if content is not None else existing["content"]
        new_ts = ts if ts is not None else existing["ts"]
        new_extra = extra if extra is not None else existing.get("extra", {})
        new_tool_chain = tool_chain if tool_chain is not None else existing.get("tool_chain")

        with self._lock:
            self._conn.execute(
                """
                UPDATE messages
                SET role = ?, content = ?, tool_chain = ?, extra = ?, ts = ?
                WHERE id = ?
                """,
                (
                    new_role,
                    new_content,
                    _tool_calls_dumps(new_tool_chain),
                    _json_dumps(new_extra),
                    new_ts,
                    message_id,
                ),
            )
            self._conn.commit()

        return self.get_message(message_id)

    def delete_message(self, message_id: str) -> bool:
        """删除单条消息。

        输入:
            message_id: 消息 id。

        输出:
            删除成功返回 True，消息不存在返回 False。
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def delete_messages_batch(self, ids: list[str]) -> int:
        """批量删除消息。

        输入:
            ids: 消息 id 列表。

        输出:
            实际删除条数。
        """
        clean = [i.strip() for i in ids if i.strip()]
        if not clean:
            return 0
        placeholders = ",".join("?" for _ in clean)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                clean,
            )
            self._conn.commit()
        return int(cursor.rowcount or 0)

    def delete_session(self, session_key: str, *, cascade: bool = True) -> bool:
        """删除某个 session 及其关联数据。

        输入:
            session_key: session key。
            cascade: 是否同时删除 messages 和 presence 记录。

        输出:
            True 表示 session 存在并已删除；False 表示 session 不存在。
        """
        if not self.session_exists(session_key):
            return False
        with self._lock:
            if cascade:
                self._conn.execute(
                    "DELETE FROM messages WHERE session_key = ?",
                    (session_key,),
                )
                self._conn.execute(
                    "DELETE FROM presence WHERE session_key = ?",
                    (session_key,),
                )
            self._conn.execute(
                "DELETE FROM sessions WHERE key = ?",
                (session_key,),
            )
            self._conn.commit()
        return True

    def delete_sessions_batch(
        self, keys: list[str], *, cascade: bool = True
    ) -> int:
        """批量删除 session。

        输入:
            keys: session key 列表。
            cascade: 是否级联删除 messages 和 presence。

        输出:
            实际删除的 session 数量。
        """
        clean = [k.strip() for k in keys if k.strip()]
        if not clean:
            return 0
        count = 0
        for key in clean:
            if self.delete_session(key, cascade=cascade):
                count += 1
        return count

    def update_session(
        self,
        session_key: str,
        *,
        metadata: dict[str, Any] | None = None,
        last_consolidated: int | None = None,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> dict[str, Any] | None:
        """更新 session 元数据。

        只更新传入的非 None 字段。

        输入:
            session_key: session key。
            metadata: 新 metadata 字典。
            last_consolidated: 新 consolidation 游标。
            last_user_at: 新用户最后活跃时间。
            last_proactive_at: 新 proactive 最后推送时间。

        输出:
            更新后的 session meta 字典；session 不存在时返回 None。
        """
        meta = self.get_session_meta(session_key)
        if meta is None:
            return None

        new_metadata = (
            metadata if metadata is not None else meta.get("metadata", {})
        )
        new_lc = (
            last_consolidated
            if last_consolidated is not None
            else meta.get("last_consolidated", 0)
        )

        with self._lock:
            # 更新 sessions 表
            self._conn.execute(
                """
                UPDATE sessions
                SET metadata = ?,
                    last_consolidated = ?,
                    updated_at = ?
                WHERE key = ?
                """,
                (
                    _json_dumps(new_metadata),
                    int(new_lc),
                    _now_iso(),
                    session_key,
                ),
            )
            # 更新 presence 表中的字段
            if last_user_at is not None or last_proactive_at is not None:
                existing_presence = self._conn.execute(
                    "SELECT last_user_at, last_proactive_at FROM presence WHERE session_key = ?",
                    (session_key,),
                ).fetchone()
                cur_user = (
                    last_user_at
                    if last_user_at is not None
                    else (existing_presence["last_user_at"] if existing_presence else None)
                )
                cur_proactive = (
                    last_proactive_at
                    if last_proactive_at is not None
                    else (existing_presence["last_proactive_at"] if existing_presence else None)
                )
                self._conn.execute(
                    """
                    INSERT INTO presence(session_key, last_user_at, last_proactive_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET
                        last_user_at = excluded.last_user_at,
                        last_proactive_at = excluded.last_proactive_at
                    """,
                    (session_key, cur_user, cur_proactive),
                )
            self._conn.commit()

        return self.get_session_meta(session_key)