from __future__ import annotations

import sqlite3
from pathlib import Path

# schema 内嵌在代码里，避免运行时依赖外部 .sql 文件。
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    session_key       TEXT NOT NULL,
    channel           TEXT,
    chat_id           TEXT,
    user_msg          TEXT,
    reply             TEXT NOT NULL DEFAULT '',
    tools_used        TEXT,
    iterations        INTEGER,
    cited_memory_ids  TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS ix_turns_sk_ts ON turns (session_key, ts);

CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    session_key   TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    arguments     TEXT,
    status        TEXT NOT NULL,
    plugin_source TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_tool_calls_sk_ts ON tool_calls (session_key, ts);

CREATE TABLE IF NOT EXISTS rag_queries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    caller         TEXT    NOT NULL,   -- 'passive' | 'proactive' | 'explicit'
    session_key    TEXT    NOT NULL,
    query          TEXT    NOT NULL,   -- rewrite 后的检索 query
    orig_query     TEXT,              -- 改写前原文，NULL = 未改写
    aux_queries    TEXT,              -- JSON: ["hypothesis1", ...] HyDE 假想条目
    hits_json      TEXT,              -- JSON: [{id, type, score, summary, injected}]
    injected_count INTEGER NOT NULL DEFAULT 0,
    route_decision TEXT,              -- 'RETRIEVE' | 'NO_RETRIEVE' | NULL
    error          TEXT
);
CREATE INDEX IF NOT EXISTS ix_rq_sk_ts  ON rag_queries (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_rq_caller ON rag_queries (caller, ts);

CREATE TABLE IF NOT EXISTS memory_writes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    session_key     TEXT    NOT NULL,
    source_ref      TEXT,
    action          TEXT    NOT NULL,  -- 'write' | 'supersede'
    memory_type     TEXT,             -- write 时填写
    item_id         TEXT,             -- write: 'new:xxx' or 'reinforced:xxx'
    summary         TEXT,             -- write 时填写
    superseded_ids  TEXT,             -- supersede: JSON 数组
    error           TEXT
);
CREATE INDEX IF NOT EXISTS ix_mw_sk_ts ON memory_writes (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_mw_action ON memory_writes (action, ts);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    """打开（或新建）observe.db 并初始化 schema。

    输入:
        db_path: observe 数据库文件路径。

    输出:
        已初始化 schema 的 SQLite 连接。会创建父目录。
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn