from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from raven_agent.plugins.builtins.observe.db import open_db
from raven_agent.plugins.builtins.observe.events import (
    MemoryWriteTrace, RagQueryLog, ToolCallTrace, TurnTrace,
)

logger = logging.getLogger("observe.writer")

_QUEUE_MAX = 500


def _now_iso() -> str:
    """生成当前 UTC ISO 时间字符串。

    输入:
        无。

    输出:
        ISO8601 UTC 时间字符串。
    """

    return datetime.now(timezone.utc).isoformat()


class TraceWriter:
    """把 TurnTrace / ToolCallTrace 异步写入 SQLite 的写入器。

    输入:
        db_path: observe 数据库路径。

    输出:
        TraceWriter 实例。
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._queue: asyncio.Queue[
            TurnTrace | ToolCallTrace | RagQueryLog | MemoryWriteTrace
        ] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._dropped = 0

    def emit(self, event: TurnTrace | ToolCallTrace) -> None:
        """非阻塞入队一条 observe 事件。

        输入:
            event: TurnTrace 或 ToolCallTrace。

        输出:
            None。队列满时丢弃并计数。
        """

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("observe queue full, total_dropped=%d", self._dropped)

    async def drain(self) -> None:
        """等待已入队事件全部写完。

        输入:
            无。

        输出:
            None。
        """

        await self._queue.join()

    async def run(self) -> None:
        """后台循环消费队列写库。作为 asyncio task 运行。

        输入:
            无。

        输出:
            None。被取消时 flush 剩余事件并关闭连接。
        """

        conn = open_db(self._db_path)
        logger.info("observe writer started: %s", self._db_path)
        try:
            while True:
                event = await self._queue.get()
                try:
                    self._write_one(conn, event)
                except Exception:
                    logger.exception("observe write failed: %s", type(event).__name__)
                finally:
                    self._queue.task_done()
        finally:
            while not self._queue.empty():
                try:
                    pending = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    self._write_one(conn, pending)
                except Exception:
                    pass
                finally:
                    self._queue.task_done()
            conn.close()
            logger.info("observe writer stopped")

    def _write_one(self, conn: object, event: TurnTrace | ToolCallTrace) -> None:
        """把单条事件写入对应表。

        输入:
            conn: SQLite 连接。
            event: TurnTrace 或 ToolCallTrace。

        输出:
            None。
        """

        ts = _now_iso()
        if isinstance(event, TurnTrace):
            _write_turn(conn, event, ts)
        elif isinstance(event, ToolCallTrace):
            _write_tool_call(conn, event, ts)
        elif isinstance(event, RagQueryLog):
            _write_rag_query(conn, event, ts)
        elif isinstance(event, MemoryWriteTrace):
            _write_memory_write(conn, event, ts)

def _write_rag_query(conn: object, event: RagQueryLog, ts: str) -> None:
    """写入一条 RAG 检索记录到 rag_queries 表。

    输入:
        conn: SQLite 连接。
        event: RagQueryLog 事件。
        ts: ISO 时间字符串。

    输出:
        None。
    """

    # 将 hits 序列化为紧凑 JSON
    hits_json = (
        json.dumps(
            [
                {
                    "id": h.item_id,
                    "type": h.memory_type,
                    "score": round(h.score, 4),
                    "summary": h.summary[:120],
                    "injected": h.injected,
                    "forced": h.forced,
                }
                for h in event.hits
            ],
            ensure_ascii=False,
        )
        if event.hits
        else None
    )

    aux_json = (
        json.dumps(event.aux_queries, ensure_ascii=False)
        if event.aux_queries
        else None
    )

    with conn:  # type: ignore[union-attr]
        conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO rag_queries (
                ts, caller, session_key, query, orig_query,
                aux_queries, hits_json, injected_count, route_decision, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event.caller,
                event.session_key,
                event.query,
                event.orig_query,
                aux_json,
                hits_json,
                event.injected_count,
                event.route_decision,
                event.error,
            ),
        )


def _write_memory_write(conn: object, event: MemoryWriteTrace, ts: str) -> None:
    """写入一条记忆操作记录到 memory_writes 表。

    输入:
        conn: SQLite 连接。
        event: MemoryWriteTrace 事件。
        ts: ISO 时间字符串。

    输出:
        None。
    """

    superseded_json = (
        json.dumps(event.superseded_ids, ensure_ascii=False)
        if event.superseded_ids
        else None
    )

    with conn:  # type: ignore[union-attr]
        conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO memory_writes (
                ts, session_key, source_ref, action, memory_type,
                item_id, summary, superseded_ids, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event.session_key,
                event.source_ref,
                event.action,
                event.memory_type,
                event.item_id,
                event.summary,
                superseded_json,
                event.error,
            ),
        )

def _write_turn(conn: object, event: TurnTrace, ts: str) -> None:
    """写入一条 turn 记录。

    输入:
        conn: SQLite 连接。
        event: TurnTrace。
        ts: ISO 时间字符串。

    输出:
        None。
    """

    with conn:  # type: ignore[union-attr]
        conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO turns (
                ts, session_key, channel, chat_id, user_msg, reply,
                tools_used, iterations, cited_memory_ids, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event.session_key,
                event.channel,
                event.chat_id,
                event.user_msg,
                event.reply,
                json.dumps(event.tools_used, ensure_ascii=False) if event.tools_used else None,
                event.iterations,
                json.dumps(event.cited_memory_ids, ensure_ascii=False) if event.cited_memory_ids else None,
                event.error,
            ),
        )


def _write_tool_call(conn: object, event: ToolCallTrace, ts: str) -> None:
    """写入一条 tool_call 记录。

    输入:
        conn: SQLite 连接。
        event: ToolCallTrace。
        ts: ISO 时间字符串。

    输出:
        None。
    """

    with conn:  # type: ignore[union-attr]
        conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO tool_calls (
                ts, session_key, tool_name, arguments, status, plugin_source, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event.session_key,
                event.tool_name,
                json.dumps(event.arguments, ensure_ascii=False) if event.arguments else None,
                event.status,
                event.plugin_source or None,
                event.error,
            ),
        )