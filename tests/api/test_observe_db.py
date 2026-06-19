"""Observe DB 扩展 schema 测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestObserveExtendedSchema:
    """验证扩展后的 observe schema（4 张表）正确创建。"""

    @pytest.fixture
    def db_path(self):
        """创建临时 observe.db。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "observe.db"
            yield p

    def test_all_tables_created(self, db_path):
        from raven_agent.plugins.builtins.observe.db import open_db
        conn = open_db(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "turns" in tables
        assert "tool_calls" in tables
        assert "rag_queries" in tables
        assert "memory_writes" in tables
        conn.close()

    def test_rag_queries_columns(self, db_path):
        from raven_agent.plugins.builtins.observe.db import open_db
        conn = open_db(db_path)
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(rag_queries)").fetchall()
        }
        expected = {
            "id", "ts", "caller", "session_key", "query",
            "orig_query", "aux_queries", "hits_json",
            "injected_count", "route_decision", "error",
        }
        assert expected.issubset(cols)
        conn.close()

    def test_memory_writes_columns(self, db_path):
        from raven_agent.plugins.builtins.observe.db import open_db
        conn = open_db(db_path)
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(memory_writes)").fetchall()
        }
        expected = {
            "id", "ts", "session_key", "source_ref", "action",
            "memory_type", "item_id", "summary", "superseded_ids", "error",
        }
        assert expected.issubset(cols)
        conn.close()


class TestTraceWriterExtended:
    """验证扩展后的 TraceWriter 能写入新表。"""

    @pytest.fixture
    async def writer_and_db(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from raven_agent.plugins.builtins.observe.writer import TraceWriter

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "observe.db"
            writer = TraceWriter(db_path)
            task = asyncio.create_task(writer.run(), name="test_observe_writer")
            await asyncio.sleep(0.05)  # 等 writer 启动
            yield writer, db_path
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_write_rag_query(self, writer_and_db):
        from raven_agent.plugins.builtins.observe.events import RagHitLog, RagQueryLog

        writer, db_path = writer_and_db
        event = RagQueryLog(
            caller="passive",
            session_key="cli:test",
            query="用户喜欢吃川菜吗",
            orig_query="我喜欢吃辣的东西",
            aux_queries=["用户喜欢麻辣口味", "用户偏好川菜"],
            hits=[
                RagHitLog(
                    item_id="mem:001",
                    memory_type="preference",
                    score=0.85,
                    summary="用户喜欢川菜和麻辣口味",
                    injected=True,
                ),
            ],
            injected_count=1,
            route_decision="RETRIEVE",
        )
        writer.emit(event)
        await writer.drain()

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM rag_queries").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["caller"] == "passive"
        assert row["query"] == "用户喜欢吃川菜吗"
        assert row["injected_count"] == 1

        hits = json.loads(row["hits_json"])
        assert len(hits) == 1
        assert hits[0]["id"] == "mem:001"
        assert hits[0]["score"] == 0.85
        conn.close()

    @pytest.mark.asyncio
    async def test_write_memory_write(self, writer_and_db):
        from raven_agent.plugins.builtins.observe.events import MemoryWriteTrace

        writer, db_path = writer_and_db
        event = MemoryWriteTrace(
            session_key="cli:test",
            source_ref="session:cli:test:turn:3",
            action="write",
            memory_type="preference",
            item_id="new:pref:001",
            summary="用户喜欢川菜",
        )
        writer.emit(event)
        await writer.drain()

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memory_writes").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["action"] == "write"
        assert row["memory_type"] == "preference"
        assert row["summary"] == "用户喜欢川菜"
        conn.close()

    @pytest.mark.asyncio
    async def test_write_memory_supersede(self, writer_and_db):
        from raven_agent.plugins.builtins.observe.events import MemoryWriteTrace

        writer, db_path = writer_and_db
        event = MemoryWriteTrace(
            session_key="cli:test",
            source_ref="session:cli:test:turn:5",
            action="supersede",
            superseded_ids=["mem:old:001", "mem:old:002"],
        )
        writer.emit(event)
        await writer.drain()

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM memory_writes").fetchone()
        assert row["action"] == "supersede"
        superseded = json.loads(row["superseded_ids"])
        assert len(superseded) == 2
        conn.close()