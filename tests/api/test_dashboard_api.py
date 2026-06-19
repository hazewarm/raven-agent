"""Dashboard API 的单元测试与集成测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from raven_agent.session_store import SessionStore


class TestSessionStoreDashboard:
    """测试 SessionStore 中为 Dashboard 新增的查询与 CRUD 方法。"""

    @pytest.fixture
    def store(self):
        """创建临时数据库的 SessionStore。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            s = SessionStore(db_path)
            # 预置测试数据
            s.upsert_session(
                "cli:test1",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-06-01T12:00:00",
                last_consolidated=0,
                metadata={"source": "cli"},
            )
            s.insert_message(
                "cli:test1", role="user", content="Hello",
                ts="2025-06-01T12:00:00", seq=0,
            )
            s.insert_message(
                "cli:test1", role="assistant", content="Hi there!",
                ts="2025-06-01T12:00:01", seq=1,
            )
            s.upsert_session(
                "telegram:12345",
                created_at="2025-02-01T00:00:00",
                updated_at="2025-06-02T08:00:00",
                last_consolidated=0,
                metadata={"source": "telegram"},
            )
            s.insert_message(
                "telegram:12345", role="user", content="What time is it?",
                ts="2025-06-02T08:00:00", seq=0,
            )
            yield s
            s.close()

    # ── list_sessions_for_dashboard ──

    def test_list_all_sessions(self, store):
        items, total = store.list_sessions_for_dashboard()
        assert total == 2
        assert len(items) == 2
        keys = {item["key"] for item in items}
        assert keys == {"cli:test1", "telegram:12345"}

    def test_list_sessions_by_channel(self, store):
        items, total = store.list_sessions_for_dashboard(channel="cli")
        assert total == 1
        assert items[0]["key"] == "cli:test1"

    def test_list_sessions_by_search(self, store):
        items, total = store.list_sessions_for_dashboard(q="tele")
        assert total == 1
        assert items[0]["key"] == "telegram:12345"

    def test_list_sessions_pagination(self, store):
        items, total = store.list_sessions_for_dashboard(page=1, page_size=1)
        assert total == 2
        assert len(items) == 1

    def test_list_sessions_message_count(self, store):
        items, _ = store.list_sessions_for_dashboard()
        for item in items:
            if item["key"] == "cli:test1":
                assert item["message_count"] == 2
            elif item["key"] == "telegram:12345":
                assert item["message_count"] == 1

    # ── list_messages_for_dashboard ──

    def test_list_all_messages(self, store):
        items, total = store.list_messages_for_dashboard()
        assert total == 3

    def test_list_messages_by_session(self, store):
        items, total = store.list_messages_for_dashboard(
            session_key="cli:test1"
        )
        assert total == 2
        assert all(item["session_key"] == "cli:test1" for item in items)

    def test_list_messages_by_role(self, store):
        items, total = store.list_messages_for_dashboard(role="user")
        assert total == 2
        assert all(item["role"] == "user" for item in items)

    def test_list_messages_by_search(self, store):
        items, total = store.list_messages_for_dashboard(q="Hello")
        assert total >= 1
        assert any("Hello" in item["content"] for item in items)

    # ── CRUD ──

    def test_get_message(self, store):
        msg = store.get_message("cli:test1:0")
        assert msg is not None
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_get_message_not_found(self, store):
        assert store.get_message("nonexistent:0") is None

    def test_update_message(self, store):
        updated = store.update_message("cli:test1:0", content="Updated hello")
        assert updated is not None
        assert updated["content"] == "Updated hello"

    def test_delete_message(self, store):
        assert store.delete_message("cli:test1:1") is True
        assert store.get_message("cli:test1:1") is None

    def test_delete_messages_batch(self, store):
        count = store.delete_messages_batch(["cli:test1:0", "cli:test1:1"])
        assert count == 2

    def test_delete_session(self, store):
        assert store.delete_session("cli:test1", cascade=True) is True
        assert not store.session_exists("cli:test1")

    def test_delete_session_not_found(self, store):
        assert store.delete_session("nonexistent") is False

    def test_update_session(self, store):
        updated = store.update_session(
            "cli:test1",
            metadata={"new_key": "new_value"},
        )
        assert updated is not None
        assert updated["metadata"]["new_key"] == "new_value"

    # ── 辅助方法 ──

    def test_count_messages(self, store):
        assert store.count_messages("cli:test1") == 2
        assert store.count_messages("telegram:12345") == 1

    def test_session_exists(self, store):
        assert store.session_exists("cli:test1") is True
        assert store.session_exists("nonexistent") is False


class TestDashboardAPIEndToEnd:
    """Dashboard API 的 HTTP 端到端测试。

    需要 fastapi 和 httpx 已安装。
    """

    @pytest.fixture
    async def client(self):
        """创建测试用的 FastAPI TestClient。"""
        try:
            from fastapi.testclient import TestClient
            from raven_agent.api.dashboard import create_dashboard_app
        except ImportError:
            pytest.skip("fastapi 未安装")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            store = SessionStore(ws / "sessions.db")

            # 预置数据
            store.upsert_session(
                "cli:default",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-06-01T12:00:00",
                last_consolidated=0,
                metadata={},
            )
            store.insert_message(
                "cli:default", role="user", content="Hello world",
                ts="2025-06-01T12:00:00", seq=0,
            )

            from raven_agent.memory import DisabledMemoryEngine
            memory_admin = DisabledMemoryEngine()

            from raven_agent.session import SessionManager
            sessions = SessionManager(store)

            app = create_dashboard_app(
                workspace=ws,
                store=store,
                sessions=sessions,
                memory_admin=memory_admin,
            )

            with TestClient(app) as client:
                yield client

            store.close()

    def test_list_sessions(self, client):
        resp = client.get("/api/dashboard/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["items"]) >= 1


    def test_list_messages(self, client):
        resp = client.get("/api/dashboard/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_list_session_messages(self, client):
        resp = client.get("/api/dashboard/sessions/cli:default/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any("Hello world" in item["content"] for item in data["items"])

    def test_get_message(self, client):
        resp = client.get("/api/dashboard/messages/cli:default:0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Hello world"

    def test_update_message(self, client):
        resp = client.patch(
            "/api/dashboard/messages/cli:default:0",
            json={"content": "Updated content"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Updated content"

    def test_delete_message(self, client):
        # 删除 cli:default:0
        resp = client.delete("/api/dashboard/messages/cli:default:0")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # 确认已删除
        resp = client.get("/api/dashboard/messages/cli:default:0")
        assert resp.status_code == 404

    def test_runtime_status(self, client):
        resp = client.get("/api/dashboard/runtime/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "session_count" in data

    def test_memory_engine_info(self, client):
        resp = client.get("/api/dashboard/memory/engine-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data

    def test_openapi_docs(self, client):
        """验证 OpenAPI 文档可访问。"""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        # 验证关键路由存在
        paths = data["paths"]
        assert "/api/dashboard/sessions" in paths
        assert "/api/dashboard/messages" in paths
        assert "/api/dashboard/memories" in paths
        assert "/api/dashboard/runtime/status" in paths


class TestObserveRoutesExtended:
    """Observe 扩展路由（rag_queries / memory_writes）的 HTTP 测试。"""

    @pytest.fixture
    async def client(self):
        try:
            from fastapi.testclient import TestClient
            from raven_agent.api.dashboard import create_dashboard_app
        except ImportError:
            pytest.skip("fastapi 未安装")

        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            store = SessionStore(ws / "sessions.db")

            # 预置 observe.db（含新表）
            obs_db = ws / "observe.db"
            from raven_agent.plugins.builtins.observe.db import open_db as open_obs
            conn = open_obs(obs_db)
            conn.execute(
                "INSERT INTO rag_queries(ts, caller, session_key, query) "
                "VALUES ('2025-06-01T00:00:00', 'passive', 'cli:default', 'test query')"
            )
            conn.execute(
                "INSERT INTO memory_writes(ts, session_key, action, summary) "
                "VALUES ('2025-06-01T00:00:00', 'cli:default', 'write', 'test summary')"
            )
            conn.commit()
            conn.close()

            from raven_agent.memory import DisabledMemoryEngine
            from raven_agent.session import SessionManager

            memory_admin = DisabledMemoryEngine()
            sessions = SessionManager(store)

            app = create_dashboard_app(
                workspace=ws,
                store=store,
                sessions=sessions,
                memory_admin=memory_admin,
                observe_db_path=obs_db,
            )

            with TestClient(app) as client:
                yield client

            store.close()

    def test_list_rag_queries(self, client):
        resp = client.get("/api/dashboard/observe/rag-queries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_get_rag_query(self, client):
        resp = client.get("/api/dashboard/observe/rag-queries/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"

    def test_get_rag_query_not_found(self, client):
        resp = client.get("/api/dashboard/observe/rag-queries/99999")
        assert resp.status_code == 404

    def test_list_memory_writes(self, client):
        resp = client.get("/api/dashboard/observe/memory-writes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_get_memory_write(self, client):
        resp = client.get("/api/dashboard/observe/memory-writes/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "test summary"

    def test_get_memory_write_not_found(self, client):
        resp = client.get("/api/dashboard/observe/memory-writes/99999")
        assert resp.status_code == 404

    def test_trigger_backup(self, client):
        resp = client.post("/api/dashboard/backup")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "completed"
        assert "backups" in data