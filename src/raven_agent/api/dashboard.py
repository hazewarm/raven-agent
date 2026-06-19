"""Dashboard HTTP API —— raven-agent 的管理与可观测后端。

通过 FastAPI 暴露 RESTful 接口，供前端 Dashboard 或管理工具消费。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from raven_agent.memory import MemoryAdminApi, MemoryOptimizer
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore
import re

logger = logging.getLogger(__name__)

# ── Dashboard 访问日志过滤器 ────────────────────────────────────────
# Dashboard 前端频繁轮询，访问日志只在 DEBUG 模式保留。

_DASHBOARD_ACCESS_PREFIXES = ("/api/dashboard", "/assets", "/plugins/")


def _is_dashboard_access_record(record: logging.LogRecord) -> bool:
    """判断日志记录是否为 Dashboard 相关的 HTTP 访问。

    输入:
        record: logging.LogRecord。

    输出:
        True 表示该记录是 Dashboard 的 HTTP 请求日志。
    """
    args = record.args
    if not isinstance(args, tuple) or len(args) < 3:
        return False
    path = args[2]
    if not isinstance(path, str):
        return False
    return path == "/" or any(
        path.startswith(prefix) for prefix in _DASHBOARD_ACCESS_PREFIXES
    )


class DashboardAccessLogFilter(logging.Filter):
    """Dashboard 访问日志过滤器。

    非 DEBUG 模式下，将 Dashboard 相关请求日志降级为 DEBUG
    （uvicorn.access 默认级别为 INFO，降级后不输出）。
    DEBUG 模式下保留所有日志。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录。

        输入:
            record: 日志记录。

        输出:
            True 保留记录，False 丢弃。
        """
        if not _is_dashboard_access_record(record):
            return True
        debug_enabled = (
            logging.getLogger().isEnabledFor(logging.DEBUG)
            or logging.getLogger("uvicorn.access").isEnabledFor(logging.DEBUG)
        )
        if not debug_enabled:
            return False
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
        return True


def _install_dashboard_access_log_filter() -> None:
    """安装 Dashboard 访问日志过滤器（幂等）。

    输入:
        无。

    输出:
        None。
    """
    access_logger = logging.getLogger("uvicorn.access")
    if any(
        isinstance(f, DashboardAccessLogFilter)
        for f in access_logger.filters
    ):
        return
    access_logger.addFilter(DashboardAccessLogFilter())

# ── Pydantic 请求/响应模型 ──────────────────────────────────────────

class SessionUpdatePayload(BaseModel):
    """PATCH /api/dashboard/sessions/{key} 的请求体。

    所有字段可选——只更新传入的非 None 字段。
    """

    metadata: dict[str, Any] | None = None
    last_consolidated: int | None = None
    last_user_at: str | None = None
    last_proactive_at: str | None = None


class SessionBatchDeletePayload(BaseModel):
    """POST /api/dashboard/sessions/batch-delete 的请求体。"""

    keys: list[str]
    cascade: bool = True


class SessionConsolidatePayload(BaseModel):
    """POST /api/dashboard/sessions/{key}/consolidate 的请求体。"""

    archive_all: bool = False
    force: bool = True


class MessageUpdatePayload(BaseModel):
    """PATCH /api/dashboard/messages/{id} 的请求体。"""

    role: str | None = None
    content: str | None = None
    tool_chain: Any | None = None
    extra: dict[str, Any] | None = None
    ts: str | None = None


class MessageBatchDeletePayload(BaseModel):
    """POST /api/dashboard/messages/batch-delete 的请求体。"""

    ids: list[str]


class MemoryUpdatePayload(BaseModel):
    """PATCH /api/dashboard/memories/{id} 的请求体。"""

    status: str | None = None
    extra_json: dict[str, Any] | None = None
    source_ref: str | None = None
    happened_at: str | None = None
    emotional_weight: int | None = None


class MemoryBatchDeletePayload(BaseModel):
    """POST /api/dashboard/memories/batch-delete 的请求体。"""

    ids: list[str]


class ProactiveDeletePayload(BaseModel):
    """DELETE /api/dashboard/proactive/seen_items/batch 的请求体。"""

    source_key: str | None = None
    item_ids: list[str] | None = None


# ── Pydantic 响应模型（核心查询端点）────────────────────────────────

class SessionSummary(BaseModel):
    """GET /api/dashboard/sessions 返回的单条 session 摘要。"""

    key: str
    channel: str
    chat_id: str
    created_at: str | None
    updated_at: str | None
    last_consolidated: int
    metadata: dict[str, Any]
    last_user_at: str | None
    last_proactive_at: str | None
    message_count: int


class SessionListResponse(BaseModel):
    """GET /api/dashboard/sessions 的响应体。"""

    items: list[SessionSummary]
    total: int
    page: int
    page_size: int


class MessageSummary(BaseModel):
    """GET /api/dashboard/messages 返回的单条 message 摘要。"""

    id: str
    session_key: str
    seq: int
    role: str
    content: str
    tool_chain: Any | None
    extra: dict[str, Any]
    ts: str | None


class MessageListResponse(BaseModel):
    """GET /api/dashboard/messages 的响应体。"""

    items: list[MessageSummary]
    total: int
    page: int
    page_size: int


class MemoryListResponse(BaseModel):
    """GET /api/dashboard/memories 的响应体。"""

    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class RuntimeStatusResponse(BaseModel):
    """GET /api/dashboard/runtime/status 的响应体。"""

    status: str
    scheduler: dict[str, Any]
    background_runtime: dict[str, Any]
    proactive: dict[str, Any]
    mcp: dict[str, Any]
    vision: dict[str, Any]
    audio: dict[str, Any]
    session_count: int


# ── API Key 认证依赖 — 始终生效，每个路由组统一注入 ───────────────

def _make_auth_dependency(api_key: str):
    """构造 API Key 校验依赖。

    Dashboard 是管理后台，认证始终开启——不存在"可选"模式。
    健康检查端点 /api/dashboard/runtime/status 硬编码豁免。

    输入:
        api_key: 配置中的 api_key。

    输出:
        FastAPI Depends 对象。所有非健康检查端点必须注入此依赖。
    """

    import hmac

    from fastapi import Depends, Request
    from fastapi.security import APIKeyHeader

    header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _verify(request: Request, key: str = header_scheme) -> None:
        if request.url.path == "/api/dashboard/runtime/status":
            return
        if key is None:
            raise HTTPException(status_code=401, detail="缺少 X-API-Key 认证头")
        if not hmac.compare_digest(str(key), api_key):
            raise HTTPException(status_code=401, detail="X-API-Key 认证失败")

    return Depends(_verify)


# ── Protocol 类型：Dashboard 通过协议引用 AppRuntime 能力，避免循环导入 ──

class ManualConsolidator(Protocol):
    """手动触发 memory consolidation 的协议。

    AppRuntime 中 MarkdownMemoryMaintenance 的 trigger_memory_consolidation
    方法满足此协议。
    """

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool: ...


class ManualMemoryOptimizer(Protocol):
    """手动触发 memory optimizer 的协议。

    MemoryOptimizer 满足此协议。
    """

    @property
    def is_running(self) -> bool: ...

    async def optimize(self) -> None: ...

# ── ProactiveDashboardReader ────────────────────────────────────────

class ProactiveDashboardReader:
    """Proactive 数据的 Dashboard 只读查询器。

    使用独立的 SQLite 只读连接访问 proactive_state.db，
    避免与 ProactiveLoop 的写连接产生锁冲突。

    输入:
        db_path: proactive_state.db 文件路径。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._db.close()

    # ── 概览 ──

    def get_overview(self) -> dict[str, Any]:
        """返回 Proactive 系统的概览统计数据。

        输出:
            包含各表行数、最近 tick 信息、最近投递时间的字典。
            proactive 系统未启动（表不存在）时返回空状态。
        """
        counts = {
            "seen_items": self._count("seen_items"),
            "deliveries": self._count("deliveries"),
            "tick_logs": self._count("tick_log"),
        }
        recent_tick = None
        last_send = None
        if self._has_table("tick_log"):
            with self._lock:
                recent_tick = self._db.execute(
                    """
                    SELECT tick_id, session_key, started_at, finished_at,
                           terminal_action, skip_reason, steps_taken
                    FROM tick_log
                    ORDER BY started_at DESC LIMIT 1
                    """
                ).fetchone()
        if self._has_table("deliveries"):
            with self._lock:
                last_send = self._db.execute(
                    "SELECT sent_at FROM deliveries ORDER BY sent_at DESC LIMIT 1"
                ).fetchone()
        return {
            "counts": counts,
            "result_counts": self._result_counts(),
            "flow_counts": self._flow_counts(),
            "last_tick_at": recent_tick["started_at"] if recent_tick else None,
            "last_send_at": last_send["sent_at"] if last_send else None,
            "last_skip_reason": (
                recent_tick["skip_reason"]
                if recent_tick and recent_tick["terminal_action"] != "reply"
                else None
            ),
            "recent_tick": self._row_to_dict(recent_tick) if recent_tick else None,
        }

    # ── 投递记录 ──

    def list_deliveries(
        self,
        *,
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 Proactive 投递记录。

        输入:
            session_key: 可选 session 过滤。
            sent_from: 可选发送时间起点（ISO）。
            sent_to: 可选发送时间终点（ISO）。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("sent_at >= ?", sent_from),
            ("sent_at <= ?", sent_to),
        )
        return self._list_rows(
            table="deliveries",
            where=where,
            params=params,
            order_by="sent_at DESC",
            page=page,
            page_size=page_size,
            columns="session_key, delivery_key, sent_at",
        )

    # ── 已见事件 ──

    def list_seen_items(
        self,
        *,
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 Proactive 已见事件（用于去重）。

        输入:
            source_key: 可选来源过滤。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        where, params = self._build_filters(("source_key = ?", source_key))
        return self._list_rows(
            table="seen_items",
            where=where,
            params=params,
            order_by="seen_at DESC",
            page=page,
            page_size=page_size,
            columns="source_key, item_id, seen_at",
        )

    # ── Tick 日志 ──

    def list_tick_logs(
        self,
        *,
        session_key: str = "",
        terminal_action: str = "",
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 Proactive tick 执行日志。

        输入:
            session_key: 可选 session 过滤。
            terminal_action: 可选最终动作过滤（reply / skip / drift）。
            started_from: 可选 tick 开始时间起点。
            started_to: 可选 tick 开始时间终点。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("terminal_action = ?", terminal_action),
            ("started_at >= ?", started_from),
            ("started_at <= ?", started_to),
        )
        return self._list_rows(
            table="tick_log",
            where=where,
            params=params,
            order_by="started_at DESC, tick_id DESC",
            page=page,
            page_size=page_size,
            columns=(
                "tick_id, session_key, started_at, finished_at, "
                "terminal_action, skip_reason, steps_taken"
            ),
        )

    def get_tick_log(self, tick_id: str) -> dict[str, Any] | None:
        """获取单条 tick 日志。

        输入:
            tick_id: tick 唯一 ID。

        输出:
            tick 日志字典；不存在时返回 None。
        """
        with self._lock:
            row = self._db.execute(
                """
                SELECT tick_id, session_key, started_at, finished_at,
                       terminal_action, skip_reason, steps_taken
                FROM tick_log WHERE tick_id = ?
                """,
                (tick_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    # ── 删除操作 ──

    def delete_seen_items(
        self,
        *,
        source_key: str = "",
        item_ids: list[str] | None = None,
    ) -> int:
        """删除 seen_items 中的记录。

        输入:
            source_key: 可选，按来源删除。
            item_ids: 可选，按 item ID 列表删除。

        输出:
            实际删除条数。

        异常:
            ValueError: 当 source_key 和 item_ids 均为空时。
        """
        if not source_key and not item_ids:
            raise ValueError("至少提供 source_key 或 item_ids")
        clauses: list[str] = []
        params: list[Any] = []
        if source_key:
            clauses.append("source_key = ?")
            params.append(source_key)
        if item_ids:
            placeholders = ", ".join("?" for _ in item_ids)
            clauses.append(f"item_id IN ({placeholders})")
            params.extend(item_ids)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = self._db.execute(
                f"DELETE FROM seen_items{where}", tuple(params)
            )
            self._db.commit()
        return int(cursor.rowcount or 0)

    # ── 内部辅助 ──

    def _has_table(self, table: str) -> bool:
        """检查某表是否已存在。

        输入:
            table: 表名。

        输出:
            True 表示表存在。
        """
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
        return row is not None

    def _count(self, table: str) -> int:
        """返回某表的行数。表不存在时返回 0。

        输入:
            table: 表名。

        输出:
            行数或 0。
        """
        if not self._has_table(table):
            return 0
        with self._lock:
            row = self._db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
        return int(row[0]) if row else 0

    def _result_counts(self) -> dict[str, int]:
        """按 terminal_action / gate_exit 统计 tick 数量。

        前端侧边栏用于显示 reply / skip / busy / cooldown / presence 分组计数。
        """
        result: dict[str, int] = {}
        if not self._has_table("tick_log"):
            return result
        with self._lock:
            rows = self._db.execute(
                """SELECT terminal_action, COUNT(*) AS n
                   FROM tick_log GROUP BY terminal_action"""
            ).fetchall()
        for row in rows:
            if row["terminal_action"]:
                result[row["terminal_action"]] = int(row["n"])
        return result

    def _flow_counts(self) -> dict[str, int]:
        """按是否进入 drift 统计 tick 数量。

        前端侧边栏用于显示 drift / proactive 分组计数。
        """
        counts = {"drift": 0, "proactive": 0}
        if not self._has_table("tick_log"):
            return counts
        with self._lock:
            drift_row = self._db.execute(
                "SELECT COUNT(*) AS n FROM tick_log WHERE drift_entered IS NOT NULL AND drift_entered != 0"
            ).fetchone()
            total_row = self._db.execute(
                "SELECT COUNT(*) AS n FROM tick_log"
            ).fetchone()
        if drift_row:
            counts["drift"] = int(drift_row["n"])
        if total_row:
            counts["proactive"] = int(total_row["n"]) - counts["drift"]
        return counts

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """把 sqlite3.Row 转换为普通字典。"""
        return {key: row[key] for key in row.keys()}

    def _list_rows(
        self,
        *,
        table: str,
        where: str,
        params: tuple[Any, ...],
        order_by: str,
        page: int,
        page_size: int,
        columns: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """通用的分页列表查询。表不存在时返回空。

        输入:
            table: 表名。
            where: WHERE 子句（可为空字符串）。
            params: WHERE 参数元组。
            order_by: ORDER BY 子句。
            page: 页码。
            page_size: 每页数量。
            columns: SELECT 列名。

        输出:
            (items, total)。
        """
        if not self._has_table(table):
            return [], 0
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size
        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM {table}{where}", params
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT {columns}
                FROM {table}{where}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row else 0
        return [self._row_to_dict(row) for row in rows], total

    def _build_filters(
        self, *filters: tuple[str, Any]
    ) -> tuple[str, tuple[Any, ...]]:
        """把过滤条件转换为 SQL WHERE 子句和参数元组。

        输入:
            *filters: 若干 (clause, value) 二元组。value 为空字符串或 None 时跳过。

        输出:
            (WHERE 子句, 参数元组)。
        """
        clauses: list[str] = []
        params: list[Any] = []
        for clause, value in filters:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            clauses.append(clause)
            params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, tuple(params)

# ── ObserveDashboardReader ──────────────────────────────────────────

class ObserveDashboardReader:
    """Observe 插件数据的 Dashboard 只读查询器。

    使用独立的 SQLite 只读连接访问 observe.db。

    输入:
        db_path: observe.db 文件路径。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._db.close()

    def list_turns(
        self,
        *,
        session_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 observe 记录的 turns。

        输入:
            session_key: 可选 session 过滤。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        clauses: list[str] = []
        params: list[Any] = []
        if session_key.strip():
            clauses.append("session_key = ?")
            params.append(session_key.strip())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM turns{where}", tuple(params)
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT id, ts, session_key, channel, chat_id,
                       user_msg, reply, tools_used, iterations,
                       cited_memory_ids, error
                FROM turns{where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row else 0
        items = [
            {key: row[key] for key in row.keys()} for row in rows
        ]
        return items, total

    def list_tool_calls(
        self,
        *,
        session_key: str = "",
        tool_name: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 observe 记录的 tool_calls。

        输入:
            session_key: 可选 session 过滤。
            tool_name: 可选工具名过滤。
            status: 可选状态过滤（success / error / denied）。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        clauses: list[str] = []
        params: list[Any] = []
        if session_key.strip():
            clauses.append("session_key = ?")
            params.append(session_key.strip())
        if tool_name.strip():
            clauses.append("tool_name = ?")
            params.append(tool_name.strip())
        if status.strip():
            clauses.append("status = ?")
            params.append(status.strip())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM tool_calls{where}", tuple(params)
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT id, ts, session_key, tool_name, arguments,
                       status, plugin_source, error
                FROM tool_calls{where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row else 0
        items = [
            {key: row[key] for key in row.keys()} for row in rows
        ]
        return items, total
    
    # ── RAG Queries ──

    def list_rag_queries(
        self,
        *,
        session_key: str = "",
        caller: str = "",
        route_decision: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 observe 记录的 RAG 检索事件。

        输入:
            session_key: 可选 session 过滤。
            caller: 可选调用来源过滤（passive / proactive / explicit）。
            route_decision: 可选路由决策过滤（RETRIEVE / NO_RETRIEVE）。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        import json as _json

        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        clauses: list[str] = []
        params: list[Any] = []
        if session_key.strip():
            clauses.append("session_key = ?")
            params.append(session_key.strip())
        if caller.strip():
            clauses.append("caller = ?")
            params.append(caller.strip())
        if route_decision.strip():
            clauses.append("route_decision = ?")
            params.append(route_decision.strip())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM rag_queries{where}", tuple(params)
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT id, ts, caller, session_key, query, orig_query,
                       aux_queries, hits_json, injected_count, route_decision, error
                FROM rag_queries{where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row else 0

        items: list[dict[str, Any]] = []
        for row in rows:
            item = {key: row[key] for key in row.keys()}
            # 解析 JSON 字段
            for json_field in ("aux_queries", "hits_json"):
                raw = item.get(json_field)
                if isinstance(raw, str) and raw:
                    try:
                        item[json_field] = _json.loads(raw)
                    except _json.JSONDecodeError:
                        pass
                elif raw is None:
                    item[json_field] = None
            items.append(item)

        return items, total

    def get_rag_query(self, query_id: int) -> dict[str, Any] | None:
        """获取单条 RAG 检索记录。

        输入:
            query_id: rag_queries 表的自增 id。

        输出:
            检索记录字典；不存在时返回 None。
        """
        import json as _json

        with self._lock:
            row = self._db.execute(
                """
                SELECT id, ts, caller, session_key, query, orig_query,
                       aux_queries, hits_json, injected_count, route_decision, error
                FROM rag_queries WHERE id = ?
                """,
                (query_id,),
            ).fetchone()
        if row is None:
            return None
        item = {key: row[key] for key in row.keys()}
        for json_field in ("aux_queries", "hits_json"):
            raw = item.get(json_field)
            if isinstance(raw, str) and raw:
                try:
                    item[json_field] = _json.loads(raw)
                except _json.JSONDecodeError:
                    pass
            elif raw is None:
                item[json_field] = None
        return item

    # ── Memory Writes ──

    def list_memory_writes(
        self,
        *,
        session_key: str = "",
        action: str = "",
        memory_type: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出 observe 记录的记忆写入/替换事件。

        输入:
            session_key: 可选 session 过滤。
            action: 可选操作过滤（write / supersede）。
            memory_type: 可选记忆类型过滤（event / profile / preference / procedure）。
            page: 页码。
            page_size: 每页数量。

        输出:
            (items, total)。
        """
        import json as _json

        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size

        clauses: list[str] = []
        params: list[Any] = []
        if session_key.strip():
            clauses.append("session_key = ?")
            params.append(session_key.strip())
        if action.strip():
            clauses.append("action = ?")
            params.append(action.strip())
        if memory_type.strip():
            clauses.append("memory_type = ?")
            params.append(memory_type.strip())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM memory_writes{where}", tuple(params)
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT id, ts, session_key, source_ref, action,
                       memory_type, item_id, summary, superseded_ids, error
                FROM memory_writes{where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row else 0

        items: list[dict[str, Any]] = []
        for row in rows:
            item = {key: row[key] for key in row.keys()}
            raw = item.get("superseded_ids")
            if isinstance(raw, str) and raw:
                try:
                    item["superseded_ids"] = _json.loads(raw)
                except _json.JSONDecodeError:
                    pass
            elif raw is None:
                item["superseded_ids"] = None
            items.append(item)

        return items, total

    def get_memory_write(self, write_id: int) -> dict[str, Any] | None:
        """获取单条记忆写入记录。

        输入:
            write_id: memory_writes 表的自增 id。

        输出:
            写入记录字典；不存在时返回 None。
        """
        import json as _json

        with self._lock:
            row = self._db.execute(
                """
                SELECT id, ts, session_key, source_ref, action,
                       memory_type, item_id, summary, superseded_ids, error
                FROM memory_writes WHERE id = ?
                """,
                (write_id,),
            ).fetchone()
        if row is None:
            return None
        item = {key: row[key] for key in row.keys()}
        raw = item.get("superseded_ids")
        if isinstance(raw, str) and raw:
            try:
                item["superseded_ids"] = _json.loads(raw)
            except _json.JSONDecodeError:
                pass
        elif raw is None:
            item["superseded_ids"] = None
        return item

# ── FastAPI App 工厂 ────────────────────────────────────────────────

def create_dashboard_app(
    workspace: Path,
    *,
    store: SessionStore,
    sessions: SessionManager,
    memory_admin: MemoryAdminApi,
    api_key: str = "",
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    observe_db_path: Path | None = None,
    proactive_db_path: Path | None = None,
    project_root: Path | None = None,
    plugins_root: Path | None = None,
    static_dir: Path | None = None,
    trips_dir: Path | None = None,
) -> FastAPI:
    """创建 Dashboard API 的 FastAPI app。

    输入:
        workspace: raven-agent 的工作区根目录。
        store: SQLite SessionStore，提供 session/message 的 CRUD。
        sessions: SessionManager，提供运行时 session 缓存。
        memory_admin: MemoryEngine 的管理 API（MemoryAdminApi 协议）。
        api_key: API 认证密钥。Dashboard 始终要求认证——所有非健康检查
            端点必须携带 X-API-Key 头。设为 "" 等价于无保护（仅本地开发）。
        manual_consolidator: 可选的手动 consolidation 触发器。
        manual_memory_optimizer: 可选的手动 memory optimizer 触发器。
        observe_db_path: observe.db 文件路径；为 None 时不注册 observe 路由。
        proactive_db_path: proactive_state.db 文件路径；为 None 时不注册 proactive 路由。
        project_root: 项目根目录（用于查找 esbuild 编译插件面板）。
            为 None 时不注册插件面板路由。
        plugins_root: 用户插件根目录（如 plugins/）。
            为 None 时不注册插件面板路由。
        static_dir: 前端静态资源目录（static/dashboard/）。
            为 None 时不注册静态文件挂载和首页路由。
        trips_dir: 旅行攻略 HTML 输出目录（plugins/travel/output/）。
            为 None 时不注册 /trips 静态文件挂载。

    输出:
        配置好所有路由的 FastAPI app 实例。
    """

    proactive_reader: ProactiveDashboardReader | None = None
    observe_reader: ObserveDashboardReader | None = None
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None
    compile_task: asyncio.Task[None] | None = None

    # ── API Key 认证依赖（始终开启）──
    _auth = _make_auth_dependency(api_key)

    def _get_proactive_reader() -> ProactiveDashboardReader:
        """惰性初始化 ProactiveDashboardReader（延迟连接）。"""
        nonlocal proactive_reader
        if proactive_reader is None and proactive_db_path is not None:
            proactive_reader = ProactiveDashboardReader(proactive_db_path)
        if proactive_reader is None:
            raise HTTPException(status_code=503, detail="proactive 数据不可用")
        return proactive_reader

    def _get_observe_reader() -> ObserveDashboardReader:
        """惰性初始化 ObserveDashboardReader。"""
        nonlocal observe_reader
        if observe_reader is None and observe_db_path is not None:
            observe_reader = ObserveDashboardReader(observe_db_path)
        if observe_reader is None:
            raise HTTPException(status_code=503, detail="observe 数据不可用")
        return observe_reader

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """FastAPI lifespan：在 app 启动/关闭时管理资源。"""
        # ── 等待插件面板延迟编译完成 ──
        if compile_task is not None:
            try:
                await compile_task
            except Exception:
                logger.warning("插件面板编译任务异常", exc_info=True)
        try:
            yield
        except asyncio.CancelledError:
            pass
        finally:
            store.close()
            if proactive_reader is not None:
                proactive_reader.close()
            if observe_reader is not None:
                observe_reader.close()

    app = FastAPI(
        title="Raven Agent Dashboard API",
        version="0.1.0",
        lifespan=lifespan,
        dependencies=[_auth] if api_key else None,
    )

    # ═══════════════════════════════════════════════════════════════
    # Sessions 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/sessions")
    def list_sessions(
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> SessionListResponse:
        """列出所有 session，支持搜索、过滤、分页和排序。

        返回:
            SessionListResponse: 分页的 session 摘要列表。
        """
        items_raw, total = store.list_sessions_for_dashboard(
            q=q,
            channel=channel,
            updated_from=updated_from,
            updated_to=updated_to,
            has_proactive=has_proactive,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return SessionListResponse(
            items=[SessionSummary(**item) for item in items_raw],
            total=total,
            page=max(1, page),
            page_size=max(1, min(page_size, 200)),
        )

    @app.get("/api/dashboard/sessions/{session_key:path}/messages")
    def list_session_messages(
        session_key: str,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        """列出某个 session 的所有消息。

        异常:
            404: session 不存在。
        """
        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        items, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.delete("/api/dashboard/sessions/{session_key:path}")
    def delete_session(session_key: str, cascade: bool = True) -> dict[str, Any]:
        """删除指定 session 及其关联数据。

        异常:
            404: session 不存在。
        """
        deleted = store.delete_session(session_key, cascade=cascade)
        if not deleted:
            raise HTTPException(status_code=404, detail="session 不存在")
        return {"deleted": True, "key": session_key}

    @app.post("/api/dashboard/sessions/batch-delete")
    def delete_sessions_batch(payload: SessionBatchDeletePayload) -> dict[str, Any]:
        """批量删除 session。

        返回:
            {"deleted_count": N}
        """
        count = store.delete_sessions_batch(payload.keys, cascade=payload.cascade)
        return {"deleted_count": count}

    # ═══════════════════════════════════════════════════════════════
    # Messages 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/messages")
    def list_messages(
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> MessageListResponse:
        """列出消息（可跨 session 或限定 session）。

        返回:
            MessageListResponse: 分页的 message 摘要列表。
        """
        items_raw, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return MessageListResponse(
            items=[MessageSummary(**item) for item in items_raw],
            total=total,
            page=max(1, page),
            page_size=max(1, min(page_size, 200)),
        )

    @app.get("/api/dashboard/messages/{message_id:path}")
    def get_message(message_id: str) -> dict[str, Any]:
        """获取单条消息。

        异常:
            404: message 不存在。
        """
        msg = store.get_message(message_id)
        if msg is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return msg

    @app.patch("/api/dashboard/messages/{message_id:path}")
    def update_message(
        message_id: str,
        payload: MessageUpdatePayload,
    ) -> dict[str, Any]:
        """更新单条消息。

        异常:
            404: message 不存在。
        """
        msg = store.update_message(
            message_id,
            role=payload.role,
            content=payload.content,
            tool_chain=payload.tool_chain,
            extra=payload.extra,
            ts=payload.ts,
        )
        if msg is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return msg

    @app.delete("/api/dashboard/messages/{message_id:path}")
    def delete_message(message_id: str) -> dict[str, Any]:
        """删除单条消息。

        异常:
            404: message 不存在。
        """
        deleted = store.delete_message(message_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="message 不存在")
        return {"deleted": True, "id": message_id}

    @app.post("/api/dashboard/messages/batch-delete")
    def delete_messages_batch(
        payload: MessageBatchDeletePayload,
    ) -> dict[str, Any]:
        """批量删除消息。

        返回:
            {"deleted_count": N}
        """
        deleted_count = store.delete_messages_batch(payload.ids)
        return {"deleted_count": deleted_count}


    # ═══════════════════════════════════════════════════════════════
    # Memory 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/memory/engine-info")
    def get_memory_engine_info() -> dict[str, Any]:
        """返回当前 MemoryEngine 的描述信息。"""
        desc = memory_admin.describe()
        return {"name": desc.name}

    @app.get("/api/dashboard/memory/optimizer")
    async def get_memory_optimizer_status() -> dict[str, Any]:
        """返回 Memory Optimizer 的运行状态。"""
        running = bool(
            manual_memory_optimizer is not None
            and (
                (optimizer_task is not None and not optimizer_task.done())
                or manual_memory_optimizer.is_running
            )
        )
        return {
            "enabled": manual_memory_optimizer is not None,
            "running": running,
            "last_status": "running" if running else optimizer_last_status,
            "last_error": optimizer_last_error,
        }

    @app.post("/api/dashboard/memory/optimize", status_code=202)
    async def trigger_memory_optimizer() -> dict[str, Any]:
        """手动触发一次 memory optimizer 执行。

        异常:
            503: optimizer 未启用。
            409: optimizer 正在运行。
        """
        nonlocal optimizer_last_error, optimizer_last_status, optimizer_task

        if manual_memory_optimizer is None:
            raise HTTPException(
                status_code=503, detail="memory optimizer 未启用"
            )
        if (
            optimizer_task is not None and not optimizer_task.done()
        ) or manual_memory_optimizer.is_running:
            raise HTTPException(
                status_code=409, detail="memory optimizer 正在运行"
            )

        logger.info("Manual memory optimizer triggered via dashboard")

        async def _run_optimizer() -> None:
            nonlocal optimizer_last_error, optimizer_last_status
            assert manual_memory_optimizer is not None
            optimizer_last_status = "running"
            optimizer_last_error = None
            try:
                await manual_memory_optimizer.optimize()
                optimizer_last_status = "succeeded"
            except asyncio.CancelledError:
                optimizer_last_status = "failed"
                optimizer_last_error = "已取消"
                raise
            except Exception as exc:
                optimizer_last_status = "failed"
                optimizer_last_error = str(exc)
                logger.exception("manual memory optimizer failed: %s", exc)

        optimizer_last_status = "running"
        optimizer_last_error = None
        optimizer_task = asyncio.create_task(
            _run_optimizer(), name="manual_memory_optimizer"
        )
        return {"status": "started", "message": "Memory optimizer started"}

    @app.get("/api/dashboard/memories")
    def list_memories(
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> MemoryListResponse:
        """列出 Memory2 记忆条目。

        支持按类型、状态、来源、作用域、embedding 有无等过滤。
        """
        items, total = memory_admin.list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return MemoryListResponse(
            items=items,
            total=total,
            page=max(1, page),
            page_size=max(1, min(page_size, 200)),
        )

    @app.get("/api/dashboard/memories/{memory_id:path}")
    def get_memory(
        memory_id: str,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        """获取单条记忆条目。

        异常:
            404: 记忆条目不存在。
        """
        item = memory_admin.get_item_for_dashboard(
            memory_id, include_embedding=include_embedding
        )
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @app.get("/api/dashboard/memories/{memory_id:path}/similar")
    def list_similar_memories(
        memory_id: str,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """查找与指定记忆条目相似的条目。

        异常:
            404: 源记忆条目不存在。
            400: 参数无效。
        """
        try:
            items = memory_admin.find_similar_items_for_dashboard(
                memory_id,
                top_k=top_k,
                memory_type=memory_type,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="memory 不存在"
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc)
            ) from exc
        return {
            "items": items,
            "total": len(items),
            "source_id": memory_id,
        }

    @app.patch("/api/dashboard/memories/{memory_id:path}")
    def update_memory(
        memory_id: str,
        payload: MemoryUpdatePayload,
    ) -> dict[str, Any]:
        """更新记忆条目。

        异常:
            400: 参数无效。
            404: 记忆条目不存在。
        """
        try:
            item = memory_admin.update_item_for_dashboard(
                memory_id,
                status=payload.status,
                extra_json=payload.extra_json,
                source_ref=payload.source_ref,
                happened_at=payload.happened_at,
                emotional_weight=payload.emotional_weight,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @app.delete("/api/dashboard/memories/{memory_id:path}")
    def delete_memory(memory_id: str) -> dict[str, Any]:
        """删除单条记忆条目。

        异常:
            404: 记忆条目不存在。
        """
        deleted = memory_admin.delete_item(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return {"deleted": True, "id": memory_id}

    @app.post("/api/dashboard/memories/batch-delete")
    def delete_memories_batch(
        payload: MemoryBatchDeletePayload,
    ) -> dict[str, Any]:
        """批量删除记忆条目。

        返回:
            {"deleted_count": N}
        """
        deleted_count = memory_admin.delete_items_batch(payload.ids)
        return {"deleted_count": deleted_count}
    
    # ═══════════════════════════════════════════════════════════════
    # Proactive 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/proactive/overview")
    def get_proactive_overview() -> dict[str, Any]:
        """返回 Proactive 系统的统计数据概览。"""
        return _get_proactive_reader().get_overview()

    @app.get("/api/dashboard/proactive/deliveries")
    def list_proactive_deliveries(
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 Proactive 投递记录。"""
        items, total = _get_proactive_reader().list_deliveries(
            session_key=session_key,
            sent_from=sent_from,
            sent_to=sent_to,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/seen_items")
    def list_proactive_seen_items(
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 Proactive 已见事件（去重记录）。"""
        items, total = _get_proactive_reader().list_seen_items(
            source_key=source_key,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/tick_logs")
    def list_proactive_tick_logs(
        session_key: str = "",
        terminal_action: str = "",
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 Proactive tick 执行日志。"""
        items, total = _get_proactive_reader().list_tick_logs(
            session_key=session_key,
            terminal_action=terminal_action,
            started_from=started_from,
            started_to=started_to,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/tick_logs/{tick_id}")
    def get_proactive_tick_log(tick_id: str) -> dict[str, Any]:
        """获取单条 Proactive tick 日志。

        异常:
            404: tick 不存在。
        """
        item = _get_proactive_reader().get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        return item

    @app.delete("/api/dashboard/proactive/seen_items/batch")
    def delete_proactive_seen_items(
        payload: ProactiveDeletePayload,
    ) -> dict[str, Any]:
        """批量删除 Proactive 已见事件。

        异常:
            400: source_key 和 item_ids 均为空。
        """
        try:
            deleted_count = _get_proactive_reader().delete_seen_items(
                source_key=str(payload.source_key or "").strip(),
                item_ids=payload.item_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted_count": deleted_count}

    # ═══════════════════════════════════════════════════════════════
    # Observe 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/observe/turns")
    def list_observe_turns(
        session_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 observe 记录的 turn 数据。

        每个 turn 记录包含用户消息、助手回复、使用工具、检索到的记忆 ID 等。
        """
        items, total = _get_observe_reader().list_turns(
            session_key=session_key,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/observe/tool_calls")
    def list_observe_tool_calls(
        session_key: str = "",
        tool_name: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 observe 记录的 tool_call 数据。

        支持按 session、工具名、执行状态过滤。
        """
        items, total = _get_observe_reader().list_tool_calls(
            session_key=session_key,
            tool_name=tool_name,
            status=status,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }
    
    # ── Observe: RAG Queries ──

    @app.get("/api/dashboard/observe/rag-queries")
    def list_observe_rag_queries(
        session_key: str = "",
        caller: str = "",
        route_decision: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 observe 记录的 RAG 检索事件。

        支持按 session、调用来源、路由决策过滤。
        每条记录包含：检索 query、改写前 query、HyDE 假想条目、
        命中列表（ID、类型、分数、是否注入）、注入数量、路由决策。
        """
        items, total = _get_observe_reader().list_rag_queries(
            session_key=session_key,
            caller=caller,
            route_decision=route_decision,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/observe/rag-queries/{query_id:int}")
    def get_observe_rag_query(query_id: int) -> dict[str, Any]:
        """获取单条 RAG 检索记录。

        异常:
            404: 记录不存在。
        """
        item = _get_observe_reader().get_rag_query(query_id)
        if item is None:
            raise HTTPException(status_code=404, detail="rag query 不存在")
        return item

    # ── Observe: Memory Writes ──

    @app.get("/api/dashboard/observe/memory-writes")
    def list_observe_memory_writes(
        session_key: str = "",
        action: str = "",
        memory_type: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """列出 observe 记录的记忆写入/替换事件。

        支持按 session、操作类型（write/supersede）、记忆类型过滤。
        每条记录包含：操作类型、记忆类型、条目 ID、摘要、被替换 ID 列表。
        """
        items, total = _get_observe_reader().list_memory_writes(
            session_key=session_key,
            action=action,
            memory_type=memory_type,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/observe/memory-writes/{write_id:int}")
    def get_observe_memory_write(write_id: int) -> dict[str, Any]:
        """获取单条记忆写入记录。

        异常:
            404: 记录不存在。
        """
        item = _get_observe_reader().get_memory_write(write_id)
        if item is None:
            raise HTTPException(status_code=404, detail="memory write 不存在")
        return item

    # ── 备份操作 ──

    @app.post("/api/dashboard/backup", status_code=202)
    async def trigger_backup() -> dict[str, Any]:
        """手动触发一次全量数据库备份。

        备份 sessions.db、observe.db、proactive_state.db 三个数据库。
        备份文件存放在 {workspace}/backups/latest/ 目录下，覆盖上一次备份。
        """
        from raven_agent.plugins.builtins.observe.backup import backup_databases

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, backup_databases, workspace)
        return {
            "status": "completed",
            "backups": results,
        }

    # ═══════════════════════════════════════════════════════════════
    # Runtime 路由组
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/dashboard/runtime/status")
    def get_runtime_status() -> RuntimeStatusResponse:
        """返回 raven-agent 运行时整体状态。

        这是一个健康检查 + 状态聚合端点，汇总各子系统的运行情况。
        响应包含 scheduler、background_runtime、proactive、mcp、vision、audio 等子系统的状态。

        注意：此端点始终免 API Key 认证，可用作健康检查探针。
        """
        # 注意：这些状态字段来自调用方通过 app.state 注入的运行时对象。
        # 实现见第 4.4 节——AppRuntime 集成。
        scheduler_info = getattr(app.state, "scheduler_info", None)
        background_info = getattr(app.state, "background_info", None)
        proactive_info = getattr(app.state, "proactive_info", None)
        mcp_info = getattr(app.state, "mcp_info", None)
        vision_info = getattr(app.state, "vision_info", None)
        audio_info = getattr(app.state, "audio_info", None)

        return RuntimeStatusResponse(
            status="running",
            scheduler=scheduler_info or {"enabled": False},
            background_runtime=background_info or {"enabled": False},
            proactive=proactive_info or {"enabled": False},
            mcp=mcp_info or {"enabled": False},
            vision=vision_info or {"enabled": False},
            audio=audio_info or {"enabled": False},
            session_count=len(store.list_session_keys()),
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 静态文件与插件面板（Ch38 新增）
    # ═══════════════════════════════════════════════════════════════

    # ── 安装 Dashboard 访问日志过滤器 ──
    _install_dashboard_access_log_filter()

    compile_task: asyncio.Task[None] | None = None

    # ── 插件面板路由 ──
    if project_root is not None and plugins_root is not None:
        from raven_agent.api.plugin_panels import register_plugin_panels

        compile_task = register_plugin_panels(app, plugins_root, project_root)

    # ── 静态文件挂载 ──
    if static_dir is not None and static_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(static_dir)),
            name="dashboard-assets",
        )
        logger.info("Dashboard 静态文件已挂载: %s", static_dir)

    # ── 旅行攻略 HTML 输出目录（static/dynamic 模式共用） ──
    if trips_dir is not None:
        trips_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/trips",
            StaticFiles(directory=str(trips_dir), html=True),
            name="trips",
        )
        logger.info("Trips 输出目录已挂载: %s", trips_dir)

    # ── 首页路由 ──
    if static_dir is not None and static_dir.is_dir():
        index_html = static_dir / "index.html"

        @app.get("/")
        def dashboard_index() -> Response:
            """返回 Dashboard SPA 入口 HTML。

            自动注入 app.js 和 styles.css 的版本号（mtime_ns）
            实现 cache-busting。浏览器每次刷新都会获取最新版本。

            输出:
                HTML Response。
            """
            html = index_html.read_text(encoding="utf-8")
            app_js = static_dir / "app.js"
            styles_css = static_dir / "styles.css"
            if app_js.exists():
                app_v = str(int(app_js.stat().st_mtime_ns))
                html = re.sub(
                    r'(/assets/app\.js)(\?[^"]*)?',
                    rf'\1?v={app_v}',
                    html,
                )
            if styles_css.exists():
                css_v = str(int(styles_css.stat().st_mtime_ns))
                html = re.sub(
                    r'(/assets/styles\.css)(\?[^"]*)?',
                    rf'\1?v={css_v}',
                    html,
                )
            return Response(content=html, media_type="text/html")

        logger.info("Dashboard 首页路由已注册: /")

    return app
