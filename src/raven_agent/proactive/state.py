"""
proactive/state.py —— Proactive 运行期状态持久化（SQLite）。

跨进程重启保留的 Proactive 关键状态：
  - seen_items：已见外部事件 ID，用于去重（"这条新闻推过了"）
  - deliveries：已发送的主动消息记录，用于疲劳度 + 重复检测

与第 8 章 SessionStore 的关系：
  ProactiveStateStore 有自己独立的 SQLite 数据库（proactive_state.db），
  原因是 proactive 状态的读写模式完全不同——高频 insert + 定期 cleanup，
  不适合混入 session 表（轮次插入 + 历史查询）。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """返回当前 UTC 时间（aware）。"""
    return datetime.now().astimezone()


def _parse_iso(raw: str | None) -> datetime | None:
    """解析 ISO 时间字符串为 aware datetime。

    输入:
        raw: ISO 字符串或 None。

    输出:
        aware datetime；解析失败或 raw 为空返回 None。
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class ProactiveStateStore:
    """Proactive 运行期跨重启状态管理器。

    参数:
        db_path: SQLite 数据库文件路径。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        logger.info("[proactive.state] 初始化完成 db=%s", self.db_path)

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    # ── seen_items（外部事件去重）─────────────────────────────────

    def is_item_seen(
        self,
        source_key: str,
        item_id: str,
        ttl_hours: int,
        now: datetime | None = None,
    ) -> bool:
        """判断一个 item 在 ttl_hours 内是否已被见过。

        输入:
            source_key: 来源 key（通常是 server 名称）。
            item_id: 条目唯一标识。
            ttl_hours: 去重有效期（小时）。
            now: 当前时间。

        输出:
            True 表示该条目已被见过且未过期。
        """
        now = now or _utcnow()
        cutoff = now - timedelta(hours=max(ttl_hours, 1))
        with self._lock:
            row = self._db.execute(
                "SELECT seen_at FROM seen_items WHERE source_key = ? AND item_id = ?",
                (source_key, item_id),
            ).fetchone()
        if row is None:
            return False
        ts = _parse_iso(str(row["seen_at"]))
        if ts is None or ts < cutoff:
            logger.info(
                "[proactive.state] item 过期 source=%s item_id=%s",
                source_key, item_id[:16],
            )
            return False
        return True

    def mark_items_seen(
        self,
        entries: list[tuple[str, str]],
        now: datetime | None = None,
    ) -> None:
        """标记一批 item 为已见。

        输入:
            entries: (source_key, item_id) 元组列表。
            now: 当前时间。

        输出:
            None。
        """
        if not entries:
            return
        now = now or _utcnow()
        ts = now.isoformat()
        params = [(sk, iid, ts) for sk, iid in entries]
        with self._lock:
            self._db.executemany(
                """
                INSERT INTO seen_items(source_key, item_id, seen_at)
                VALUES(?, ?, ?)
                ON CONFLICT(source_key, item_id) DO UPDATE SET seen_at = excluded.seen_at
                """,
                params,
            )
            self._db.commit()
        logger.info("[proactive.state] 已标记已见条目 count=%d", len(entries))

    # ── deliveries（推送去重 + 疲劳度）───────────────────────────

    def is_delivery_duplicate(
        self,
        session_key: str,
        delivery_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> bool:
        """判断一条消息在 window_hours 内是否已经推送过。

        输入:
            session_key: 目标 session key。
            delivery_key: 推送内容唯一标识。
            window_hours: 去重窗口（小时）。
            now: 当前时间。

        输出:
            True 表示已推送过（且在窗口内）。
        """
        now = now or _utcnow()
        cutoff = now - timedelta(hours=max(window_hours, 1))
        with self._lock:
            row = self._db.execute(
                "SELECT sent_at FROM deliveries WHERE session_key = ? AND delivery_key = ?",
                (session_key, delivery_key),
            ).fetchone()
        if row is None:
            return False
        ts = _parse_iso(str(row["sent_at"]))
        return ts is not None and ts >= cutoff

    def mark_delivery(
        self,
        session_key: str,
        delivery_key: str,
        content: str = "",
        now: datetime | None = None,
    ) -> None:
        """记录一次主动推送。

        输入:
            session_key: 目标 session key。
            delivery_key: 推送内容唯一标识。
            content: 推送的消息文本。
            now: 当前时间。

        输出:
            None。
        """
        now = now or _utcnow()
        ts = now.isoformat()
        with self._lock:
            self._db.execute(
                """
                INSERT INTO deliveries(session_key, delivery_key, content, sent_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_key, delivery_key) DO UPDATE SET sent_at = excluded.sent_at
                """,
                (session_key, delivery_key, content[:500], ts),
            )
            self._db.commit()

    def count_deliveries_in_window(
        self,
        session_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> int:
        """统计过去 window_hours 小时内对目标 session 的推送次数。

        输入:
            session_key: 目标 session key。
            window_hours: 统计窗口（小时）。
            now: 当前时间。

        输出:
            推送次数整数。
        """
        now = now or _utcnow()
        cutoff = now - timedelta(hours=window_hours)
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM deliveries WHERE session_key = ? AND sent_at >= ?",
                (session_key, cutoff.isoformat()),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def recent_deliveries(
        self,
        session_key: str,
        n: int = 5,
    ) -> list[dict[str, str]]:
        """获取最近 n 条主动推送消息（用于 Judge 语义去重）。

        输入:
            session_key: 目标 session key。
            n: 最多返回条数。

        输出:
            [{"content": "...", "sent_at": "..."}, ...] 列表。
        """
        with self._lock:
            rows = self._db.execute(
                """
                SELECT content, sent_at
                FROM deliveries
                WHERE session_key = ?
                ORDER BY sent_at DESC
                LIMIT ?
                """,
                (session_key, n),
            ).fetchall()
        return [
            {"content": str(r["content"]), "sent_at": str(r["sent_at"])}
            for r in rows
        ]

    # ── cleanup ────────────────────────────────────────────────────

    def cleanup(
        self,
        seen_ttl_hours: int = 168,
        delivery_ttl_hours: int = 720,
    ) -> None:
        """清理过期数据。

        输入:
            seen_ttl_hours: seen_items 有效期（小时），默认 7 天。
            delivery_ttl_hours: deliveries 有效期（小时），默认 30 天。

        输出:
            None。
        """
        now = _utcnow()
        seen_cutoff = (now - timedelta(hours=max(seen_ttl_hours, 1))).isoformat()
        delivery_cutoff = (
            now - timedelta(hours=max(delivery_ttl_hours, 1))
        ).isoformat()
        with self._lock:
            removed_seen = self._db.execute(
                "DELETE FROM seen_items WHERE seen_at < ?", (seen_cutoff,),
            ).rowcount
            removed_delivery = self._db.execute(
                "DELETE FROM deliveries WHERE sent_at < ?", (delivery_cutoff,),
            ).rowcount
            self._db.commit()
            # WAL 文件在 24/7 运行的守护进程中会持续追加——连接从不关闭，
            # 自动 checkpoint 的触发时机不可控。cleanup 通常伴随大量 DELETE，
            # 显式 TRUNCATE 可以立即回收 WAL 空间，避免在树莓派 SD 卡等
            # 低存储设备上意外占满磁盘。
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info(
            "[proactive.state] cleanup 完成 removed_seen=%d removed_delivery=%d",
            removed_seen, removed_delivery,
        )

    # ── schema ─────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """初始化 SQLite 表结构。"""
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS seen_items (
                source_key TEXT NOT NULL,
                item_id TEXT NOT NULL,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (source_key, item_id)
            );

            CREATE TABLE IF NOT EXISTS deliveries (
                session_key TEXT NOT NULL,
                delivery_key TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                sent_at TEXT NOT NULL,
                PRIMARY KEY (session_key, delivery_key)
            );
            CREATE INDEX IF NOT EXISTS idx_deliveries_session_sent
                ON deliveries(session_key, sent_at);
        """)
        self._db.commit()