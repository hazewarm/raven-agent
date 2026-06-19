"""
PresenceStore —— 跨 session 的用户在线心跳管理。

底层复用 SessionStore 的 SQLite 连接（sessions.db 中的 presence 表）。
不自己开数据库——一个 Agent 只应有一个 SQLite 写入者。
"""

from __future__ import annotations

import logging
from datetime import datetime

from raven_agent.session_store import SessionStore

logger = logging.getLogger(__name__)


def _parse_iso(raw: str | None) -> datetime | None:
    """将 ISO 时间字符串解析为 aware datetime。

    输入:
        raw: ISO 格式字符串或 None。

    输出:
        datetime 对象；解析失败或 raw 为 None 时返回 None。
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _utcnow() -> datetime:
    """返回当前 UTC 时间（aware）。

    输出:
        datetime 对象。
    """
    return datetime.now().astimezone()


class PresenceStore:
    """跨 session 的用户在线心跳持久化。

    record_user_message() 在每次收到用户消息时更新心跳；
    record_proactive_sent() 在每次 Agent 主动推送后记录时间戳。

    参数:
        store: SessionStore 实例（复用其 SQLite 连接和 presence 表）。
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        logger.info("[presence] 初始化完成 db=%s", store.db_path)

    # ── 写入 ───────────────────────────────────────────────────────

    def record_user_message(
        self, session_key: str, now: datetime | None = None
    ) -> None:
        """记录用户消息到达时间（更新心跳）。

        每次收到用户消息时调用——这是 Proactive 系统判断
        "用户是否在线"的核心依据。

        输入:
            session_key: 会话 key。
            now: 当前时间；默认 UTC now。

        输出:
            None。
        """
        ts = (now or _utcnow()).isoformat()
        self._store.update_presence(session_key, last_user_at=ts)
        logger.debug("[presence] 心跳更新 session=%s ts=%s", session_key, ts)

    def record_proactive_sent(
        self, session_key: str, now: datetime | None = None
    ) -> None:
        """记录 Agent 主动推送时间。

        用于计算"上次推送后用户是否回复过"——如果用户在上次推送后
        发了消息，说明推送没有打扰用户。

        输入:
            session_key: 会话 key。
            now: 当前时间；默认 UTC now。

        输出:
            None。
        """
        ts = (now or _utcnow()).isoformat()
        self._store.update_presence(session_key, last_proactive_at=ts)
        logger.debug("[presence] 主动消息记录 session=%s ts=%s", session_key, ts)

    # ── 读取 ───────────────────────────────────────────────────────

    def get_last_user_at(self, session_key: str) -> datetime | None:
        """获取指定 session 的用户最后活跃时间。

        输入:
            session_key: 会话 key。

        输出:
            aware datetime；无记录时返回 None。
        """
        row = self._store.get_presence(session_key)
        return _parse_iso(row.get("last_user_at"))

    def get_last_proactive_at(self, session_key: str) -> datetime | None:
        """获取指定 session 的 Agent 最后主动推送时间。

        输入:
            session_key: 会话 key。

        输出:
            aware datetime；无记录时返回 None。
        """
        row = self._store.get_presence(session_key)
        return _parse_iso(row.get("last_proactive_at"))

    def most_recent_user_at(self) -> datetime | None:
        """获取所有 session 中最近一次用户活跃时间。

        用于全局心跳检测——即使用户在当前 target session 静默，
        只要在其他 session 有活动，Proactive 系统就能感知。

        输出:
            aware datetime；全站无记录时返回 None。
        """
        return _parse_iso(self._store.most_recent_user_at())

    def get_all_sessions(self) -> dict[str, dict[str, datetime | None]]:
        """列出所有 session 的 presence 快照。

        输出:
            {session_key: {"last_user_at": datetime|None,
                           "last_proactive_at": datetime|None}} 字典。
        """
        rows = self._store.list_presence()
        return {
            key: {
                "last_user_at": _parse_iso(item.get("last_user_at")),
                "last_proactive_at": _parse_iso(item.get("last_proactive_at")),
            }
            for key, item in rows.items()
        }