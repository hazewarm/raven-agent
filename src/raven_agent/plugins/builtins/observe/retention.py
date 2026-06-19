from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from raven_agent.plugins.builtins.observe.db import open_db

logger = logging.getLogger("observe.retention")

_RETENTION_DAYS = {
    "turns": 180,           # 对话记录
    "tool_calls": 90,       # 工具调用记录
    "rag_queries": 90,      # RAG检索日志
    "memory_writes": 180,   # 记忆变更轨迹
}
_STAMP_FILE = ".last_cleanup"


def _stamp_path(db_path: Path) -> Path:
    """返回淘汰时间戳文件路径。

    输入:
        db_path: observe 数据库路径。

    输出:
        .last_cleanup 文件路径。
    """

    return db_path.parent / _STAMP_FILE


def _should_run(db_path: Path) -> bool:
    """判断距上次淘汰是否已超过 24 小时。

    输入:
        db_path: observe 数据库路径。

    输出:
        True 表示应执行淘汰。
    """

    stamp = _stamp_path(db_path)
    if not stamp.exists():
        return True
    import time

    age_hours = (time.time() - stamp.stat().st_mtime) / 3600
    return age_hours >= 24


def _run_cleanup(db_path: Path) -> None:
    """执行一次数据淘汰。

    输入:
        db_path: observe 数据库路径。

    输出:
        None。error 不为空的行永久保留。
    """

    conn = open_db(db_path)
    try:
        deleted: dict[str, int] = {}
        with conn:
            for table, days in _RETENTION_DAYS.items():
                cutoff = f"datetime('now', '-{days} days')"
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE ts < {cutoff} AND error IS NULL"
                )
                deleted[table] = cur.rowcount
        logger.info("observe retention done: %s", deleted)
        _stamp_path(db_path).write_text("ok", encoding="utf-8")
    except Exception:
        logger.exception("observe retention failed")
    finally:
        conn.close()


async def run_retention_if_needed(db_path: Path) -> None:
    """在后台执行 observe 数据淘汰。

    输入:
        db_path: observe 数据库路径。

    输出:
        None。数据库不存在或未到周期时直接返回。
    """

    if not db_path.exists():
        return
    if not _should_run(db_path):
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_cleanup, db_path)