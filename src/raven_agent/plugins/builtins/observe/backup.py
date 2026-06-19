"""数据库每日自动备份模块。

每天执行一次全量备份，覆盖前一天的备份。
备份范围：sessions.db、observe.db、proactive_state.db。

输入:
    workspace: 工作区根目录（Path）。

输出:
    backup_databases() — 执行一次全量备份。
    schedule_backup()  — 启动每日备份后台任务。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("observe.backup")

_BACKUP_DIR = "backups/latest"


def backup_databases(workspace: Path) -> dict[str, str | None]:
    """执行一次全量备份，覆盖上一次备份。

    输入:
        workspace: 工作区根目录。

    输出:
        字典，键为数据库名（"sessions" / "observe" / "proactive"），
        值为备份文件路径的字符串，或 None（源文件不存在时）。
    """
    import sqlite3

    dest_dir = workspace / _BACKUP_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    db_names = {
        "sessions": workspace / "sessions.db",
        "observe": workspace / "observe" / "observe.db",
        "proactive": workspace / "proactive_state.db",
    }

    results: dict[str, str | None] = {}
    for name, src_path in db_names.items():
        if not src_path.exists():
            results[name] = None
            continue

        # WAL checkpoint —— 确保主文件包含最新数据
        try:
            conn = sqlite3.connect(str(src_path))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception:
            logger.warning("WAL checkpoint failed for %s, proceeding anyway", src_path)

        dest = dest_dir / f"{name}.db"
        shutil.copy2(str(src_path), str(dest))
        results[name] = str(dest)
        logger.info(
            "backup: %s -> %s (%d bytes)",
            src_path.name, dest.name, dest.stat().st_size,
        )

    return results


async def schedule_backup(workspace: Path) -> None:
    """启动每日备份后台任务。

    每 24 小时执行一次全量备份，覆盖前一天的备份文件。

    输入:
        workspace: 工作区根目录。

    输出:
        None。永不返回（作为独立 asyncio Task 运行）。
        被取消时静默退出。
    """
    INTERVAL = 24 * 3600
    logger.info("daily backup scheduler started")
    try:
        while True:
            await asyncio.sleep(INTERVAL)
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, backup_databases, workspace)
            except Exception:
                logger.exception("daily backup failed")
    except asyncio.CancelledError:
        logger.info("daily backup scheduler stopped")