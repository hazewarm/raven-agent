"""Observe 备份模块测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestBackup:
    """验证备份模块的核心行为。"""

    @pytest.fixture
    def workspace(self):
        """创建包含模拟数据库文件的临时工作区。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)

            # 创建模拟 sessions.db
            (ws / "sessions.db").write_bytes(b"mock sessions db")

            # 创建模拟 observe.db
            obs_dir = ws / "observe"
            obs_dir.mkdir()
            (obs_dir / "observe.db").write_bytes(b"mock observe db")

            yield ws

    def test_backup_creates_files(self, workspace):
        from raven_agent.plugins.builtins.observe.backup import backup_databases

        results = backup_databases(workspace)
        assert results["sessions"] is not None
        assert results["observe"] is not None
        # proactive_state.db 不存在时返回 None
        assert results["proactive"] is None

        # 验证备份文件存在
        sessions_backup = Path(results["sessions"])  # type: ignore[arg-type]
        assert sessions_backup.exists()
        assert sessions_backup.stat().st_size > 0

    def test_backup_dir_structure(self, workspace):
        from raven_agent.plugins.builtins.observe.backup import backup_databases

        backup_databases(workspace)
        expected_dir = workspace / "backups" / "latest"
        assert expected_dir.exists()
        files = list(expected_dir.glob("*.db"))
        assert len(files) >= 1  # 至少有一个备份文件

    def test_backup_overwrites(self, workspace):
        """第二次备份覆盖第一次的备份文件。"""
        from raven_agent.plugins.builtins.observe.backup import backup_databases

        r1 = backup_databases(workspace)
        r2 = backup_databases(workspace)
        # 两次备份都应成功
        assert r1["sessions"] is not None
        assert r2["sessions"] is not None
        # 固定目录覆盖：两次备份指向同一个文件
        assert r1["sessions"] == r2["sessions"]

    def test_backup_keeps_only_one_copy(self, workspace):
        """备份目录中只保留每个库的一份文件，不堆积。"""
        from raven_agent.plugins.builtins.observe.backup import backup_databases

        # 执行多次备份
        for _ in range(3):
            backup_databases(workspace)

        latest_dir = workspace / "backups" / "latest"
        db_files = list(latest_dir.glob("*.db"))
        # 只有 sessions.db 和 observe.db（proactive 不存在）
        names = {f.name for f in db_files}
        assert names <= {"sessions.db", "observe.db", "proactive_state.db"}