from __future__ import annotations

from raven_agent.workspace import Workspace


def test_workspace_exposes_runtime_paths(tmp_path) -> None:
    """测试 Workspace 会暴露 sessions 与 memory 路径。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    workspace = Workspace(tmp_path / ".raven")

    assert workspace.root == tmp_path / ".raven"
    assert workspace.sessions_dir == tmp_path / ".raven" / "sessions"
    assert workspace.memory_dir == tmp_path / ".raven" / "memory"
    assert workspace.memory2_dir == tmp_path / ".raven" / "memory2"
    assert workspace.memory2_db_file == tmp_path / ".raven" / "memory2" / "memory2.db"


def test_workspace_ensure_creates_base_directories(tmp_path) -> None:
    """测试 ensure 会创建 workspace 基础目录。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    workspace = Workspace(tmp_path / ".raven")

    workspace.ensure()

    assert workspace.root.exists()
    assert workspace.sessions_dir.exists()
    assert workspace.memory_dir.exists()
    assert workspace.memory2_dir.exists()