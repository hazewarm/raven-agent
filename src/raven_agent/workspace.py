from __future__ import annotations

from pathlib import Path


class Workspace:
    """raven-agent 的本地运行工作区。

    参数:
        root: 工作区根目录，例如 .raven。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def sessions_dir(self) -> Path:
        """返回 SessionStore 使用的目录。

        返回:
            workspace 下的 sessions 目录路径。
        """

        return self.root / "sessions"
    
    @property
    def sessions_db_file(self) -> Path:
        """返回 SQLite SessionStore 数据库文件路径。

        参数:
            无。

        返回:
            workspace 下的 sessions.db 路径。
        """

        return self.root / "sessions.db"

    @property
    def memory_dir(self) -> Path:
        """返回 Markdown memory 使用的目录。

        返回:
            workspace 下的 memory 目录路径。
        """

        return self.root / "memory"
    
    @property
    def memory2_dir(self) -> Path:
        """返回 Memory2 使用的目录。

        返回:
            workspace 下的 memory2 目录路径。
        """

        return self.root / "memory2"

    @property
    def memory2_db_file(self) -> Path:
        """返回 Memory2 SQLite 数据库文件路径。

        返回:
            workspace 下的 memory2/memory2.db 路径。
        """

        return self.memory2_dir / "memory2.db"
    
    @property
    def schedules_file(self) -> Path:
        """返回 SchedulerService 的 JSON 持久化文件路径。

        返回:
            workspace 下的 schedules.json 路径。
        """

        return self.root / "schedules.json"

    @property
    def mcp_servers_file(self) -> Path:
        """返回 MCP server 持久化配置文件的路径。

        返回:
            workspace 下的 mcp_servers.json 路径。
        """

        return self.root / "mcp_servers.json"
    
    @property
    def drift_dir(self) -> Path:
        """返回 Drift 工作目录路径。

        返回:
            workspace 下的 drift 目录路径。
        """
        return self.root / "drift"

    @property
    def proactive_sources_file(self) -> Path:
        """返回 proactive_sources.json 的路径。

        输出:
            workspace 下的 proactive_sources.json 路径。
        """
        return self.root / "proactive_sources.json"


    @property
    def proactive_state_db_file(self) -> Path:
        """返回 ProactiveStateStore SQLite 数据库文件路径。

        输出:
            workspace 下的 proactive_state.db 路径。
        """
        return self.root / "proactive_state.db"
    
    @property
    def peer_agents_dir(self) -> Path:
        """返回 Peer Agent 子进程日志目录。

        输出:
            workspace 下的 peer_agents 目录路径。
        """
        return self.root / "peer_agents"

    @property
    def subagents_dir(self) -> Path:
        """返回本地 SubAgent 任务目录根路径。

        输出:
            workspace 下的 subagent-runs 目录路径。
        """
        return self.root / "subagent-runs"
    
    def ensure(self) -> None:
        """确保 workspace 的基础目录存在。

        返回:
            None。
        """

        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory2_dir.mkdir(parents=True, exist_ok=True)
        # 不要初始化 schedules_file

        self.drift_dir.mkdir(parents=True, exist_ok=True)
        self.peer_agents_dir.mkdir(parents=True, exist_ok=True)
        self.subagents_dir.mkdir(parents=True, exist_ok=True)