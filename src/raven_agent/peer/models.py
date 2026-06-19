"""
peer/models.py —— Peer Agent 数据模型。

三个核心数据结构：
  - AgentSkill：Peer Agent 声明的单项能力
  - AgentCard：Peer Agent 的元数据名片（从 TOML 配置字段构建）
  - PeerProcessConfig：子进程启动参数
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentSkill:
    """Peer Agent 的一项能力声明。

    字段:
        id: 技能唯一标识。
        name: 技能名称。
        description: 技能描述。
        tags: 标签列表（用于 LLM 路由匹配）。
        examples: 使用示例列表。
    """

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """Peer Agent 的名片——从 TOML 配置字段构建。

    字段:
        name: Agent 名称。
        url: Agent A2A 服务端点 URL。
        description: 整体描述。
        skills: 能力列表。

    属性:
        primary_skill: 返回第一个 skill（用于工具描述生成）。
    """

    name: str
    url: str
    description: str = ""
    skills: list[AgentSkill] = field(default_factory=list)

    @property
    def primary_skill(self) -> AgentSkill | None:
        """返回第一个 skill，用于生成工具描述。

        输出:
            AgentSkill 或 None（无 skill 时）。
        """
        return self.skills[0] if self.skills else None


@dataclass(frozen=True)
class PeerProcessConfig:
    """Peer Agent 子进程的启动配置。

    字段:
        name: Agent 名称。
        base_url: A2A 服务 HTTP 端点 URL。
        launcher: 启动命令列表（如 ["uv", "run", "python", "-m", "app"]）。
        cwd: 子进程工作目录；None 表示继承父进程。
        health_path: 健康检查路径，默认 "/health"。
        startup_timeout_s: 启动后等待健康检查的最长时间（秒）。
        shutdown_timeout_s: 终止后等待进程退出的最长时间（秒）。

    注意：description 不在此处——它属于 AgentCard/PeerAgentConfig，
    ProcessManager 只关心进程生命周期，不关心语义描述。
    """

    name: str
    base_url: str
    launcher: list[str]
    cwd: str | None = None
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10