"""
mcp_add / mcp_remove / mcp_list：AI 用于动态管理 MCP server 的三个工具。
"""

from __future__ import annotations

from typing import Any

from raven_agent.tools.base import Tool


class McpAddTool(Tool):
    """连接并注册一个 MCP server。

    参数:
        registry: McpServerRegistry 实例。
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    name = "mcp_add"
    description = (
        "连接并注册一个 MCP server。"
        "command 是可选的启动命令列表（适用于本地 MCP server），env 是可选的额外环境变量，url 是 MCP server 的 HTTP/SSE 端点 URL（适用于远程 MCP server）。"
        "连接成功后，该 server 的所有工具立即可用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "给这个 MCP server 起一个唯一短名称，如 'calendar'",
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "启动命令列表，如 "
                    "['python', '/home/user/.raven/mcp/calendar-mcp/run_server.py']"
                    "与 url 互斥——如果提供了 command，就不需要 url。"
                ),
            },
            "env": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "可选的额外环境变量，如 {'GOOGLE_CLIENT_ID': 'xxx'}",
            },
            "cwd": {
                "type": "string",
                "description": "可选的工作目录。MCP server 子进程将在这个目录下启动。",
            },
            "url": {
                "type": "string",
                "description": (
                    "MCP server 的 HTTP/SSE 端点 URL，如 'https://mcp.amap.com/mcp?key=xxx'。"
                    "与 command 互斥——如果提供了 url，就不需要 command。"
                ),
            },
        },
        "required": ["name"],
    }

    async def execute(
        self,
        name: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        url: str | None = None,
        **_: Any,
    ) -> str:
        """连接并注册 MCP server。

        输入:
            name: server 唯一短名称。
            command: 启动命令列表。
            env: 可选的额外环境变量。
            cwd: 可选的工作目录。
            url: MCP server 的 HTTP/SSE 端点 URL。

        输出:
            描述注册结果的字符串。
        """
        return await self._registry.add(name, command, env, cwd, url)


class McpRemoveTool(Tool):
    """注销并断开一个已注册的 MCP server。

    参数:
        registry: McpServerRegistry 实例。
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    name = "mcp_remove"
    description = "注销并断开一个已注册的 MCP server，同时移除其所有工具。"
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "要注销的 MCP server 名称",
            },
        },
        "required": ["name"],
    }

    async def execute(self, name: str, **_: Any) -> str:
        """注销 MCP server。

        输入:
            name: 要注销的 server 名称。

        输出:
            描述注销结果的字符串。
        """
        return await self._registry.remove(name)


class McpListTool(Tool):
    """列出当前所有已注册的 MCP server 及其工具。

    参数:
        registry: McpServerRegistry 实例。
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    name = "mcp_list"
    description = "列出当前所有已注册的 MCP server 及其工具名称。"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        """列出所有 MCP server。

        输出:
            格式化的 server 列表字符串。
        """
        return self._registry.list_servers()