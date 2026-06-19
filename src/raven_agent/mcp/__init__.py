"""raven-agent MCP (Model Context Protocol) 模块。

基于 fastmcp 库，提供 MCP server 的连接管理、工具包装和 AI 管理能力。

包含:
  - McpClient: 基于 fastmcp.Client 的多传输客户端（stdio/HTTP/SSE/Memory）
  - McpServerRegistry: 多 server 连接管理 + 工具同步 + 持久化
  - McpToolWrapper: 远端工具 → 本地 Tool 适配器
  - McpAddTool / McpRemoveTool / McpListTool: AI 管理工具
"""

from raven_agent.mcp.client import McpClient
from raven_agent.mcp.manage_tools import McpAddTool, McpListTool, McpRemoveTool
from raven_agent.mcp.registry import McpServerRegistry
from raven_agent.mcp.tool import McpToolWrapper

__all__ = [
    "McpAddTool",
    "McpClient",
    "McpListTool",
    "McpRemoveTool",
    "McpServerRegistry",
    "McpToolWrapper",
]