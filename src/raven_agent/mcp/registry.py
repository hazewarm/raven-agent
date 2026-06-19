"""
McpServerRegistry: 管理多个 MCP server 连接，持久化到 mcp_servers.json。

职责:
  - 管理多个 McpClient 的连接生命周期
  - 将远端工具同步注册到 ToolRegistry
  - 持久化 server 配置到 mcp_servers.json
  - 启动时自动重连已保存的 server
  - 关闭时优雅断开所有连接

持久化格式（mcp_servers.json）：
{
  "servers": {
    "calendar": {
      "command": ["python", "/path/to/calendar-mcp/run_server.py"],
      "env": {"GOOGLE_CLIENT_ID": "xxx"},
      "cwd": "/path/to/calendar-mcp"
    }
  }
}
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from raven_agent.mcp.client import McpClient
from raven_agent.mcp.tool import McpToolWrapper
from raven_agent.persistence import load_json, save_json

logger = logging.getLogger(__name__)


class McpServerRegistry:
    """管理 MCP server 连接生命周期，并将工具同步进 ToolRegistry。

    参数:
        config_path: mcp_servers.json 的完整路径。
        tool_registry: raven-agent 的工具注册表，远端工具会注册到其中。
        auto_connect: 启动时自动连接的 server 名称白名单。
            空集合表示连接 mcp_servers.json 中的全部 server（向后兼容）。
            非空时只连接白名单内的 server，其余 server 的配置仍保留在
            mcp_servers.json 中但不会自动连接。
    """

    def __init__(
        self,
        config_path: Path,
        tool_registry: Any,
        auto_connect: tuple[str, ...] = (),
    ) -> None:
        self._config_path = config_path
        self._tool_registry = tool_registry
        self._auto_connect: set[str] = set(auto_connect)
        self._clients: dict[str, McpClient] = {}
        # server_name → 该 server 已注册的工具名列表
        self._server_tools: dict[str, list[str]] = {}
        # server_name → 原始配置（command / env / cwd），用于持久化
        self._server_configs: dict[str, dict[str, Any]] = {}
        self._connect_task: asyncio.Task[None] | None = None

    # ── 生命周期 ──────────────────────────────────────────────────

    async def load_and_connect_all(self) -> None:
        """启动时读取持久化配置，按 auto_connect 白名单重连 server。

        如果 auto_connect 非空，只连接白名单内的 server；为空则全量连接
        （向后兼容）。

        收集所有连接失败的 server 名称，完成后输出汇总日志——
        避免关键 server 静默失败，用户和 AI 都不知道工具缺失。

        输出:
            None。连接失败只记日志不抛异常——单个 server 失败不应影响 Agent 启动。
        """

        configs = self._load_raw_configs()
        if not configs:
            return

        # 白名单过滤：非空时只连接白名单内的 server
        if self._auto_connect:
            filtered = {
                k: v for k, v in configs.items() if k in self._auto_connect
            }
            skipped = set(configs) - set(filtered)
            if skipped:
                logger.info(
                    "[mcp] auto_connect 白名单生效，跳过 %d 个 server：%s",
                    len(skipped), ", ".join(sorted(skipped)),
                )
            configs = filtered

        if not configs:
            return

        failed: list[str] = []

        async def connect_one(name: str, cfg: dict[str, Any]) -> None:
            try:
                await self._connect(
                    name,
                    cfg.get("command"),
                    cfg.get("env"),
                    cfg.get("cwd"),
                    cfg.get("url", ""),
                )
            except Exception as e:
                logger.error("[mcp] 重连 %r 失败: %s", name, e)
                failed.append(name)

        await asyncio.gather(
            *(connect_one(name, cfg) for name, cfg in configs.items())
        )

        succeeded = len(configs) - len(failed)
        if failed:
            logger.warning(
                "[mcp] 重连完成：%d 成功，%d 失败（%s）。"
                "失败的 server 工具在对话中不可用。",
                succeeded, len(failed), ", ".join(failed),
            )
        else:
            logger.info(
                "[mcp] 重连完成：%d/%d 个 server 全部连接成功",
                succeeded, len(configs),
            )

    def start_connect_all_background(self) -> None:
        """后台重连所有 server，不阻塞主服务启动。

        MCP server 子进程启动可能很慢（需要初始化数据库连接、加载配置等），
        在后台 asyncio Task 中执行可以避免 Agent 服务长时间无法接受消息。

        输出:
            None。
        """
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(
                self.load_and_connect_all(),
                name="mcp_connect_all",
            )

    async def shutdown(self) -> None:
        """关闭所有 MCP server 连接。

        先取消后台重连任务（如果还在跑），再逐一断开所有已连接的 server。

        输出:
            None。
        """
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        clients = list(self._clients.values())
        self._clients.clear()
        self._server_tools.clear()
        await asyncio.gather(
            *(client.disconnect() for client in clients),
            return_exceptions=True,
        )

    # ── 管理 API ──────────────────────────────────────────────────
    # 面向大模型（LLM Tool 调用）和用户的公共交互层

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        url: str = "",
    ) -> str:
        """连接并注册一个新的 MCP server。

        输入:
            name: server 唯一短名称（如 "calendar"）。
            command: 启动命令列表。
            env: 可选的额外环境变量。
            cwd: 可选的工作目录。
            url: MCP server 的 HTTP/SSE 端点 URL。
        输出:
            描述注册结果的字符串（含已注册的工具名列表）。
        """
        if name in self._clients:
            return (
                f"MCP server {name!r} 已存在。"
                "如需更新，请先 mcp_remove 再重新添加。"
            )
        try:
            tool_names = await self._connect(name, command, env, cwd, url)
        except Exception as e:
            return f"连接 MCP server {name!r} 失败：{e}"
        self._save()
        return (
            f"已连接 MCP server {name!r}，"
            f"注册了 {len(tool_names)} 个工具：\n"
            + "\n".join(f"- {n}" for n in tool_names)
        )

    async def remove(self, name: str) -> str:
        """注销并断开一个已注册的 MCP server。

        会同时从 ToolRegistry 中移除该 server 的所有工具。

        输入:
            name: server 名称。

        输出:
            描述注销结果的字符串。
        """
        if name not in self._clients:
            return (
                f"MCP server {name!r} 不存在，"
                f"当前已注册：{list(self._clients.keys()) or '无'}"
            )
        # 先注销工具，再断开连接
        for tool_name in self._server_tools.pop(name, []):
            self._tool_registry.unregister(tool_name)
        await self._clients.pop(name).disconnect()
        self._save()
        return f"已注销 MCP server {name!r}。"

    def list_servers(self) -> str:
        """列出当前所有已注册的 MCP server 及其工具。

        输出:
            格式化的 server 列表字符串。
        """
        if not self._clients:
            return "当前没有已注册的 MCP server。"
        lines = []
        for name in self._clients:
            tools = self._server_tools.get(name, [])
            lines.append(
                f"- {name}（{len(tools)} 个工具）："
                f"{', '.join(tools) or '无'}"
            )
        return "\n".join(lines)

    # ── 内部方法 ──────────────────────────────────────────────────

    def connected_server_names(self) -> set[str]:
        """返回当前已连接的 MCP server 名称集合。

        输出:
            set[str]。
        """
        return set(self._clients.keys())


    def tool_names_for_server(self, name: str) -> list[str]:
        """返回指定 server 已注册的全部工具名称。

        输入:
            name: server 名称。

        输出:
            工具名称列表。server 未连接时返回空列表。
        """
        return list(self._server_tools.get(name, []))


    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """在指定 MCP server 上调用一个工具。

        输入:
            server: server 名称。
            tool_name: 远端工具名称（不含 mcp_{server}__ 前缀）。
            arguments: 工具参数。

        输出:
            McpClient.call_tool() 的返回值（可能为 str / list / None）。

        异常:
            RuntimeError: server 未连接时抛出。
        """
        client = self._clients.get(server)
        if client is None:
            raise RuntimeError(f"MCP server {server!r} 未连接")
        return await client.call_tool(tool_name, arguments, **kwargs)

    async def _connect(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None,
        cwd: str | None = None,
        url: str = "",
    ) -> list[str]:
        """创建 McpClient、连接、获取工具列表、注册到 ToolRegistry。

        输入:
            name: server 名称。
            command: 启动命令列表。
            env: 额外环境变量。
            cwd: 工作目录。

        输出:
            已注册的工具名列表。
        """
        if url:
            # HTTP/SSE 模式：直接传 URL 字符串，McpClient._prepare_server_source()
            # 自动识别 https:// 前缀并创建 StreamableHttpTransport。
            client = McpClient(url, env=env)
        elif cwd:
            # stdio 模式（含工作目录）
            client = McpClient({
                "transport": "stdio",
                "command": command[0],
                "args": command[1:],
                "env": env,
                "cwd": cwd,
            }, env=env)
        else:
            # stdio 模式（无工作目录）
            client = McpClient(command, env=env)
        await client.connect()
        tool_infos = await client.list_tools()
        tool_names: list[str] = []
        for info in tool_infos:
            wrapper = McpToolWrapper(client, name, info)
            self._tool_registry.register(
                wrapper,
                risk="external-side-effect",
                source_type="mcp",
                source_name=name,
            )
            tool_names.append(wrapper.name)
        self._clients[name] = client
        self._server_tools[name] = tool_names
        self._server_configs[name] = {
            "command": command,
            "env": env or {},
            "cwd": cwd,
            "url": url,
        }
        return tool_names

    def _load_raw_configs(self) -> dict[str, Any]:
        """从 mcp_servers.json 加载 server 配置。

        输出:
            {"server_name": {"command": [...], "env": {...}, "cwd": "..."}} 字典。
            文件不存在或损坏时返回空字典。
        """
        data = load_json(self._config_path, default={})
        if not isinstance(data, dict):
            return {}
        return data.get("servers", {})

    def _save(self) -> None:
        """将当前已连接的 server 配置持久化到 mcp_servers.json。

        使用 _server_configs 而非 client 的属性——fastmcp 的 Client
        不暴露原始 command/env/cwd，所以用 add() 时保存的原始配置写回。

        使用 persistence.save_json 做原子写入，避免进程崩溃导致文件损坏。
        """
        save_json(self._config_path, {"servers": dict(self._server_configs)})