"""
peerAgent/travel-planner/mcp_manager.py —— MCP 工具管理器。

通过 fastmcp.Client 管理 Amap + XHS 两个 MCP server 子进程。
工具 schema 缓存后复用，避免每个 ReAct 轮次重复 list_tools。
amap-mcp-server 偶发输出非法 JSON 时自动重连并重试。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from config import AmapConfig, XHSConfig

logger = logging.getLogger(__name__)


def _get_tool_name(tool: Any) -> str:
    if hasattr(tool, "name"):
        return tool.name
    if isinstance(tool, dict):
        return tool.get("name", "unknown")
    return str(tool)


def _get_tool_description(tool: Any) -> str:
    if hasattr(tool, "description"):
        return tool.description or ""
    if isinstance(tool, dict):
        return tool.get("description", "")
    return ""


def _get_tool_input_schema(tool: Any) -> dict[str, Any]:
    for attr in ("inputSchema", "input_schema"):
        if hasattr(tool, attr):
            schema = getattr(tool, attr)
            if isinstance(schema, dict):
                return schema
            if hasattr(schema, "model_dump"):
                return schema.model_dump(mode="json")
    return {"type": "object", "properties": {}}


def _parse_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content") and result.content:
        texts: list[str] = []
        for c in result.content:
            if hasattr(c, "text"):
                texts.append(str(c.text))
            elif hasattr(c, "data"):
                texts.append(str(c.data))
            else:
                texts.append(str(c))
        return "\n".join(texts)
    return str(result) if result is not None else "工具返回空结果"


async def _start_client(command: tuple[str, ...], env: dict[str, str]) -> Client:
    """启动一个 MCP server 子进程并完成握手。

    输入:
        command: 启动命令 (如 ("uvx", "amap-mcp-server"))。
        env: 环境变量 dict。

    输出:
        已连接的 fastmcp.Client 实例。
    """
    transport = StdioTransport(
        command=command[0],
        args=list(command[1:]),
        env=env,
    )
    client = Client(transport)
    await client.__aenter__()
    return client


async def _collect_tools(
    client: Client,
    server_name: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """从 MCP client 收集工具列表，返回 (OpenAI schemas, 路由表)。

    输入:
        client: 已连接的 fastmcp.Client。
        server_name: "amap" 或 "xhs"。

    输出:
        (tools, routes) — tools 是 OpenAI function schema 列表，
        routes 是 {tool_name: server_name} 映射。
    """
    result = await client.list_tools()
    raw_tools = (
        result.tools
        if hasattr(result, "tools")
        else (result if isinstance(result, list) else [])
    )
    tools: list[dict[str, Any]] = []
    routes: dict[str, str] = {}
    blocked = 0
    blocklist = _AMAP_BLOCKED_TOOLS if server_name == "amap" else _XHS_BLOCKED_TOOLS
    for t in raw_tools:
        name = _get_tool_name(t)
        if name in blocklist:
            blocked += 1
            continue
        routes[name] = server_name
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": _get_tool_description(t),
                "parameters": _get_tool_input_schema(t),
            },
        })
    if blocked:
        logger.info(
            "[McpManager] %s 过滤了 %d 个不稳定工具",
            server_name, blocked,
        )
    return tools, routes


# ★ amap-mcp-server 的一些工具输出 Python 单引号 JSON，
#   会触发新版 mcp 库的 JSON-RPC 严格校验导致连接崩溃。
#   这些工具对旅行规划不是必需的，直接过滤掉。
_AMAP_BLOCKED_TOOLS = frozenset({
    # 公交/骑行/驾车路线规划 — 经常触发单引号 JSON bug，且 LLM 不会用
    "maps_direction_transit_integrated_by_coordinates",
    "maps_direction_transit_integrated_by_address",
    "maps_direction_transit_integrated",
    "maps_direction_walking_by_address",
    "maps_direction_walking_by_coordinates",
    "maps_direction_walking",
    "maps_direction_bicycling_by_address",
    "maps_direction_bicycling_by_coordinates",
    "maps_direction_bicycling",
    "maps_direction_driving_by_address",
    "maps_direction_driving_by_coordinates",
    "maps_direction_driving",
    # IP 定位 — 旅行规划用不到
    "maps_ip_location",
})

# ★ stride28-search-mcp 的敏感工具——绝对不能给 LLM
_XHS_BLOCKED_TOOLS = frozenset({
    "login_xiaohongshu",       # 登录 — 会覆盖现有 cookie
    "reset_xiaohongshu_login", # 重置登录态 — 会清空 cookie
    "login_zhihu",             # 知乎登录 — 不需要
    "search_zhihu",            # 知乎搜索 — 旅行规划不需要
})


class McpManager:
    """MCP 工具管理器。

    参数:
        amap_cfg: AmapConfig 实例。
        xhs_cfg: XHSConfig 实例。
    """

    def __init__(self, amap_cfg: AmapConfig, xhs_cfg: XHSConfig) -> None:
        self._amap_cfg = amap_cfg
        self._xhs_cfg = xhs_cfg

        self._amap_client: Client | None = None
        self._xhs_client: Client | None = None

        # 工具路由表: {tool_name: server_name}
        self._tool_routes: dict[str, str] = {}

        # ★ 缓存: {server: [tool_schemas]}
        self._tool_cache: dict[str, list[dict[str, Any]]] = {}

    # ── 生命周期 ───────────────────────────────────────────

    async def start(self) -> None:
        """启动 Amap + XHS MCP server 并缓存工具 schema。

        输入: 无。
        输出: None。
        异常: RuntimeError — 任一 MCP server 启动失败。
        """
        # ── 高德 ──
        logger.info("[McpManager] 启动 Amap MCP server...")
        try:
            self._amap_client = await _start_client(
                self._amap_cfg.server_command,
                {"AMAP_MAPS_API_KEY": self._amap_cfg.api_key},
            )
        except Exception as exc:
            raise RuntimeError(f"Amap MCP server 启动失败: {exc}") from exc

        amap_tools, amap_routes = await _collect_tools(self._amap_client, "amap")
        self._tool_routes.update(amap_routes)
        self._tool_cache["amap"] = amap_tools
        logger.info(
            "[McpManager] Amap MCP 就绪: %d 个工具 %s",
            len(amap_tools), list(amap_routes.keys()),
        )

        # ── 小红书 ──
        logger.info("[McpManager] 启动 XHS MCP server...")
        try:
            self._xhs_client = await _start_client(
                self._xhs_cfg.server_command,
                {
                    "STRIDE28_SEARCH_MCP_HOME": self._xhs_cfg.search_mcp_home,
                    "STRIDE28_XHS_HEADLESS": self._xhs_cfg.headless,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"XHS MCP server 启动失败: {exc}") from exc

        xhs_tools, xhs_routes = await _collect_tools(self._xhs_client, "xhs")
        self._tool_routes.update(xhs_routes)
        self._tool_cache["xhs"] = xhs_tools
        logger.info(
            "[McpManager] XHS MCP 就绪: %d 个工具 %s",
            len(xhs_tools), list(xhs_routes.keys()),
        )

    async def close(self) -> None:
        """关闭所有 MCP server 子进程。

        输入: 无。输出: None。幂等。
        """
        for name, client in [("Amap", self._amap_client), ("XHS", self._xhs_client)]:
            if client is not None:
                try:
                    await client.__aexit__(None, None, None)
                    logger.info("[McpManager] %s MCP 已关闭", name)
                except Exception as exc:
                    logger.warning("[McpManager] 关闭 %s MCP 时出错: %s", name, exc)
        self._amap_client = None
        self._xhs_client = None
        self._tool_routes.clear()
        self._tool_cache.clear()

    # ── 工具发现 ★ 优先走缓存 ──────────────────────────────

    async def list_tools(self, server: str = "all") -> list[dict[str, Any]]:
        """列出 MCP 工具 schema（优先缓存）。

        输入:
            server: "amap" / "xhs" / "all"（默认全部）。

        输出:
            OpenAI function calling tool schema 列表。
        """
        tools: list[dict[str, Any]] = []
        for svr in ("amap", "xhs"):
            if server not in (svr, "all"):
                continue
            if svr in self._tool_cache:
                tools.extend(self._tool_cache[svr])
                continue
            # 缓存未命中 → 实时查询
            client = self._amap_client if svr == "amap" else self._xhs_client
            if client is None:
                continue
            try:
                new_tools, new_routes = await _collect_tools(client, svr)
                self._tool_routes.update(new_routes)
                self._tool_cache[svr] = new_tools
                tools.extend(new_tools)
            except Exception as exc:
                logger.warning("[McpManager] 列出 %s 工具失败: %s", svr, exc)
        return tools

    # ── 工具调用 ★ 超时保护 + 自动重连 ────────────────────

    _CALL_TIMEOUT_S = 30.0          # 单次调用超时
    _RECONNECT_RETRIES = 1          # 重连后重试次数

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """调用 MCP 工具，带超时和自动重连。

        输入:
            tool_name: MCP 工具名。
            arguments: 工具参数字典。

        输出:
            文本结果（供 LLM 阅读）。永远不抛异常——所有错误都返回文本。
        """
        server = self._tool_routes.get(tool_name)
        if server is None:
            return f"未知工具: {tool_name}（可用: {sorted(self._tool_routes.keys())}）"

        cfg = self._amap_cfg if server == "amap" else self._xhs_cfg
        client_ref = "_amap_client" if server == "amap" else "_xhs_client"

        for retry in range(self._RECONNECT_RETRIES + 1):
            client: Client | None = getattr(self, client_ref)
            if client is None:
                return f"{server} MCP server 未连接"

            try:
                result = await asyncio.wait_for(
                    client.call_tool(tool_name, arguments),
                    timeout=self._CALL_TIMEOUT_S,
                )
                return _parse_tool_result(result)
            except asyncio.TimeoutError:
                logger.warning(
                    "[McpManager] %s call_tool(%s) 超时 %.0fs，子进程可能已崩溃",
                    server, tool_name, self._CALL_TIMEOUT_S,
                )
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(
                    "[McpManager] %s call_tool(%s) 异常: %s",
                    server, tool_name, err_msg[:150],
                )

            # ── 到达这里说明调用失败，尝试重连 ──
            if retry < self._RECONNECT_RETRIES:
                logger.info("[McpManager] 正在重连 %s ...", server)
                try:
                    # 关旧连接（静默忽略关闭时的异常）
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception:
                        pass

                    # 重连
                    env = (
                        {"AMAP_MAPS_API_KEY": self._amap_cfg.api_key}
                        if server == "amap"
                        else {
                            "STRIDE28_SEARCH_MCP_HOME": self._xhs_cfg.search_mcp_home,
                            "STRIDE28_XHS_HEADLESS": self._xhs_cfg.headless,
                        }
                    )
                    new_client = await _start_client(cfg.server_command, env)
                    setattr(self, client_ref, new_client)

                    # 刷新工具缓存
                    new_tools, new_routes = await _collect_tools(new_client, server)
                    self._tool_routes.update(new_routes)
                    self._tool_cache[server] = new_tools

                    logger.info(
                        "[McpManager] %s 重连成功 (%d tools)",
                        server, len(new_tools),
                    )
                except Exception as exc2:
                    logger.error(
                        "[McpManager] %s 重连失败: %s", server, exc2,
                    )
                    return f"工具调用失败 ({tool_name})，{server} MCP 连接丢失且重连失败: {exc2}"

        return f"工具调用失败 ({tool_name})，{server} MCP 重试 {self._RECONNECT_RETRIES + 1} 次均未成功"

