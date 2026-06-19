"""
增强的 MCP 客户端实现

支持多种传输方式的 MCP 客户端，用于教学和实际应用。
这个实现展示了如何使用不同的传输方式连接到 MCP 服务器。

支持的传输方式：
1. Memory: 内存传输（用于测试，直接传递 FastMCP 实例）
2. Stdio: 标准输入输出传输（本地进程，Python/Node.js 脚本）
3. HTTP: HTTP 传输（远程服务器）
4. SSE: Server-Sent Events 传输（实时通信）

使用示例：
```python
# 1. 内存传输（测试）
from fastmcp import FastMCP
server = FastMCP("TestServer")
client = MCPClient(server)

# 2. Stdio 传输（本地脚本）
client = MCPClient("server.py")
client = MCPClient(["python", "server.py"])

# 3. HTTP 传输（远程服务器）
client = MCPClient("https://api.example.com/mcp")

# 4. SSE 传输（实时通信）
client = MCPClient("https://api.example.com/mcp", transport_type="sse")

# 5. 配置传输（高级用法）
config = {
    "transport": "stdio",
    "command": "python",
    "args": ["server.py"],
    "env": {"DEBUG": "1"}
}
client = MCPClient(config)
```
"""

from __future__ import annotations

import logging
from typing import Any
import asyncio


# 将 fastmcp 作为一个可选依赖。如果在启动阶段没有安装，不会立刻引发进程崩溃（ImportError），
# 而是将 FASTMCP_AVAILABLE 置为 False，把报错时机延后到用户真正实例化 McpClient 时。
try:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import (
        PythonStdioTransport,
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    # 使用 type: ignore 压制静态类型检查器的报错，提供伪占位符
    Client = None  # type: ignore[assignment]
    FastMCP = None  # type: ignore[assignment]
    PythonStdioTransport = None  # type: ignore[assignment]
    SSETransport = None  # type: ignore[assignment]
    StdioTransport = None  # type: ignore[assignment]
    StreamableHttpTransport = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class McpClient:
    """MCP 客户端，基于 fastmcp.Client，支持多种传输方式。

    参数:
        server_source: MCP server 源，支持多种格式：
            - FastMCP 实例 → Memory 传输（测试用，直接传递 server 对象）
            - str 路径 → 以 .py 结尾则 PythonStdioTransport
            - str HTTP URL → StreamableHttpTransport
            - list[str] 命令 → StdioTransport（通用命令）
            - dict 配置 → 根据 "transport" 字段创建对应传输
        server_args: 额外的命令行参数列表（仅 stdio 传输有效）。
        transport_type: 强制指定传输类型（"stdio"/"http"/"sse"/"memory"）。
        env: 额外的环境变量字典（仅 stdio 传输有效）。
        **transport_kwargs: 透传给传输层的额外参数。

    异常:
        ImportError: fastmcp 库未安装时抛出。
    """

    def __init__(
        self,
        server_source: str | list[str] | dict[str, Any],
        server_args: list[str] | None = None,
        transport_type: str | None = None,
        env: dict[str, str] | None = None,
        **transport_kwargs: Any,
    ) -> None:
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "McpClient 需要 fastmcp 库（>=2.0.0）。"
                "请执行: pip install fastmcp>=2.0.0"
            )
        self.server_args = server_args or []
        self.transport_type = transport_type
        self.env = env or {}
        self.transport_kwargs = transport_kwargs

        # 在初始化时就完成多态传输对象的创建，但不立即连接
        self.server_source = self._prepare_server_source(server_source)
        self._client: Client | None = None

        # 在断开连接时，能够正确调用其底层 __aexit__ 以回收子进程资源
        self._ctx: Any = None  # fastmcp.Client 的上下文管理器返回值

    # ── async with 支持 ──────────────────────────────────────────

    # 实现异步上下文管理器协议，确保资源的绝对安全释放
    async def __aenter__(self) -> McpClient:
        """异步上下文管理器入口：自动连接。

        输出:
            self，连接已就绪。
        """
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """异步上下文管理器出口：自动断开。"""
        # 无论内部是否发生异常，退出 with 块时必然执行 disconnect
        await self.disconnect()

    # ── 连接管理 ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """连接到 MCP server 并完成握手。

        输出: None。连接成功后 self._client 可用。

        异常:
            ImportError: fastmcp 未安装。
            ConnectionError: 子进程启动失败或握手超时。
        """
        logger.info("[mcp] 连接到 MCP 服务器...")
        # 实例化底层的 fastmcp.Client
        self._client = Client(self.server_source)
        self._ctx = self._client
        # 显式调用底层的异步上下文入口，这一步会真正启动子进程(stdio)或建立网络连接(http/sse)
        await self._ctx.__aenter__()
        logger.info("[mcp] 连接成功 (%s)", self.get_transport_info())

    async def disconnect(self) -> None:
        """断开连接并清理子进程。

        输出: None。幂等——重复调用无副作用。
        """
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._client = None
            self._ctx = None
        logger.info("[mcp] 连接已断开")

    # ── 工具操作 ──────────────────────────────────────────────────

    async def list_tools(self) -> list[dict[str, Any]]:
        """列出远端 MCP server 提供的所有工具。

        输出:
            工具信息字典列表，每项含 name / description / input_schema。
            格式已归一化，与 raven-agent 的 Tool schema 约定一致。

        异常:
            RuntimeError: 尚未连接时抛出。
        """
        if not self._client:
            raise RuntimeError(
                "McpClient 尚未连接，请先调用 connect() 或使用 async with"
            )
        result = await self._client.list_tools()

        # 处理 fastmcp 的不同返回格式
        if hasattr(result, "tools"):
            tools = result.tools
        elif isinstance(result, list):
            tools = result
        else:
            tools = []

        # 将 fastmcp 的内部对象转换为标准的 Python 字典列表
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": (
                    tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> Any:
        """调用远端工具，原样返回 fastmcp 的 ToolResult。

        输入:
            tool_name: 远端工具名称。
            arguments: 工具参数字典。

        输出:
            fastmcp ToolResult 的解析结果：
            - 单 text block → 返回其文本字符串
            - 单 data block → 返回其原始数据
            - 多 block → 返回 list[str|Any]
            - 无 content → 返回 None

        异常:
            RuntimeError: 尚未连接时抛出。
        """
        if not self._client:
            raise RuntimeError(
                "McpClient 尚未连接，请先调用 connect() 或使用 async with"
            )
        try:
            return await self._call_tool_inner(tool_name, arguments, **kwargs)
        except RuntimeError as e:
            # 底层 fastmcp session 可能因子进程崩溃 / 空闲超时而断开。
            # McpClient._client 非 None 但 session 已无效 → 尝试重连一次。
            if "not connected" not in str(e).lower():
                raise
            logger.warning(
                "[mcp] %s.%s 调用时连接丢失，尝试重连并重试",
                self.get_transport_info().get("transport_info", "?"),
                tool_name,
            )
            await self._reconnect()
            return await self._call_tool_inner(tool_name, arguments, **kwargs)

    async def _call_tool_inner(
        self, tool_name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> Any:
        """call_tool 的无重连实现，供 call_tool 内部复用。"""
        timeout = kwargs.get("timeout")
        if timeout:
            result = await asyncio.wait_for(
                self._client.call_tool(tool_name, arguments), timeout=timeout
            )
        else:
            result = await self._client.call_tool(tool_name, arguments)

        # 解析 fastmcp 的 ToolResult
        if hasattr(result, "content") and result.content:
            if len(result.content) == 1:
                content = result.content[0]
                if hasattr(content, "text"):
                    return content.text
                elif hasattr(content, "data"):
                    return content.data
            return [
                getattr(c, "text", getattr(c, "data", str(c)))
                for c in result.content
            ]
        return None

    async def _reconnect(self) -> None:
        """断开旧连接并重新连接。

        disconnect() 会清理旧子进程（__aexit__），connect() 会启动新子进程
        并完成 MCP 握手。失败时 _client 保持为 None，后续调用会得到明确的
        "尚未连接"错误。
        """
        try:
            await self.disconnect()
        except Exception:
            pass
        await self.connect()

    # ── 扩展能力：resources / prompts / ping ─────────────────────
    # 薄封装 fastmcp 的 list_resources / read_resource / list_prompts / get_prompt / ping 方法，
    async def list_resources(self) -> list[dict[str, Any]]:
        """列出所有可用的资源。

        输出:
            资源信息字典列表，每项含 uri / name / description / mime_type。
        """
        if not self._client:
            raise RuntimeError("McpClient 尚未连接")
        result = await self._client.list_resources()
        return [
            {
                "uri": r.uri,
                "name": r.name or "",
                "description": r.description or "",
                "mime_type": getattr(r, "mimeType", None),
            }
            for r in result.resources
        ]

    async def read_resource(self, uri: str) -> str | None:
        """读取资源内容。

        输入:
            uri: 资源 URI。

        输出:
            资源文本内容；无内容时返回 None。
        """
        if not self._client:
            raise RuntimeError("McpClient 尚未连接")
        result = await self._client.read_resource(uri)
        if hasattr(result, "contents") and result.contents:
            if len(result.contents) == 1:
                c = result.contents[0]
                return str(getattr(c, "text", getattr(c, "blob", "")))
            return "\n".join(
                str(getattr(c, "text", getattr(c, "blob", str(c))))
                for c in result.contents
            )
        return None

    async def list_prompts(self) -> list[dict[str, Any]]:
        """列出所有可用的提示词模板。

        输出:
            提示词信息字典列表。
        """
        if not self._client:
            raise RuntimeError("McpClient 尚未连接")
        result = await self._client.list_prompts()
        return [
            {
                "name": p.name,
                "description": p.description or "",
                "arguments": getattr(p, "arguments", []),
            }
            for p in result.prompts
        ]

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """获取提示词内容。

        输入:
            prompt_name: 提示词模板名称。
            arguments: 模板参数。

        输出:
            提示词消息列表。
        """
        if not self._client:
            raise RuntimeError("McpClient 尚未连接")
        result = await self._client.get_prompt(prompt_name, arguments or {})
        if hasattr(result, "messages") and result.messages:
            return [
                {
                    "role": m.role,
                    "content": (
                        m.content.text
                        if hasattr(m.content, "text")
                        else str(m.content)
                    ),
                }
                for m in result.messages
            ]
        return []

    async def ping(self) -> bool:
        """测试服务器连接是否正常。

        输出:
            True 表示服务器可达。
        """
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    def get_transport_info(self) -> dict[str, Any]:
        """获取当前传输信息。

        输出:
            包含 status / transport_type 等字段的字典。
        """
        if not self._client:
            return {"status": "not_connected"}
        transport = getattr(self._client, "transport", None)
        if transport:
            return {
                "status": "connected",
                "transport_type": type(transport).__name__,
                "transport_info": str(transport),
            }
        return {"status": "unknown"}

    # ── 内部 ──────────────────────────────────────────────────────

    def _prepare_server_source(
        self,
        server_source: str | list[str] | FastMCP | dict[str, Any],
    ) -> Any:
        """根据输入类型自动创建 fastmcp 传输对象。

        输入:
            server_source: 原始的 server 源。

        输出:
            fastmcp 传输对象（直接传给 Client 构造函数）。
        """
        s = server_source
        extra = {**self.transport_kwargs}

        # 1. FastMCP 实例 → Memory 传输（测试用）
        # 便利了单元测试，无需真正起端口或子进程
        if isinstance(s, FastMCP):
            logger.info("[mcp] 使用 Memory 传输: %s", s.name)
            return s

        # 2. dict 配置 → 根据 transport 字段选择
        if isinstance(s, dict):
            ttype = s.get("transport", "stdio")
            logger.info("[mcp] 使用配置传输 (%s)", ttype)
            if ttype == "sse":
                return SSETransport(url=s["url"], **extra)
            if ttype == "http":
                return StreamableHttpTransport(url=s["url"], **extra)
            # stdio 分支处理
            args = s.get("args", [])
            # 启发式判断：如果 args 第一个是 .py 文件，优化为 PythonStdioTransport
            if args and args[0].endswith(".py"):
                return PythonStdioTransport(
                    script_path=args[0],
                    args=args[1:] + self.server_args,
                    env=s.get("env"),
                    cwd=s.get("cwd"),
                    **extra,
                )
            # 通用二进制或其他语言的 stdio 命令（如 node, cargo 等）
            return StdioTransport(
                command=s.get("command", "python"),
                args=args + self.server_args,
                env=s.get("env"),
                cwd=s.get("cwd"),
                **extra,
            )

        # 3. HTTP URL → HTTP/SSE 传输（如果传入的是 URL 字符串，自动转为网络传输）
        if isinstance(s, str) and (
            s.startswith("http://") or s.startswith("https://")
        ):
            ttype = self.transport_type or "http"
            logger.info("[mcp] 使用 %s 传输: %s", ttype.upper(), s)
            if ttype == "sse":
                return SSETransport(url=s, **extra)
            return StreamableHttpTransport(url=s, **extra)

        # 4. .py 路径 → PythonStdioTransport
        if isinstance(s, str) and s.endswith(".py"):
            logger.info("[mcp] 使用 Stdio 传输 (Python): %s", s)
            return PythonStdioTransport(
                script_path=s,
                args=self.server_args,
                env=self.env if self.env else None,
                **extra,
            )

        # 5. 命令列表（例如 ['node', 'build/index.js']） → StdioTransport
        if isinstance(s, list) and s:
            logger.info(
                "[mcp] 使用 Stdio 传输 (命令): %s", " ".join(s)
            )
            if s[0] == "python" and len(s) > 1 and s[1].endswith(".py"):
                return PythonStdioTransport(
                    script_path=s[1],
                    args=s[2:] + self.server_args,
                    env=self.env if self.env else None,
                    **extra,
                )
            return StdioTransport(
                command=s[0],
                args=s[1:] + self.server_args,
                env=self.env if self.env else None,
                **extra,
            )

        # 6. 其他情况 → 直接返回，让 fastmcp 自动推断
        logger.info("[mcp] 自动推断传输: %s", s)
        return s