"""Tests for McpClient: 薄封装逻辑、传输解析、上下文管理器。

使用 mock fastmcp.Client 而非真实子进程——
fastmcp 的协议正确性由它自己的测试保证，我们只测封装层。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raven_agent.mcp.client import McpClient


# ── autouse fixture: mock 掉 _prepare_server_source ──────────────
# McpClient.__init__ 中会调用 _prepare_server_source() 创建 fastmcp
# 传输对象，PythonStdioTransport 构造时会检查脚本文件是否存在。
# 测试场景下不需要真实传输对象——connect() 里 fastmcp.Client 也被 mock 了。

@pytest.fixture(autouse=True)
def _mock_prepare_source() -> MagicMock:
    """自动 mock _prepare_server_source，避免文件存在性检查。"""
    transport = MagicMock()
    with patch.object(McpClient, "_prepare_server_source", return_value=transport):
        yield transport


# ── Helpers ──────────────────────────────────────────────────────


class _FakeTool:
    """模拟 fastmcp 的 Tool 对象。"""

    def __init__(self, name: str, description: str = "", input_schema: dict | None = None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class _FakeToolsResult:
    """模拟 fastmcp list_tools() 的返回值。"""

    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


class _FakeContent:
    """模拟 fastmcp ToolResult 的 content block。"""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolResult:
    """模拟 fastmcp call_tool() 的返回值。"""

    def __init__(self, content_blocks: list[_FakeContent]) -> None:
        self.content = content_blocks


def _make_mock_client(
    tools: list[_FakeTool] | None = None,
    call_results: list[str] | None = None,
) -> MagicMock:
    """创建一个 mock 的 fastmcp.Client。

    输入:
        tools: list_tools() 返回的 _FakeTool 列表。
        call_results: call_tool() 按顺序返回的文本列表。

    输出:
        MagicMock，其 list_tools() / call_tool() 返回模拟数据，
        且支持 async with 上下文管理器。
    """
    mock = MagicMock()
    mock.list_tools = AsyncMock(
        return_value=_FakeToolsResult(tools or [])
    )
    if call_results:
        mock.call_tool = AsyncMock(
            side_effect=[
                _FakeToolResult([_FakeContent(text)])
                for text in call_results
            ]
        )
    else:
        mock.call_tool = AsyncMock(
            return_value=_FakeToolResult([_FakeContent("default")])
        )
    # async with 支持
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


# ── connect / disconnect ──────────────────────────────────────────


async def test_connect_creates_fastmcp_client() -> None:
    """connect() 创建 fastmcp.Client 并进入上下文管理器。"""
    mock = _make_mock_client(tools=[_FakeTool("echo")])

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()

    # 验证 fastmcp.Client 的上下文管理器被调用
    mock.__aenter__.assert_called_once()
    # 连接后 get_transport_info 返回 connected
    info = client.get_transport_info()
    assert info["status"] == "connected"


async def test_init_stores_raw_source(_mock_prepare_source: MagicMock) -> None:
    """__init__ 将原始 source 传给 _prepare_server_source。"""
    client = McpClient(["python", "server.py"])
    # _prepare_server_source 被调用了，返回值存为 server_source
    assert client.server_source is _mock_prepare_source


async def test_disconnect_after_connect() -> None:
    """连接后 disconnect 正常关闭。"""
    mock = _make_mock_client(tools=[_FakeTool("echo")])

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        await client.disconnect()

    mock.__aexit__.assert_called_once()


async def test_disconnect_idempotent() -> None:
    """重复调用 disconnect() 不报错。"""
    client = McpClient(["python", "server.py"])
    await client.disconnect()
    await client.disconnect()


# ── list_tools ───────────────────────────────────────────────────


async def test_list_tools_returns_normalized_dicts() -> None:
    """list_tools() 将 fastmcp 的 Tool 对象归一化为 dict 列表。"""
    mock = _make_mock_client(tools=[
        _FakeTool("echo", "Echo back", {"type": "object", "properties": {"msg": {"type": "string"}}}),
        _FakeTool("add", "Add numbers", {"type": "object", "properties": {"a": {"type": "number"}}}),
    ])

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        tools = await client.list_tools()

    assert len(tools) == 2
    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "Echo back"
    assert "msg" in tools[0]["input_schema"]["properties"]
    assert tools[1]["name"] == "add"


async def test_list_tools_empty() -> None:
    """远端返回空工具列表时返回空列表。"""
    mock = _make_mock_client(tools=[])

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        tools = await client.list_tools()

    assert tools == []


async def test_list_tools_before_connect_raises() -> None:
    """未连接时调用 list_tools 抛出 RuntimeError。"""
    client = McpClient(["python", "server.py"])
    with pytest.raises(RuntimeError, match="尚未连接"):
        await client.list_tools()


# ── call_tool ────────────────────────────────────────────────────


async def test_call_tool_returns_text() -> None:
    """call_tool() 返回远端工具的文本结果。"""
    mock = _make_mock_client(
        tools=[_FakeTool("echo")],
        call_results=["hello world"],
    )

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        result = await client.call_tool("echo", {"message": "hello"})

    assert result == "hello world"
    mock.call_tool.assert_called_once_with("echo", {"message": "hello"})


async def test_call_tool_multi_content() -> None:
    """多个 content block 时返回 list（保留与 hello_agents 一致的行为）。"""
    mock = _make_mock_client(tools=[_FakeTool("multi")])
    mock.call_tool = AsyncMock(return_value=_FakeToolResult([
        _FakeContent("line 1"),
        _FakeContent("line 2"),
        _FakeContent("line 3"),
    ]))

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        result = await client.call_tool("multi", {})

    # hello_agents 行为：多 block 返回 list[str]
    assert isinstance(result, list)
    assert result == ["line 1", "line 2", "line 3"]


async def test_call_tool_no_content_returns_none() -> None:
    """远端返回空 content 时 call_tool() 返回 None。"""
    mock = _make_mock_client(tools=[_FakeTool("empty")])
    mock.call_tool = AsyncMock(return_value=_FakeToolResult([]))

    client = McpClient(["python", "server.py"])
    with patch("raven_agent.mcp.client.Client", return_value=mock):
        await client.connect()
        result = await client.call_tool("empty", {})

    assert result is None


async def test_call_tool_before_connect_raises() -> None:
    """未连接时调用 call_tool 抛出 RuntimeError。"""
    client = McpClient(["python", "server.py"])
    with pytest.raises(RuntimeError, match="尚未连接"):
        await client.call_tool("echo", {})


# ── Async Context Manager ────────────────────────────────────────


async def test_async_context_manager_auto_connect_disconnect() -> None:
    """async with 自动调用 connect 和 disconnect。"""
    client = McpClient(["python", "server.py"])
    mock_connect = AsyncMock()
    mock_disconnect = AsyncMock()

    with (
        patch.object(client, "connect", mock_connect),
        patch.object(client, "disconnect", mock_disconnect),
        patch.object(client, "list_tools", AsyncMock(return_value=[])),
    ):
        async with client:
            pass

    mock_connect.assert_called_once()
    mock_disconnect.assert_called_once()


async def test_async_context_manager_cleans_up_on_error() -> None:
    """即使内部抛异常，async with 也会断开连接。"""
    client = McpClient(["python", "server.py"])
    mock_disconnect = AsyncMock()

    async def _boom() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    with (
        patch.object(client, "connect", AsyncMock()),
        patch.object(client, "disconnect", mock_disconnect),
    ):
        try:
            async with client:
                await _boom()
        except RuntimeError:
            pass

    mock_disconnect.assert_called_once()


# ── get_transport_info ───────────────────────────────────────────


def test_get_transport_info_not_connected() -> None:
    """未连接时返回 status=not_connected。"""
    client = McpClient(["python", "server.py"])
    assert client.get_transport_info() == {"status": "not_connected"}