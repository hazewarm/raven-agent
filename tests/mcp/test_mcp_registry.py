"""Tests for McpServerRegistry: 增删 server、持久化、工具同步。"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raven_agent.tools.registry import ToolRegistry


# ── autouse: mock McpClient._prepare_server_source ───────────────
# test_tool_wrapper_* 直接创建 McpClient(["python", "server.py"])，
# __init__ 中 PythonStdioTransport 会检查文件是否存在。autouse
# fixture 全局 mock 掉传输解析，避免 FileNotFoundError。

@pytest.fixture(autouse=True)
def _mock_prepare_source() -> MagicMock:
    from raven_agent.mcp.client import McpClient
    with patch.object(McpClient, "_prepare_server_source", return_value=MagicMock()):
        yield


# ── Helpers ──────────────────────────────────────────────────────


def _make_mock_registry() -> ToolRegistry:
    """创建一个真实的 ToolRegistry 用于测试。"""
    return ToolRegistry()


def _make_mock_client(
    tool_infos: list[str | dict[str, object]] | None = None,
) -> MagicMock:
    """创建一个 mock 的 McpClient。

    输入:
        tool_infos: list_tools() 返回的工具信息列表。
            可以是字符串列表（仅 tool name），也可以是字典列表。

    输出:
        MagicMock，其 connect() / list_tools() / disconnect() 为 AsyncMock。
    """
    mock = MagicMock()
    mock.connect = AsyncMock()
    if tool_infos is None:
        tool_infos = []
    mock.list_tools = AsyncMock(
        return_value=[
            {
                "name": str(t if isinstance(t, str) else t["name"]),
                "description": (
                    f"Tool: {t}"
                    if isinstance(t, str)
                    else str(t.get("description", ""))
                ),
                "input_schema": (
                    {"type": "object", "properties": {}}
                    if isinstance(t, str)
                    else dict(t.get("input_schema", {}))
                ),
            }
            for t in tool_infos
        ]
    )
    mock.disconnect = AsyncMock()
    return mock


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ToolRegistry:
    """返回空 ToolRegistry。"""
    return _make_mock_registry()


@pytest.fixture
def servers_path(tmp_path: Path) -> Path:
    """返回测试用的 mcp_servers.json 路径。"""
    return tmp_path / "mcp_servers.json"


# ── Add ──────────────────────────────────────────────────────────


async def test_add_registers_tools(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """add() 连接成功后远端工具注册到 ToolRegistry。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(
            ["get_forecast", "get_alerts"]
        )
        MockClient.return_value = mock_client

        result = await mcp_registry.add(
            "weather", ["python", "weather_server.py"]
        )

    assert "已连接" in result
    assert "weather" in result
    assert "get_forecast" in result
    assert "get_alerts" in result
    # 工具应注册到 ToolRegistry
    assert registry.has_tool("mcp_weather__get_forecast")
    assert registry.has_tool("mcp_weather__get_alerts")
    # 验证 source_type
    doc = registry.get_document("mcp_weather__get_forecast")
    assert doc is not None
    assert doc.source_type == "mcp"
    assert doc.source_name == "weather"


async def test_add_duplicate_returns_error(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """重复添加同名 server 返回错误信息。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        MockClient.return_value = mock_client

        # 第一次添加
        await mcp_registry.add("weather", ["python", "server.py"])
        # 第二次添加同名
        result = await mcp_registry.add(
            "weather", ["python", "server.py"]
        )

    assert "已存在" in result


async def test_add_connect_failure_returns_error(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """连接失败时返回错误信息而不抛异常。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client()
        mock_client.connect = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        mock_client.disconnect = AsyncMock()
        MockClient.return_value = mock_client

        result = await mcp_registry.add(
            "bad", ["python", "bad_server.py"]
        )

    assert "失败" in result
    assert "Connection refused" in result


# ── Remove ───────────────────────────────────────────────────────


async def test_remove_unregisters_tools(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """remove() 后远端工具从 ToolRegistry 中注销。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        MockClient.return_value = mock_client

        await mcp_registry.add("weather", ["python", "server.py"])
        result = await mcp_registry.remove("weather")

    assert "已注销" in result
    assert not registry.has_tool("mcp_weather__get_forecast")


async def test_remove_nonexistent_returns_message(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """移除不存在的 server 返回提示信息。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)
    result = await mcp_registry.remove("nonexistent")

    assert "不存在" in result


# ── List ─────────────────────────────────────────────────────────


async def test_list_servers_empty(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """无已注册 server 时返回提示信息。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)
    result = mcp_registry.list_servers()

    assert "没有" in result


async def test_list_servers_with_items(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """有已注册 server 时列出名称和工具。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)
    # 手动注入状态（绕过真实的 connect 流程）
    # list_servers() 迭代 _clients，_server_tools 提供工具名映射
    fake_client = MagicMock()
    mcp_registry._clients["weather"] = fake_client
    mcp_registry._server_tools["weather"] = [
        "mcp_weather__get_forecast",
        "mcp_weather__get_alerts",
    ]
    mcp_registry._clients["calendar"] = fake_client
    mcp_registry._server_tools["calendar"] = [
        "mcp_calendar__list_events"
    ]

    result = mcp_registry.list_servers()

    assert "weather" in result
    assert "get_forecast" in result
    assert "calendar" in result
    assert "list_events" in result


# ── Persistence ──────────────────────────────────────────────────


async def test_persist_saves_config(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """add() 后配置持久化到 mcp_servers.json。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        MockClient.return_value = mock_client

        await mcp_registry.add(
            "weather",
            ["python", "/path/to/server.py"],
            env={"API_KEY": "secret"},
            cwd="/path/to",
        )

    assert servers_path.exists()
    data = json.loads(servers_path.read_text(encoding="utf-8"))
    assert "servers" in data
    assert "weather" in data["servers"]
    assert data["servers"]["weather"]["command"] == [
        "python",
        "/path/to/server.py",
    ]
    assert data["servers"]["weather"]["env"] == {"API_KEY": "secret"}
    assert data["servers"]["weather"]["cwd"] == "/path/to"


async def test_load_and_connect_all_restores(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """load_and_connect_all() 从持久化文件恢复 server 连接。"""
    from raven_agent.mcp.registry import McpServerRegistry

    # 预先写入持久化文件
    servers_path.write_text(
        json.dumps(
            {
                "servers": {
                    "weather": {
                        "command": ["python", "weather_server.py"],
                        "env": {},
                        "cwd": None,
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        mock_client.disconnect = AsyncMock()
        MockClient.return_value = mock_client

        await mcp_registry.load_and_connect_all()

    assert registry.has_tool("mcp_weather__get_forecast")


async def test_load_nonexistent_file_no_error(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """mcp_servers.json 不存在时不报错。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)
    # 不应该抛异常
    await mcp_registry.load_and_connect_all()
    # 应该为空
    assert "没有" in mcp_registry.list_servers()


# ── Shutdown ─────────────────────────────────────────────────────


async def test_shutdown_disconnects_all(
    registry: ToolRegistry, servers_path: Path
) -> None:
    """shutdown() 断开所有已连接 server。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch(
        "raven_agent.mcp.registry.McpClient"
    ) as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        MockClient.return_value = mock_client

        await mcp_registry.add("weather", ["python", "server.py"])
        await mcp_registry.shutdown()

    mock_client.disconnect.assert_called_once()


# ── McpToolWrapper ───────────────────────────────────────────────


async def test_tool_wrapper_naming() -> None:
    """McpToolWrapper 按 mcp_{server}__{tool} 格式命名。"""
    from raven_agent.mcp.client import McpClient
    from raven_agent.mcp.tool import McpToolWrapper

    # 创建一个未连接的 client（仅测试命名）
    client = McpClient(["python", "server.py"])
    info_dict = {
        "name": "list_events",
        "description": "List calendar events",
        "input_schema": {"type": "object", "properties": {}},
    }
    wrapper = McpToolWrapper(client, "calendar", info_dict)

    assert wrapper.name == "mcp_calendar__list_events"
    assert "[MCP:calendar]" in wrapper.description
    assert wrapper.parameters == info_dict["input_schema"]


async def test_tool_wrapper_execute_proxies_to_client() -> None:
    """McpToolWrapper.execute() 代理到 McpClient.call_tool()。"""
    from raven_agent.mcp.client import McpClient
    from raven_agent.mcp.tool import McpToolWrapper

    client = McpClient(["python", "server.py"])
    client.call_tool = AsyncMock(return_value="Sunny, 25°C")

    info_dict = {
        "name": "get_forecast",
        "description": "Get weather forecast",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    }
    wrapper = McpToolWrapper(client, "weather", info_dict)

    result = await wrapper.execute(city="Beijing")

    assert result == "Sunny, 25°C"
    client.call_tool.assert_called_once_with("get_forecast", {"city": "Beijing"})


# ── Manage Tools ────────────────────────────────────────────────


async def test_mcp_add_tool_schema_has_required_fields() -> None:
    """McpAddTool 的 schema 声明 name 和 command 为必填。"""
    from raven_agent.mcp.manage_tools import McpAddTool

    # 用 MagicMock 代替 registry（只测试 schema，不执行）
    tool = McpAddTool(MagicMock())
    assert tool.name == "mcp_add"
    assert "name" in tool.parameters.get("required", [])


async def test_mcp_remove_tool_schema_has_required_name() -> None:
    """McpRemoveTool 的 schema 声明 name 为必填。"""
    from raven_agent.mcp.manage_tools import McpRemoveTool

    tool = McpRemoveTool(MagicMock())
    assert tool.name == "mcp_remove"
    assert "name" in tool.parameters.get("required", [])


async def test_mcp_list_tool_no_required_params() -> None:
    """McpListTool 的 schema 无必填参数。"""
    from raven_agent.mcp.manage_tools import McpListTool

    tool = McpListTool(MagicMock())
    assert tool.name == "mcp_list"
    assert tool.parameters.get("required") is None or len(
        tool.parameters.get("required", [])
    ) == 0


# ── Source helpers & call_tool tests ──────────────────────────


async def test_get_tool_names_by_source(registry, servers_path):
    """按 source 查询工具名称。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch("raven_agent.mcp.registry.McpClient") as MockClient:
        mock_client = _make_mock_client(["get_forecast", "get_alerts"])
        MockClient.return_value = mock_client
        await mcp_registry.add("weather", ["python", "server.py"])

    names = registry.get_tool_names_by_source("mcp", "weather")
    assert "mcp_weather__get_forecast" in names
    assert "mcp_weather__get_alerts" in names


async def test_get_mcp_server_names(registry, servers_path):
    """列出已连接 MCP server 名称。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch("raven_agent.mcp.registry.McpClient") as MockClient:
        mock_client = _make_mock_client(["get_forecast"])
        MockClient.return_value = mock_client
        await mcp_registry.add("weather", ["python", "server.py"])

    servers = registry.get_mcp_server_names()
    assert "weather" in servers


async def test_connected_server_names(registry, servers_path):
    """connected_server_names 返回已连接的 server。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch("raven_agent.mcp.registry.McpClient") as MockClient:
        mock_client = _make_mock_client(["tool_a"])
        MockClient.return_value = mock_client
        await mcp_registry.add("srv", ["python", "server.py"])

    assert "srv" in mcp_registry.connected_server_names()


async def test_call_tool_proxies_to_client(registry, servers_path):
    """call_tool() 代理到已连接 client。"""
    from raven_agent.mcp.registry import McpServerRegistry

    mcp_registry = McpServerRegistry(servers_path, registry)

    with patch("raven_agent.mcp.registry.McpClient") as MockClient:
        mock_client = _make_mock_client(["echo"])
        mock_client.call_tool = AsyncMock(return_value="hello")
        MockClient.return_value = mock_client
        await mcp_registry.add("echo-srv", ["python", "server.py"])

        result = await mcp_registry.call_tool("echo-srv", "echo", {"msg": "hi"})
        assert result == "hello"
        mock_client.call_tool.assert_called_once_with("echo", {"msg": "hi"})