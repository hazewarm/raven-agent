"""
proactive_source_add / proactive_source_remove / proactive_source_list
—— AI 可调用的 proactive 数据源管理工具。
"""

from __future__ import annotations

from typing import Any

from raven_agent.proactive.mcp_sources import SourceStore
from raven_agent.tools.base import Tool


class ProactiveSourceAddTool(Tool):
    """添加或更新一个 proactive 数据源。

    参数:
        store: SourceStore 实例。
    """

    def __init__(self, store: SourceStore) -> None:
        self._store = store

    name = "proactive_source_add"
    description = (
        "添加或更新一个 Proactive 外部数据源。source 声明了一个已连接 MCP server "
        "的哪个工具提供哪类数据（alert/content/context）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "source 名称，如 'github-alerts'",
            },
            "server": {
                "type": "string",
                "description": "已连接的 MCP server 名称，如 'github'",
            },
            "channel": {
                "type": "string",
                "enum": ["alert", "content", "context"],
                "description": "数据通道类型",
            },
            "get_tool": {
                "type": "string",
                "description": (
                    "MCP server 上用于获取数据的工具名。"
                    "alert/content 默认 get_proactive_events，context 默认 get_context"
                ),
            },
            "ack_tool": {
                "type": "string",
                "description": "可选的 ack 工具名。Proactive 处理完事件后调用以标记已读",
            },
            "poll_tool": {
                "type": "string",
                "description": "可选的内容预抓取工具名",
            },
            "args": {
                "type": "object",
                "description": "调用 get_tool 时的默认参数",
            },
            "enabled": {
                "type": "boolean",
                "description": "是否启用该 source，默认 true",
                "default": True,
            },
        },
        "required": ["name", "server", "channel"],
    }

    async def execute(
        self,
        name: str,
        server: str,
        channel: str,
        get_tool: str = "",
        ack_tool: str = "",
        poll_tool: str = "",
        args: dict[str, Any] | None = None,
        enabled: bool = True,
        **_: Any,
    ) -> str:
        """添加或更新 proactive source。

        输入:
            name: source 名称。
            server: 已连接 MCP server 名称。
            channel: "alert" / "content" / "context"。
            get_tool: 获取数据的远端工具名。
            ack_tool: ack 工具名。
            poll_tool: 预抓取工具名。
            args: 默认工具参数。
            enabled: 是否启用。

        输出:
            描述操作结果的字符串。
        """
        status = self._store.add_source({
            "name": str(name).strip(),
            "server": str(server).strip(),
            "channel": str(channel).strip(),
            "get_tool": str(get_tool or "").strip(),
            "ack_tool": str(ack_tool or "").strip(),
            "poll_tool": str(poll_tool or "").strip(),
            "args": dict(args or {}),
            "enabled": bool(enabled),
        })
        if status == "ok":
            return f"已保存 proactive source {name!r}。"
        return status


class ProactiveSourceRemoveTool(Tool):
    """移除一个 proactive 数据源。

    参数:
        store: SourceStore 实例。
    """

    def __init__(self, store: SourceStore) -> None:
        self._store = store

    name = "proactive_source_remove"
    description = "移除一个 Proactive 外部数据源（不会断开对应的 MCP server 连接）。"
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "要移除的 source 名称",
            },
        },
        "required": ["name"],
    }

    async def execute(self, name: str, **_: Any) -> str:
        """移除 proactive source。

        输入:
            name: source 名称。

        输出:
            描述操作结果的字符串。
        """
        return self._store.remove_source(name)


class ProactiveSourceListTool(Tool):
    """列出所有 proactive 数据源。

    参数:
        store: SourceStore 实例。
    """

    def __init__(self, store: SourceStore) -> None:
        self._store = store

    name = "proactive_source_list"
    description = "列出当前所有已配置的 Proactive 外部数据源及其 channel。"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        """列出所有 proactive source。

        输出:
            格式化的 source 列表字符串。
        """
        return self._store.list_sources()