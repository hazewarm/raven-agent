"""McpToolWrapper: 把 MCP server 的远端工具包装成本地 Tool。"""

from __future__ import annotations

from typing import Any

from raven_agent.mcp.client import McpClient
from raven_agent.tools.base import Tool


class McpToolWrapper(Tool):
    """将单个 MCP 远端工具暴露为标准本地 Tool。

    工具名格式：mcp_{server_name}__{tool_name}
    用双下划线分隔 server 名和工具名，避免与内置工具冲突，
    也方便按 server 批量识别和注销。

    参数:
        client: 已连接的 McpClient 实例。
        server_name: MCP server 的逻辑名称（用于构造工具前缀）。
        info: 远端工具的元数据字典（来自 client.list_tools() 的返回值）。
    """

    def __init__(
        self,
        client: McpClient,
        server_name: str,
        info: dict[str, Any],
    ) -> None:
        self._client = client
        self._server_name = server_name
        self._info = info

    @property
    def name(self) -> str:
        """返回 raven-agent 内部的工具名。

        输出:
            "mcp_{server_name}__{tool_name}" 格式的字符串。
        """
        return f"mcp_{self._server_name}__{self._info['name']}"

    @property
    def description(self) -> str:
        """返回带 server 前缀的工具描述。

        输出:
            以 "[MCP:{server_name}]" 为前缀的描述字符串，
            帮助模型理解该工具的来源。
        """
        desc = self._info.get("description", "")
        return f"[MCP:{self._server_name}] {desc}"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回远端工具声明的 JSON Schema 参数定义。

        输出:
            JSON Schema 字典（来自 MCP server 的 inputSchema）。
        """
        return self._info.get(
            "input_schema", {"type": "object", "properties": {}}
        )

    async def execute(self, **kwargs: Any) -> str:
        """代理到 McpClient.call_tool()，调用远端工具并转为文本。

        输入:
            **kwargs: 模型传入的工具参数，直接透传给远端。

        输出:
            远端工具返回的文本结果。McpClient.call_tool() 返回 Any
            （可能是 str / list / None / 二进制数据），本方法负责将其
            归一化为 LLM 可消费的字符串。多段 content 用换行符拼接。
        
        """
        raw = await self._client.call_tool(self._info["name"], kwargs)

        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return "\n".join(str(item) for item in raw)
        # 其他类型（int / dict / bytes 等）→ 字符串兜底
        return str(raw)