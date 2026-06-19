from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from raven_agent.plugins import Plugin, on_tool_pre
from raven_agent.plugins.context import PluginToolHookEvent
from raven_agent.tools.hooks import ToolHookOutcome

_DEFAULT_REPEAT_LIMIT = 3
_DENY_PREFIX = "tool_loop_guard:"
# tool_search 不计入循环：它本来就可能被连续调用以解锁不同工具。
_EXCLUDED_TOOLS = frozenset({"tool_search"})


@dataclass
class _LoopState:
    """单个 session 的工具循环状态。

    输入:
        signature: 上一次工具调用签名。
        repeat_count: 连续相同签名次数。

    输出:
        _LoopState 实例。
    """

    signature: str = ""
    repeat_count: int = 0


class ToolLoopGuardPlugin(Plugin):
    """检测连续重复工具调用并提前截断的内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        ToolLoopGuardPlugin 实例。
    """

    name = "tool_loop_guard"
    version = "0.1.0"
    desc = "检测连续重复的工具调用并提前截断"

    def __init__(self) -> None:
        """初始化插件状态。

        输入:
            无。

        输出:
            None。
        """

        self._states: dict[str, _LoopState] = {}
        self._repeat_limit = _DEFAULT_REPEAT_LIMIT

    @on_tool_pre()
    async def detect_repeated_tool_call(self, event: PluginToolHookEvent) -> ToolHookOutcome | None:
        """检测连续重复工具调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            连续重复超过阈值时返回 decision="deny"；否则返回 None 放行。
        """

        if event.tool_name in _EXCLUDED_TOOLS:
            return None
        signature = self._signature(event.tool_name, event.arguments)
        state = self._states.setdefault(self._state_key(event), _LoopState())
        if signature == state.signature:
            state.repeat_count += 1
        else:
            state.signature = signature
            state.repeat_count = 1
        if state.repeat_count < self._repeat_limit:
            return None
        return ToolHookOutcome(
            decision="deny",
            reason=(
                f"{_DENY_PREFIX}连续重复调用工具 {state.repeat_count} 次，"
                "已截断，请基于已有结果直接回答用户。"
            ),
        )

    def _state_key(self, event: PluginToolHookEvent) -> str:
        """计算循环状态 key。

        输入:
            event: PluginToolHookEvent。

        输出:
            按 session_key 区分的状态 key。
        """

        return event.session_key or "__default__"

    def _signature(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """计算一次工具调用的签名。

        输入:
            tool_name: 工具名。
            arguments: 工具参数。

        输出:
            tool_name 与排序后参数 JSON 组成的签名字符串。
        """

        args = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"{tool_name}:{args}"