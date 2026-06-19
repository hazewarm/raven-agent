from __future__ import annotations

import asyncio
import json
from typing import Any

from raven_agent.tools.base import Tool
from raven_agent.tools.registry import ToolRegistry
from raven_agent.tools.search import ToolDiscoveryState, ToolSearchTool


class _FakeTool(Tool):
    """测试用假工具。

    参数:
        name: 工具名称。
        description: 工具描述。
    """

    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    async def execute(self, **kwargs: Any) -> str:
        """返回固定测试文本。

        参数:
            **kwargs: 测试调用参数，本工具不会使用。

        返回:
            字符串 ok。
        """

        return "ok"


def _run(coro: Any) -> Any:
    """同步运行一个异步调用。

    参数:
        coro: 要运行的 coroutine。

    返回:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def _make_registry() -> tuple[ToolRegistry, ToolSearchTool]:
    """构建测试用工具注册表。

    返回:
        二元组，第一项是 ToolRegistry，第二项是 ToolSearchTool。
    """

    registry = ToolRegistry()
    search_tool = ToolSearchTool(registry)
    registry.register(search_tool, always_on=True, risk="read-only")
    registry.register(
        _FakeTool("read_note", "读取笔记内容"),
        always_on=True,
        risk="read-only",
        search_hint="查看 笔记",
    )
    registry.register(
        _FakeTool("write_note", "保存内容到笔记文件"),
        risk="write",
        search_hint="写入 创建 记录",
    )
    registry.register(
        _FakeTool("send_message", "向用户发送一条外部通知"),
        risk="external-side-effect",
        search_hint="推送 通知",
    )
    return registry, search_tool


def test_visible_schemas_only_include_always_on_tools() -> None:
    """验证默认 schema 只包含 always-on 工具。

    返回:
        None。
    """

    registry, _ = _make_registry()

    schema_names = [
        schema["function"]["name"]
        for schema in registry.get_visible_schemas(loaded_names=set())
    ]

    assert schema_names == ["tool_search", "read_note"]
    assert registry.get_deferred_names(loaded_names=set())["builtin"] == [
        "send_message",
        "write_note",
    ]


def test_keyword_search_finds_deferred_tool() -> None:
    """验证关键词搜索可以命中 deferred 工具。

    返回:
        None。
    """

    registry, _ = _make_registry()

    results = registry.search("保存笔记", excluded_names=registry.get_visible_names())
    names = [item["name"] for item in results]

    assert "write_note" in names
    assert "read_note" not in names


def test_tool_search_select_unlocks_named_tool() -> None:
    """验证 select: 可以精确解锁指定工具。

    返回:
        None。
    """

    registry, search_tool = _make_registry()
    search_tool.set_excluded_names(registry.get_visible_names())

    result = _run(search_tool.execute("select:write_note"))
    payload = json.loads(result.text)

    assert payload["unlocked"] == ["write_note"]
    assert result.metadata["unlocked"] == ["write_note"]


def test_tool_search_respects_allowed_risk() -> None:
    """验证 allowed_risk 会阻止不允许的风险等级。

    返回:
        None。
    """

    registry, search_tool = _make_registry()
    search_tool.set_excluded_names(registry.get_visible_names())

    result = _run(
        search_tool.execute(
            "select:write_note,send_message",
            allowed_risk=["read-only"],
        )
    )
    payload = json.loads(result.text)

    assert payload["unlocked"] == []
    assert payload["risk_blocked"] == ["write_note", "send_message"]


def test_discovery_state_lru_is_session_scoped() -> None:
    """验证 ToolDiscoveryState 按 session 保存最近工具。

    返回:
        None。
    """

    state = ToolDiscoveryState(capacity=2)
    state.update("cli:1", ["a", "b"], always_on={"always"})
    state.update("cli:1", ["c"], always_on={"always"})
    state.update("cli:2", ["x"], always_on={"always"})

    assert state.get_preloaded("cli:1") == {"b", "c"}
    assert state.get_preloaded_ordered("cli:1") == ["b", "c"]
    assert state.get_preloaded("cli:2") == {"x"}


def test_discovery_state_skips_always_on_and_tool_search() -> None:
    """验证 LRU 不保存 always-on 工具和 tool_search。

    返回:
        None。
    """

    state = ToolDiscoveryState()
    state.update(
        "cli:1",
        ["tool_search", "read_note", "write_note"],
        always_on={"read_note"},
    )

    assert state.get_preloaded("cli:1") == {"write_note"}