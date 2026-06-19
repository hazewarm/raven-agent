from __future__ import annotations

from typing import cast

from raven_agent.memory.engine import MemoryEngine, MemoryToolSpec
from raven_agent.tools.base import Tool
from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome
from raven_agent.tools.memory_tools import ForgetMemoryTool, MemorizeTool, RecallMemoryTool
from raven_agent.tools.registry import ToolRegistry

MEMORY_TOOL_NAMES: frozenset[str] = frozenset({"recall_memory", "memorize", "forget_memory"})
_MEMORY_CONTEXT_KEYS: tuple[str, ...] = ("current_user_source_ref", "channel", "chat_id")


def register_memory_tools(registry: ToolRegistry, engine: MemoryEngine, *, event_bus: object | None = None,) -> list[str]:
    """根据 engine.tool_profile() 把记忆工具注册进 registry。

    参数:
        registry: 目标 ToolRegistry。
        engine: 提供 tool_profile() 的 MemoryEngine。

    返回:
        实际注册成功的工具名列表（按 memorize / forget / recall 顺序）。
    """

    profile = engine.tool_profile()
    registered: list[str] = []

    if profile.memorize is not None:
        registered += _register_one(registry, engine, profile.memorize, MemorizeTool, event_bus=event_bus,)
    if profile.forget is not None:
        registered += _register_one(registry, engine, profile.forget, ForgetMemoryTool)
    if profile.recall is not None:
        registered += _register_one(registry, engine, profile.recall, RecallMemoryTool, event_bus=event_bus,)
    return registered


def _register_one(
    registry: ToolRegistry,
    engine: MemoryEngine,
    spec: MemoryToolSpec,
    default_cls: type,
    *,
    event_bus: object | None = None,
) -> list[str]:
    """构造并注册单个记忆工具。

    参数:
        registry: 目标 ToolRegistry。
        engine: 传给工具的 MemoryEngine。
        spec: 该工具的 MemoryToolSpec。
        default_cls: spec.tool_class 为空时使用的默认工具类。

    返回:
        注册成功返回 [工具名]；工具名非法或已存在返回 []。
    """

    tool = _build_tool(engine, spec, default_cls, event_bus=event_bus)
    if tool.name not in MEMORY_TOOL_NAMES:
        raise ValueError(f"未知 memory 工具: {tool.name}")
    if registry.has_tool(tool.name):
        return []
    registry.register(
        tool,
        risk=spec.risk,
        always_on=True,
        search_hint=spec.search_hint or None,
    )
    return [tool.name]


def _build_tool(engine: MemoryEngine, spec: MemoryToolSpec, default_cls: type, event_bus: object | None = None) -> Tool:
    """实例化工具，支持 spec.tool_class 覆盖默认类。

    参数:
        engine: 传给工具的 MemoryEngine。
        spec: MemoryToolSpec。
        default_cls: 默认工具类。
        event_bus: 事件总线。

    返回:
        Tool 实例。
    """

    cls = spec.tool_class if spec.tool_class is not None else default_cls
    return cast(Tool, cls(engine, spec, event_bus=event_bus))


class MemoryToolContextHook(ToolHook):
    """向记忆工具注入每轮运行时上下文的 pre hook。

    参数:
        无。

    返回:
        MemoryToolContextHook 实例。
    """

    name = "memory_tool_context"
    event = "pre_tool_use"

    def matches(self, context: ToolHookContext) -> bool:
        """仅匹配记忆工具。

        参数:
            context: 当前 hook 上下文。

        返回:
            工具名属于记忆工具时返回 True。
        """

        return context.request.tool_name in MEMORY_TOOL_NAMES

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """把每轮上下文并入工具参数。

        参数:
            context: 当前 hook 上下文，request.metadata 携带每轮上下文。

        返回:
            ToolHookOutcome；当有字段注入时返回 updated_arguments。
        """

        turn_context = context.request.metadata or {}
        merged = dict(context.current_arguments)
        changed = False
        for key in _MEMORY_CONTEXT_KEYS:
            value = turn_context.get(key)
            if value and not merged.get(key):
                merged[key] = value
                changed = True
        if not changed:
            return ToolHookOutcome()
        return ToolHookOutcome(updated_arguments=merged)