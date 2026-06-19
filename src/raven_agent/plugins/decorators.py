from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_origin

import docstring_parser

from raven_agent.plugins.registry import (
    PluginEventName,
    PluginHandlerKind,
    PluginHandlerMetadata,
    plugin_registry,
)


def _event_decorator(
    event_name: PluginEventName,
    **options: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """创建事件 handler 装饰器。

    输入:
        event_name: 插件事件名。
        **options: 装饰器选项，例如 priority。

    输出:
        可作用于函数的装饰器。
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        """登记事件 handler 元数据。

        输入:
            func: 被装饰的插件方法。

        输出:
            原函数。
        """

        existing = plugin_registry._handlers.get_by_name(
            event_name,
            func.__name__,
            func.__module__,
        )
        if existing is not None:
            return func
        plugin_registry._handlers.append(
            PluginHandlerMetadata(
                kind=PluginHandlerKind.EVENT,
                event_name=event_name,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                priority=int(options.get("priority", 0) or 0),
            )
        )
        return func

    return decorate



def on_turn_started(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 TurnStarted 事件 handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.TURN_STARTED, **options)


def on_turn_completed(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 TurnCompleted 观察事件 handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.TURN_COMPLETED, **options)


def on_before_turn(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 BeforeTurnCtx GATE handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.BEFORE_TURN, **options)


def on_before_reasoning(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 BeforeReasoningCtx GATE handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.BEFORE_REASONING, **options)


def on_prompt_render(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 PromptRenderCtx GATE handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.PROMPT_RENDER, **options)


def on_before_step(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 BeforeStepCtx GATE handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.BEFORE_STEP, **options)


def on_after_step(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 AfterStepCtx TAP handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.AFTER_STEP, **options)


def on_after_reasoning(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 AfterReasoningCtx GATE handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.AFTER_REASONING, **options)


def on_after_turn(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 AfterTurnCtx TAP handler。

    输入:
        **options: 装饰器选项，例如 priority。

    输出:
        函数装饰器。
    """

    return _event_decorator(PluginEventName.AFTER_TURN, **options)


# 兼容旧项目命名。
# on_before_turn = on_turn_started
# on_after_turn = on_turn_completed


def _tool_hook_decorator(
    *,
    hook_event: str,
    tool_name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """创建工具 Hook 装饰器。

    输入:
        hook_event: Hook 事件名。
        tool_name: 只匹配某个工具；None 表示匹配所有工具。
        priority: 优先级，越大越靠前。

    输出:
        函数装饰器。
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        """登记 tool hook metadata。

        输入:
            func: 被装饰的插件方法。

        输出:
            原函数。
        """

        plugin_registry._handlers.append(
            PluginHandlerMetadata(
                kind=PluginHandlerKind.TOOL_HOOK,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                hook_event=hook_event,
                hook_tool_name=tool_name,
                priority=priority,
            )
        )
        return func

    return decorate


def on_tool_pre(
    *,
    tool_name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 pre_tool_use 工具 Hook。

    输入:
        tool_name: 只匹配某个工具；None 表示匹配所有工具。
        priority: 优先级，越大越靠前。

    输出:
        函数装饰器。
    """

    return _tool_hook_decorator(
        hook_event="pre_tool_use",
        tool_name=tool_name,
        priority=priority,
    )


def on_tool_post(
    *,
    tool_name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 post_tool_use 工具 Hook。

    输入:
        tool_name: 只匹配某个工具；None 表示匹配所有工具。
        priority: 优先级，越大越靠前。

    输出:
        函数装饰器。
    """

    return _tool_hook_decorator(
        hook_event="post_tool_use",
        tool_name=tool_name,
        priority=priority,
    )


def on_tool_error(
    *,
    tool_name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明 post_tool_error 工具 Hook。

    输入:
        tool_name: 只匹配某个工具；None 表示匹配所有工具。
        priority: 优先级，越大越靠前。

    输出:
        函数装饰器。
    """

    return _tool_hook_decorator(
        hook_event="post_tool_error",
        tool_name=tool_name,
        priority=priority,
    )

def tool(
    name: str,
    *,
    risk: str = "read-only",
    always_on: bool = False,
    search_hint: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明一个插件工具。

    输入:
        name: 暴露给模型的工具名。
        risk: 工具风险等级，使用 ToolRegistry 支持的 risk 值。
        always_on: 是否每轮默认暴露给模型。
        search_hint: 工具搜索提示。

    输出:
        函数装饰器。
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        """登记工具 metadata。

        输入:
            func: 被装饰的插件方法。前两个参数必须是 self 和 event。

        输出:
            原函数。
        """

        params = list(inspect.signature(func).parameters.keys())
        if len(params) < 2 or params[0] != "self" or params[1] != "event":
            raise TypeError(f"@tool handler 前两个参数必须是 self 和 event: {func.__qualname__}")
        plugin_registry._handlers.append(
            PluginHandlerMetadata(
                kind=PluginHandlerKind.TOOL,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                tool_name=name,
                tool_schema=_derive_parameters_schema(func),
                tool_risk=risk,
                tool_always_on=always_on,
                tool_search_hint=search_hint,
            )
        )
        return func

    return decorate


_PY_TO_JSON: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _json_type_from_annotation(annotation: object) -> str:
    """把 Python 类型注解转换成 JSON Schema type。

    输入:
        annotation: 函数参数注解。

    输出:
        JSON Schema type 字符串。
    """

    if annotation is inspect.Parameter.empty:
        return "string"
    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _PY_TO_JSON.get(getattr(annotation, "__name__", ""), "string")


def _derive_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """从插件工具函数签名和 docstring 推导 JSON Schema。

    输入:
        func: 插件工具方法。

    输出:
        JSON Schema object。
    """

    signature = inspect.signature(func)
    docs = docstring_parser.parse(func.__doc__ or "")
    param_docs = {item.arg_name: item.description for item in docs.params}
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, parameter in signature.parameters.items():
        if param_name in {"self", "event"}:
            continue
        prop: dict[str, Any] = {"type": _json_type_from_annotation(parameter.annotation)}
        description = param_docs.get(param_name)
        if description:
            prop["description"] = description
        properties[param_name] = prop
        if parameter.default is inspect.Parameter.empty:
            required.append(param_name)

    return {"type": "object", "properties": properties, "required": required}