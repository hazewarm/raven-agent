from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class PluginHandlerKind(Enum):
    """插件 handler 类型。

    输入:
        无。

    输出:
        PluginHandlerKind 枚举值。
    """

    EVENT = auto()
    TOOL = auto()
    TOOL_HOOK = auto()


class PluginEventName(Enum):
    """插件支持的事件名称。

    输入:
        无。

    输出:
        PluginEventName 枚举值。
    """

    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    BEFORE_TURN = "before_turn"
    BEFORE_REASONING = "before_reasoning"
    PROMPT_RENDER = "prompt_render"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    AFTER_REASONING = "after_reasoning"
    AFTER_TURN = "after_turn"


@dataclass(frozen=True)
class PluginHandlerMetadata:
    """装饰器产生的 handler 元数据。

    输入:
        kind: handler 类型。
        handler: 原始函数对象。
        handler_name: 函数名。
        plugin_module_path: 函数所属模块名。
        event_name: lifecycle / turn event 名；仅 EVENT handler 使用。
        tool_name / tool_schema / tool_risk / tool_always_on / tool_search_hint: tool metadata。
        hook_event: 工具 hook 事件，pre_tool_use / post_tool_use / post_tool_error。
        hook_tool_name: tool hook 匹配的工具名；None 表示匹配所有工具。
        priority: 优先级，越大越靠前。

    输出:
        PluginHandlerMetadata 实例。
    """

    kind: PluginHandlerKind
    handler: Callable[..., Any]
    handler_name: str
    plugin_module_path: str
    event_name: PluginEventName | None = None
    tool_name: str | None = None
    tool_schema: dict[str, Any] | None = None
    tool_risk: str | None = None
    tool_always_on: bool = False
    tool_search_hint: str | None = None
    hook_event: str = "pre_tool_use"
    hook_tool_name: str | None = None
    priority: int = 0


class PluginHandlerRegistry:
    """保存插件装饰器产生的 handler metadata。

    输入:
        无。

    输出:
        PluginHandlerRegistry 实例。
    """

    def __init__(self) -> None:
        self._handlers: list[PluginHandlerMetadata] = []

    def append(self, metadata: PluginHandlerMetadata) -> None:
        """追加 metadata 并按 priority 降序排列。

        输入:
            metadata: 要保存的 handler 元数据。

        输出:
            None。
        """

        self._handlers.append(metadata)
        self._handlers.sort(key=lambda item: -item.priority)

    def get_by_name(
        self,
        event_name: PluginEventName,
        handler_name: str,
        module_path: str,
    ) -> PluginHandlerMetadata | None:
        """按事件、函数名、模块名查找已有 metadata。

        输入:
            event_name: 事件名。
            handler_name: 函数名。
            module_path: 模块名。

        输出:
            找到时返回 metadata，否则返回 None。
        """

        for item in self._handlers:
            if (
                item.event_name == event_name
                and item.handler_name == handler_name
                and item.plugin_module_path == module_path
            ):
                return item
        return None

    def get_by_module_path(self, module_path: str) -> list[PluginHandlerMetadata]:
        """返回某插件模块声明的所有 metadata。

        输入:
            module_path: 插件 import module name。

        输出:
            metadata 列表。
        """

        return [item for item in self._handlers if item.plugin_module_path == module_path]

    def remove_by_module_path(self, module_path: str) -> None:
        """移除某插件模块的所有 metadata。

        输入:
            module_path: 插件 import module name。

        输出:
            None。
        """

        self._handlers = [
            item for item in self._handlers if item.plugin_module_path != module_path
        ]

    def clear(self) -> None:
        """清空所有 metadata。

        输入:
            无。

        输出:
            None。
        """

        self._handlers.clear()


class PluginRegistry:
    """插件类、实例和 handler metadata 的全局注册表。

    输入:
        无。

    输出:
        PluginRegistry 实例。
    """

    def __init__(self) -> None:
        self._handlers = PluginHandlerRegistry()
        self._classes: dict[str, type] = {}
        self._instances: dict[str, object] = {}

    def register_class(self, plugin_class: type) -> None:
        """登记一个 Plugin 子类。

        输入:
            plugin_class: 被导入模块中定义的 Plugin 子类。

        输出:
            None。
        """

        self._classes[plugin_class.__module__] = plugin_class

    def get_class(self, module_path: str) -> type | None:
        """按模块名获取 Plugin 子类。

        输入:
            module_path: 插件 import module name。

        输出:
            找到时返回 class，否则返回 None。
        """

        return self._classes.get(module_path)

    def register_instance(self, module_path: str, instance: object) -> None:
        """登记一个已加载插件实例。

        输入:
            module_path: 插件 import module name。
            instance: Plugin 实例。

        输出:
            None。
        """

        self._instances[module_path] = instance

    def get_instance(self, module_path: str) -> object | None:
        """按模块名获取插件实例。

        输入:
            module_path: 插件 import module name。

        输出:
            找到时返回插件实例，否则返回 None。
        """

        return self._instances.get(module_path)

    def get_handlers_by_module_path(self, module_path: str) -> list[PluginHandlerMetadata]:
        """返回某插件模块声明的所有 metadata。

        输入:
            module_path: 插件 import module name。

        输出:
            metadata 列表。
        """

        return self._handlers.get_by_module_path(module_path)

    def remove_plugin(self, module_path: str) -> None:
        """移除某插件的类、实例和 metadata。

        输入:
            module_path: 插件 import module name。

        输出:
            None。
        """

        self._handlers.remove_by_module_path(module_path)
        self._classes.pop(module_path, None)
        self._instances.pop(module_path, None)

    def clear(self) -> None:
        """清空 registry，主要用于测试隔离。

        输入:
            无。

        输出:
            None。
        """

        self._handlers.clear()
        self._classes.clear()
        self._instances.clear()


plugin_registry = PluginRegistry()