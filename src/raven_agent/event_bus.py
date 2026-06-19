from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast


E = TypeVar("E")
EventHandler = Callable[[E], E | None | Awaitable[E | None]]


class EventBus:
    """轻量事件总线。

    参数:
        无。通过 on() 注册 handler，再通过 emit() 或 observe() 触发。
    """

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[EventHandler[object]]] = {}

    def on(self, event_type: type[E], handler: EventHandler[E]) -> None:
        """注册事件处理函数。

        参数:
            event_type: 要监听的事件类型。
            handler: 处理函数。emit 模式下可返回新事件替换旧事件。

        返回:
            None。
        """

        handlers = self._handlers.setdefault(cast(type[object], event_type), [])
        handlers.append(cast(EventHandler[object], handler))

    def off(self, event_type: type[E], handler: EventHandler[E]) -> None:
        """注销事件处理函数。

        输入:
            event_type: 要解绑的事件类型。
            handler: 之前通过 on() 注册的处理函数。

        输出:
            None。handler 不存在时直接返回。
        """

        handlers = self._handlers.get(cast(type[object], event_type))
        if not handlers:
            return
        target = cast(EventHandler[object], handler)
        self._handlers[cast(type[object], event_type)] = [
            item for item in handlers if item != target
        ]
    
    async def emit(self, event: E) -> E:
        """按注册顺序触发可干预事件。

        参数:
            event: 要触发的事件对象。

        返回:
            最终事件对象。handler 返回非 None 时会替换当前事件。
        """

        current = event
        for raw_handler in self._handlers.get(cast(type[object], type(event)), []):
            handler = cast(EventHandler[E], raw_handler)
            result = handler(current)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                current = result
        return current

    async def observe(self, event: E) -> None:
        """按注册顺序触发观察事件。

        参数:
            event: 要观察的事件对象。

        返回:
            None。观察者返回值会被忽略。
        """

        for raw_handler in self._handlers.get(cast(type[object], type(event)), []):
            handler = cast(EventHandler[E], raw_handler)
            result = handler(event)
            if inspect.isawaitable(result):
                await result