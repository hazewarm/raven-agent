from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raven_agent.plugins.context import PluginContext


class Plugin(ABC):
    """所有插件的基类。

    输入:
        无。PluginManager 会实例化插件并注入 context。

    输出:
        Plugin 子类实例。
    """

    name: str | None = None
    version: str | None = None
    desc: str | None = None
    author: str | None = None
    context: PluginContext

    def __init_subclass__(cls, **kwargs: object) -> None:
        """插件子类定义时自动注册到全局 plugin_registry。

        输入:
            **kwargs: 类创建协议传入的扩展参数。

        输出:
            None。
        """

        super().__init_subclass__(**kwargs)
        from raven_agent.plugins.registry import plugin_registry

        plugin_registry.register_class(cls)

    async def initialize(self) -> None:
        """插件初始化入口。

        输入:
            无。

        输出:
            None。
        """

        return None

    async def terminate(self) -> None:
        """插件停止入口。

        输入:
            无。

        输出:
            None。
        """

        return None

    @property
    def plugin_dir(self):
        """返回当前插件目录。

        输入:
            无。

        输出:
            插件目录 Path；context 尚未注入时返回 None。
        """

        context = getattr(self, "context", None)
        return getattr(context, "plugin_dir", None)
    
    def before_turn_modules(self) -> list[object]:
        """返回 before_turn 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def before_reasoning_modules(self) -> list[object]:
        """返回 before_reasoning 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def prompt_render_modules(self) -> list[object]:
        """返回 prompt_render 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def before_step_modules(self) -> list[object]:
        """返回 before_step 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def after_step_modules(self) -> list[object]:
        """返回 after_step 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def after_reasoning_modules(self) -> list[object]:
        """返回 after_reasoning 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []

    def after_turn_modules(self) -> list[object]:
        """返回 after_turn 阶段模块。

        输入:
            无。

        输出:
            PhaseModule 列表。
        """

        return []