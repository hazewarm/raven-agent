from __future__ import annotations

from raven_agent.config import BuiltinPluginsConfig
from raven_agent.plugins.manager import BuiltinPluginSpec

from raven_agent.plugins.builtins.citation import CitationPlugin
from raven_agent.plugins.builtins.context_pressure import ContextPressurePlugin
from raven_agent.plugins.builtins.memory_rollup import MemoryRollupPlugin
from raven_agent.plugins.builtins.observe.plugin import ObservePlugin
from raven_agent.plugins.builtins.shell_safety import ShellSafetyPlugin
from raven_agent.plugins.builtins.status_commands import StatusCommandsPlugin
from raven_agent.plugins.builtins.tool_loop_guard import ToolLoopGuardPlugin


# name -> Plugin 子类。顺序决定加载顺序（shell_safety / tool_loop_guard 先于业务插件）。
_BUILTIN_PLUGIN_CLASSES: dict[str, type] = {
    "shell_safety": ShellSafetyPlugin,
    "tool_loop_guard": ToolLoopGuardPlugin,
    "context_pressure": ContextPressurePlugin,
    "status_commands": StatusCommandsPlugin,
    "citation": CitationPlugin,
    "memory_rollup": MemoryRollupPlugin,
    "observe": ObservePlugin,
}


def load_builtin_plugin_specs(
    config: BuiltinPluginsConfig,
) -> list[BuiltinPluginSpec]:
    """根据配置生成启用的内置插件 spec 列表。

    输入:
        config: BuiltinPluginsConfig 开关。

    输出:
        BuiltinPluginSpec 列表，按 _BUILTIN_PLUGIN_CLASSES 顺序，只包含启用项。
    """

    specs: list[BuiltinPluginSpec] = []
    for name, plugin_class in _BUILTIN_PLUGIN_CLASSES.items():
        if config.is_enabled(name):
            specs.append(BuiltinPluginSpec(name=name, plugin_class=plugin_class))
    return specs


def builtin_plugin_names() -> list[str]:
    """返回所有可用内置插件名。

    输入:
        无。

    输出:
        内置插件名列表，便于文档或诊断命令展示。
    """

    return list(_BUILTIN_PLUGIN_CLASSES.keys())