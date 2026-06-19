from __future__ import annotations


from raven_agent.config import BuiltinPluginsConfig
from raven_agent.plugins import load_builtin_plugin_specs


def test_load_specs_respects_default_switches() -> None:
    """测试默认配置只加载默认启用的内置插件。

    输入:
        无。

    输出:
        None。
    """

    specs = load_builtin_plugin_specs(BuiltinPluginsConfig())
    names = {spec.name for spec in specs}

    assert names == {"shell_safety", "tool_loop_guard", "context_pressure", "status_commands"}


def test_load_specs_can_enable_optional_plugins() -> None:
    """测试显式启用 citation / observe / memory_rollup。

    输入:
        无。

    输出:
        None。
    """

    config = BuiltinPluginsConfig(citation=True, observe=True, memory_rollup=True)
    names = {spec.name for spec in load_builtin_plugin_specs(config)}

    assert {"citation", "observe", "memory_rollup"} <= names


def test_load_specs_can_disable_default_plugins() -> None:
    """测试关闭默认插件后不加载它。

    输入:
        无。

    输出:
        None。
    """

    config = BuiltinPluginsConfig(shell_safety=False)
    names = {spec.name for spec in load_builtin_plugin_specs(config)}

    assert "shell_safety" not in names