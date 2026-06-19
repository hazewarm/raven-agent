from __future__ import annotations

from typing import Any


class PluginConfig:
    """插件私有配置对象。

    输入:
        values: _conf_schema.json 默认值与 plugin_config.json 覆盖值合并后的字典。

    输出:
        PluginConfig 实例。
    """

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def get(self, key: str, default: Any = None) -> Any:
        """按 key 读取配置。

        输入:
            key: 配置键。
            default: 缺失时返回的默认值。

        输出:
            配置值或 default。
        """

        return self._values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """返回配置字典副本。

        输入:
            无。

        输出:
            配置字典副本。
        """

        return dict(self._values)

    def __getattr__(self, key: str) -> Any:
        """允许通过属性读取配置。

        输入:
            key: 属性名。

        输出:
            配置值。

        异常:
            AttributeError: 配置不存在时抛出。
        """

        try:
            return self._values[key]
        except KeyError as exc:
            raise AttributeError(key) from exc