from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raven_agent.plugins.config import PluginConfig


@dataclass
class PluginContext:
    """插件运行上下文。

    输入:
        event_bus: 当前 EventBus。
        tool_registry: 当前 ToolRegistry。
        plugin_id: 插件 ID。
        plugin_dir: 插件目录。
        kv_store: 插件私有 KV 存储。
        config: 插件私有配置。
        workspace: raven-agent workspace 根目录。
        session_manager: 当前 SessionManager。
        memory_engine: 当前结构化语义 MemoryEngine。
        memory_maintenance: 当前 MarkdownMemoryMaintenance；用于手动 consolidation。
        memory_optimizer: 当前 MemoryOptimizer；用于手动 PENDING -> MEMORY 归档。

    输出:
        PluginContext 实例。
    """

    event_bus: Any
    tool_registry: Any
    plugin_id: str
    plugin_dir: Path
    kv_store: PluginKVStore
    config: PluginConfig | None = None
    workspace: Path | None = None
    session_manager: Any = None
    memory_engine: Any = None
    memory_maintenance: Any = None
    memory_optimizer: Any = None


@dataclass(frozen=True)
class PluginToolEvent:
    """插件工具执行事件对象。

    输入:
        plugin: 当前插件实例。
        context: PluginContext。
        tool_name: 工具名称。
        arguments: 工具参数。

    输出:
        PluginToolEvent 实例。
    """

    plugin: Any
    context: PluginContext
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PluginToolHookEvent:
    """插件工具 Hook 事件对象。

    输入:
        context: PluginContext。
        event: Hook 事件名，pre_tool_use / post_tool_use / post_tool_error。
        session_key: 当前会话 key。
        tool_name: 工具名。
        arguments: 当前工具参数。
        call_id: 模型 tool call id。
        metadata: ToolExecutionRequest.metadata。
        result: post_tool_use 时的工具结果对象。
        error: post_tool_error 时的错误文本。

    输出:
        PluginToolHookEvent 实例。
    """

    context: PluginContext
    event: str
    session_key: str
    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    metadata: dict[str, Any] | None = None
    result: Any = None
    error: str = ""


class PluginKVStore:
    """插件私有 JSON KV 存储。

    输入:
        path: KV JSON 文件路径，通常是 plugin_dir/.kv.json。

    输出:
        PluginKVStore 实例。
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, key: str, default: Any = None) -> Any:
        """读取 KV 值。

        输入:
            key: 键名。
            default: 缺失时返回的值。

        输出:
            KV 值或 default。不会创建文件。
        """

        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """写入 KV 值。

        输入:
            key: 键名。
            value: JSON 可序列化值。

        输出:
            None。会创建或覆盖 .kv.json。
        """

        data = self._read()
        data[key] = value
        self._write(data)

    def increment(self, key: str, delta: int = 1) -> int:
        """递增整数 KV 值。

        输入:
            key: 键名。
            delta: 增量。

        输出:
            递增后的整数。会创建或覆盖 .kv.json。
        """

        data = self._read()
        new_value = int(data.get(key, 0)) + delta
        data[key] = new_value
        self._write(data)
        return new_value

    def _read(self) -> dict[str, Any]:
        """读取 JSON 文件。

        输入:
            无。

        输出:
            字典。文件不存在时返回空字典。
        """

        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        """写入 JSON 文件。

        输入:
            data: 要写入的字典。

        输出:
            None。
        """

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )