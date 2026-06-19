from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass

import yaml

from raven_agent.events import TurnCompleted, TurnStarted
from raven_agent.lifecycle import LifecycleModules
from raven_agent.plugins.config import PluginConfig
from raven_agent.plugins.context import (
    PluginContext,
    PluginKVStore,
    PluginToolEvent,
    PluginToolHookEvent,
)
from raven_agent.plugins.registry import (
    PluginEventName,
    PluginHandlerKind,
    plugin_registry,
)
from raven_agent.tools.base import Tool
from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome

from raven_agent.lifecycle import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
    LifecycleModules,
    PromptRenderCtx,
)

logger = logging.getLogger(__name__)

_EVENT_TYPE_MAP: dict[PluginEventName, type] = {
    PluginEventName.TURN_STARTED: TurnStarted,
    PluginEventName.TURN_COMPLETED: TurnCompleted,
    PluginEventName.BEFORE_TURN: BeforeTurnCtx,
    PluginEventName.BEFORE_REASONING: BeforeReasoningCtx,
    PluginEventName.PROMPT_RENDER: PromptRenderCtx,
    PluginEventName.BEFORE_STEP: BeforeStepCtx,
    PluginEventName.AFTER_STEP: AfterStepCtx,
    PluginEventName.AFTER_REASONING: AfterReasoningCtx,
    PluginEventName.AFTER_TURN: AfterTurnCtx,
}

_PHASE_PROVIDERS: tuple[str, ...] = (
    "before_turn_modules",
    "before_reasoning_modules",
    "prompt_render_modules",
    "before_step_modules",
    "after_step_modules",
    "after_reasoning_modules",
    "after_turn_modules",
)

@dataclass(frozen=True)
class BuiltinPluginSpec:
    """内置插件加载描述。

    输入:
        name: 内置插件名，例如 "shell_safety"。
        plugin_class: 已 import 的 Plugin 子类。

    输出:
        BuiltinPluginSpec 实例。
    """

    name: str
    plugin_class: type


class PluginManager:
    """插件管理器。

    输入:
        plugin_dirs: 插件根目录列表。
        event_bus: 当前 EventBus。
        tool_registry: 当前 ToolRegistry。
        workspace: workspace 根目录。
        session_manager: 当前 SessionManager。
        memory_engine: 当前 MemoryEngine。

    输出:
        PluginManager 实例。
    """

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: Any,
        tool_registry: Any = None,
        workspace: Path | None = None,
        session_manager: Any = None,
        memory_engine: Any = None,
        memory_maintenance: Any = None,
        memory_optimizer: Any = None,
        builtin_specs: list[BuiltinPluginSpec] | None = None,
    ) -> None:
        self._dirs = list(plugin_dirs)
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._session_manager = session_manager
        self._memory_engine = memory_engine
        self._memory_maintenance = memory_maintenance
        self._memory_optimizer = memory_optimizer
        self._builtin_specs = list(builtin_specs or [])
        self._loaded: set[str] = set()
        self._plugin_tools: dict[str, list[str]] = {}
        self._event_handlers: dict[str, list[tuple[type, Any]]] = {}
        self._tool_hooks: list[ToolHook] = []
        self._modules: dict[str, list[object]] = {
            provider: [] for provider in _PHASE_PROVIDERS
        }

    @property
    def loaded_count(self) -> int:
        """返回已加载插件数量。

        输入:
            无。

        输出:
            已加载插件数量。
        """

        return len(self._loaded)

    @property
    def tool_hooks(self) -> list[ToolHook]:
        """返回插件注册的工具 Hook 列表副本。

        输入:
            无。

        输出:
            ToolHook 列表。
        """

        return list(self._tool_hooks)

    def lifecycle_modules(self) -> LifecycleModules:
        """把收集到的 phase modules 打包成 LifecycleModules。

        输入:
            无。

        输出:
            LifecycleModules，供 PassiveTurnPipeline 使用。
        """

        return LifecycleModules(
            before_turn=list(self._modules["before_turn_modules"]),
            before_reasoning=list(self._modules["before_reasoning_modules"]),
            prompt_render=list(self._modules["prompt_render_modules"]),
            before_step=list(self._modules["before_step_modules"]),
            after_step=list(self._modules["after_step_modules"]),
            after_reasoning=list(self._modules["after_reasoning_modules"]),
            after_turn=list(self._modules["after_turn_modules"]),
        )
    
    def discover(self) -> list[dict[str, str]]:
        """扫描插件目录，发现可加载插件。

        输入:
            无。

        输出:
            插件描述列表，每项含 name / module_path / import_path。
        """

        modules: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for root in self._dirs:
            if not root.is_dir():
                continue
            source = root.name
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                main = child / "plugin.py"
                if not main.exists():
                    continue
                if child.name in seen_names:
                    logger.warning("插件名重复，跳过: %s (%s)", child.name, main)
                    continue
                seen_names.add(child.name)
                modules.append(
                    {
                        "name": child.name,
                        "module_path": str(main),
                        "import_path": f"raven_plugin_{source}_{child.name}",
                    }
                )
        return modules

    async def load_all(self) -> None:
        """加载所有内置插件和外部目录插件。

        输入:
            无。

        输出:
            None。
        """

        for spec in self._builtin_specs:
            await self._load_builtin(spec)
        for module in self.discover():
            await self._load_one(module)
    
    async def _load_builtin(self, spec: BuiltinPluginSpec) -> None:
        """加载单个内置插件。

        输入:
            spec: 内置插件 spec。

        输出:
            None。失败时记录 warning 并跳过。
        """

        import_path = f"builtin:{spec.name}"
        if import_path in self._loaded:
            return

        plugin_dir = self._builtin_plugin_dir(spec.name)
        instance = spec.plugin_class()
        if not getattr(instance, "name", None):
            instance.name = spec.name  # type: ignore[attr-defined]
        instance.context = self._make_context(instance, plugin_dir, import_path)  # type: ignore[attr-defined]
        plugin_registry.register_instance(import_path, instance)

        snapshot = self._snapshot_counts(import_path)
        tool_names: list[str] = []
        try:
            self._bind_event_handlers(instance, spec.plugin_class.__module__)
            tool_names = self._register_tools(instance, spec.plugin_class.__module__)
            self._bind_tool_hooks(instance, spec.plugin_class.__module__)
            self._collect_phase_modules(instance)
            await instance.initialize()
        except Exception as exc:
            logger.warning("内置插件 %s 初始化失败，回滚: %s", spec.name, exc)
            self._rollback_plugin(import_path, tool_names, snapshot)
            return

        self._plugin_tools[import_path] = tool_names
        self._loaded.add(import_path)
        logger.info("内置插件已加载: %s", spec.name)

    def _builtin_plugin_dir(self, name: str) -> Path:
        """计算内置插件的私有数据目录。

        输入:
            name: 内置插件名。

        输出:
            workspace 下的 builtin_plugins/<name> 目录；无 workspace 时退化到 .raven 相对目录。
        """

        base = self._workspace if self._workspace is not None else Path(".raven")
        return base / "builtin_plugins" / name

    def _make_context(self, instance: Any, plugin_dir: Path, import_path: str) -> PluginContext:
        """为内置插件构造 PluginContext。

        输入:
            instance: 插件实例。
            plugin_dir: 插件私有数据目录。
            import_path: 插件在 registry 中的 key。

        输出:
            PluginContext。
        """

        plugin_id = str(getattr(instance, "name", None) or import_path)
        return PluginContext(
            event_bus=self._event_bus,
            tool_registry=self._tool_registry,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            kv_store=PluginKVStore(plugin_dir / ".kv.json"),
            config=None,
            workspace=self._workspace,
            session_manager=self._session_manager,
            memory_engine=self._memory_engine,
            memory_maintenance=self._memory_maintenance,
            memory_optimizer=self._memory_optimizer,
        )

    async def _load_one(self, module: dict[str, str]) -> None:
        """加载单个插件。

        输入:
            module: discover() 产生的插件描述。

        输出:
            None。失败时记录 warning 并跳过。
        """

        import_path = module["import_path"]
        if import_path in self._loaded:
            return
        plugin_dir = Path(module["module_path"]).parent
        if _is_plugin_disabled(plugin_dir):
            logger.info("插件已禁用（plugin.disabled）: %s", module["name"])
            return

        try:
            self._import_plugin(import_path, Path(module["module_path"]))
        except Exception as exc:
            logger.warning("插件 %s 导入失败: %s", module["name"], exc)
            plugin_registry.remove_plugin(import_path)
            return

        plugin_class = plugin_registry.get_class(import_path)
        if plugin_class is None:
            logger.warning("插件 %s 未注册 Plugin 子类", module["name"])
            plugin_registry.remove_plugin(import_path)
            return

        instance = plugin_class()
        _apply_manifest(instance, plugin_dir)
        plugin_id = str(getattr(instance, "name", None) or module["name"])
        instance.context = PluginContext(  # type: ignore[attr-defined]
            event_bus=self._event_bus,
            tool_registry=self._tool_registry,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            kv_store=PluginKVStore(plugin_dir / ".kv.json"),
            config=_load_plugin_config(plugin_dir),
            workspace=self._workspace,
            session_manager=self._session_manager,
            memory_engine=self._memory_engine,
            memory_maintenance=self._memory_maintenance,
            memory_optimizer=self._memory_optimizer,
        )
        plugin_registry.register_instance(import_path, instance)

        snapshot = self._snapshot_counts(import_path)
        tool_names: list[str] = []
        try:
            self._bind_event_handlers(instance, import_path)
            tool_names = self._register_tools(instance, import_path)
            self._bind_tool_hooks(instance, import_path)
            self._collect_phase_modules(instance)
            await instance.initialize()
        except Exception as exc:
            logger.warning("插件 %s 初始化失败，回滚: %s", module["name"], exc)
            self._rollback_plugin(import_path, tool_names, snapshot)
            return

        self._plugin_tools[import_path] = tool_names
        self._loaded.add(import_path)
        logger.info("插件已加载: %s", module["name"])
    
    def _import_plugin(self, module_name: str, path: Path) -> None:
        """从文件路径导入插件模块。

        输入:
            module_name: 动态 import 使用的模块名。
            path: plugin.py 文件路径。

        输出:
            None。
        """

        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=[str(path.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件文件: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    def _bind_event_handlers(self, instance: Any, module_path: str) -> None:
        """把插件事件 handler 注册到 EventBus。

        输入:
            instance: 插件实例。
            module_path: 插件 import module name。

        输出:
            None。
        """

        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind != PluginHandlerKind.EVENT or metadata.event_name is None:
                continue
            event_type = _EVENT_TYPE_MAP.get(metadata.event_name)
            if event_type is None:
                continue
            bound = functools.partial(metadata.handler, instance)
            self._event_bus.on(event_type, bound)
            self._event_handlers.setdefault(module_path, []).append((event_type, bound))

    def _register_tools(self, instance: Any, module_path: str) -> list[str]:
        """把 @tool 声明的插件工具注册到 ToolRegistry。

        输入:
            instance: 插件实例。
            module_path: 插件 import module name。

        输出:
            实际注册的工具名列表。
        """

        tool_names: list[str] = []
        if self._tool_registry is None:
            return tool_names
        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind != PluginHandlerKind.TOOL:
                continue
            tool_name = metadata.tool_name or metadata.handler_name
            bound = functools.partial(metadata.handler, instance)
            description = (metadata.handler.__doc__ or "").strip() or tool_name
            schema = metadata.tool_schema or {"type": "object", "properties": {}, "required": []}
            plugin_context = instance.context
            tool_class = type(
                f"PluginTool_{tool_name}",
                (Tool,),
                {
                    "name": tool_name,
                    "description": description,
                    "parameters": schema,
                    "execute": _make_execute(bound, plugin_context, tool_name, instance),
                },
            )
            plugin_name = str(getattr(instance, "name", None) or module_path)
            self._tool_registry.register(
                tool_class(),
                risk=metadata.tool_risk or "read-only",
                always_on=bool(metadata.tool_always_on),
                search_hint=metadata.tool_search_hint,
                source_type="plugin",
                source_name=plugin_name,
            )
            tool_names.append(tool_name)
        return tool_names

    def _bind_tool_hooks(self, instance: Any, module_path: str) -> None:
        """把 @on_tool_* 声明转换为 ToolHook。

        输入:
            instance: 插件实例。
            module_path: 插件 import module name。

        输出:
            None。
        """

        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind != PluginHandlerKind.TOOL_HOOK:
                continue
            bound = functools.partial(metadata.handler, instance)
            hook = _PluginToolHook(
                name=f"plugin:{getattr(instance, 'name', module_path)}:{metadata.handler_name}",
                event=metadata.hook_event,
                handler=bound,
                plugin_context=instance.context,
                handler_name=metadata.handler_name,
                tool_name_filter=metadata.hook_tool_name,
            )
            self._tool_hooks.append(hook)

    def _collect_phase_modules(self, instance: Any) -> None:
        """收集插件暴露的 phase modules。

        输入:
            instance: 插件实例。

        输出:
            None。
        """

        for provider in _PHASE_PROVIDERS:
            self._modules[provider].extend(_load_module_list(instance, provider))

    def _snapshot_counts(self, module_path: str) -> dict[str, int]:
        """记录加载某插件前各集合的长度，用于回滚。

        输入:
            module_path: 插件 import module name。

        输出:
            包含 hooks 与各 phase modules 当前长度的字典。
        """

        counts = {"tool_hooks": len(self._tool_hooks)}
        for provider in _PHASE_PROVIDERS:
            counts[provider] = len(self._modules[provider])
        return counts

    def _rollback_plugin(
        self,
        module_path: str,
        tool_names: list[str],
        snapshot: dict[str, int],
    ) -> None:
        """回滚一次失败的插件加载。

        输入:
            module_path: 插件 import module name。
            tool_names: 已注册工具名。
            snapshot: 加载前各集合长度。

        输出:
            None。
        """

        if self._tool_registry is not None:
            for tool_name in tool_names:
                self._tool_registry.unregister(tool_name)
        for event_type, handler in self._event_handlers.pop(module_path, []):
            off = getattr(self._event_bus, "off", None)
            if callable(off):
                off(event_type, handler)
        del self._tool_hooks[snapshot["tool_hooks"]:]
        for provider in _PHASE_PROVIDERS:
            del self._modules[provider][snapshot[provider]:]
        self._plugin_tools.pop(module_path, None)
        self._loaded.discard(module_path)
        plugin_registry.remove_plugin(module_path)

    async def terminate_all(self) -> None:
        """停止所有已加载插件并注销插件资源。

        输入:
            无。

        输出:
            None。
        """

        for module_path in list(self._loaded):
            instance = plugin_registry.get_instance(module_path)
            if instance is not None:
                try:
                    await instance.terminate()
                except Exception as exc:
                    logger.warning("插件 terminate 失败 (%s): %s", module_path, exc)
            if self._tool_registry is not None:
                for tool_name in self._plugin_tools.get(module_path, []):
                    self._tool_registry.unregister(tool_name)
            for event_type, handler in self._event_handlers.get(module_path, []):
                off = getattr(self._event_bus, "off", None)
                if callable(off):
                    off(event_type, handler)
            plugin_registry.remove_plugin(module_path)
        self._loaded.clear()
        self._plugin_tools.clear()
        self._event_handlers.clear()
        self._tool_hooks.clear()
        for provider in _PHASE_PROVIDERS:
            self._modules[provider].clear()
    

_MANIFEST_FIELDS = ("name", "version", "desc", "author")


def _is_plugin_disabled(plugin_dir: Path) -> bool:
    """判断插件是否被本地 disabled marker 禁用。

    输入:
        plugin_dir: 插件目录。

    输出:
        True 表示存在 plugin.disabled，应跳过加载。
    """

    return (plugin_dir / "plugin.disabled").exists()


def _apply_manifest(instance: Any, plugin_dir: Path) -> None:
    """读取 manifest.yaml 并覆盖插件元信息。

    输入:
        instance: 插件实例。
        plugin_dir: 插件目录。

    输出:
        None。
    """

    manifest_path = plugin_dir / "manifest.yaml"
    if not manifest_path.exists():
        return
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        logger.warning("manifest.yaml 格式错误，期望 dict (%s)", plugin_dir)
        return
    raw = cast(dict[str, object], loaded)
    for field_name in _MANIFEST_FIELDS:
        value = raw.get(field_name)
        if value is not None:
            setattr(instance, field_name, str(value))


def _load_plugin_config(plugin_dir: Path) -> PluginConfig | None:
    """读取插件配置 schema 和本地覆盖。

    输入:
        plugin_dir: 插件目录。

    输出:
        PluginConfig；没有 _conf_schema.json 时返回 None。
    """

    schema_path = plugin_dir / "_conf_schema.json"
    if not schema_path.exists():
        return None
    loaded = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        logger.warning("_conf_schema.json 格式错误，期望 dict (%s)", plugin_dir)
        return None
    values: dict[str, Any] = {}
    for key, spec in cast(dict[str, object], loaded).items():
        if isinstance(key, str) and isinstance(spec, dict) and "default" in spec:
            values[key] = spec["default"]

    override_path = plugin_dir / "plugin_config.json"
    if override_path.exists():
        override = json.loads(override_path.read_text(encoding="utf-8"))
        if isinstance(override, dict):
            for key, value in cast(dict[str, object], override).items():
                if isinstance(key, str):
                    values[key] = value
        else:
            logger.warning("plugin_config.json 格式错误，期望 dict (%s)", plugin_dir)
    return PluginConfig(values)


def _load_module_list(instance: Any, method_name: str) -> list[object]:
    """调用插件的 phase module provider。

    输入:
        instance: 插件实例。
        method_name: provider 方法名。

    输出:
        module 列表。方法不存在或返回非法值时返回空列表。
    """

    provider = getattr(instance, method_name, None)
    if not callable(provider):
        return []
    loaded = provider()
    if loaded is None:
        return []
    if not isinstance(loaded, list):
        logger.warning("插件 %s.%s 返回值不是 list", type(instance).__name__, method_name)
        return []
    return loaded


def _make_execute(
    bound: Callable[..., Any],
    context: PluginContext,
    tool_name: str,
    plugin_instance: Any,
) -> Callable[..., Any]:
    """把插件方法适配成 Tool.execute。

    输入:
        bound: 已绑定插件实例的工具方法。
        context: 插件上下文。
        tool_name: 工具名称。
        plugin_instance: 当前插件实例。

    输出:
        可赋给动态 Tool 类的 async execute 方法。
    """

    signature = inspect.signature(bound)
    accepted = frozenset(name for name in signature.parameters if name not in {"self", "event"})

    async def execute(self: Any, **kwargs: Any) -> str:
        """执行插件工具。

        输入:
            **kwargs: ToolRegistry 传入的工具参数。

        输出:
            字符串形式的工具结果。
        """

        filtered = {key: value for key, value in kwargs.items() if key in accepted}
        event = PluginToolEvent(
            plugin=plugin_instance,
            context=context,
            tool_name=tool_name,
            arguments=dict(filtered),
        )
        result = bound(event, **filtered)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    return execute


class _PluginToolHook(ToolHook):
    """把插件 @on_tool_* 方法适配为 ToolHook。

    输入:
        name: hook 名称。
        event: hook 事件名。
        handler: 已绑定插件实例的 handler。
        plugin_context: 插件上下文。
        handler_name: 插件方法名。
        tool_name_filter: 只匹配某个工具；None 表示匹配全部。

    输出:
        _PluginToolHook 实例。
    """

    def __init__(
        self,
        *,
        name: str,
        event: str,
        handler: Callable[..., Any],
        plugin_context: PluginContext,
        handler_name: str,
        tool_name_filter: str | None = None,
    ) -> None:
        self.name = name
        self.event = cast(Any, event)
        self.trace_metadata = {
            "source_type": "plugin",
            "plugin_id": plugin_context.plugin_id,
            "handler": handler_name,
        }
        self._handler = handler
        self._plugin_context = plugin_context
        self._tool_name_filter = tool_name_filter

    def matches(self, context: ToolHookContext) -> bool:
        """判断当前 hook 是否匹配工具调用。

        输入:
            context: ToolHookContext。

        输出:
            True 表示匹配，应执行 run()。
        """

        if self._tool_name_filter is None:
            return True
        return context.request.tool_name == self._tool_name_filter

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """执行插件 hook。

        输入:
            context: ToolHookContext。

        输出:
            ToolHookOutcome。pre hook 返回 dict 表示改写参数，返回 ToolHookOutcome 表示直接使用其结果。
        """

        event = PluginToolHookEvent(
            context=self._plugin_context,
            event=context.event,
            session_key=context.request.session_key,
            tool_name=context.request.tool_name,
            arguments=dict(context.current_arguments),
            call_id=context.request.call_id,
            metadata=dict(context.request.metadata),
            result=context.result,
            error=context.error,
        )
        result = self._handler(event)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return ToolHookOutcome()
        if isinstance(result, ToolHookOutcome):
            return result
        if isinstance(result, dict) and context.event == "pre_tool_use":
            return ToolHookOutcome(updated_arguments=cast(dict[str, Any], result))
        return ToolHookOutcome()


def _builtin_plugin_dir(self, name: str) -> Path:
    """计算内置插件的私有数据目录。

    输入:
        name: 内置插件名。

    输出:
        workspace 下的 builtin_plugins/<name> 目录；无 workspace 时退化到 .raven 相对目录。
    """

    base = self._workspace if self._workspace is not None else Path(".raven")
    return base / "builtin_plugins" / name


def _make_context(self, instance: Any, plugin_dir: Path, import_path: str) -> PluginContext:
    """为插件构造 PluginContext。

    输入:
        instance: 插件实例。
        plugin_dir: 插件私有数据目录。
        import_path: 插件在 registry 中的 key。

    输出:
        PluginContext。
    """

    plugin_id = str(getattr(instance, "name", None) or import_path)
    return PluginContext(
        event_bus=self._event_bus,
        tool_registry=self._tool_registry,
        plugin_id=plugin_id,
        plugin_dir=plugin_dir,
        kv_store=PluginKVStore(plugin_dir / ".kv.json"),
        config=None,
        workspace=self._workspace,
        session_manager=self._session_manager,
        memory_engine=self._memory_engine,
        memory_maintenance=self._memory_maintenance,
        memory_optimizer=self._memory_optimizer,
    )