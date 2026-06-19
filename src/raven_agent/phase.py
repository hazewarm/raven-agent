from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast


I = TypeVar("I")
O = TypeVar("O")
# F 是 PhaseFrame 类型或其子类
F = TypeVar("F", bound="PhaseFrame[Any, Any]")


@dataclass
class PhaseFrame(Generic[I, O]):
    """Phase 执行期间共享的数据帧。

    参数:
        input: Phase 的输入对象。
        slots: 模块之间共享的临时数据。
        output: Phase 的输出对象；必须由某个模块写入。
    """

    input: I
    slots: dict[str, Any] = field(default_factory=dict)
    output: O | None = None


class PhaseModule(Protocol[F]):
    """Phase 模块协议。

    参数:
        无。模块可以通过类属性 slot / requires / produces 声明依赖。
    """

    async def run(self, frame: F) -> F:
        """执行模块逻辑。

        参数:
            frame: 当前 PhaseFrame。

        返回:
            修改后的 PhaseFrame。
        """

        ...


def _module_slot(module: object) -> str:
    """读取模块 slot。

    参数:
        module: Phase 模块对象。

    返回:
        模块 slot 字符串。

    异常:
        RuntimeError: 当模块缺少 slot 或 slot 为空时抛出。
    """

    slot = getattr(module, "slot", "")
    if not isinstance(slot, str) or not slot.strip():
        raise RuntimeError(f"Phase 模块缺少 slot: {type(module).__name__}")
    return slot


def _module_requires(module: object) -> tuple[str, ...]:
    """读取模块 requires 声明。

    参数:
        module: Phase 模块对象。

    返回:
        依赖 slot 元组。
    """

    return tuple(str(item) for item in getattr(module, "requires", ()))


def _module_produces(module: object) -> tuple[str, ...]:
    """读取模块 produces 声明。

    参数:
        module: Phase 模块对象。

    返回:
        产出 slot 元组。
    """

    return tuple(str(item) for item in getattr(module, "produces", ()))


def topo_sort_modules(modules: Sequence[object]) -> list[object]:
    """按模块依赖拓扑排序。

    参数:
        modules: Phase 模块列表。每个模块必须有唯一 slot。

    返回:
        按依赖顺序排列的模块列表。

    异常:
        RuntimeError: 当 slot 重复、依赖缺失或存在循环依赖时抛出。
    """

    slot_to_module: dict[str, object] = {}
    slot_order: dict[str, int] = {}
    for index, module in enumerate(modules):
        slot = _module_slot(module)
        if slot in slot_to_module:
            raise RuntimeError(f"Phase 模块 slot 重复: {slot}")
        slot_to_module[slot] = module
        slot_order[slot] = index

    in_degree = {slot: 0 for slot in slot_to_module}
    dependents: dict[str, list[str]] = {slot: [] for slot in slot_to_module}

    # 1. 先建立 provider 表：每个 slot 由哪个模块 slot 提供。
    #    一个模块提供：它自己的 slot，以及它声明的所有 produces。
    provider_of: dict[str, str] = {}
    for slot, module in slot_to_module.items():
        provider_of[slot] = slot
        for produced in _module_produces(module):
            # 数据 slot 由它的模块负责提供；同名 produces 以最后一个为准。
            provider_of[produced] = slot

    # 2. 再按 requires 建图，requires 既可指向模块 slot，也可指向数据 slot。
    for slot, module in slot_to_module.items():
        for required in _module_requires(module):
            provider = provider_of.get(required)
            if provider is None:
                raise RuntimeError(
                    f"Phase 模块依赖不存在: module={slot} requires={required}"
                )
            if provider == slot:
                # 模块依赖自己的 produces，不构成跨模块依赖。
                continue
            in_degree[slot] += 1
            dependents[provider].append(slot)

    queue = [slot for slot, degree in in_degree.items() if degree == 0]
    sorted_slots: list[str] = []

    while queue:
        queue.sort(key=lambda item: slot_order[item])
        slot = queue.pop(0)
        sorted_slots.append(slot)
        for dependent in dependents[slot]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_slots) != len(slot_to_module):
        unresolved = sorted(slot for slot, degree in in_degree.items() if degree > 0)
        raise RuntimeError(f"Phase 模块循环依赖: {', '.join(unresolved)}")

    return [slot_to_module[slot] for slot in sorted_slots]


class Phase(Generic[I, O, F]):
    """可组合的异步模块链。

    参数:
        modules: Phase 模块列表。
        frame_factory: 根据 input 创建 PhaseFrame 的函数或类。
    """

    def __init__(
        self,
        modules: Sequence[PhaseModule[F]],
        *,
        frame_factory: Callable[[I], F],
    ) -> None:
        self._modules = cast(list[PhaseModule[F]], topo_sort_modules(list(modules)))
        self._frame_factory = frame_factory
        self._validate_produces()

    async def run(self, input: I) -> O:
        """执行 Phase。

        参数:
            input: Phase 输入对象。

        返回:
            Phase 输出对象。

        异常:
            RuntimeError: 当模块链没有产生 output 时抛出。
        """

        frame = self._frame_factory(input)
        for module in self._modules:
            frame = await module.run(frame)
        if frame.output is None:
            raise RuntimeError("Phase 模块链未产生 output")
        return frame.output

    def _validate_produces(self) -> None:
        """验证 requires 是否由前序模块声明产生。

        返回:
            None。

        异常:
            RuntimeError: 当 requires 没有被前序模块 produces 或 slot 满足时抛出。
        """

        provided: set[str] = set()
        for module in self._modules:
            slot = _module_slot(module)
            for required in _module_requires(module):
                if required not in provided:
                    raise RuntimeError(f"Phase slot 未闭合: module={slot} requires={required}")
            provided.add(slot)
            provided.update(_module_produces(module))


def collect_prefixed_slots(slots: dict[str, Any], prefix: str) -> dict[str, Any]:
    """收集所有以 prefix 开头的 slot，并去掉 prefix 返回。

    输入:
        slots: 当前 PhaseFrame.slots。
        prefix: slot 前缀，例如 "prompt:section_bottom:"。

    输出:
        去掉 prefix 后的 {子键: 值} 字典，按 slot 名排序。
    """

    result: dict[str, Any] = {}
    for key in sorted(slots.keys()):
        if key.startswith(prefix):
            result[key[len(prefix):]] = slots[key]
    return result


def append_string_exports(target: list[str], exports: dict[str, Any]) -> None:
    """把 export 字典里的非空字符串追加到目标列表。

    输入:
        target: 要追加内容的字符串列表，例如 ctx.extra_hints。
        exports: collect_prefixed_slots 的返回值。

    输出:
        None。会就地修改 target。
    """

    for value in exports.values():
        if isinstance(value, str) and value.strip():
            target.append(value)