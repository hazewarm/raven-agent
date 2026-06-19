from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from raven_agent.phase import Phase, PhaseFrame, topo_sort_modules


@dataclass
class TextFrame(PhaseFrame[str, str]):
    """测试用文本 PhaseFrame。

    参数:
        input: 输入文本。
        slots: 模块之间共享的中间数据。
        output: 输出文本，由 FinalizeModule 写入。

    返回:
        TextFrame 实例。
    """


class SetupModule:
    """测试用初始化模块。

    参数:
        无。

    返回:
        SetupModule 实例。
    """

    slot = "setup"

    async def run(self, frame: TextFrame) -> TextFrame:
        """写入初始文本 slot。

        参数:
            frame: 当前 TextFrame。

        返回:
            写入 text slot 后的 TextFrame。
        """

        frame.slots["text"] = f"setup:{frame.input}"
        return frame


class MutateModule:
    """测试用文本变换模块。

    参数:
        无。

    返回:
        MutateModule 实例。
    """

    slot = "mutate"
    requires = ("setup",)

    async def run(self, frame: TextFrame) -> TextFrame:
        """读取并修改 setup 写入的 text slot。

        参数:
            frame: 当前 TextFrame。

        返回:
            更新 text slot 后的 TextFrame。
        """

        frame.slots["text"] = frame.slots["text"] + ":mutated"
        return frame


class FinalizeModule:
    """测试用收尾模块。

    参数:
        无。

    返回:
        FinalizeModule 实例。
    """

    slot = "finalize"
    requires = ("mutate",)

    async def run(self, frame: TextFrame) -> TextFrame:
        """把 text slot 写入 Phase output。

        参数:
            frame: 当前 TextFrame。

        返回:
            写入 output 后的 TextFrame。
        """

        frame.output = frame.slots["text"] + ":finalized"
        return frame


class NoOutputModule:
    """测试用空模块。

    参数:
        无。

    返回:
        NoOutputModule 实例。
    """

    slot = "noop"

    async def run(self, frame: TextFrame) -> TextFrame:
        """不写 output，原样返回 frame。

        参数:
            frame: 当前 TextFrame。

        返回:
            未写 output 的 TextFrame。
        """

        return frame


def test_topo_sort_modules_orders_by_requires() -> None:
    """测试 topo_sort_modules 会按 requires 排序。

    参数:
        无。

    返回:
        None。
    """

    modules = topo_sort_modules([FinalizeModule(), MutateModule(), SetupModule()])

    assert [module.slot for module in modules] == ["setup", "mutate", "finalize"]


def test_phase_runs_modules_and_returns_output() -> None:
    """测试 Phase 会执行模块链并返回 output。

    参数:
        无。

    返回:
        None。
    """

    async def run() -> None:
        """执行异步 Phase 测试。

        参数:
            无。

        返回:
            None。
        """

        phase = Phase[str, str, TextFrame](
            [SetupModule(), MutateModule(), FinalizeModule()],
            frame_factory=TextFrame,
        )

        result = await phase.run("hello")

        assert result == "setup:hello:mutated:finalized"

    asyncio.run(run())


def test_phase_requires_output() -> None:
    """测试模块链没有 output 时会报错。

    参数:
        无。

    返回:
        None。
    """

    async def run() -> None:
        """执行缺失 output 的异步测试。

        参数:
            无。

        返回:
            None。
        """

        phase = Phase[str, str, TextFrame]([NoOutputModule()], frame_factory=TextFrame)

        with pytest.raises(RuntimeError, match="未产生 output"):
            await phase.run("hello")

    asyncio.run(run())


def test_phase_rejects_missing_slot() -> None:
    """测试模块缺少 slot 时会报错。

    参数:
        无。

    返回:
        None。
    """

    class MissingSlotModule:
        """缺少 slot 的非法模块。

        参数:
            无。

        返回:
            MissingSlotModule 实例。
        """

        async def run(self, frame: TextFrame) -> TextFrame:
            """原样返回 frame。

            参数:
                frame: 当前 TextFrame。

            返回:
                原样返回的 TextFrame。
            """

            return frame

    with pytest.raises(RuntimeError, match="缺少 slot"):
        topo_sort_modules([MissingSlotModule()])


def test_phase_rejects_duplicate_slot() -> None:
    """测试重复 slot 会报错。

    参数:
        无。

    返回:
        None。
    """

    class AnotherSetupModule(SetupModule):
        """复用 setup slot 的非法模块。

        参数:
            无。

        返回:
            AnotherSetupModule 实例。
        """

    with pytest.raises(RuntimeError, match="slot 重复"):
        topo_sort_modules([SetupModule(), AnotherSetupModule()])


def test_phase_rejects_missing_dependency() -> None:
    """测试缺失 requires 依赖会报错。

    参数:
        无。

    返回:
        None。
    """

    with pytest.raises(RuntimeError, match="依赖不存在"):
        topo_sort_modules([MutateModule()])


def test_phase_rejects_dependency_cycle() -> None:
    """测试循环依赖会报错。

    参数:
        无。

    返回:
        None。
    """

    class AModule:
        """循环依赖测试模块 A。

        参数:
            无。

        返回:
            AModule 实例。
        """

        slot = "a"
        requires = ("b",)

        async def run(self, frame: TextFrame) -> TextFrame:
            """原样返回 frame。

            参数:
                frame: 当前 TextFrame。

            返回:
                原样返回的 TextFrame。
            """

            return frame

    class BModule:
        """循环依赖测试模块 B。

        参数:
            无。

        返回:
            BModule 实例。
        """

        slot = "b"
        requires = ("a",)

        async def run(self, frame: TextFrame) -> TextFrame:
            """原样返回 frame。

            参数:
                frame: 当前 TextFrame。

            返回:
                原样返回的 TextFrame。
            """

            return frame

    with pytest.raises(RuntimeError, match="循环依赖"):
        topo_sort_modules([AModule(), BModule()])