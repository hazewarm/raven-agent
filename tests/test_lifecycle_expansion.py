from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raven_agent.phase import (
    Phase,
    PhaseFrame,
    collect_prefixed_slots,
    topo_sort_modules,
)


@dataclass
class _TextFrame(PhaseFrame[str, str]):
    """测试用文本 PhaseFrame。

    输入:
        input: 输入文本。

    输出:
        _TextFrame 实例。
    """


class _ProducerModule:
    """产出数据 slot 的模块。

    输入:
        无。

    输出:
        _ProducerModule 实例。
    """

    slot = "producer"
    produces = ("data:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        """写入 data:value 数据 slot。

        输入:
            frame: 当前 _TextFrame。

        输出:
            写入数据 slot 后的 _TextFrame。
        """

        frame.slots["data:value"] = frame.input
        return frame


class _ConsumerModule:
    """依赖数据 slot 的模块。

    输入:
        无。

    输出:
        _ConsumerModule 实例。
    """

    slot = "consumer"
    requires = ("data:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        """读取数据 slot 并写入 output。

        输入:
            frame: 当前 _TextFrame。

        输出:
            写入 output 后的 _TextFrame。
        """

        frame.output = f"consumed:{frame.slots['data:value']}"
        return frame


def test_topo_sort_supports_data_slot_requires() -> None:
    """测试 requires 指向 produces 数据 slot 时也能排序。

    输入:
        无。

    输出:
        None。
    """

    modules = topo_sort_modules([_ConsumerModule(), _ProducerModule()])
    assert [m.slot for m in modules] == ["producer", "consumer"]


def test_phase_runs_with_data_slot_dependency() -> None:
    """测试基于数据 slot 依赖的 Phase 能正确执行。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        phase = Phase[str, str, _TextFrame](
            [_ConsumerModule(), _ProducerModule()],
            frame_factory=_TextFrame,
        )
        result = await phase.run("hello")
        assert result == "consumed:hello"

    asyncio.run(run())


def test_collect_prefixed_slots_strips_prefix() -> None:
    """测试 collect_prefixed_slots 去掉 prefix 并按名排序。

    输入:
        无。

    输出:
        None。
    """

    slots = {"p:b": "B", "p:a": "A", "other": "X"}
    assert collect_prefixed_slots(slots, "p:") == {"a": "A", "b": "B"}



from raven_agent.agent import ReactAgent
from raven_agent.lifecycle import AfterStepCtx, BeforeStepCtx
from raven_agent.messages import user_message
from raven_agent.tools import ToolRegistry


class _FakeResponse:
    """fake LLM 响应。

    输入:
        content: 回复文本。
        tool_calls: 工具调用列表。

    输出:
        _FakeResponse 实例。
    """

    def __init__(self, content, tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = ""


class _NoToolProvider:
    """直接返回文本、不调工具的 fake provider。

    输入:
        无。

    输出:
        _NoToolProvider 实例。
    """

    async def chat(self, messages, tools, tool_choice="auto", **kwargs):
        """返回一条纯文本响应。

        输入:
            messages: 消息列表。
            tools: 工具 schema 列表。
            tool_choice: 工具选择策略。

        输出:
            _FakeResponse。
        """

        return _FakeResponse(content="done")


class _RecordingLifecycle:
    """记录 step 调用次数的 lifecycle。

    输入:
        early_stop_at: 在第几个 before_step 触发 early_stop；0 表示不触发。

    输出:
        _RecordingLifecycle 实例。
    """

    def __init__(self, early_stop_at: int = 0) -> None:
        self.before_calls: list[int] = []
        self.after_calls: list[int] = []
        self._early_stop_at = early_stop_at

    async def before_step(self, ctx: BeforeStepCtx) -> BeforeStepCtx:
        """记录 before_step 调用，按需 early_stop。

        输入:
            ctx: 当前 BeforeStepCtx。

        输出:
            可能置 early_stop 的 BeforeStepCtx。
        """

        self.before_calls.append(ctx.iteration)
        if self._early_stop_at and ctx.iteration == self._early_stop_at:
            ctx.early_stop = True
            ctx.early_stop_reply = "stopped"
        return ctx

    async def after_step(self, ctx: AfterStepCtx) -> AfterStepCtx:
        """记录 after_step 调用。

        输入:
            ctx: 当前 AfterStepCtx。

        输出:
            原样返回的 AfterStepCtx。
        """

        self.after_calls.append(ctx.iteration)
        return ctx


def test_react_agent_invokes_step_lifecycle() -> None:
    """测试 ReactAgent 在每个 step 调用 before_step / after_step。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        agent = ReactAgent(provider=_NoToolProvider(), tools=ToolRegistry())
        lifecycle = _RecordingLifecycle()

        result = await agent.run(
            [user_message("hi")],
            session_key="cli:default",
            lifecycle=lifecycle,
        )

        assert result.content == "done"
        assert lifecycle.before_calls == [1]
        assert lifecycle.after_calls == [1]

    asyncio.run(run())


def test_react_agent_early_stop_from_before_step() -> None:
    """测试 before_step early_stop 能提前结束工具循环。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。

        输入:
            无。

        输出:
            None。
        """

        agent = ReactAgent(provider=_NoToolProvider(), tools=ToolRegistry())
        lifecycle = _RecordingLifecycle(early_stop_at=1)

        result = await agent.run(
            [user_message("hi")],
            session_key="cli:default",
            lifecycle=lifecycle,
        )

        assert result.content == "stopped"
        # early_stop 发生在 before_step，本 step 不应再调模型或 after_step。
        assert lifecycle.after_calls == []

    asyncio.run(run())