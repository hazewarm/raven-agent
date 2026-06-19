from __future__ import annotations

import asyncio

from raven_agent.lifecycle import AfterStepCtx
from raven_agent.plugins.builtins.context_pressure import (
    ContextPressureStopModule,
    _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS,
)
from raven_agent.turn_pipeline import AfterStepFrame


def test_context_pressure_sets_early_stop_when_over_threshold() -> None:
    """测试压力超阈值时写 early_stop_reason 与 telemetry。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        ctx = AfterStepCtx(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            iteration=1,
            tools_called=("echo",),
            partial_reply="",
            has_more=True,
            context_tokens_estimate=_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS + 1,
        )
        frame = AfterStepFrame(input=ctx)
        frame.slots["after_step:ctx"] = ctx
        module = ContextPressureStopModule()

        await module.run(frame)

        assert frame.slots["after_step:early_stop_reason"] == "context_pressure"
        assert frame.slots["after_step:telemetry:context_pressure_tokens"] > 0

    asyncio.run(run())


def test_context_pressure_noop_when_under_threshold() -> None:
    """测试压力未超阈值时不写 early_stop_reason。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        ctx = AfterStepCtx(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            iteration=1,
            tools_called=("echo",),
            partial_reply="",
            has_more=True,
            context_tokens_estimate=10,
        )
        frame = AfterStepFrame(input=ctx)
        frame.slots["after_step:ctx"] = ctx
        module = ContextPressureStopModule()

        await module.run(frame)

        assert "after_step:early_stop_reason" not in frame.slots

    asyncio.run(run())
