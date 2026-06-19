from __future__ import annotations

from typing import cast

from raven_agent.lifecycle import AfterStepCtx
from raven_agent.plugins import Plugin

# after_step phase 中第 22 章约定的 slot 命名。
_AFTER_STEP_CTX_SLOT = "after_step:ctx"
_EARLY_STOP_REASON_SLOT = "after_step:early_stop_reason"
_TELEMETRY_PREFIX = "after_step:telemetry:"

# 模型上下文窗口估算与压力阈值。
_MODEL_CONTEXT_WINDOW_TOKENS = 128_000
_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS = _MODEL_CONTEXT_WINDOW_TOKENS * 80 // 100


class ContextPressureStopModule:
    """在 after_step 阶段判断上下文压力的 PhaseModule。

    输入:
        无。模块依赖 frame.slots["after_step:ctx"]。

    输出:
        ContextPressureStopModule 实例。
    """

    slot = "context_pressure.stop"
    requires = ("after_step.copy_ctx", _AFTER_STEP_CTX_SLOT)
    produces = (
        _EARLY_STOP_REASON_SLOT,
        f"{_TELEMETRY_PREFIX}context_pressure_tokens",
        f"{_TELEMETRY_PREFIX}context_pressure_threshold",
    )

    async def run(self, frame: object) -> object:
        """在压力超阈值时写入 early_stop_reason 与 telemetry slot。

        输入:
            frame: 当前 AfterStepFrame。

        输出:
            可能写入 early_stop_reason / telemetry slot 后的 AfterStepFrame。
        """

        slots = cast("dict[str, object]", getattr(frame, "slots"))
        ctx = slots.get(_AFTER_STEP_CTX_SLOT)
        if not isinstance(ctx, AfterStepCtx) or not ctx.has_more:
            return frame
        tokens = ctx.context_tokens_estimate
        if tokens <= _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS:
            return frame
        slots[_EARLY_STOP_REASON_SLOT] = "context_pressure"
        slots[f"{_TELEMETRY_PREFIX}context_pressure_tokens"] = tokens
        slots[f"{_TELEMETRY_PREFIX}context_pressure_threshold"] = (
            _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS
        )
        return frame


class ContextPressurePlugin(Plugin):
    """上下文压力过高时请求工具循环阶段性收尾的内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        ContextPressurePlugin 实例。
    """

    name = "context_pressure"
    version = "0.1.0"
    desc = "上下文压力过高时请求工具循环阶段性收尾"

    def after_step_modules(self) -> list[object]:
        """返回 after_step 阶段模块。

        输入:
            无。

        输出:
            含 ContextPressureStopModule 的列表。
        """

        return [ContextPressureStopModule()]