from __future__ import annotations

import asyncio
from typing import Any

from raven_agent.tools import ToolResult
from raven_agent.tools.executor import ToolExecutor
from raven_agent.tools.hooks import (
    ToolExecutionRequest,
    ToolHook,
    ToolHookContext,
    ToolHookOutcome,
)


class SpyHook(ToolHook):
    """测试用 Hook。

    参数:
        name: Hook 名称。
        event: Hook 事件。
        matched: matches() 返回值。
        outcome: run() 返回值。
    """

    def __init__(
        self,
        *,
        name: str,
        event: str,
        matched: bool = True,
        outcome: ToolHookOutcome | None = None,
    ) -> None:
        self.name = name
        self.event = event  # type: ignore[assignment]
        self._matched = matched
        self._outcome = outcome or ToolHookOutcome()
        self.calls: list[ToolHookContext] = []
        self.run_error: Exception | None = None

    def matches(self, context: ToolHookContext) -> bool:
        """返回预设 matched 值。

        参数:
            context: Hook 上下文。

        返回:
            是否匹配。
        """

        return self._matched

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """记录调用并返回预设 outcome。

        参数:
            context: Hook 上下文。

        返回:
            ToolHookOutcome。
        """

        if self.run_error is not None:
            raise self.run_error
        self.calls.append(context)
        return self._outcome


async def invoke(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    """测试用工具执行函数。

    参数:
        tool_name: 工具名称。
        arguments: 工具参数。

    返回:
        ToolResult。
    """

    return ToolResult(text=f"{tool_name}:{arguments}", metadata={"ok": True})


def test_tool_executor_pre_hook_can_update_arguments() -> None:
    """测试 pre hook 可以改写参数。

    返回:
        None。
    """

    hook = SpyHook(
        name="rewrite",
        event="pre_tool_use",
        outcome=ToolHookOutcome(updated_arguments={"x": 2}),
    )
    executor = ToolExecutor([hook])

    result = asyncio.run(
        executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="dummy", arguments={"x": 1}),
            invoke,
        )
    )

    assert result.status == "success"
    assert result.final_arguments == {"x": 2}
    assert result.output == "dummy:{'x': 2}"
    assert hook.calls[0].request.arguments == {"x": 1}


def test_tool_executor_pre_hook_can_deny_execution() -> None:
    """测试 pre hook 可以拒绝工具执行。

    返回:
        None。
    """

    hook = SpyHook(
        name="deny",
        event="pre_tool_use",
        outcome=ToolHookOutcome(decision="deny", reason="blocked"),
    )
    executor = ToolExecutor([hook])

    result = asyncio.run(
        executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="dummy", arguments={"x": 1}),
            invoke,
        )
    )

    assert result.status == "denied"
    assert result.output == "blocked"
    assert result.final_arguments == {"x": 1}


def test_tool_executor_post_hook_adds_extra_message() -> None:
    """测试 post hook 可以追加 extra message。

    返回:
        None。
    """

    hook = SpyHook(
        name="post",
        event="post_tool_use",
        outcome=ToolHookOutcome(extra_message="hint"),
    )
    executor = ToolExecutor([hook])

    result = asyncio.run(
        executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="dummy", arguments={}),
            invoke,
        )
    )

    assert result.status == "success"
    assert result.extra_messages == ["hint"]
    assert result.post_hook_trace[0].hook_name == "post"


def test_tool_executor_post_error_hook_runs_on_tool_error() -> None:
    """测试工具失败时会触发 post_tool_error hook。

    返回:
        None。
    """

    hook = SpyHook(
        name="post_error",
        event="post_tool_error",
        outcome=ToolHookOutcome(extra_message="logged"),
    )
    executor = ToolExecutor([hook])

    async def broken(_tool_name: str, _arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(text="工具执行失败: boom", metadata={"ok": False})

    result = asyncio.run(
        executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="dummy", arguments={}),
            broken,
        )
    )

    assert result.status == "error"
    assert result.output == "工具执行失败: boom"
    assert result.extra_messages == ["logged"]


def test_tool_executor_post_hook_failure_does_not_pollute_success() -> None:
    """测试 post_tool_use hook 失败不会污染成功工具结果。

    返回:
        None。
    """

    hook = SpyHook(name="boom", event="post_tool_use")
    hook.run_error = RuntimeError("post boom")
    executor = ToolExecutor([hook])

    result = asyncio.run(
        executor.execute(
            ToolExecutionRequest(call_id="c1", tool_name="dummy", arguments={}),
            invoke,
        )
    )

    assert result.status == "success"
    assert result.output == "dummy:{}"
    assert result.post_hook_trace[0].reason == "hook failed: post boom"