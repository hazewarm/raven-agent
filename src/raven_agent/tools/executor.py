from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from raven_agent.tools.base import ToolResult, normalize_tool_result
from raven_agent.tools.hooks import (
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolHook,
    ToolHookContext,
    ToolHookOutcome,
    ToolHookTrace,
)

ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


class ToolHookError(RuntimeError):
    """Hook 执行失败。

    参数:
        hook_name: 失败的 Hook 名称。
        event: Hook 事件。
        cause: 原始异常。
    """

    def __init__(self, hook_name: str, event: str, cause: Exception) -> None:
        self.hook_name = hook_name
        self.event = event
        self.cause = cause
        super().__init__(f"hook {hook_name} ({event}) failed: {cause}")


class ToolExecutor:
    """带 Hook 的工具执行器。

    参数:
        hooks: 初始 Hook 列表。
    """

    def __init__(self, hooks: Sequence[ToolHook] | None = None) -> None:
        self._hooks = list(hooks or [])

    def add_hooks(self, hooks: Sequence[ToolHook]) -> None:
        """追加 Hook。

        参数:
            hooks: 要追加的 Hook 列表。

        返回:
            None。
        """

        self._hooks.extend(hooks)
    
    def remove_hooks_by_prefix(self, prefix: str) -> None:
        """按名称前缀移除 Hook。

        输入:
            prefix: Hook 名称前缀，例如 "plugin:"。

        输出:
            None。会就地更新 Hook 列表。
        """

        self._hooks = [hook for hook in self._hooks if not hook.name.startswith(prefix)]

    async def execute(
        self,
        request: ToolExecutionRequest,
        invoker: ToolInvoker,
    ) -> ToolExecutionResult:
        """执行一次工具调用。

        参数:
            request: 工具执行请求。
            invoker: 真正执行工具的函数，通常是 ToolRegistry.execute。

        返回:
            ToolExecutionResult。
        """

        current_arguments = dict(request.arguments)
        extra_messages: list[str] = []
        pre_trace: list[ToolHookTrace] = []
        post_trace: list[ToolHookTrace] = []

        try:
            denied_reason, current_arguments = await self._run_pre_hooks(
                request=request,
                current_arguments=current_arguments,
                extra_messages=extra_messages,
                traces=pre_trace,
            )
        except ToolHookError as exc:
            return ToolExecutionResult(
                status="error",
                output=f"工具执行出错: {exc}",
                final_arguments=dict(current_arguments),
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
                post_hook_trace=post_trace,
            )

        final_arguments = dict(current_arguments)
        if denied_reason:
            return ToolExecutionResult(
                status="denied",
                output=denied_reason,
                final_arguments=final_arguments,
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
                post_hook_trace=post_trace,
            )

        result = await invoker(request.tool_name, final_arguments)
        normalized = normalize_tool_result(result)
        if not normalized.metadata.get("ok", True):
            await self._run_post_error_hooks(
                request=request,
                final_arguments=final_arguments,
                error=normalized.text,
                extra_messages=extra_messages,
                traces=post_trace,
            )
            return ToolExecutionResult(
                status="error",
                output=normalized.text,
                final_arguments=final_arguments,
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
                post_hook_trace=post_trace,
            )

        await self._run_post_use_hooks(
            request=request,
            final_arguments=final_arguments,
            result=normalized,
            extra_messages=extra_messages,
            traces=post_trace,
        )
        return ToolExecutionResult(
            status="success",
            output=normalized.text,
            final_arguments=final_arguments,
            extra_messages=extra_messages,
            pre_hook_trace=pre_trace,
            post_hook_trace=post_trace,
        )
    
    # pre 方法
    async def _run_pre_hooks(
        self,
        *,
        request: ToolExecutionRequest,
        current_arguments: dict[str, Any],
        extra_messages: list[str],
        traces: list[ToolHookTrace],
    ) -> tuple[str, dict[str, Any]]:
        """执行 pre_tool_use hooks。

        参数:
            request: 原始工具执行请求。
            current_arguments: 当前参数。
            extra_messages: 附加提示收集列表。
            traces: trace 收集列表。

        返回:
            二元组，第一项为拒绝原因，第二项为最终参数。
        """

        for hook in self._hooks:
            if hook.event != "pre_tool_use":
                continue
            context = ToolHookContext(
                event="pre_tool_use",
                request=request,
                current_arguments=dict(current_arguments),
            )
            try:
                matched = hook.matches(context)
            except Exception as exc:
                raise ToolHookError(hook.name, hook.event, exc) from exc
            if not matched:
                traces.append(
                    ToolHookTrace(
                        hook_name=hook.name,
                        event=hook.event,
                        matched=False,
                        metadata=_hook_metadata(hook),
                    )
                )
                continue
            try:
                outcome = await hook.run(context)
            except Exception as exc:
                raise ToolHookError(hook.name, hook.event, exc) from exc
            if outcome.updated_arguments is not None:
                current_arguments = dict(outcome.updated_arguments)
            if outcome.extra_message:
                extra_messages.append(outcome.extra_message)
            traces.append(_trace_from_outcome(hook, matched=True, outcome=outcome))
            if outcome.decision == "deny":
                return outcome.reason.strip() or "工具调用被拦截", current_arguments
        return "", current_arguments
    
    # post 方法
    async def _run_post_use_hooks(
        self,
        *,
        request: ToolExecutionRequest,
        final_arguments: dict[str, Any],
        result: ToolResult,
        extra_messages: list[str],
        traces: list[ToolHookTrace],
    ) -> None:
        """执行 post_tool_use hooks。

        参数:
            request: 工具执行请求。
            final_arguments: 最终参数。
            result: 工具结果。
            extra_messages: 附加提示收集列表。
            traces: trace 收集列表。

        返回:
            None。
        """

        await self._run_post_hooks(
            context=ToolHookContext(
                event="post_tool_use",
                request=request,
                current_arguments=final_arguments,
                result=result,
            ),
            extra_messages=extra_messages,
            traces=traces,
            fail_open=True,
        )

    async def _run_post_error_hooks(
        self,
        *,
        request: ToolExecutionRequest,
        final_arguments: dict[str, Any],
        error: str,
        extra_messages: list[str],
        traces: list[ToolHookTrace],
    ) -> None:
        """执行 post_tool_error hooks。

        参数:
            request: 工具执行请求。
            final_arguments: 最终参数。
            error: 错误文本。
            extra_messages: 附加提示收集列表。
            traces: trace 收集列表。

        返回:
            None。
        """

        await self._run_post_hooks(
            context=ToolHookContext(
                event="post_tool_error",
                request=request,
                current_arguments=final_arguments,
                error=error,
            ),
            extra_messages=extra_messages,
            traces=traces,
            fail_open=True,
        )

    async def _run_post_hooks(
        self,
        *,
        context: ToolHookContext,
        extra_messages: list[str],
        traces: list[ToolHookTrace],
        fail_open: bool,
    ) -> None:
        """执行指定 post hook。

        参数:
            context: Hook 上下文。
            extra_messages: 附加提示收集列表。
            traces: trace 收集列表。
            fail_open: True 时 Hook 失败只记录 trace，不影响工具成功结果。

        返回:
            None。
        """

        for hook in self._hooks:
            if hook.event != context.event:
                continue
            try:
                matched = hook.matches(context)
            except Exception as exc:
                if fail_open:
                    traces.append(_failed_trace(hook, str(exc)))
                    continue
                raise ToolHookError(hook.name, hook.event, exc) from exc
            if not matched:
                traces.append(
                    ToolHookTrace(
                        hook_name=hook.name,
                        event=hook.event,
                        matched=False,
                        metadata=_hook_metadata(hook),
                )
            )
                continue
            try:
                outcome = await hook.run(context)
            except Exception as exc:
                if fail_open:
                    traces.append(_failed_trace(hook, str(exc), matched=True))
                    continue
                raise ToolHookError(hook.name, hook.event, exc) from exc
            if outcome.extra_message:
                extra_messages.append(outcome.extra_message)
            traces.append(_trace_from_outcome(hook, matched=True, outcome=outcome))



def _trace_from_outcome(
    hook: ToolHook,
    *,
    matched: bool,
    outcome: ToolHookOutcome,
) -> ToolHookTrace:
    """根据 HookOutcome 创建 ToolHookTrace。

    参数:
        hook: 当前 Hook。
        matched: Hook 是否匹配。
        outcome: Hook 执行结果。

    返回:
        ToolHookTrace。
    """

    return ToolHookTrace(
        hook_name=hook.name,
        event=hook.event,
        matched=matched,
        decision=outcome.decision,
        reason=outcome.reason,
        extra_message=outcome.extra_message,
        metadata=_hook_metadata(hook),
    )


def _failed_trace(
    hook: ToolHook,
    reason: str,
    *,
    matched: bool = False,
) -> ToolHookTrace:
    """创建 Hook 失败 trace。

    参数:
        hook: 当前 Hook。
        reason: 失败原因。
        matched: Hook 是否已经匹配。

    返回:
        ToolHookTrace。
    """

    return ToolHookTrace(
        hook_name=hook.name,
        event=hook.event,
        matched=matched,
        reason=f"hook failed: {reason}",
        metadata=_hook_metadata(hook),
    )

def _hook_metadata(hook: ToolHook) -> dict[str, Any]:
    """读取 Hook 上的 trace metadata。

    输入:
        hook: 当前 Hook。

    输出:
        metadata 字典；Hook 未提供时返回空字典。
    """

    raw = getattr(hook, "trace_metadata", {})
    return dict(raw) if isinstance(raw, dict) else {}