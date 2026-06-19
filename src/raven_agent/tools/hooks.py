from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


ToolHookEvent = Literal["pre_tool_use", "post_tool_use", "post_tool_error"]
ToolHookDecision = Literal["pass", "deny"]
ToolExecutionStatus = Literal["success", "denied", "error"]


def _empty_metadata() -> dict[str, Any]:
    """创建空 metadata 字典。

    返回:
        新的空字典。
    """

    return {}


@dataclass(frozen=True)
class ToolExecutionRequest:
    """一次工具执行请求。

    参数:
        call_id: 模型生成的 tool call id。
        tool_name: 要执行的工具名称。
        arguments: 工具参数。
        session_key: 当前会话 key。
        metadata: 附加信息。
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    session_key: str = ""
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True)
class ToolHookContext:
    """Hook 执行上下文。

    参数:
        event: 当前 hook 事件。
        request: 原始工具执行请求。
        current_arguments: 当前参数，可能已被前序 pre hook 改写。
        result: 工具成功执行后的结果。
        error: 工具执行失败时的错误文本。
    """

    event: ToolHookEvent
    request: ToolExecutionRequest
    current_arguments: dict[str, Any]
    result: Any = None
    error: str = ""


@dataclass(frozen=True)
class ToolHookOutcome:
    """Hook 执行结果。

    参数:
        decision: pass 表示放行，deny 表示拒绝工具执行。
        updated_arguments: pre hook 返回的新参数；只在 pre_tool_use 中生效。
        extra_message: 附加提示，会写入 ToolExecutionResult。
        reason: deny 或 trace 记录使用的原因。
    """

    decision: ToolHookDecision = "pass"
    updated_arguments: dict[str, Any] | None = None
    extra_message: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ToolHookTrace:
    """单个 hook 的执行轨迹。

    输入:
        hook_name: Hook 名称。
        event: Hook 事件。
        matched: Hook 是否匹配本次调用。
        decision: Hook 决策。
        reason: Hook 原因。
        extra_message: Hook 附加提示。
        metadata: Hook 来源元数据，例如 plugin_id / handler。

    输出:
        ToolHookTrace 实例。
    """

    hook_name: str
    event: ToolHookEvent
    matched: bool
    decision: ToolHookDecision = "pass"
    reason: str = ""
    extra_message: str = ""
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True)
class ToolExecutionResult:
    """ToolExecutor 的统一执行结果。

    参数:
        status: success / denied / error。
        output: 给模型回填的文本。
        final_arguments: 最终传给工具的参数。
        extra_messages: hooks 产生的附加提示。
        pre_hook_trace: pre hooks 的执行轨迹。
        post_hook_trace: post hooks 的执行轨迹。
    """

    status: ToolExecutionStatus
    output: str
    final_arguments: dict[str, Any]
    extra_messages: list[str] = field(default_factory=list)
    pre_hook_trace: list[ToolHookTrace] = field(default_factory=list)
    post_hook_trace: list[ToolHookTrace] = field(default_factory=list)


class ToolHook(ABC):
    """工具 Hook 抽象基类。

    参数:
        无。具体 Hook 可以在子类 __init__ 中定义自己的参数。
    """

    name: str
    event: ToolHookEvent

    @abstractmethod
    def matches(self, context: ToolHookContext) -> bool:
        """判断 Hook 是否作用于当前工具调用。

        参数:
            context: 当前 Hook 上下文。

        返回:
            True 表示执行 run()，False 表示跳过。
        """

    @abstractmethod
    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """执行 Hook。

        参数:
            context: 当前 Hook 上下文。

        返回:
            ToolHookOutcome。
        """