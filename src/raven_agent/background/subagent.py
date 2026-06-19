from __future__ import annotations

import json
import logging
from typing import Any

from raven_agent.llm import LLMProvider
from raven_agent.messages import ChatMessage, assistant_message, tool_message, user_message, system_message
from raven_agent.tools.base import Tool
from raven_agent.tools.executor import ToolExecutor
from raven_agent.tools.hooks import ToolExecutionRequest, ToolHook
from raven_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = (
    "根据上述工具结果，决定下一步操作。"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。"
)
_REFLECT_PROMPT_WARN = (
    "步骤预算剩余 {remaining} 步，请优先完成核心目标，跳过非必要步骤。"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。"
)
_FORCED_FINAL_SUMMARY_PROMPT = (
    "你已用完任务执行预算，禁止再调用工具。现在必须直接输出中文最终总结，"
    "供主 agent 回传给用户。必须覆盖：1) 已完成内容；2) 未完成内容；"
    "3) 产出文件路径（如果有）；4) 下一步建议。"
)
_WARN_THRESHOLD = 5
_MAX_TOOL_RESULT_CHARS = 60_000
_RECENT_TOOL_ROUNDS = 10
_TRIM_HEAD_LINES = 3
_TRIM_TAIL_LINES = 3
_TRIM_OMITTED_MARKER = "中间省略"


class SubAgent:
    """一次性本地子 Agent：固定工具集 + 独立上下文 + 有界 ReAct loop。

    输入:
        provider: LLMProvider 实例。
        model: 使用的模型名；为空时 LLMProvider 使用默认模型。
        tools: 子 Agent 可使用的 Tool 列表。
        system_prompt: 子 Agent system prompt。
        max_iterations: 最大 ReAct 轮数。
        tool_hooks: 可选工具 hook 列表。

    输出:
        SubAgent 实例。调用 run(task) 后返回文本结果。
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str = "",
        tools: list[Tool],
        system_prompt: str = "",
        max_iterations: int = 30,
        tool_hooks: list[ToolHook] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model.strip() or None
        self._system_prompt = system_prompt
        self._max_iterations = max(1, int(max_iterations))
        self.last_exit_reason: str = "idle"
        self.iterations_used: int = 0
        self.tools_called: list[str] = []
        self._registry = ToolRegistry()
        for tool in tools:
            if tool.name in {"spawn", "spawn_manage"}:
                continue
            if not self._registry.has_tool(tool.name):
                self._registry.register(tool, risk="read-only", always_on=True)
        self._tool_executor = ToolExecutor(tool_hooks or [])

    async def run(self, task: str) -> str:
        """执行一次子任务并返回文本总结。

        输入:
            task: 主 Agent 交给子 Agent 的完整任务描述。

        输出:
            子 Agent 的最终文本结果。异常时返回错误摘要字符串。
        """
        messages: list[ChatMessage] = []
        if self._system_prompt:
            messages.append(system_message(self._system_prompt))
        messages.append(user_message(task))

        self.last_exit_reason = "running"
        self.iterations_used = 0
        self.tools_called = []
        tool_session_key = f"subagent:{id(self)}"

        for iteration in range(1, self._max_iterations + 1):
            self.iterations_used = iteration
            try:
                response = await self._provider.chat(
                    messages=_trim_tool_results(messages),
                    tools=self._registry.get_schemas(),
                    model=self._model,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.exception("[subagent] LLM 调用失败 iteration=%d", iteration)
                self.last_exit_reason = "error"
                return f"子任务执行失败：{exc}"

            if not response.tool_calls:
                self.last_exit_reason = "completed"
                return (response.content or "").strip() or "（子任务无输出）"

            messages.append(
                assistant_message(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
            )

            for call in response.tool_calls:
                logger.info(
                    "[subagent] 调用工具 %s args=%s",
                    call.name,
                    json.dumps(call.arguments, ensure_ascii=False)[:200],
                )
                result = await self._tool_executor.execute(
                    ToolExecutionRequest(
                        call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        session_key=tool_session_key,
                        metadata={"source": "subagent"},
                    ),
                    self._registry.execute,
                )
                output = result.output
                if len(output) > _MAX_TOOL_RESULT_CHARS:
                    original_len = len(output)
                    output = (
                        output[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n...[工具结果已截断，原始长度 {original_len} 字符]"
                    )
                if result.status == "success" and call.name not in self.tools_called:
                    self.tools_called.append(call.name)
                messages.append(tool_message(tool_call_id=call.id, content=output))

            remaining = self._max_iterations - iteration
            if remaining <= 0:
                break
            reflect = (
                _REFLECT_PROMPT_WARN.format(remaining=remaining)
                if remaining <= _WARN_THRESHOLD
                else _REFLECT_PROMPT
            )
            messages.append(user_message(reflect))

        self.last_exit_reason = "max_iterations"
        return await self._force_final_summary(messages)

    async def _force_final_summary(self, messages: list[ChatMessage]) -> str:
        """达到迭代上限后强制生成最终进度总结。

        输入:
            messages: 子 Agent 当前上下文消息列表。

        输出:
            中文总结文本；LLM 失败时返回兜底说明。
        """
        try:
            response = await self._provider.chat(
                messages=[*messages, user_message(_FORCED_FINAL_SUMMARY_PROMPT)],
                tools=[],
                model=self._model,
            )
            text = (response.content or "").strip()
            if text:
                self.last_exit_reason = "forced_summary"
                return text
        except Exception as exc:
            logger.warning("[subagent] 强制总结失败: %s", exc)
        self.last_exit_reason = "forced_summary_fallback"
        return "子任务已达到步骤预算：已完成部分关键步骤，但仍有未完成项。"


def _trim_tool_results(messages: list[ChatMessage]) -> list[ChatMessage]:
    """清理较早轮次的 tool result，控制子 Agent 上下文增长。

    保留最近 _RECENT_TOOL_ROUNDS 轮的工具结果完整不变。
    更早轮次的 tool 结果保留首尾各若干行作为上下文线索，
    中间用占位符替换——LLM 仍能判断"之前查出了什么类型的信息"，
    但不会被几千行的旧输出占据上下文。

    输入:
        messages: 当前消息列表。

    输出:
        新的消息列表。
    """
    tool_round_indices = [
        index
        for index, message in enumerate(messages)
        if message.role == "assistant" and message.tool_calls
    ]
    if len(tool_round_indices) <= _RECENT_TOOL_ROUNDS:
        return list(messages)
    cutoff = tool_round_indices[-_RECENT_TOOL_ROUNDS]
    trimmed: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if message.role == "tool" and index < cutoff:
            trimmed.append(
                tool_message(
                    tool_call_id=message.tool_call_id,
                    content=_summarize_trimmed(message.content),
                )
            )
        else:
            trimmed.append(message)
    return trimmed


def _summarize_trimmed(content: str) -> str:
    """保留首尾各若干行，中间替换为省略标记。

    输入:
        content: 工具返回的原始文本。

    输出:
        首 _TRIM_HEAD_LINES 行 + 省略标记 + 尾 _TRIM_TAIL_LINES 行的文本。
        总行数不超过 8 行时不做处理，直接返回原文。
    """
    lines = content.split("\n")
    total = len(lines)
    keep = _TRIM_HEAD_LINES + _TRIM_TAIL_LINES
    if total <= keep + 2:
        return content
    head = "\n".join(lines[:_TRIM_HEAD_LINES])
    tail = "\n".join(lines[-_TRIM_TAIL_LINES:])
    omitted = total - keep
    return f"{head}\n\n[{_TRIM_OMITTED_MARKER} {omitted} 行]\n\n{tail}"