from __future__ import annotations

from dataclasses import dataclass, field

from raven_agent.llm import LLMProvider
from raven_agent.messages import ChatMessage, assistant_message, tool_message, user_message
from raven_agent.tools import ToolExecutor, ToolExecutionRequest, ToolRegistry
from raven_agent.tools.search import ToolDiscoveryState, ToolSearchTool

from raven_agent.events import StreamToken, ToolCallCompleted, ToolCallStarted
from raven_agent.event_bus import EventBus

from raven_agent.lifecycle import AfterStepCtx, BeforeStepCtx, ReasoningLifecycle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from raven_agent.background.interrupt import InterruptManager


@dataclass(frozen=True)
class AgentRunResult:
    """Agent 单轮运行结果。

    参数:
        content: 最终要展示给用户的文本回复。
        iterations: 本轮 LLM 调用次数。
        tools_used: 本轮执行过的工具名称列表。
    """

    content: str
    iterations: int
    tools_used: list[str] = field(default_factory=list)


class ReactAgent:
    """最小 ReAct 工具循环 Agent。

    参数:
        provider: LLMProvider，负责调用大模型。
        tools: ToolRegistry，负责提供工具 schema 和执行工具。
        tool_executor: ToolExecutor，负责执行工具 hook 链路。
        max_iterations: 单轮最多允许多少次 LLM 调用。
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = 20,
        interrupt_manager: "InterruptManager | None" = None,
        event_bus: EventBus | None = None,
        streaming_enabled: bool = True,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._tool_executor = tool_executor or ToolExecutor()
        self._max_iterations = max(1, int(max_iterations))
        self._interrupt_manager = interrupt_manager
        self._tool_discovery = ToolDiscoveryState()
        self._event_bus = event_bus or EventBus()
        self._streaming_enabled = streaming_enabled

    async def run(
        self,
        messages: list[ChatMessage],
        session_key: str = "__default__",
        tool_context: dict[str, object] | None = None,
        lifecycle: ReasoningLifecycle | None = None,
        channel: str = "",
        chat_id: str = "",
    ) -> AgentRunResult:
        """运行一轮 ReAct 推理。

        输入:
            messages: 当前轮初始消息，通常来自 PromptBuilder。
            session_key: 当前会话 key，用于隔离工具发现 LRU 状态。
            tool_context: 每轮工具上下文，会作为 ToolExecutionRequest.metadata 传给工具 hook。
            lifecycle: 可选 step 生命周期；提供时每个 step 前后运行 before_step / after_step。
            channel: 当前渠道，用于构造 step ctx。
            chat_id: 当前聊天标识，用于构造 step ctx。

        输出:
            AgentRunResult，包含最终回复、迭代次数和工具使用列表。
        """

        working_messages = list(messages)
        turn_context = dict(tool_context or {})
        tools_used: list[str] = []
        tools_unlocked: list[str] = []
        preloaded_order = self._tool_discovery.get_preloaded_ordered(session_key)
        loaded_names = set(preloaded_order)

        # —— 构造 on_content_delta 回调 ——
        bus = self._event_bus
        async def on_delta(token: str) -> None:
            if bus is not None:
                await bus.observe(
                    StreamToken(
                        session_key=session_key,
                        channel=channel,
                        chat_id=chat_id,
                        token=token,
                    )
                )
        on_delta = on_delta if self._streaming_enabled else None
        
        
        
        for iteration in range(1, self._max_iterations + 1):
            visible_names = self._tools.get_visible_names(loaded_names)

            if lifecycle is not None:
                before_ctx = await lifecycle.before_step(
                    BeforeStepCtx(
                        session_key=session_key,
                        channel=channel,
                        chat_id=chat_id,
                        iteration=iteration,
                        visible_tool_names=frozenset(visible_names),
                        context_tokens_estimate=_estimate_tokens(working_messages),
                    )
                )
                for hint in before_ctx.extra_hints:
                    if hint.strip():
                        working_messages.append(user_message(f"[hint] {hint}"))
                if before_ctx.early_stop:
                    # 插件（如 tool_loop_guard）请求提前结束工具循环。
                    return AgentRunResult(
                        content=before_ctx.early_stop_reply or "（工具循环已提前结束）",
                        iterations=iteration,
                        tools_used=tools_used,
                    )

            tool_schemas = self._tools.get_visible_schemas(loaded_names)
            # ── 过滤 disabled_tools：从发给 LLM 的 schema 中移除禁用工具 ──
            disabled = turn_context.get("disabled_tools")
            if isinstance(disabled, (list, set)):
                tool_schemas = [
                    schema for schema in tool_schemas if schema.get("function", {}).get("name") not in disabled
                ]
            response = await self._provider.chat(
                working_messages,
                tools=tool_schemas,
                tool_choice="auto",
                on_content_delta=on_delta,
            )

            if not response.tool_calls:
                if lifecycle is not None:
                    await lifecycle.after_step(
                        AfterStepCtx(
                            session_key=session_key,
                            channel=channel,
                            chat_id=chat_id,
                            iteration=iteration,
                            tools_called=(),
                            partial_reply=response.content or "",
                            has_more=False,
                            context_tokens_estimate=_estimate_tokens(working_messages),
                        )
                    )
                self._tool_discovery.update(
                    session_key,
                    [*tools_unlocked, *tools_used],
                    self._tools.get_always_on_names(),
                )
                return AgentRunResult(
                    content=response.content or "（无响应）",
                    iterations=iteration,
                    tools_used=tools_used,
                )

            working_messages.append(
                assistant_message(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
            )

            for call in response.tool_calls:
                guard_message = self._tools.get_execution_guard_message(
                    call.name,
                    loaded_names,
                )
                if guard_message is not None:
                    working_messages.append(
                        tool_message(
                            tool_call_id=call.id,
                            content=guard_message,
                        )
                    )
                    continue

                tool = self._tools.get(call.name)
                if call.name == "tool_search" and isinstance(tool, ToolSearchTool):
                    tool.set_excluded_names(self._tools.get_visible_names(loaded_names))

                # —— 发射 ToolCallStarted ——
                if self._event_bus is not None:
                    await self._event_bus.observe(
                        ToolCallStarted(
                            session_key=session_key,
                            channel=channel,
                            chat_id=chat_id,
                            iteration=iteration,
                            call_id=call.id,
                            tool_name=call.name,
                            arguments=dict(call.arguments),
                        )
                    )
                
                result = await self._tool_executor.execute(
                    ToolExecutionRequest(
                        call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        session_key=session_key,
                        metadata=turn_context,
                    ),
                    self._tools.execute,
                )

                if result.status == "success":
                    tools_used.append(call.name)
                    if self._interrupt_manager is not None:
                        self._interrupt_manager.record_tool_call(
                            session_key=session_key,
                            tool_name=call.name,
                            arguments=call.arguments,
                        )

                if call.name == "tool_search" and result.status == "success":
                    unlocked = self._tool_discovery.unlock_names_from_result(result.output)
                    new_unlocked = [name for name in unlocked if name not in loaded_names]
                    loaded_names.update(new_unlocked)
                    tools_unlocked.extend(new_unlocked)

                # —— 发射 ToolCallCompleted ——
                if self._event_bus is not None:
                    await self._event_bus.observe(
                        ToolCallCompleted(
                            session_key=session_key,
                            channel=channel,
                            chat_id=chat_id,
                            iteration=iteration,
                            call_id=call.id,
                            tool_name=call.name,
                            arguments=dict(call.arguments),
                            final_arguments=dict(call.arguments),
                            status=result.status,
                            result_preview=_truncate(result.output),
                        )
                    )
                
                
                working_messages.append(
                    tool_message(
                        tool_call_id=call.id,
                        content=result.output,
                    )
                )
            
            if lifecycle is not None:
                step_tools = tuple(call.name for call in response.tool_calls)
                after_ctx = await lifecycle.after_step(
                    AfterStepCtx(
                        session_key=session_key,
                        channel=channel,
                        chat_id=chat_id,
                        iteration=iteration,
                        tools_called=step_tools,
                        partial_reply=response.content or "",
                        has_more=True,
                        context_tokens_estimate=_estimate_tokens(working_messages),
                    )
                )
                early_stop_reason = after_ctx.extra_metadata.get("early_stop_reason")
                if isinstance(early_stop_reason, str) and early_stop_reason.strip():
                    # 插件（如 context_pressure）请求阶段性收尾：让模型基于已有结果直接回答。
                    working_messages.append(
                        user_message(
                            "上下文已接近上限，请基于已有工具结果直接回答用户，不要再调用工具。"
                        )
                    )
                    final_response = await self._provider.chat(working_messages, tools=[])
                    self._tool_discovery.update(
                        session_key,
                        [*tools_unlocked, *tools_used],
                        self._tools.get_always_on_names(),
                    )
                    return AgentRunResult(
                        content=final_response.content or "（上下文压力收尾未能生成回复）",
                        iterations=iteration,
                        tools_used=tools_used,
                    )

        working_messages.append(
            user_message(
                "工具调用已经达到本轮上限。请基于已有工具结果直接回答用户，不要再调用工具。"
            )
        )
        final_response = await self._provider.chat(working_messages, tools=[])
        self._tool_discovery.update(
            session_key,
            [*tools_unlocked, *tools_used],
            self._tools.get_always_on_names(),
        )
        return AgentRunResult(
            content=final_response.content or "工具调用达到上限，未能生成最终回复。",
            iterations=self._max_iterations,
            tools_used=tools_used,
        )


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    """粗略估算一组消息的 token 数。

    输入:
        messages: 当前 ReAct 工作消息列表。

    输出:
        token 估算值。使用“字符数整除 4”的常见近似，不追求精确。
    """

    total_chars = 0
    for message in messages:
        total_chars += len(message.content or "")
        for call in message.tool_calls:
            total_chars += len(call.name) + len(str(call.arguments))
    return total_chars // 4

def _truncate(text: str, limit: int = 200) -> str:
    """截断工具结果，避免 ToolCallCompleted 事件的 result_preview 过长。

    输入:
        text: 工具结果文本。
        limit: 最大字符数，默认 200。

    输出:
        截断后的文本。
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."