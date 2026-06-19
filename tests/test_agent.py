from __future__ import annotations

import asyncio
from typing import Any

from raven_agent.agent import ReactAgent
from raven_agent.llm import LLMResponse
from raven_agent.messages import ChatMessage, ToolCall, user_message
from raven_agent.tools import Tool, ToolRegistry, ToolExecutor
from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome
from raven_agent.tools.search import ToolSearchTool


class EchoTool(Tool):
    """测试用 echo 工具。

    参数:
        无。
    """

    name = "echo"
    description = "返回输入文本。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要返回的文本。"},
        },
        "required": ["text"],
    }

    async def execute(self, text: str, **kwargs: Any) -> str:
        """返回输入文本。

        参数:
            text: 要返回的文本。
            **kwargs: 预留参数，本测试工具不使用。

        返回:
            echo 文本。
        """

        return f"echo: {text}"

class RewriteEchoHook(ToolHook):
    """把 echo 工具参数改写为 hooked。

    参数:
        无。
    """

    name = "rewrite_echo"
    event = "pre_tool_use"

    def matches(self, context: ToolHookContext) -> bool:
        """只匹配 echo 工具。

        参数:
            context: Hook 上下文。

        返回:
            当前工具是 echo 时返回 True。
        """

        return context.request.tool_name == "echo"

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """改写 echo 参数。

        参数:
            context: Hook 上下文。

        返回:
            带 updated_arguments 的 ToolHookOutcome。
        """

        return ToolHookOutcome(updated_arguments={"text": "hooked"})


class FakeProvider:
    """测试用假模型 Provider。

    参数:
        无。内部会记录每次 chat() 调用的消息与工具 schema。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        **kwargs
    ) -> LLMResponse:
        """模拟模型调用。

        参数:
            messages: 当前传给模型的消息。
            tools: 当前传给模型的工具 schema。
            tool_choice: 工具选择策略。

        返回:
            第一次返回工具调用，第二次返回最终文本。
        """

        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
                "tool_choice": tool_choice,
            }
        )
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                reasoning_content="model reasoning",
            )
        return LLMResponse(content=f"final: {messages[-1].content}")


def test_react_agent_executes_tool_and_returns_final_answer() -> None:
    """测试 ReactAgent 会执行工具并继续请求最终回答。

    返回:
        None。
    """

    provider = FakeProvider()
    registry = ToolRegistry()
    registry.register(EchoTool(), always_on=True)
    agent = ReactAgent(provider=provider, tools=registry)

    result = asyncio.run(agent.run([user_message("use echo")]))

    assert result.content == "final: echo: hello"
    assert result.tools_used == ["echo"]
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"][0]["function"]["name"] == "echo"
    assert provider.calls[1]["messages"][-1].role == "tool"
    assert provider.calls[1]["messages"][-2].reasoning_content == "model reasoning"


def test_react_agent_returns_direct_answer_without_tool_call() -> None:
    """测试模型不请求工具时，ReactAgent 直接返回最终回答。

    返回:
        None。
    """

    class DirectProvider:
        """测试用直接回复 Provider。

        参数:
            无。
        """

        async def chat(
            self,
            messages: list[ChatMessage],
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] = "auto",
            **kwargs
        ) -> LLMResponse:
            """直接返回文本回复。

            参数:
                messages: 当前传给模型的消息。
                tools: 当前传给模型的工具 schema。
                tool_choice: 工具选择策略。

            返回:
                不包含 tool_calls 的 LLMResponse。
            """

            return LLMResponse(content="direct answer")

    registry = ToolRegistry()
    registry.register(EchoTool(), always_on=True)
    agent = ReactAgent(provider=DirectProvider(), tools=registry)

    result = asyncio.run(agent.run([user_message("hello")]))

    assert result.content == "direct answer"
    assert result.tools_used == []
    assert result.iterations == 1


def test_react_agent_rewrite_EchoHook_and_returns_final_answer() -> None:
    """测试 ReactAgent 会执行工具并继续请求最终回答。

    返回:
        None。
    """

    provider = FakeProvider()
    registry = ToolRegistry()
    registry.register(EchoTool(), always_on=True)
    agent = ReactAgent(
        provider=provider,
        tools=registry,
        tool_executor=ToolExecutor([RewriteEchoHook()]),
    )

    result = asyncio.run(agent.run([user_message("use echo")]))

    assert result.content == "final: echo: hooked"
    assert result.tools_used == ["echo"]
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"][0]["function"]["name"] == "echo"
    assert provider.calls[1]["messages"][-1].role == "tool"
    assert provider.calls[1]["messages"][-2].reasoning_content == "model reasoning"

def test_react_agent_blocks_invisible_deferred_tool() -> None:
    """测试 deferred 工具未解锁时不会被直接执行。

    返回:
        None。
    """

    class DirectEchoProvider(FakeProvider):
        pass

    provider = DirectEchoProvider()
    registry = ToolRegistry()
    registry.register(ToolSearchTool(registry), always_on=True)
    registry.register(EchoTool())
    agent = ReactAgent(provider=provider, tools=registry)

    result = asyncio.run(agent.run([user_message("use echo")], session_key="cli:default"))

    assert "select:echo" in provider.calls[1]["messages"][-1].content
    assert "echo" not in result.tools_used


def test_react_agent_tool_search_unlocks_deferred_tool() -> None:
    """测试 tool_search 解锁后，下一轮 LLM 可以看到 deferred 工具。

    返回:
        None。
    """

    class SearchThenEchoProvider:
        """先请求 tool_search，再请求 echo 的测试 Provider。

        参数:
            无。内部记录每次 chat() 的 tools。
        """

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def chat(
            self,
            messages: list[ChatMessage],
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] = "auto",
            **kwargs
        ) -> LLMResponse:
            """按调用次数返回不同模型响应。

            参数:
                messages: 当前消息。
                tools: 当前可见工具 schema。
                tool_choice: 工具选择策略。

            返回:
                LLMResponse。
            """

            self.calls.append({"messages": messages, "tools": tools or []})
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="s1", name="tool_search", arguments={"query": "select:echo"})],
                )
            if len(self.calls) == 2:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="e1", name="echo", arguments={"text": "hello"})],
                )
            return LLMResponse(content=f"final: {messages[-1].content}")

    provider = SearchThenEchoProvider()
    registry = ToolRegistry()
    registry.register(ToolSearchTool(registry), always_on=True)
    registry.register(EchoTool())
    agent = ReactAgent(provider=provider, tools=registry)

    result = asyncio.run(agent.run([user_message("use echo")], session_key="cli:default"))

    first_tools = [schema["function"]["name"] for schema in provider.calls[0]["tools"]]
    second_tools = [schema["function"]["name"] for schema in provider.calls[1]["tools"]]

    assert first_tools == ["tool_search"]
    assert "echo" in second_tools
    assert result.content == "final: echo: hello"
    assert result.tools_used == ["tool_search", "echo"]


def test_react_agent_passes_tool_context_to_execution_request() -> None:
    """测试 ReactAgent.run 把 tool_context 透传进 ToolExecutionRequest.metadata。"""

    from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome

    captured: dict[str, object] = {}

    class _CaptureHook(ToolHook):
        name = "capture"
        event = "pre_tool_use"

        def matches(self, context: ToolHookContext) -> bool:
            return True

        async def run(self, context: ToolHookContext) -> ToolHookOutcome:
            captured.update(context.request.metadata)
            return ToolHookOutcome()

    provider = FakeProvider()
    registry = ToolRegistry()
    registry.register(EchoTool(), always_on=True)
    agent = ReactAgent(
        provider=provider,
        tools=registry,
        tool_executor=ToolExecutor([_CaptureHook()]),
    )

    asyncio.run(
        agent.run(
            [user_message("use echo")],
            session_key="cli:default",
            tool_context={"current_user_source_ref": '["cli:default:0"]'},
        )
    )

    assert captured["current_user_source_ref"] == '["cli:default:0"]'