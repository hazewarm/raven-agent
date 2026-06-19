from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raven_agent.memory2.query_rewriter import GateDecision, QueryRewriter


@dataclass(frozen=True)
class FakeResponse:
    """测试用 LLM 响应。"""

    content: str


class FakeProvider:
    """按 prompt 内容返回固定响应的测试 provider。

    参数:
        history_response: history gate 响应。
        procedure_response: procedure rewrite 响应。
        raise_error: 是否抛异常。

    返回:
        FakeProvider 实例。
    """

    def __init__(
        self,
        *,
        history_response: str,
        procedure_response: str = "",
        raise_error: bool = False,
    ) -> None:
        self.history_response = history_response
        self.procedure_response = procedure_response
        self.raise_error = raise_error

    async def chat(self, *, messages, **kwargs: object) -> FakeResponse:
        """根据 prompt 返回测试响应。

        参数:
            messages: provider messages。
            kwargs: 其他 provider 参数。

        返回:
            FakeResponse。
        """

        if self.raise_error:
            raise RuntimeError("boom")
        prompt = messages[0].content
        if "只输出一行检索 query" in prompt:
            return FakeResponse(self.procedure_response)
        return FakeResponse(self.history_response)


def test_gate_decision_fields() -> None:
    """测试 GateDecision 字段。"""

    decision = GateDecision(
        needs_episodic=True,
        episodic_query="用户设备型号",
        procedure_query="用户发送设备问题时 agent 应如何处理",
        latency_ms=12,
    )

    assert decision.needs_episodic is True
    assert decision.episodic_query == "用户设备型号"
    assert decision.procedure_query
    assert decision.latency_ms == 12


def test_query_rewriter_retrieve_decision() -> None:
    """测试 QueryRewriter 解析 RETRIEVE 输出。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(
            history_response="<decision>RETRIEVE</decision><history_query>用户的 Fitbit 设备型号</history_query>",
            procedure_response="用户询问记忆内容时 agent 应如何查找依据",
        )
        rewriter = QueryRewriter(provider=provider)  # type: ignore[arg-type]

        result = await rewriter.decide("你还记得我用什么 Fitbit 吗？", "")

        assert result.needs_episodic is True
        assert result.episodic_query == "用户的 Fitbit 设备型号"
        assert result.procedure_query == "用户询问记忆内容时 agent 应如何查找依据"

    asyncio.run(run())


def test_query_rewriter_no_retrieve_decision() -> None:
    """测试 QueryRewriter 解析 NO_RETRIEVE 输出。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(
            history_response="<decision>NO_RETRIEVE</decision><history_query></history_query>",
            procedure_response="None",
        )
        rewriter = QueryRewriter(provider=provider)  # type: ignore[arg-type]

        result = await rewriter.decide("你好", "")

        assert result.needs_episodic is False
        assert result.procedure_query == ""

    asyncio.run(run())


def test_query_rewriter_fails_open_on_bad_output() -> None:
    """测试 malformed 输出时 fail-open。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(history_response="garbage")
        rewriter = QueryRewriter(provider=provider)  # type: ignore[arg-type]

        result = await rewriter.decide("帮我查以前说过什么", "")

        assert result.needs_episodic is True
        assert result.episodic_query == "帮我查以前说过什么"

    asyncio.run(run())


def test_query_rewriter_fails_open_on_exception() -> None:
    """测试 LLM 异常时 fail-open。"""

    async def run() -> None:
        """执行异步测试主体。

        返回:
            None。
        """

        provider = FakeProvider(history_response="", raise_error=True)
        rewriter = QueryRewriter(provider=provider)  # type: ignore[arg-type]

        result = await rewriter.decide("帮我查以前说过什么", "")

        assert result.needs_episodic is True
        assert result.episodic_query == "帮我查以前说过什么"

    asyncio.run(run())