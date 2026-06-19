from __future__ import annotations

import asyncio

from raven_agent.lifecycle import AfterReasoningCtx, PromptRenderCtx
from raven_agent.plugins.builtins.citation import (
    CitationAfterReasoningModule,
    CitationPromptModule,
    extract_cited_ids,
)
from raven_agent.turn_pipeline import AfterReasoningFrame, PromptRenderFrame


def test_extract_cited_ids_parses_and_cleans() -> None:
    """测试 extract_cited_ids 解析并清理引用行。

    输入:
        无。

    输出:
        None。
    """

    cleaned, ids = extract_cited_ids("我记得你喜欢茶\n§cited:[a,b]§")

    assert cleaned == "我记得你喜欢茶"
    assert ids == ["a", "b"]


def test_citation_after_reasoning_writes_metadata() -> None:
    """测试 after_reasoning module 写 cited_memory_ids 并清理 reply。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        ctx = AfterReasoningCtx(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            tools_used=(),
            reply="好的\n§cited:[m1]§",
        )
        frame = AfterReasoningFrame(input=None)  # type: ignore[arg-type]
        frame.slots["after_reasoning:ctx"] = ctx
        await CitationAfterReasoningModule().run(frame)

        assert ctx.reply == "好的"
        assert ctx.outbound_metadata["cited_memory_ids"] == ["m1"]

    asyncio.run(run())


def test_citation_prompt_module_injects_protocol() -> None:
    """测试 prompt_render module 注入引用协议 section。

    输入:
        无。

    输出:
        None。
    """

    async def run() -> None:
        """执行异步测试主体。"""

        ctx = PromptRenderCtx(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            content="hi",
        )
        frame = PromptRenderFrame(input=None)  # type: ignore[arg-type]
        frame.slots["prompt_render:ctx"] = ctx
        await CitationPromptModule().run(frame)

        names = [section.name for section in ctx.system_sections_bottom]
        assert "citation_protocol" in names

    asyncio.run(run())