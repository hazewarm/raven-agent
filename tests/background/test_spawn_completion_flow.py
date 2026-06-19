"""test_spawn_completion_flow.py —— 完成回流时 session 只保存短 marker。

覆盖：
  - Prompt 中看到完整 raw result
  - session 只保存 persist_user_content marker
  - 主模型回复正确入库
"""

from pathlib import Path

import pytest

from raven_agent.events import InboundMessage
from raven_agent.turn_pipeline import PassiveTurnPipeline, PassiveTurnPipelineDeps
from raven_agent.event_bus import EventBus
from raven_agent.prompt import PromptBuilder
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore
from raven_agent.agent import AgentRunResult


class FakeAgent:
    """记录 LLM 收到的输入并返回固定回复的 fake ReactAgent。"""
    def __init__(self) -> None:
        self.seen_contents: list[str] = []

    async def run(self, messages, **kwargs):
        self.seen_contents = [m.content for m in messages]
        return AgentRunResult(
            content="我已经整理完后台结果，结论如下。",
            iterations=1,
            tools_used=[],
        )


@pytest.mark.asyncio
async def test_completion_persists_marker_not_raw_result(tmp_path: Path) -> None:
    """完成回流时 Prompt 含完整 raw result，但 session 只存短 marker。"""
    sessions = SessionManager(SessionStore(tmp_path / "sessions.db"))
    prompt_builder = PromptBuilder(system_prompt="You are Raven.")
    agent = FakeAgent()
    pipeline = PassiveTurnPipeline(
        PassiveTurnPipelineDeps(
            sessions=sessions,
            prompt_builder=prompt_builder,
            agent=agent,  # type: ignore[arg-type]
            event_bus=EventBus(),
        )
    )

    inbound = InboundMessage(
        channel="telegram",
        sender="spawn",
        chat_id="123",
        content="[后台任务回传]\n执行结果:\n原始后台结果：文件位于 /tmp/report.md\n详细分析如下...",
        metadata={
            "spawn_completion": True,
            "persist_user_content": (
                "[后台任务完成] 整理任务"
                " — 整理资料并对比三个文件的实现差异..."
                " (incomplete) [forced_summary] profile=research"
            ),
            "skip_post_memory": True,
        },
    )

    outbound = await pipeline.run(inbound)
    session = sessions.get_or_create("telegram:123")

    expected_marker = (
        "[后台任务完成] 整理任务"
        " — 整理资料并对比三个文件的实现差异..."
        " (incomplete) [forced_summary] profile=research"
    )

    # 主模型应正常回复
    assert outbound.content == "我已经整理完后台结果，结论如下。"
    # Prompt 中包含 raw result（完整的 inbound.content，不受 persist_user_content 影响）
    assert any("原始后台结果" in c for c in agent.seen_contents)
    # session 中 user message 是 persist_user_content（含任务目标、状态、exit_reason、profile）
    assert session.messages[-2].content == expected_marker
    assert session.messages[-1].content == "我已经整理完后台结果，结论如下。"
    # raw result 没有写入 session
    assert all("原始后台结果" not in m.content for m in session.messages)