from __future__ import annotations

import asyncio

from raven_agent.agent import AgentRunResult
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, TurnCompleted, TurnStarted
from raven_agent.messages import ChatMessage
from raven_agent.prompt import PromptBuilder
from raven_agent.runtime import handle_inbound_message
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore

class FakeAgent:
    """测试用假 Agent。

    参数:
        无。内部记录收到的 messages。
    """

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def run(
        self,
        messages: list[ChatMessage],
        session_key: str = "__default__",
        tool_context: dict[str, object] | None = None,
        lifecycle: object | None = None,
        channel: str = "",
        chat_id: str = "",
    ) -> AgentRunResult:
        """模拟 Agent 单轮运行。

        参数:
            messages: Runtime 构造出的模型输入。
            session_key: 当前会话 key，供测试观察。

        返回:
            固定 AgentRunResult。
        """

        self.messages = messages
        return AgentRunResult(content="pong", iterations=1, tools_used=["fake_tool"])


def test_handle_inbound_message_returns_outbound_and_updates_session(tmp_path) -> None:
    """测试 runtime 可以处理入站消息、返回出站消息并更新 Session。

    参数:
        tmp_path: pytest 提供的临时目录。
    
    返回:
        None。
    """

    async def run() -> None:
        store = SessionStore(tmp_path / "sessions.db")
        sessions = SessionManager(store)
        try:
            prompt_builder = PromptBuilder(system_prompt="You are Raven.")
            agent = FakeAgent()
            inbound = InboundMessage(
                channel="cli",
                sender="local",
                chat_id="default",
                content="ping",
            )

            outbound = await handle_inbound_message(
                inbound=inbound,
                sessions=sessions,
                prompt_builder=prompt_builder,
                agent=agent,  # type: ignore[arg-type]
            )
            session = sessions.get_or_create("cli:default")
            persisted = store.fetch_session_messages("cli:default")


            assert outbound.channel == "cli"
            assert outbound.chat_id == "default"
            assert outbound.content == "pong"
            assert outbound.metadata["iterations"] == 1
            assert outbound.metadata["tools_used"] == ["fake_tool"]
            assert [message.content for message in session.messages] == ["ping", "pong"]
            assert [message.id for message in session.messages] == [
                "cli:default:0",
                "cli:default:1",
            ]
            assert [message.content for message in persisted] == ["ping", "pong"]
            msgs = [message.content for message in agent.messages]
            assert "You are Raven." in msgs[0]
            assert msgs[1] == "ping"
            assert [message.id for message in persisted] == [
                "cli:default:0",
                "cli:default:1",
            ]
        finally:
            sessions.close()

    asyncio.run(run())


def test_handle_inbound_message_emits_turn_events(tmp_path) -> None:
    """测试 runtime 会触发 TurnStarted 和 TurnCompleted。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    async def run() -> None:
        store = SessionStore(tmp_path / "sessions.db")
        sessions = SessionManager(store)
        try:
            prompt_builder = PromptBuilder(system_prompt="sys")
            agent = FakeAgent()
            event_bus = EventBus()
            seen: list[str] = []

            def on_started(event: TurnStarted) -> None:
                seen.append(f"started:{event.session_key}")

            def on_completed(event: TurnCompleted) -> None:
                seen.append(f"completed:{event.outbound.content}")

            event_bus.on(TurnStarted, on_started)
            event_bus.on(TurnCompleted, on_completed)
            inbound = InboundMessage(
                channel="cli",
                sender="local",
                chat_id="default",
                content="ping",
            )

            await handle_inbound_message(
                inbound=inbound,
                sessions=sessions,
                prompt_builder=prompt_builder,
                agent=agent,  # type: ignore[arg-type]
                event_bus=event_bus,
            )

            assert seen == ["started:cli:default", "completed:pong"]
        finally:
            sessions.close()

    asyncio.run(run())


def test_turn_started_handler_can_rewrite_inbound_content(tmp_path) -> None:
    """测试 TurnStarted handler 可以修改进入 Agent 的消息内容。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        None。
    """

    async def run() -> None:
        store = SessionStore(tmp_path / "sessions.db")
        sessions = SessionManager(store)
        try:
            prompt_builder = PromptBuilder(system_prompt="sys")
            agent = FakeAgent()
            event_bus = EventBus()

            def rewrite(event: TurnStarted) -> TurnStarted:
                rewritten = InboundMessage(
                    channel=event.inbound.channel,
                    sender=event.inbound.sender,
                    chat_id=event.inbound.chat_id,
                    content="rewritten",
                    timestamp=event.inbound.timestamp,
                    metadata=event.inbound.metadata,
                )
                return TurnStarted(session_key=event.session_key, inbound=rewritten)

            event_bus.on(TurnStarted, rewrite)
            inbound = InboundMessage(
                channel="cli",
                sender="local",
                chat_id="default",
                content="original",
            )

            await handle_inbound_message(
                inbound=inbound,
                sessions=sessions,
                prompt_builder=prompt_builder,
                agent=agent,  # type: ignore[arg-type]
                event_bus=event_bus,
            )

            session = sessions.get_or_create("cli:default")
            assert session.messages[0].content == "rewritten"
            assert agent.messages[-1].content == "rewritten"
        finally:
            sessions.close()

    asyncio.run(run())