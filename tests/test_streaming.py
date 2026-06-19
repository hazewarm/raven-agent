"""流式输出与 Live 消息集成测试。"""

import asyncio
import pytest
from raven_agent.events import (
    StreamStart,
    StreamToken,
    StreamEnd,
    ToolCallStarted,
    ToolCallCompleted,
)
from raven_agent.event_bus import EventBus


class TestStreamEvents:
    """流式事件基本行为测试。"""

    def test_stream_events_are_frozen(self):
        """StreamToken 等事件是不可变的 frozen dataclass。"""
        token = StreamToken(
            session_key="cli:default",
            channel="cli",
            chat_id="default",
            token="hello",
        )
        with pytest.raises(Exception):
            token.token = "world"  # frozen dataclass 禁止修改

    async def test_event_bus_observe_stream(self):
        """EventBus.observe() 可以传递流式事件。"""
        bus = EventBus()
        received: list[str] = []

        async def on_token(event: StreamToken) -> None:
            received.append(event.token)

        bus.on(StreamToken, on_token)

        await bus.observe(
            StreamToken(
                session_key="test", channel="cli",
                chat_id="x", token="a",
            )
        )
        await bus.observe(
            StreamToken(
                session_key="test", channel="cli",
                chat_id="x", token="b",
            )
        )

        assert received == ["a", "b"]


class TestToolLiveFormatting:
    """工具进度格式化测试。"""

    def test_format_empty(self):
        """空工具行和空回复返回提示文本。"""
        from raven_agent.channels.telegram.utils import _format_turn_live
        result = _format_turn_live([], "", terminal=True)
        assert "本轮预览完成" in result

    def test_format_with_tools(self):
        """有工具调用时显示工具列表和状态。"""
        from raven_agent.channels.telegram.utils import _format_turn_live
        lines = [
            {"tool_name": "web_search", "intent": "天气",
             "target": "", "status": "done"},
            {"tool_name": "read_file", "intent": "config",
             "target": '"config.toml"', "status": "running"},
        ]
        result = _format_turn_live(lines, "", terminal=False)
        assert "web_search" in result
        assert "✅" in result
        assert "read_file" in result
        assert "config.toml" in result

    def test_tool_emoji_mapping(self):
        """工具名到 emoji 的映射正确。"""
        from raven_agent.channels.telegram.utils import _tool_emoji
        assert _tool_emoji("web_search") == "🔍"
        assert _tool_emoji("read_file") == "📄"
        assert _tool_emoji("shell") == "⚙"
        assert _tool_emoji("spawn") == "🤖"
        assert _tool_emoji("unknown_tool") == "🔧"