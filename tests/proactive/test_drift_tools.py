"""
test_drift_tools.py —— Drift 工具单元测试。

覆盖：
  - FinishDriftTool: 正常结束 / 未知 skill / 空字段 /
    message_result 一致性校验 / drift_finished 标志
  - DriftSendMessageTool: 正常发送 / 重复拒绝 / 空消息拒绝
  - DriftWebFetchTool: 正常结果 / 截断逻辑
  - build_drift_tool_registry: 组装完整性
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore
from raven_agent.proactive.drift_tools import (
    DriftSendMessageTool,
    DriftWebFetchTool,
    FinishDriftTool,
    build_drift_tool_registry,
)


def _make_skill(source_dir: Path, state_dir: Path, name: str) -> DriftStateStore:
    """创建包含单个 skill 的 DriftStateStore。SKILL.md 写到 source_dir，状态写入 state_dir。"""
    skill_dir = source_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: {name} 的测试描述",
        "---",
        "",
        "# 目标",
    ]
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return DriftStateStore(source_dir=source_dir, state_dir=state_dir)


class TestFinishDriftTool:
    """FinishDriftTool 行为测试。"""

    @pytest.mark.asyncio
    async def test_finish_ok_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext(session_key="cli:default")

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="完成测试",
                    next="继续测试",
                    message_result="silent",
                )
            )
            assert result["ok"] is True
            assert ctx.drift_finished is True

    @pytest.mark.asyncio
    async def test_finish_ok_sent_after_message_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext(session_key="cli:default")
            ctx.drift_message_sent = True  # 模拟已推送

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="完成并推送",
                    next="等待回复",
                    message_result="sent",
                )
            )
            assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_finish_rejects_unknown_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "known-skill")
            ctx = DriftAgentTickContext()

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="unknown-skill",
                    one_line="test",
                    next="test",
                    message_result="silent",
                )
            )
            assert "error" in result
            assert "未知 skill" in result["error"]

    @pytest.mark.asyncio
    async def test_finish_rejects_empty_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="",
                    next="test",
                    message_result="silent",
                )
            )
            assert "error" in result

    @pytest.mark.asyncio
    async def test_finish_rejects_empty_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="test",
                    next="",
                    message_result="silent",
                )
            )
            assert "error" in result

    @pytest.mark.asyncio
    async def test_finish_rejects_sent_without_push(self):
        """message_result=sent 但未调用过 message_push → 拒绝"""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="完成",
                    next="继续",
                    message_result="sent",
                )
            )
            assert "error" in result
            assert "sent" in result["error"]

    @pytest.mark.asyncio
    async def test_finish_rejects_silent_after_push(self):
        """message_result=silent 但调用过 message_push → 拒绝"""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()
            ctx.drift_message_sent = True

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="完成",
                    next="继续",
                    message_result="silent",
                )
            )
            assert "error" in result
            assert "冲突" in result["error"]

    @pytest.mark.asyncio
    async def test_finish_rejects_invalid_message_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            tool = FinishDriftTool(ctx, store)
            result = json.loads(
                await tool.execute(
                    skill_used="test-skill",
                    one_line="完成",
                    next="继续",
                    message_result="invalid",
                )
            )
            assert "error" in result
            assert "sent 或 silent" in result["error"]


class TestDriftSendMessageTool:
    """DriftSendMessageTool 行为测试。"""

    @pytest.mark.asyncio
    async def test_send_ok(self):
        sent: list[tuple[str, list[str]]] = []

        async def mock_send(text: str, media: list[str]) -> bool:
            sent.append((text, media))
            return True

        ctx = DriftAgentTickContext()
        tool = DriftSendMessageTool(ctx, mock_send)
        result = json.loads(await tool.execute(message="Hello"))
        assert result["ok"] is True
        assert ctx.drift_message_sent is True
        assert len(sent) == 1
        assert sent[0][0] == "Hello"

    @pytest.mark.asyncio
    async def test_send_rejects_duplicate(self):
        async def mock_send(text: str, media: list[str]) -> bool:
            return True

        ctx = DriftAgentTickContext()
        tool = DriftSendMessageTool(ctx, mock_send)

        # 第一次成功
        r1 = json.loads(await tool.execute(message="First"))
        assert r1["ok"] is True

        # 第二次拒绝
        r2 = json.loads(await tool.execute(message="Second"))
        assert "error" in r2
        assert "已使用" in r2["error"]

    @pytest.mark.asyncio
    async def test_send_rejects_empty(self):
        async def mock_send(text: str, media: list[str]) -> bool:
            return True

        ctx = DriftAgentTickContext()
        tool = DriftSendMessageTool(ctx, mock_send)
        result = json.loads(await tool.execute(message=""))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_send_handles_failure(self):
        async def mock_send(text: str, media: list[str]) -> bool:
            return False

        ctx = DriftAgentTickContext()
        tool = DriftSendMessageTool(ctx, mock_send)
        result = json.loads(await tool.execute(message="Hello"))
        assert "error" in result
        assert ctx.drift_message_sent is False  # 失败不标记


class TestDriftWebFetchTool:
    """DriftWebFetchTool 截断行为测试。"""

    @pytest.mark.asyncio
    async def test_no_truncation_when_under_limit(self):
        from raven_agent.tools.base import Tool

        class FakeWebFetch(Tool):
            @property
            def name(self) -> str:
                return "web_fetch"

            @property
            def description(self) -> str:
                return "fake"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> str:
                return json.dumps({"text": "短文本"})

        wrapped = DriftWebFetchTool(FakeWebFetch(), max_chars=100)
        result = await wrapped.execute()
        payload = json.loads(result)
        assert payload["text"] == "短文本"
        assert payload.get("truncated") is None

    @pytest.mark.asyncio
    async def test_truncation_when_over_limit(self):
        from raven_agent.tools.base import Tool

        class FakeWebFetch(Tool):
            @property
            def name(self) -> str:
                return "web_fetch"

            @property
            def description(self) -> str:
                return "fake"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> str:
                return json.dumps({"text": "A" * 200})

        wrapped = DriftWebFetchTool(FakeWebFetch(), max_chars=100)
        result = await wrapped.execute()
        payload = json.loads(result)
        assert len(payload["text"]) == 100
        assert payload["truncated"] is True


class TestBuildDriftToolRegistry:
    """build_drift_tool_registry 组装测试。"""

    def test_minimal_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            registry = build_drift_tool_registry(
                ctx=ctx,
                store=store,
                state_dir=state_dir,
            )

            # 至少包含核心 drift 工具
            names = set(registry.list_names())
            assert "read_file" in names
            assert "write_text_file" in names
            assert "edit_file" in names
            assert "message_push" in names
            assert "finish_drift" in names

    def test_registry_with_shared_tools(self):
        from raven_agent.tools.registry import ToolRegistry
        from raven_agent.tools.base import Tool

        class FakeMemoryTool(Tool):
            @property
            def name(self) -> str:
                return "recall_memory"

            @property
            def description(self) -> str:
                return "fake"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> str:
                return "ok"

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "test-skill")
            ctx = DriftAgentTickContext()

            shared = ToolRegistry()
            shared.register(FakeMemoryTool(), risk="read-only")

            registry = build_drift_tool_registry(
                ctx=ctx,
                store=store,
                state_dir=state_dir,
                shared_tools=shared,
            )

            names = set(registry.list_names())
            assert "recall_memory" in names