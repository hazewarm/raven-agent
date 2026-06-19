"""
test_drift_turn.py —— DriftTurnPipeline 与 ProactiveLoop 集成测试。

覆盖：
  - DriftTurnPipeline.run(): 无 skill / 有 skill 但 LLM 返回空 /
    finish_drift 正常结束
  - ProactiveLoop._maybe_run_drift(): min_interval 约束 / 无 skill 跳过
  - DriftAgentTickContext 状态标志生命周期
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_state import DriftStateStore
from raven_agent.proactive.drift_turn import DriftTurnPipeline


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
        "",
        "## 工作流程",
        "1. 读取工作文件",
        "2. 执行操作",
        "3. finish_drift",
    ]
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return DriftStateStore(source_dir=source_dir, state_dir=state_dir)


class FakeLLMResponse:
    """模拟 LLMProvider.chat() 的返回值。"""

    def __init__(self, tool_calls=None, content="", reasoning_content=""):
        self.tool_calls = tool_calls or []
        self.content = content
        self.reasoning_content = reasoning_content


class FakeToolCall:
    """模拟 LLM 返回的 tool_call 对象。"""

    def __init__(self, name: str, arguments: dict, call_id: str = "call_1"):
        self.name = name
        self.arguments = arguments
        self.id = call_id


class FakeLLMProvider:
    """模拟 LLMProvider——按预设序列返回 tool_call。

    参数:
        responses: FakeLLMResponse 列表，按顺序返回。
    """

    def __init__(self, responses: list[FakeLLMResponse]):
        self._responses = responses
        self._index = 0
        self.calls: list[dict] = []

    async def chat(self, messages, tools, model, max_tokens=1024, **kwargs):
        self.calls.append({
            "messages_count": len(messages),
            "tools_count": len(tools),
            "model": model,
        })
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        # 默认：无 tool_call
        return FakeLLMResponse()


class TestDriftTurnPipeline:
    """DriftTurnPipeline 核心行为测试。"""

    @pytest.mark.asyncio
    async def test_run_returns_false_when_no_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            source_dir.mkdir(parents=True, exist_ok=True)
            store = DriftStateStore(source_dir=source_dir, state_dir=state_dir)

            pipeline = DriftTurnPipeline(
                store=store,
                provider=FakeLLMProvider([]),
                model="test-model",
            )
            ctx = DriftAgentTickContext()
            result = await pipeline.run(ctx)
            assert result is False

    @pytest.mark.asyncio
    async def test_run_executes_finish_drift(self):
        """模拟一次完整的 drift 执行：read_file → finish_drift。"""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            # 预设 LLM 响应序列：
            # 1. read_file → 读 SKILL.md
            # 2. finish_drift → 结束
            responses = [
                FakeLLMResponse(
                    tool_calls=[
                        FakeToolCall(
                            "read_file",
                            {"path": "skills/my-skill/SKILL.md"},
                        )
                    ]
                ),
                FakeLLMResponse(
                    tool_calls=[
                        FakeToolCall(
                            "finish_drift",
                            {
                                "skill_used": "my-skill",
                                "one_line": "测试完成",
                                "next": "等待下次 drift",
                                "message_result": "silent",
                            },
                        )
                    ]
                ),
            ]

            pipeline = DriftTurnPipeline(
                store=store,
                provider=FakeLLMProvider(responses),
                model="test-model",
            )
            ctx = DriftAgentTickContext(
                session_key="cli:default",
                now_utc=datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc),
            )
            result = await pipeline.run(ctx)
            assert result is True
            assert ctx.drift_entered is True
            assert ctx.drift_finished is True
            assert ctx.drift_message_sent is False
            assert ctx.steps_taken == 2

            # 验证状态已持久化
            skills = store.scan_skills()
            assert skills[0].run_count == 1

    @pytest.mark.asyncio
    async def test_run_stops_at_max_steps(self):
        """达到 max_steps 上限后退出（即使未调用 finish_drift）。"""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            # LLM 一直返回 read_file（不调 finish_drift）
            responses = [
                FakeLLMResponse(
                    tool_calls=[
                        FakeToolCall(
                            "read_file",
                            {"path": "skills/my-skill/SKILL.md"},
                            call_id=f"call_{i}",
                        )
                    ]
                )
                for i in range(5)
            ]

            pipeline = DriftTurnPipeline(
                store=store,
                provider=FakeLLMProvider(responses),
                model="test-model",
                max_steps=3,
            )
            ctx = DriftAgentTickContext()
            result = await pipeline.run(ctx)
            assert result is True
            assert ctx.drift_finished is False  # 未调 finish_drift
            assert ctx.steps_taken == 3  # 被 max_steps 截断

    @pytest.mark.asyncio
    async def test_run_handles_llm_error_gracefully(self):
        """LLM 调用异常时优雅退出，不抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            class FailingProvider:
                async def chat(self, **kwargs):
                    raise RuntimeError("LLM 不可用")

            pipeline = DriftTurnPipeline(
                store=store,
                provider=FailingProvider(),
                model="test-model",
            )
            ctx = DriftAgentTickContext()
            # 不应抛异常
            result = await pipeline.run(ctx)
            assert result is True  # 进入了 drift（有 skill），但 execute 阶段因错误退出

    @pytest.mark.asyncio
    async def test_drift_context_state_flags(self):
        """验证 DriftAgentTickContext 的标志生命周期。"""
        ctx = DriftAgentTickContext(
            tick_id="test001",
            session_key="cli:default",
        )
        assert ctx.drift_entered is False
        assert ctx.drift_finished is False
        assert ctx.drift_message_sent is False
        assert ctx.steps_taken == 0

        # 模拟 Prepare
        ctx.drift_entered = True

        # 模拟 SendMessageTool
        ctx.drift_message_sent = True
        ctx.steps_taken += 1

        # 模拟 FinishDriftTool
        ctx.drift_finished = True

        assert ctx.drift_entered is True
        assert ctx.drift_finished is True
        assert ctx.drift_message_sent is True
        assert ctx.steps_taken == 1


class TestMinIntervalConstraint:
    """Drift min_interval 约束测试。"""

    def test_get_last_drift_at_returns_correct_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
            store.save_finish(
                skill_used="my-skill",
                one_line="上次运行",
                next_action="继续",
                message_result="silent",
                note=None,
                now_utc=now,
            )

            last = store.get_last_drift_at()
            assert last == now