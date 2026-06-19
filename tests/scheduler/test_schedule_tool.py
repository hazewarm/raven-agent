"""Tests for ScheduleTool, ListSchedulesTool, CancelScheduleTool."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from raven_agent.scheduler import LatencyTracker, SchedulerService
from raven_agent.tools.schedule import (
    CancelScheduleTool,
    ListSchedulesTool,
    ScheduleTool,
)

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_svc(tmp_path, mock_push, mock_loop) -> SchedulerService:
    """创建测试用 SchedulerService。

    输入:
        tmp_path: pytest 临时目录路径。
        mock_push: mock MessagePushTool。
        mock_loop: mock agent loop。

    输出:
        SchedulerService 实例，时间固定为 _NOW。
    """
    return SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        tracker=LatencyTracker(default=25.0),
        _now_fn=lambda: _NOW,
    )


def make_job(**kwargs) -> object:
    """创建测试用 ScheduledJob 的便捷工厂。"""
    from raven_agent.scheduler import ScheduledJob
    defaults = {
        "trigger": "at",
        "tier": "instant",
        "fire_at": _NOW + timedelta(minutes=5),
        "channel": "telegram",
        "chat_id": "123",
        "message": "hello",
    }
    defaults.update(kwargs)
    return ScheduledJob(**defaults)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_push() -> AsyncMock:
    m = AsyncMock()
    m.execute = AsyncMock(return_value="文本已发送")
    return m


@pytest.fixture
def mock_loop() -> AsyncMock:
    m = AsyncMock()
    m.process_direct = AsyncMock(return_value="AI response")
    return m


# ── ScheduleTool: validation ──────────────────────────────────────


async def test_invalid_tier_returns_error(tmp_path, mock_push, mock_loop) -> None:
    """无效的 tier 返回中文错误信息。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="precise", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "tier" in result


async def test_invalid_trigger_returns_error(
    tmp_path, mock_push, mock_loop
) -> None:
    """无效的 trigger 返回中文错误信息。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant",
        trigger="sometime",
        when="5m",
        channel="tg",
        chat_id="1",
        message="hi",
    )
    assert "错误" in result
    assert "trigger" in result


async def test_instant_without_message_returns_error(
    tmp_path, mock_push, mock_loop
) -> None:
    """instant 模式缺少 message 返回错误。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "message" in result


async def test_soft_without_prompt_returns_error(
    tmp_path, mock_push, mock_loop
) -> None:
    """soft 模式缺少 prompt 返回错误。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="soft", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "prompt" in result


async def test_invalid_when_returns_error(
    tmp_path, mock_push, mock_loop
) -> None:
    """无效的 when 参数返回错误。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant",
        trigger="after",
        when="blah",
        channel="tg",
        chat_id="1",
        message="hi",
    )
    assert "错误" in result


# ── ScheduleTool: successful registration ────────────────────────


async def test_instant_after_registers_job(
    tmp_path, mock_push, mock_loop
) -> None:
    """instant + after 组合成功注册任务。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    result = await tool.execute(
        tier="instant",
        trigger="after",
        when="5m",
        channel="telegram",
        chat_id="123",
        message="喝水了",
        request_time=_NOW.isoformat(),
    )
    assert "错误" not in result
    assert len(svc._jobs) == 1
    job = list(svc._jobs.values())[0]
    assert job.tier == "instant"
    assert job.message == "喝水了"


async def test_after_request_time_used_for_fire_at(
    tmp_path, mock_push, mock_loop
) -> None:
    """after 模式的 fire_at 从 request_time 算起而非 now。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="after",
        when="30s",
        channel="tg",
        chat_id="1",
        message="hi",
        request_time=_NOW.isoformat(),
    )
    job = list(svc._jobs.values())[0]
    expected_fire_at = _NOW + timedelta(seconds=30)
    assert abs((job.fire_at - expected_fire_at).total_seconds()) < 1


async def test_soft_at_registers_job(
    tmp_path, mock_push, mock_loop
) -> None:
    """soft + at 组合成功注册。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    result = await tool.execute(
        tier="soft",
        trigger="at",
        when="2025-06-01T14:00:00",
        channel="telegram",
        chat_id="456",
        prompt="查询北京天气",
    )
    assert "错误" not in result
    job = list(svc._jobs.values())[0]
    assert job.tier == "soft"
    assert job.prompt == "查询北京天气"
    assert job.fire_at.hour == 14


async def test_every_interval_stores_interval_seconds(
    tmp_path, mock_push, mock_loop
) -> None:
    """every + interval 成功存储 interval_seconds。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="every",
        when="1h",
        channel="tg",
        chat_id="1",
        message="提醒",
    )
    job = list(svc._jobs.values())[0]
    assert job.interval_seconds == 3600
    assert job.cron_expr is None


async def test_every_cron_stores_cron_expr(
    tmp_path, mock_push, mock_loop
) -> None:
    """every + cron 成功存储 cron_expr。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="soft",
        trigger="every",
        when="0 9 * * *",
        channel="tg",
        chat_id="1",
        prompt="天气",
    )
    job = list(svc._jobs.values())[0]
    assert job.cron_expr == "0 9 * * *"
    assert job.interval_seconds is None


async def test_named_job(tmp_path, mock_push, mock_loop) -> None:
    """带 name 的任务成功注册。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="after",
        when="5m",
        channel="tg",
        chat_id="1",
        message="hi",
        name="my-reminder",
        request_time=_NOW.isoformat(),
    )
    job = list(svc._jobs.values())[0]
    assert job.name == "my-reminder"


# ── ListSchedulesTool ────────────────────────────────────────────


async def test_list_empty(tmp_path, mock_push, mock_loop) -> None:
    """空列表返回提示信息。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ListSchedulesTool(svc)
    result = await tool.execute()
    assert "没有" in result


async def test_list_shows_jobs(tmp_path, mock_push, mock_loop) -> None:
    """有任务时列出详细信息。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job(name="喝水提醒", tier="instant")
    svc._jobs[job.id] = job

    tool = ListSchedulesTool(svc)
    result = await tool.execute()
    assert "喝水提醒" in result


# ── CancelScheduleTool ───────────────────────────────────────────


async def test_cancel_by_id(tmp_path, mock_push, mock_loop) -> None:
    """按 ID 取消成功。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job()
    svc._jobs[job.id] = job

    tool = CancelScheduleTool(svc)
    result = await tool.execute(id=job.id)
    assert "已取消" in result
    assert job.id not in svc._jobs


async def test_cancel_by_name(tmp_path, mock_push, mock_loop) -> None:
    """按名称取消成功。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job(name="daily-report")
    svc._jobs[job.id] = job

    tool = CancelScheduleTool(svc)
    result = await tool.execute(name="daily-report")
    assert "已取消" in result
    assert job.id not in svc._jobs


async def test_cancel_nonexistent_id(
    tmp_path, mock_push, mock_loop
) -> None:
    """取消不存在的 ID 返回未找到。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = CancelScheduleTool(svc)
    result = await tool.execute(id="no-such-id")
    assert "未找到" in result


async def test_cancel_no_args_returns_error(
    tmp_path, mock_push, mock_loop
) -> None:
    """不提供 id 和 name 返回错误。"""
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = CancelScheduleTool(svc)
    result = await tool.execute()
    assert "错误" in result


# ── Time Parsing ─────────────────────────────────────────────────


def test_parse_duration_simple() -> None:
    """parse_duration 基础用例。"""
    from raven_agent.scheduler import parse_duration
    assert parse_duration("30s").total_seconds() == 30
    assert parse_duration("5m").total_seconds() == 300
    assert parse_duration("2h").total_seconds() == 7200
    assert parse_duration("1d2h").total_seconds() == 93600


def test_parse_duration_invalid() -> None:
    """parse_duration 无效输入抛 ValueError。"""
    from raven_agent.scheduler import parse_duration
    import pytest
    with pytest.raises(ValueError):
        parse_duration("blah")


def test_is_cron_expr() -> None:
    """is_cron_expr 正确判断 cron 格式。"""
    from raven_agent.scheduler import is_cron_expr
    assert is_cron_expr("0 9 * * *") is True
    assert is_cron_expr("*/5 * * * *") is True
    assert is_cron_expr("1h") is False
    assert is_cron_expr("30s") is False


def test_compute_fire_at_after_with_request_time() -> None:
    """after 模式从 request_time 算延迟。"""
    from raven_agent.scheduler import compute_fire_at
    rt = "2025-06-01T12:00:00+00:00"
    result = compute_fire_at("after", "30s", "UTC", request_time=rt)
    expected = datetime(2025, 6, 1, 12, 0, 30, tzinfo=timezone.utc)
    assert abs((result - expected).total_seconds()) < 1


def test_compute_fire_at_at_hhmm() -> None:
    """at 模式解析 HH:MM（当天或明天）。"""
    from raven_agent.scheduler import compute_fire_at

    fixed_now = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    _now_fn = lambda: fixed_now

    # 下午 2 点，还未到——在今天
    result = compute_fire_at("at", "14:00", "UTC", _now_fn=_now_fn)
    assert result.hour == 14
    assert result.day == 1

    # 早上 8 点，已过——在明天
    result = compute_fire_at("at", "08:00", "UTC", _now_fn=_now_fn)
    assert result.hour == 8
    assert result.day == 2