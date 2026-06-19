"""Tests for SchedulerService: tick, execution, misfire, rescheduling."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from raven_agent.scheduler import LatencyTracker, ScheduledJob, SchedulerService


# ── Helpers ──────────────────────────────────────────────────────


def make_job(
    trigger: str = "at",
    tier: str = "instant",
    fire_at: datetime | None = None,
    channel: str = "telegram",
    chat_id: str = "123",
    message: str | None = "hello",
    prompt: str | None = None,
    name: str | None = None,
    interval_seconds: int | None = None,
    cron_expr: str | None = None,
    timezone_: str = "UTC",
) -> ScheduledJob:
    """创建测试用 ScheduledJob。

    输入:
        trigger, tier, fire_at, channel, chat_id, message, prompt,
        name, interval_seconds, cron_expr, timezone_:
            对应 ScheduledJob 的各个字段。

    输出:
        ScheduledJob 实例。
    """
    if fire_at is None:
        fire_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    return ScheduledJob(
        trigger=trigger,
        tier=tier,
        fire_at=fire_at,
        channel=channel,
        chat_id=chat_id,
        message=message,
        prompt=prompt,
        name=name,
        interval_seconds=interval_seconds,
        cron_expr=cron_expr,
        timezone=timezone_,
    )


async def drain_tasks() -> None:
    """等待所有 pending asyncio tasks 完成。"""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        done, still_pending = await asyncio.wait(pending, timeout=1.0)
        if still_pending:
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
        if done:
            await asyncio.gather(*done, return_exceptions=True)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_push() -> AsyncMock:
    """返回 mock 的 MessagePushTool。"""
    m = AsyncMock()
    m.execute = AsyncMock(return_value="文本已发送")
    return m


@pytest.fixture
def mock_loop() -> AsyncMock:
    """返回 mock 的 agent loop。"""
    m = AsyncMock()
    m.process_direct = AsyncMock(return_value="AI response")
    return m


@pytest.fixture
def fixed_now() -> datetime:
    """固定的 '现在' 时间点。"""
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def service(
    tmp_path, mock_push, mock_loop, fixed_now
) -> SchedulerService:
    """创建配置了 mock 依赖的 SchedulerService。

    输入:
        tmp_path: pytest 临时目录。
        mock_push: mock MessagePushTool。
        mock_loop: mock agent loop。
        fixed_now: 固定时间点。

    输出:
        SchedulerService 实例。
    """
    return SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        tracker=LatencyTracker(default=25.0),
        _now_fn=lambda: fixed_now,
    )


# ── Execution: INSTANT ───────────────────────────────────────────


async def test_instant_calls_push_not_ai(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """INSTANT 任务只调用 push，不调用 AI。"""
    job = make_job(tier="instant", fire_at=fixed_now - timedelta(seconds=1))
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once()
    mock_loop.process_direct.assert_not_called()


async def test_instant_push_receives_correct_args(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """INSTANT 任务将 channel/chat_id/message 正确传给 push_tool。"""
    job = make_job(
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        channel="telegram",
        chat_id="999",
        message="喝水了",
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once_with(
        channel="telegram", chat_id="999", message="喝水了"
    )


# ── Execution: SOFT ──────────────────────────────────────────────


async def test_soft_calls_process_direct(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """SOFT 任务调用 agent loop 的 process_direct。"""
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),  # 晚于 pretrigger
        channel="telegram",
        chat_id="123",
        message=None,
        prompt="查询北京天气",
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_loop.process_direct.assert_called_once()
    call_kwargs = mock_loop.process_direct.call_args
    assert call_kwargs.kwargs["content"] == "查询北京天气"
    assert call_kwargs.kwargs["channel"] == "telegram"
    assert call_kwargs.kwargs["chat_id"] == "123"
    assert call_kwargs.kwargs["skip_post_memory"] is True
    assert call_kwargs.kwargs["disabled_tools"] == ["message_push"]


async def test_soft_sends_ai_response_via_push(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """SOFT 任务结束后通过 push_tool 发送 AI 生成的回复。"""
    mock_loop.process_direct = AsyncMock(return_value="北京今天晴，15°C")
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        prompt="查询北京天气",
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once_with(
        channel=job.channel, chat_id=job.chat_id, message="北京今天晴，15°C"
    )


async def test_soft_records_latency(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """SOFT 执行后 LatencyTracker 记录耗时。"""
    tracker = LatencyTracker(default=25.0)
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        tracker=tracker,
        _now_fn=lambda: fixed_now,
    )
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        prompt="天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert len(tracker._samples) == 1


# ── Timing: pre-trigger ──────────────────────────────────────────


async def test_soft_not_fired_before_pretrigger(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """SOFT 任务在预触发窗口之前不会被触发。"""
    # fire_at 在 60s 后；pretrigger = fire_at - 25s = now + 35s，未到
    job = make_job(
        tier="soft",
        fire_at=fixed_now + timedelta(seconds=60),
        prompt="天气",
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_loop.process_direct.assert_not_called()


async def test_instant_not_fired_before_fire_at(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """INSTANT 任务在 fire_at 之前不会被触发。"""
    job = make_job(
        tier="instant", fire_at=fixed_now + timedelta(seconds=10)
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    mock_push.execute.assert_not_called()


# ── One-shot jobs removed after firing ───────────────────────────


async def test_at_job_removed_after_fire(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """at 触发的一次性任务执行后被移除。"""
    job = make_job(
        trigger="at", tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    assert job.id not in service._jobs


async def test_after_job_removed_after_fire(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """after 触发的一次性任务执行后被移除。"""
    job = make_job(
        trigger="after", tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    assert job.id not in service._jobs


# ── Every: rescheduling ───────────────────────────────────────────


async def test_every_job_rescheduled_after_fire(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """every 任务执行后自动重调度到下一个触发时间。"""
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=3600,
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    # Job 仍应存在
    assert job.id in service._jobs
    # fire_at 应该推进到 now + ~1h
    new_fire_at = service._jobs[job.id].fire_at
    assert new_fire_at > fixed_now


async def test_every_run_count_increments(
    service, mock_push, mock_loop, fixed_now
) -> None:
    """every 任务每次执行后 run_count 递增。"""
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    service._jobs[job.id] = job

    await service._tick()
    await drain_tasks()

    assert service._jobs[job.id].run_count == 1


async def test_every_soft_cron_advances_past_nominal(
    tmp_path, mock_push, mock_loop
) -> None:
    """SOFT cron 任务不会在同一 nominal boundary 重复触发。"""
    now_ref = {"value": datetime(2025, 6, 1, 7, 59, 40, tzinfo=timezone.utc)}
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        tracker=LatencyTracker(default=25.0),
        _now_fn=lambda: now_ref["value"],
    )
    fire_at = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    job = make_job(
        trigger="every",
        tier="soft",
        fire_at=fire_at,
        cron_expr="0 8 * * *",
        timezone_="UTC",
        message=None,
        prompt="查询北京天气",
    )
    svc._jobs[job.id] = job

    # 第一次 tick：预触发触发
    await svc._tick()
    await drain_tasks()
    assert mock_loop.process_direct.call_count == 1
    assert svc._jobs[job.id].fire_at > fire_at

    # 第二次 tick：时间不变，不应再触发
    now_ref["value"] = datetime(2025, 6, 1, 7, 59, 46, tzinfo=timezone.utc)
    await svc._tick()
    await drain_tasks()
    assert mock_loop.process_direct.call_count == 1  # 仍为 1


# ── Misfire handling ─────────────────────────────────────────────


def test_misfire_within_grace_loaded(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """宽限期内（<5min）的 misfire 任务在启动时保留。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    job = make_job(
        trigger="at",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=100),  # 100s < 300s
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id in svc._jobs


def test_misfire_beyond_grace_discarded(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """超出宽限期（>5min）的 misfire 任务在启动时丢弃。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    job = make_job(
        trigger="at",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=400),  # 400s > 300s
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id not in svc._jobs


def test_every_misfire_advances_to_future(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """every 任务启动时 misfire 被推进到未来时间。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(hours=3),
        interval_seconds=3600,
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id in svc._jobs
    assert svc._jobs[job.id].fire_at > fixed_now


# ── Cancel ───────────────────────────────────────────────────────


def test_cancel_job_by_id(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """按 ID 取消任务成功。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    job = make_job()
    svc._jobs[job.id] = job

    result = svc.cancel_job(job.id)

    assert result is True
    assert job.id not in svc._jobs


def test_cancel_nonexistent_returns_false(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """取消不存在的任务返回 False。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    assert svc.cancel_job("nonexistent-id") is False


def test_cancel_by_name(
    tmp_path, mock_push, mock_loop, fixed_now
) -> None:
    """按名称取消任务成功。"""
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop_provider=lambda: mock_loop,
        _now_fn=lambda: fixed_now,
    )
    j1 = make_job(name="daily-weather")
    j2 = make_job(name="other")
    svc._jobs[j1.id] = j1
    svc._jobs[j2.id] = j2

    cancelled = svc.cancel_job_by_name("daily-weather")

    assert len(cancelled) == 1
    assert j1.id not in svc._jobs
    assert j2.id in svc._jobs


# ── JobStore ─────────────────────────────────────────────────────


def test_job_store_persist_and_load(tmp_path, fixed_now) -> None:
    """JobStore 能正确持久化和加载任务列表。"""
    from raven_agent.scheduler import JobStore

    store = JobStore(tmp_path / "jobs.json")
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now + timedelta(hours=1),
        interval_seconds=3600,
        name="test-job",
    )
    store.save({job.id: job})

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].name == "test-job"
    assert loaded[0].trigger == "every"
    assert loaded[0].interval_seconds == 3600


def test_job_store_load_missing_file(tmp_path) -> None:
    """JobStore 加载不存在的文件返回空列表。"""
    from raven_agent.scheduler import JobStore

    store = JobStore(tmp_path / "nonexistent.json")
    result = store.load()
    assert result == []