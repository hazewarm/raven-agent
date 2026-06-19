"""Tests for BackgroundRuntime: queue, concurrency, cancel, callbacks."""

import asyncio
import types
from unittest.mock import AsyncMock

import pytest

from raven_agent.background.runtime import (
    BackgroundJob,
    BackgroundJobRunner,
    BackgroundRuntime,
)


# ── Helpers ──────────────────────────────────────────────────────


def make_job(
    task: str = "test task",
    channel: str = "cli",
    chat_id: str = "default",
    **kwargs,
) -> BackgroundJob:
    """创建测试用 BackgroundJob。

    输入:
        task: 任务描述。
        channel: 来源渠道。
        chat_id: 来源会话 ID。
        **kwargs: 其他 BackgroundJob 字段。

    输出:
        BackgroundJob 实例。
    """
    return BackgroundJob(
        task=task,
        channel=channel,
        chat_id=chat_id,
        **kwargs,
    )


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline() -> AsyncMock:
    """返回 mock pipeline（process_direct 返回固定结果）。"""
    m = AsyncMock()
    m.process_direct = AsyncMock(return_value="AI 完成了任务")
    return m


@pytest.fixture
def runner(mock_pipeline: AsyncMock) -> BackgroundJobRunner:
    """返回使用 mock pipeline 的 BackgroundJobRunner。"""
    return BackgroundJobRunner(lambda: mock_pipeline)


# ── Basic execution ──────────────────────────────────────────────


async def test_submit_and_execute(runner: BackgroundJobRunner) -> None:
    """提交后任务被放入队列并执行。"""
    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()

    job = make_job(task="查询北京天气")
    job_id = await rt.submit(job)

    # 等待任务完成
    await asyncio.sleep(0.2)

    completed = rt.get(job_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.result_summary == "AI 完成了任务"

    await rt.stop()


async def test_runner_sets_fields(runner: BackgroundJobRunner) -> None:
    """runner 正确设置 job 的 started_at / finished_at / exit_reason。"""
    job = make_job(task="test")
    result = await runner.run(job)

    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.exit_reason == "completed"
    assert result.status == "completed"


# ── Concurrency ──────────────────────────────────────────────────


async def test_max_concurrent_respected() -> None:
    """并发上限生效：提交超过 max_concurrent 的任务，只有最大并发数个同时运行。"""
    running_count = 0
    max_observed = 0

    async def slow_run(self, job: BackgroundJob, **kwargs: object) -> BackgroundJob:
        nonlocal running_count, max_observed
        job.status = "running"
        running_count += 1
        max_observed = max(max_observed, running_count)
        await asyncio.sleep(0.05)
        running_count -= 1
        job.status = "completed"
        job.exit_reason = "completed"
        return job

    # 注入慢执行 runner
    runner = BackgroundJobRunner(lambda: AsyncMock())
    runner.run = types.MethodType(slow_run, runner)  # type: ignore[method-assign]

    rt = BackgroundRuntime(runner=runner, max_concurrent=2)
    await rt.start()

    # 提交 5 个任务
    for i in range(5):
        await rt.submit(make_job(task=f"task-{i}"))

    # 等待所有任务完成
    await asyncio.sleep(0.5)

    all_jobs = rt.list_jobs()
    completed_count = sum(1 for j in all_jobs if j.status == "completed")
    assert completed_count >= 3  # 至少完成了部分（由于并发 2）
    assert max_observed <= 2  # 最大并发 <= 2

    await rt.stop()


# ── Cancel ───────────────────────────────────────────────────────


async def test_cancel_running_job(runner: BackgroundJobRunner) -> None:
    """取消正在运行的任务。"""
    # 让 runner 执行很慢
    async def very_slow(self, job: BackgroundJob, **kwargs: object) -> BackgroundJob:
        job.status = "running"
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.exit_reason = "cancelled"
            raise
        return job

    runner.run = types.MethodType(very_slow, runner)  # type: ignore[method-assign]

    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()

    job = make_job(task="very slow task")
    job_id = await rt.submit(job)

    # 等待任务开始执行
    await asyncio.sleep(0.1)

    result = await rt.cancel(job_id)
    assert result is True

    await asyncio.sleep(0.1)
    cancelled = rt.get(job_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"

    await rt.stop()


async def test_cancel_pending_job(runner: BackgroundJobRunner) -> None:
    """取消还在队列中的任务。"""
    # runner 永远不释放槽位
    async def block_forever(self, job: BackgroundJob, **kwargs: object) -> BackgroundJob:
        job.status = "running"
        await asyncio.Event().wait()
        return job

    runner.run = types.MethodType(block_forever, runner)  # type: ignore[method-assign]

    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()

    # 第一个任务占据槽位
    job1 = make_job(task="blocker")
    await rt.submit(job1)

    # 第二个任务在队列中
    job2 = make_job(task="pending task")
    job2_id = await rt.submit(job2)

    await asyncio.sleep(0.1)

    # 取消队列中的任务
    result = await rt.cancel(job2_id)
    assert result is True

    cancelled = rt.get(job2_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"

    await rt.stop()


async def test_cancel_completed_returns_false(
    runner: BackgroundJobRunner,
) -> None:
    """取消已完成的任务返回 False。"""
    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()

    job = make_job(task="quick task")
    job_id = await rt.submit(job)

    await asyncio.sleep(0.2)

    # 任务应已完成
    completed = rt.get(job_id)
    assert completed is not None
    assert completed.status == "completed"

    result = await rt.cancel(job_id)
    assert result is False

    await rt.stop()


# ── Callbacks ────────────────────────────────────────────────────


async def test_on_complete_callback_fires(
    runner: BackgroundJobRunner,
) -> None:
    """完成回调在任务完成后被执行。"""
    callback_results: list[str] = []

    def on_done(job: BackgroundJob) -> None:
        callback_results.append(job.job_id)

    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    rt.on_complete(on_done)
    await rt.start()

    job = make_job(task="callback test")
    job_id = await rt.submit(job)

    await asyncio.sleep(0.2)

    assert job_id in callback_results

    await rt.stop()


async def test_multiple_callbacks_all_fire(
    runner: BackgroundJobRunner,
) -> None:
    """多个完成回调全部被执行。"""
    fired: set[int] = set()

    def cb1(job: BackgroundJob) -> None:
        fired.add(1)

    async def cb2(job: BackgroundJob) -> None:
        fired.add(2)

    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    rt.on_complete(cb1)
    rt.on_complete(cb2)
    await rt.start()

    await rt.submit(make_job(task="multi callback"))
    await asyncio.sleep(0.2)

    assert fired == {1, 2}

    await rt.stop()


# ── Listing ──────────────────────────────────────────────────────


async def test_list_jobs_returns_all(runner: BackgroundJobRunner) -> None:
    """list_jobs 返回所有已提交的任务（含已完成的）。"""
    rt = BackgroundRuntime(runner=runner, max_concurrent=2)
    await rt.start()

    for i in range(3):
        await rt.submit(make_job(task=f"task-{i}"))

    await asyncio.sleep(0.3)

    all_jobs = rt.list_jobs()
    assert len(all_jobs) == 3
    for j in all_jobs:
        assert j.status == "completed"

    await rt.stop()


async def test_list_running_only_active(runner: BackgroundJobRunner) -> None:
    """list_running 只返回正在运行的任务。"""
    async def slow_run(self, job: BackgroundJob, **kwargs: object) -> BackgroundJob:
        job.status = "running"
        await asyncio.sleep(0.1)
        job.status = "completed"
        job.exit_reason = "completed"
        return job

    runner.run = types.MethodType(slow_run, runner)  # type: ignore[method-assign]

    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()

    await rt.submit(make_job(task="first"))
    await rt.submit(make_job(task="second"))

    await asyncio.sleep(0.05)

    running = rt.list_running()
    # 并发=1，只有一个在运行
    assert len(running) == 1
    assert running[0].status == "running"

    await asyncio.sleep(0.3)
    await rt.stop()


# ── Lifecycle ────────────────────────────────────────────────────


async def test_start_stop_idempotent(runner: BackgroundJobRunner) -> None:
    """重复 start / stop 不报错。"""
    rt = BackgroundRuntime(runner=runner, max_concurrent=1)
    await rt.start()
    await rt.start()  # 第二次 start 应无操作
    await rt.stop()
    await rt.stop()  # 第二次 stop 应无操作


async def test_stop_cancels_running_jobs(runner: BackgroundJobRunner) -> None:
    """stop 时取消所有运行中的任务。"""
    async def never_finish(self, job: BackgroundJob, **kwargs: object) -> BackgroundJob:
        job.status = "running"
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        return job

    runner.run = types.MethodType(never_finish, runner)  # type: ignore[method-assign]

    rt = BackgroundRuntime(runner=runner, max_concurrent=2)
    await rt.start()

    await rt.submit(make_job(task="never"))
    await asyncio.sleep(0.1)

    await rt.stop()
    # stop() 调了 task.cancel() 但没 await task，需等 CancelledError 传播
    await asyncio.sleep(0.05)

    # 任务应被标记为 cancelled
    jobs = rt.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status in ("cancelled", "error")