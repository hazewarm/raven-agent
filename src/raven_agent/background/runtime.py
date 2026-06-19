"""
后台任务运行时 —— 独立于 Agent turn 的异步任务执行器。

组件:
  BackgroundJob          — 后台任务数据模型
  BackgroundJobRunner    — 调用 agent pipeline 的后台 job runner
  BackgroundRuntime      — 队列 + 并发控制 + 生命周期管理 + 完成回调

与 SchedulerService 的关系:
  SchedulerService 决定"什么时候做"（时间触发）
  BackgroundRuntime 决定"怎么做"（队列、并发、完成回调）
  两者正交。BackgroundRuntime 不关心任务何时被提交，
  只关心提交后的执行管理。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 默认最大并发后台任务数
_DEFAULT_MAX_CONCURRENT = 3

# 默认队列最大容量（防止 LLM 陷入循环疯狂 spawn 导致 OOM）
_DEFAULT_MAX_QUEUE_SIZE = 100

# 自动生成的 label 最大字符数（从 task 截取）
_LABEL_MAX_CHARS = 60


# ── BackgroundJob ───────────────────────────────────────────────────


@dataclass
class BackgroundJob:
    """后台任务数据模型。

    字段:
        job_id: 任务唯一 ID（8 位 hex）。
        label: 人类可读的任务标签。
        task: 任务描述 / prompt。
        channel: 发起任务的来源渠道。
        chat_id: 发起任务的来源会话 ID。
        status: 任务状态 —— "pending" | "running" | "completed" | "error" | "cancelled"。
        created_at: 任务创建时间（UTC）。
        started_at: 任务开始执行时间；pending 时为 None。
        finished_at: 任务完成时间；未完成时为 None。
        result_summary: 结果摘要（由 runner 写入）。
        exit_reason: 退出原因 —— "completed" | "error" | "cancelled" | "max_iterations"。
        metadata: 附加元数据（预留扩展）。
    """

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""
    task: str = ""
    channel: str = ""
    chat_id: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_summary: str | None = None
    exit_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── BackgroundJobRunner ─────────────────────────────────────────────


class BackgroundJobRunner:
    """调用 agent pipeline 执行后台任务的 runner。

    将 job.task 作为 prompt 发送给 agent pipeline 执行完整的 ReAct 循环，
    返回 AgentRunResult。这个 runner 是 BackgroundRuntime 和 pipeline 之间的桥接。

    输入:
        pipeline_provider: 返回 PassiveTurnPipeline 的可调用对象。
            pipeline 需有 async process_direct(content, channel, chat_id,
            session_key, omit_user_turn, skip_post_memory, disabled_tools) 方法。
    """

    def __init__(
        self,
        pipeline_provider: Callable[[], Any],
    ) -> None:
        self._pipeline_provider = pipeline_provider

    async def run(
        self,
        job: BackgroundJob,
        *,
        on_exception: Callable[[Exception], None] | None = None,
    ) -> BackgroundJob:
        """执行后台任务并填充结果。

        输入:
            job: BackgroundJob 实例（status 应为 "pending"）。
            on_exception: 异常回调，用于记录日志。

        输出:
            更新了 status / result_summary / exit_reason 的 BackgroundJob。
        """
        job.started_at = datetime.now(timezone.utc)
        job.status = "running"

        try:
            pipeline = self._pipeline_provider()
            result_content = await pipeline.process_direct(
                content=job.task,
                channel=job.channel,
                chat_id=job.chat_id,
                session_key=f"bg:{job.job_id}",
                omit_user_turn=True,
                skip_post_memory=True,
                disabled_tools=["message_push"],
            )
            job.result_summary = result_content
            job.exit_reason = "completed"
            job.status = "completed"
        except asyncio.CancelledError:
            job.result_summary = "后台任务已按请求取消。"
            job.exit_reason = "cancelled"
            job.status = "cancelled"
            raise
        except Exception as exc:
            if on_exception is not None:
                on_exception(exc)
            job.result_summary = f"后台任务执行失败: {exc}"
            job.exit_reason = "error"
            job.status = "error"
        finally:
            job.finished_at = datetime.now(timezone.utc)

        elapsed = (
            (job.finished_at - job.started_at).total_seconds()
            if job.started_at and job.finished_at
            else 0
        )
        logger.info(
            "BackgroundJobRunner completed: job_id=%s  label=%r  "
            "status=%s  exit_reason=%s  elapsed=%.1fs",
            job.job_id, job.label, job.status, job.exit_reason, elapsed,
        )
        return job


# ── BackgroundRuntime ───────────────────────────────────────────────


class BackgroundRuntime:
    """后台任务运行时管理器。

    职责:
    - 维护有界 asyncio Queue 作为任务缓冲（防止 LLM 循环 spawn 导致 OOM）
    - 控制最大并发数（max_concurrent）
    - 跟踪所有任务的生命周期（pending → running → completed/error/cancelled）
    - 支持取消（cancel asyncio Task）
    - 完成时调用 on_complete 回调链（通知感兴趣的接收方）

    参数:
        max_concurrent: 最大并发后台任务数，默认 3。
        max_queue_size: 队列最大容量，默认 100。超出时 submit() 返回错误。
        runner: BackgroundJobRunner 实例，负责实际执行任务。
    """

    def __init__(
        self,
        runner: BackgroundJobRunner | None = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._runner = runner
        self._max_concurrent = max(1, int(max_concurrent))
        self._queue: asyncio.Queue[BackgroundJob] = asyncio.Queue(
            maxsize=max(1, int(max_queue_size))
        )
        self._jobs: dict[str, BackgroundJob] = {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._on_complete_callbacks: list[
            Callable[[BackgroundJob], None | Awaitable[None]]
        ] = []
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False

    # ── Public API ───────────────────────────────────────────────

    def set_runner(self, runner: Any) -> None:
        """设置或替换 job runner。

        输入:
            runner: 任何实现 run(job, *, on_exception=None) -> BackgroundJob
                的对象；不限于 BackgroundJobRunner。

        输出:
            None。
        """
        self._runner = runner

    def on_complete(
        self,
        callback: Callable[[BackgroundJob], None | Awaitable[None]],
    ) -> None:
        """注册完成回调（所有 job 完成时调用）。

        输入:
            callback: 接收 BackgroundJob 的可调用对象。

        输出:
            None。
        """
        self._on_complete_callbacks.append(callback)

    async def submit(self, job: BackgroundJob) -> str:
        """提交一个后台任务到队列。

        队列满时使用 put_nowait() 直接返回错误信息，而非 await put() 阻塞等待。
        await put() 会阻塞当前协程，而当前协程就是 process_bus_message_once
        ——同一个 while 循环。阻塞它意味着整个消息处理循环停止，
        所有用户的消息都无法处理。因此选择响应性优先于任务完整性：
        宁可丢弃该次 spawn 请求（LLM 收到错误后可告知用户或稍后重试），
        也不能让交互式 Agent 的消息循环卡死。

        输入:
            job: BackgroundJob 实例（status 自动设为 "pending"）。

        输出:
            成功时返回 job.job_id 字符串；队列满时返回错误信息。
        """
        if not job.label:
            job.label = (
                job.task[: _LABEL_MAX_CHARS].replace("\n", " ").strip()
                or job.job_id
            )
        job.status = "pending"
        self._jobs[job.job_id] = job

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            del self._jobs[job.job_id]
            logger.warning(
                "BackgroundRuntime queue full: depth=%d  max=%d",
                self._queue.qsize(), self._queue.maxsize,
            )
            return (
                f"错误：后台任务队列已满（当前上限 {self._queue.maxsize}），"
                "请等待部分任务完成后再提交。"
            )

        logger.info(
            "BackgroundRuntime submit: job_id=%s  label=%r  queue_depth=%d",
            job.job_id, job.label, self._queue.qsize(),
        )
        return job.job_id

    async def cancel(self, job_id: str) -> bool:
        """取消一个后台任务。

        输入:
            job_id: 任务 ID。

        输出:
            True 表示成功取消，False 表示任务不存在或已完成。
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "error", "cancelled"):
            return False

        task = self._running_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            job.status = "cancelled"
            job.exit_reason = "cancelled"
            job.finished_at = datetime.now(timezone.utc)
            logger.info("BackgroundRuntime cancel: job_id=%s", job_id)
            return True

        # 还在队列中（pending）
        job.status = "cancelled"
        job.exit_reason = "cancelled"
        job.finished_at = datetime.now(timezone.utc)
        logger.info("BackgroundRuntime cancel pending: job_id=%s", job_id)
        return True

    def get(self, job_id: str) -> BackgroundJob | None:
        """按 ID 查找任务。

        输入:
            job_id: 任务 ID。

        输出:
            BackgroundJob；不存在时返回 None。
        """
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BackgroundJob]:
        """列出所有任务。

        输出:
            BackgroundJob 列表副本。
        """
        return list(self._jobs.values())

    def list_running(self) -> list[BackgroundJob]:
        """列出正在运行中的任务。

        输出:
            status="running" 的 BackgroundJob 列表。
        """
        return [j for j in self._jobs.values() if j.status == "running"]

    def running_count(self) -> int:
        """返回当前运行中的任务数。

        输出:
            int。
        """
        return len(self._running_tasks)

    # ── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """启动后台 worker 循环。

        输入:
            无。

        输出:
            None。
        """
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(), name="bg_worker")
        logger.info("BackgroundRuntime started (max_concurrent=%d)", self._max_concurrent)

    async def stop(self) -> None:
        """停止后台 worker 循环并取消所有运行中任务。

        输入:
            无。

        输出:
            None。
        """
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # 取消所有运行中的任务
        for job_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
        logger.info("BackgroundRuntime stopped")

    # ── Internal ────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        """Worker 主循环：从队列取 job，控制并发，执行。"""
        while self._running:
            # 等待有空闲槽位
            while self.running_count() >= self._max_concurrent:
                await asyncio.sleep(0.1)
                if not self._running:
                    return

            # 取一个 job（带超时，避免死等阻塞停止）
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # 跳过已取消的 job
            if job.status == "cancelled":
                self._queue.task_done()
                continue

            # 创建并追踪 task
            task = asyncio.create_task(
                self._execute_job(job), name=f"bg:{job.job_id}"
            )
            self._running_tasks[job.job_id] = task
            task.add_done_callback(
                lambda _: self._running_tasks.pop(job.job_id, None)
            )

    async def _execute_job(self, job: BackgroundJob) -> None:
        """执行单个后台任务并触发完成回调。

        输入:
            job: BackgroundJob 实例。

        输出:
            None。
        """
        try:
            if self._runner is None:
                raise RuntimeError("BackgroundRuntime runner 未设置")

            result = await self._runner.run(
                job,
                on_exception=lambda exc: logger.exception(
                    "BackgroundRuntime job error: job_id=%s  err=%s",
                    job.job_id, exc,
                ),
            )
            self._jobs[job.job_id] = result
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.exit_reason = "cancelled"
            job.finished_at = datetime.now(timezone.utc)
            self._jobs[job.job_id] = job
        except Exception as exc:
            logger.exception(
                "BackgroundRuntime unexpected error: job_id=%s  err=%s",
                job.job_id, exc,
            )
            job.status = "error"
            job.exit_reason = "error"
            job.result_summary = f"执行时发生意外错误: {exc}"
            job.finished_at = datetime.now(timezone.utc)
            self._jobs[job.job_id] = job
        finally:
            # 触发完成回调
            final_job = self._jobs.get(job.job_id, job)
            for callback in self._on_complete_callbacks:
                try:
                    result = callback(final_job)
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.warning(
                        "BackgroundRuntime on_complete callback error: %s", exc
                    )