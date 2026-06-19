from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from raven_agent.background.delegation import SpawnDecision
from raven_agent.background.runtime import BackgroundJob, BackgroundJobRunner, BackgroundRuntime
from raven_agent.background.subagent_profiles import (
    PROFILE_RESEARCH,
    SubagentRuntime,
    build_spawn_spec,
    build_spawn_subagent_prompt,
)
from raven_agent.events import InboundMessage, SpawnCompletionEvent
from raven_agent.message_bus import MessageBus

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 12_000
_SYNC_RESULT_MAX_CHARS = 100_000
_SPAWN_MAX_ITERATIONS = 50
_SYNC_MAX_ITERATIONS = 10


@dataclass(frozen=True)
class RunningSubagentJob:
    """运行中的本地 SubAgent 后台任务快照。

    字段:
        job_id: BackgroundJob ID。
        label: 任务短标签。
        task: 子任务完整描述。
        profile: 工具 profile。
        origin_channel: 原始 channel。
        origin_chat_id: 原始 chat_id。
        task_dir: 当前 job 的任务目录。
        retry_count: 已重试次数。
        status: 当前状态。
    """

    job_id: str
    label: str
    task: str
    profile: str
    origin_channel: str
    origin_chat_id: str
    task_dir: str
    retry_count: int = 0
    status: str = "running"


class SubagentManager:
    """管理本地 SubAgent 任务的生命周期。

    输入:
        runtime: SubagentRuntime，提供 provider/model/web search/hook。
        workspace: 工作区根目录。
        task_root: subagent-runs 根目录。
        bus: MessageBus，用于后台完成回流。
        background_runtime: BackgroundRuntime，用于后台队列和取消。
    """

    def __init__(
        self,
        *,
        runtime: SubagentRuntime,
        workspace: Path,
        task_root: Path,
        bus: MessageBus,
        background_runtime: BackgroundRuntime,
    ) -> None:
        self._runtime = runtime
        self._workspace = Path(workspace)
        self._task_root = Path(task_root)
        self._bus = bus
        self._background_runtime = background_runtime
        self._running_jobs: dict[str, RunningSubagentJob] = {}

    def set_tool_hooks(self, hooks: list[Any]) -> None:
        """更新传给后续 SubAgent 的工具 hook 列表。

        输入:
            hooks: ToolHook 列表。

        输出:
            None。
        """
        object.__setattr__(self._runtime, "tool_hooks", list(hooks))

    def get_running_count(self) -> int:
        """返回当前运行中或排队中的 spawn 任务数。

        输出:
            int。
        """
        return sum(
            1
            for job in self._background_runtime.list_jobs()
            if job.metadata.get("job_kind") == "conversation_spawn"
            and job.status in {"pending", "running"}
        )

    def list_running_jobs(self) -> list[dict[str, object]]:
        """列出运行中/排队中的 spawn 任务。

        输出:
            字典列表，可直接 JSON 序列化。
        """
        rows: list[dict[str, object]] = []
        for job in self._background_runtime.list_jobs():
            if job.metadata.get("job_kind") != "conversation_spawn":
                continue
            if job.status not in {"pending", "running"}:
                continue
            snapshot = self._running_jobs.get(job.job_id)
            if snapshot is not None:
                rows.append(asdict(snapshot) | {"status": job.status})
            else:
                rows.append(
                    {
                        "job_id": job.job_id,
                        "label": job.label,
                        "task": job.task,
                        "profile": job.metadata.get("profile", ""),
                        "origin_channel": job.channel,
                        "origin_chat_id": job.chat_id,
                        "task_dir": job.metadata.get("task_dir", ""),
                        "retry_count": job.metadata.get("retry_count", 0),
                        "status": job.status,
                    }
                )
        return rows

    async def cancel(self, job_id: str) -> bool:
        """取消一个后台 spawn 任务。

        输入:
            job_id: BackgroundJob ID。

        输出:
            True 表示取消请求成功；False 表示不存在或已完成。
        """
        ok = await self._background_runtime.cancel(job_id)
        if ok:
            logger.info("[spawn] cancel requested job_id=%s", job_id)
        return ok

    async def spawn_sync(
        self,
        *,
        task: str,
        label: str | None,
        profile: str = PROFILE_RESEARCH,
    ) -> str:
        """同步执行 SubAgent，阻塞当前工具调用直到完成。

        输入:
            task: 子任务完整描述。
            label: 任务短标签。
            profile: 工具 profile。

        输出:
            给主 Agent 的工具结果字符串。
        """
        job = BackgroundJob(
            label=(label or task[:30] or "subagent").strip(),
            task=task,
            channel="sync",
            chat_id="sync",
            metadata={"profile": profile},
        )
        task_dir = self._job_task_dir(job.job_id)
        subagent = self._build_subagent(
            task_dir=task_dir,
            profile=profile,
            max_iterations=_SYNC_MAX_ITERATIONS,
        )
        try:
            result = await subagent.run(task)
            exit_reason = subagent.last_exit_reason or "completed"
        except Exception as exc:
            logger.exception("[spawn_sync] subagent failed job_id=%s", job.job_id)
            result = f"执行出错：{exc}"
            exit_reason = "error"

        visible = _truncate(result, _SYNC_RESULT_MAX_CHARS)
        return f"[子任务「{job.label}」结果]\n退出原因: {exit_reason}\n任务目录: {task_dir}\n\n{visible}"

    async def spawn(
        self,
        *,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
        decision: SpawnDecision | None = None,
        profile: str = PROFILE_RESEARCH,
        retry_count: int = 0,
    ) -> str:
        """创建后台 SubAgent 任务，并立即返回确认文本。

        输入:
            task: 子任务完整描述。
            label: 任务短标签。
            origin_channel: 原始 channel。
            origin_chat_id: 原始 chat_id。
            decision: spawn 决策信息。
            profile: 工具 profile。
            retry_count: 当前重试次数。

        输出:
            给主 Agent 的确认文本。
        """
        display_label = (label or task[:30] or "subagent").strip()
        job = BackgroundJob(
            label=display_label,
            task=task,
            channel=origin_channel,
            chat_id=origin_chat_id,
            metadata={
                "job_kind": "conversation_spawn",
                "profile": profile,
                "retry_count": max(0, int(retry_count)),
                "decision": _decision_payload(decision),
            },
        )
        task_dir = self._job_task_dir(job.job_id)
        job.metadata["task_dir"] = str(task_dir)

        self._running_jobs[job.job_id] = RunningSubagentJob(
            job_id=job.job_id,
            label=display_label,
            task=task,
            profile=profile,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            task_dir=str(task_dir),
            retry_count=max(0, int(retry_count)),
            status="pending",
        )
        result = await self._background_runtime.submit(job)
        if result != job.job_id:
            self._running_jobs.pop(job.job_id, None)
            return result

        return (
            f"已创建后台任务「{display_label}」（job_id={job.job_id}）。"
            "不要等待其完成；请直接向用户说明你已开始处理，完成后会继续回复。"
        )

    async def run_background_job(self, job: BackgroundJob) -> BackgroundJob:
        """供 SpawnAwareBackgroundJobRunner 调用，执行一个后台 spawn job。

        输入:
            job: BackgroundJob，metadata.job_kind 必须是 conversation_spawn。

        输出:
            更新 status/result_summary/exit_reason 的 BackgroundJob。
        """
        task_dir = Path(str(job.metadata.get("task_dir") or self._job_task_dir(job.job_id)))
        profile = str(job.metadata.get("profile") or PROFILE_RESEARCH)
        job.status = "running"
        snapshot = self._running_jobs.get(job.job_id)
        if snapshot is not None:
            self._running_jobs[job.job_id] = RunningSubagentJob(
                **{**asdict(snapshot), "status": "running"}
            )

        subagent = self._build_subagent(
            task_dir=task_dir,
            profile=profile,
            max_iterations=_SPAWN_MAX_ITERATIONS,
        )
        try:
            result = await subagent.run(job.task)
            exit_reason = subagent.last_exit_reason or "completed"
            job.exit_reason = exit_reason
            job.status = _status_from_exit_reason(exit_reason)
            job.result_summary = result
        except Exception as exc:
            logger.exception("[spawn] subagent failed job_id=%s", job.job_id)
            job.exit_reason = "error"
            job.status = "error"
            job.result_summary = f"后台任务执行失败：{exc}"
        finally:
            self._running_jobs.pop(job.job_id, None)
        return job

    async def announce_completion(self, job: BackgroundJob) -> None:
        """把后台 spawn 完成结果注入 MessageBus。

        输入:
            job: 已完成的 BackgroundJob。

        输出:
            None。
        """
        if job.metadata.get("job_kind") != "conversation_spawn":
            return
        result_text = job.result_summary or (
            "后台任务已按请求取消。" if job.status == "cancelled" else ""
        )
        event = SpawnCompletionEvent(
            job_id=job.job_id,
            label=job.label,
            task=job.task,
            status=_semantic_status(job.status, job.exit_reason),
            exit_reason=job.exit_reason or job.status,
            result=_truncate(result_text, _RESULT_MAX_CHARS),
            retry_count=int(job.metadata.get("retry_count", 0) or 0),
            profile=str(job.metadata.get("profile", "")),
        )
        content = _build_completion_prompt(event)
        marker = _build_completion_marker(event)
        await self._bus.publish_inbound(
            InboundMessage(
                channel=job.channel,
                sender="spawn",
                chat_id=job.chat_id,
                content=content,
                metadata={
                    "system_injected": True,
                    "spawn_completion": True,
                    "spawn_event": asdict(event),
                    "persist_user_content": marker,
                    "skip_post_memory": True,
                    "disabled_tools": ["message_push"],
                },
            )
        )
        logger.info(
            "[spawn] completed job_id=%s status=%s exit_reason=%s route=%s:%s",
            job.job_id,
            event.status,
            event.exit_reason,
            job.channel,
            job.chat_id,
        )

    def _job_task_dir(self, job_id: str) -> Path:
        """返回并创建某个 job 的任务目录。

        输入:
            job_id: BackgroundJob ID。

        输出:
            任务目录 Path。
        """
        task_dir = self._task_root / job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _build_subagent(
        self,
        *,
        task_dir: Path,
        profile: str = PROFILE_RESEARCH,
        max_iterations: int = _SPAWN_MAX_ITERATIONS,
    ):
        """构建一个 SubAgent 实例。

        输入:
            task_dir: 当前任务目录。
            profile: 工具 profile。
            max_iterations: 最大 ReAct 轮数。

        输出:
            SubAgent 实例。
        """
        system_prompt = build_spawn_subagent_prompt(self._workspace, task_dir, profile)
        spec = build_spawn_spec(
            workspace=self._workspace,
            task_dir=task_dir,
            runtime=self._runtime,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            profile=profile,
        )
        return spec.build(self._runtime)


class SubagentJobRunner:
    """把 SubagentManager 适配成 BackgroundRuntime runner。

    输入:
        manager: SubagentManager。
    """

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    async def run(self, job: BackgroundJob, **_: Any) -> BackgroundJob:
        """执行后台 spawn job。

        输入:
            job: BackgroundJob。

        输出:
            更新后的 BackgroundJob。
        """
        return await self._manager.run_background_job(job)


class SpawnAwareBackgroundJobRunner:
    """根据 BackgroundJob.metadata 分流默认后台任务与 spawn 后台任务。

    输入:
        default_runner: 第 28 章已有的 BackgroundJobRunner。
        subagent_runner: SubagentJobRunner。
    """

    def __init__(
        self,
        *,
        default_runner: BackgroundJobRunner,
        subagent_runner: SubagentJobRunner,
    ) -> None:
        self._default_runner = default_runner
        self._subagent_runner = subagent_runner

    async def run(
        self,
        job: BackgroundJob,
        *,
        on_exception: Any = None,
    ) -> BackgroundJob:
        """执行 BackgroundJob。

        输入:
            job: BackgroundJob。
            on_exception: BackgroundRuntime 传入的异常日志回调。

        输出:
            BackgroundJob。spawn 任务走 SubagentJobRunner，其他任务走默认 runner。
        """
        if job.metadata.get("job_kind") == "conversation_spawn":
            return await self._subagent_runner.run(job)
        return await self._default_runner.run(job, on_exception=on_exception)


def _status_from_exit_reason(exit_reason: str) -> str:
    """把 SubAgent exit_reason 映射为 BackgroundJob.status。

    输入:
        exit_reason: 子 Agent 退出原因。

    输出:
        completed / incomplete / error / cancelled。
    """
    if exit_reason == "completed":
        return "completed"
    if exit_reason == "cancelled":
        return "cancelled"
    if exit_reason == "error":
        return "error"
    return "incomplete"


def _semantic_status(job_status: str, exit_reason: str | None) -> str:
    """把 BackgroundJob.status 映射为用户可理解的完成状态。

    输入:
        job_status: BackgroundJob.status。
        exit_reason: BackgroundJob.exit_reason。

    输出:
        completed / incomplete / error / cancelled。
    """
    if job_status == "cancelled" or exit_reason == "cancelled":
        return "cancelled"
    if job_status == "completed" and exit_reason == "completed":
        return "completed"
    if job_status == "error" or exit_reason == "error":
        return "error"
    return "incomplete"


def _build_completion_marker(event: SpawnCompletionEvent) -> str:
    """构建写入 session 历史的短 marker。

    这条 marker 是 session 中"这次 spawn 曾经发生过"的唯一记录。
    用户翻看对话历史时，通过它能回忆起当时委托了什么任务、结果如何。
    它应包含足够的上下文线索，但不应携带 raw result 正文。

    输入:
        event: SpawnCompletionEvent，包含 task / label / status / exit_reason / profile。

    输出:
        单行文本，约 120 字以内。
    """
    task_preview = (event.task or "").strip().replace("\n", " ")[:100]
    if len(event.task or "") > 100:
        task_preview += "..."
    parts = [f"[后台任务完成] {event.label}"]
    if task_preview:
        parts.append(f"— {task_preview}")
    parts.append(f"({event.status})")
    if event.exit_reason and event.exit_reason != event.status:
        parts.append(f"[{event.exit_reason}]")
    if event.profile:
        parts.append(f"profile={event.profile}")
    return " ".join(parts)


def _build_completion_prompt(event: SpawnCompletionEvent) -> str:
    """构建给主模型看的完整后台回传 prompt。

    输入:
        event: SpawnCompletionEvent。

    输出:
        包含任务、退出原因、结果和处理指引的文本。
    """
    if event.retry_count >= 1:
        guidance = (
            "⚠️ 已重试一次，不再重试。请直接将已获得的结果汇报给用户，"
            "说明已完成的部分和未完成的部分。"
        )
    else:
        guidance = (
            "处理指引：如果结果足够完整，直接向用户汇报；"
            "如果结果不完整但核心信息明显不足，可以调用 spawn 重试一次，"
            "task 中说明上次卡在哪、这次从哪继续，并设置 run_in_background=true。"
        )
    return (
        "[后台任务回传]\n"
        f"任务标签: {event.label}\n"
        f"原始任务: {event.task or '（未提供）'}\n"
        f"状态: {event.status}\n"
        f"退出原因: {event.exit_reason or '未知'}\n"
        f"profile: {event.profile or 'unknown'}\n"
        f"执行结果:\n{event.result or '（无结果）'}\n\n"
        f"{guidance}\n\n"
        "禁止在回复中提及 subagent、spawn、job_id、内部事件等内部概念。"
        "必要时可读取结果里提到的文件来补充说明。"
    )


def _truncate(text: str, limit: int) -> str:
    """截断文本并保留原始长度提示。

    输入:
        text: 原始文本。
        limit: 最大字符数。

    输出:
        截断后的文本。
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[结果已截断，原始长度 {len(text)} 字符]"


def _decision_payload(decision: SpawnDecision | None) -> dict[str, object] | None:
    """把 SpawnDecision 转换为可 JSON 序列化的字典。

    输入:
        decision: SpawnDecision 或 None。

    输出:
        dict 或 None。
    """
    if decision is None:
        return None
    return {
        "should_spawn": decision.should_spawn,
        "label": decision.label,
        "block_reason": decision.block_reason,
        "meta": {
            "source": decision.meta.source,
            "confidence": decision.meta.confidence,
            "reason_code": decision.meta.reason_code,
        },
    }