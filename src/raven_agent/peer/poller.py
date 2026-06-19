"""
peer/poller.py —— Peer Agent 任务轮询器。

后台 asyncio 任务，每 10 秒轮询所有 pending A2A 任务的状态。
完成时通过 MessageBus 注入系统消息，触发主 Agent 的新一轮处理。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from raven_agent.events import InboundMessage

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 10  # 基础轮询间隔（用于判断是否需要检查某任务）
_TASK_TIMEOUT_S = 3600  # 60 分钟硬超时
_FAST_RETRY_MAX = 3     # 网络瞬断时快速重试次数
_FAST_RETRY_DELAY_S = 2.0  # 快速重试间隔（秒）


@dataclass
class _PendingTask:
    """Poller 内部跟踪的待完成 A2A 任务。

    字段:
        task_id: A2A 任务 ID。
        agent_name: Peer Agent 名称。
        agent_url: Peer Agent A2A 端点 URL。
        channel: 发起任务的 channel。
        chat_id: 发起任务的 chat_id。
        goal: 用户原始请求文本。
        submitted_at: 提交时间（monotonic 秒）。
        last_polled_at: 上次轮询时间（monotonic 秒），用于指数退避。
    """

    task_id: str
    agent_name: str
    agent_url: str
    channel: str
    chat_id: str
    goal: str
    submitted_at: float = field(default_factory=time.monotonic)
    last_polled_at: float = field(default_factory=time.monotonic)


class PeerAgentPoller:
    """后台轮询所有 pending A2A 任务，完成后注入 MessageBus 触发新一轮处理。

    参数:
        bus: MessageBus 实例（用于注入完成/失败通知）。
        process_manager: PeerProcessManager 实例（用于销毁子进程）。
        client: httpx.AsyncClient 实例（用于 A2A JSON-RPC 调用）。
        artifacts_dir: 产出文件落盘目录（绝对路径）。
    """

    def __init__(
        self,
        bus: Any,
        process_manager: Any,
        client: httpx.AsyncClient,
        artifacts_dir: Path,
    ) -> None:
        self._bus = bus
        self._pm = process_manager
        self._client = client
        self._artifacts_dir = Path(artifacts_dir)
        self._pending: dict[str, _PendingTask] = {}
        self._task: asyncio.Task[None] | None = None

    # ── 公共 API ──────────────────────────────────────────────────

    def register(
        self,
        *,
        task_id: str,
        agent_name: str,
        agent_url: str,
        channel: str,
        chat_id: str,
        goal: str,
    ) -> None:
        """注册一个待跟踪的 A2A 任务。

        由 PeerAgentTool.execute() 在提交任务成功后调用。

        输入:
            task_id: A2A 任务 ID。
            agent_name: Peer Agent 名称。
            agent_url: Peer Agent A2A 端点 URL。
            channel: 发起任务的 channel。
            chat_id: 发起任务的 chat_id。
            goal: 用户原始请求文本。

        输出:
            None。
        """
        self._pending[task_id] = _PendingTask(
            task_id=task_id,
            agent_name=agent_name,
            agent_url=agent_url,
            channel=channel,
            chat_id=chat_id,
            goal=goal,
        )
        logger.info(
            "[Poller] 注册任务 task_id=%s agent=%s", task_id, agent_name,
        )

    def start(self) -> None:
        """启动后台轮询循环。

        输出:
            None。
        """
        self._task = asyncio.create_task(
            self._loop(), name="peer_agent_poller",
        )
        logger.info("[Poller] 后台轮询已启动")

    async def stop(self) -> None:
        """停止后台轮询循环。

        输出:
            None。
        """
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Poller] 已停止")

    # ── 内部 ──────────────────────────────────────────────────────

    @staticmethod
    def _dynamic_interval(submitted_at: float) -> float:
        """根据任务已运行时间返回动态轮询间隔。

        策略（指数退避）：
          - 前 1 分钟：10 秒（快速捕获"秒挂"——环境缺失、语法错误等）
          - 1-5 分钟：30 秒（任务已稳定运行，降低 RPC 频率）
          - 5 分钟后：  60 秒（低频保活检查）

        输入:
            submitted_at: 任务提交时间（time.monotonic() 值）。

        输出:
            当前应使用的轮询间隔（秒）。
        """
        elapsed = time.monotonic() - submitted_at
        if elapsed < 60:
            return 10.0
        elif elapsed < 300:
            return 30.0
        else:
            return 60.0

    async def _loop(self) -> None:
        """主轮询循环：每 10 秒检查 pending 任务（指数退避跳过不需要检查的）。"""
        while True:
            await asyncio.sleep(_POLL_INTERVAL_S)
            now = time.monotonic()
            for task_id, meta in list(self._pending.items()):
                # 指数退避：距上次检查不足动态间隔时跳过
                if now - meta.last_polled_at < self._dynamic_interval(
                    meta.submitted_at
                ):
                    continue
                meta.last_polled_at = now
                try:
                    await self._check(task_id, meta)
                except Exception as exc:
                    logger.warning(
                        "[Poller] 检查任务 %s 出错: %s", task_id, exc,
                    )

    async def _check(
        self,
        task_id: str,
        meta: _PendingTask,
    ) -> None:
        """检查单个 pending 任务的状态。

        输入:
            task_id: A2A 任务 ID。
            meta: _PendingTask 元数据。

        输出:
            None。完成/失败/超时时自动从 pending 字典中删除。
        """
        # 硬超时检查
        if time.monotonic() - meta.submitted_at > _TASK_TIMEOUT_S:
            logger.warning("[Poller] 任务 %s 超时（60分钟）", task_id)
            del self._pending[task_id]
            await self._inject_failure(
                meta, "调研超时（超过60分钟）",
            )
            await self._pm.terminate(meta.agent_name)
            return

        # 查询 A2A 任务状态（网络瞬断时快速重试 2-3 次）
        for attempt in range(_FAST_RETRY_MAX):
            try:
                state, artifacts, status_text = await self._get_task_status(
                    meta.agent_url, task_id,
                )
                break
            except httpx.HTTPError:
                if attempt < _FAST_RETRY_MAX - 1:
                    logger.debug(
                        "[Poller] 网络错误，%.1fs 后快速重试 "
                        "task_id=%s (attempt=%d/%d)",
                        _FAST_RETRY_DELAY_S, task_id,
                        attempt + 1, _FAST_RETRY_MAX,
                    )
                    await asyncio.sleep(_FAST_RETRY_DELAY_S)
                else:
                    raise  # 重试耗尽，抛给 _loop 保留任务到下一轮

        if state == "completed":
            logger.info(
                "[Poller] 任务 %s 完成，artifacts: %s",
                task_id, list(artifacts.keys()),
            )
            del self._pending[task_id]
            await self._inject_completion(meta, artifacts)
            await self._pm.terminate(meta.agent_name)

        elif state == "failed":
            logger.warning(
                "[Poller] 任务 %s 失败 原因: %s",
                task_id, status_text or "(无消息)",
            )
            del self._pending[task_id]
            await self._inject_failure(
                meta,
                f"调研任务执行失败：{status_text}"
                if status_text
                else "调研任务执行失败",
            )
            await self._pm.terminate(meta.agent_name)

        # 其他状态（submitted / working）静默等待下一轮

    async def _get_task_status(
        self,
        agent_url: str,
        task_id: str,
    ) -> tuple[str, dict[str, str], str]:
        """通过 A2A JSON-RPC tasks/get 查询任务状态。

        输入:
            agent_url: Peer Agent A2A 端点 URL。
            task_id: A2A 任务 ID。

        输出:
            (state, artifacts, status_text) 三元组。
            state: "submitted" | "working" | "completed" | "failed"。
            artifacts: {name → text_value} 产出文件映射。
            status_text: 状态消息文本（用于失败诊断）。

        异常:
            RuntimeError: A2A 返回错误时抛出。
            httpx.HTTPError: HTTP 请求失败时抛出。
        """
        payload = {
            "jsonrpc": "2.0",
            "id": "poll-1",
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        response = await self._client.post(
            agent_url,
            json=payload,
            timeout=8.0,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"tasks/get 错误: {data['error']}")

        result = data.get("result", {})
        status = result.get("status", {})
        state = status.get("state", "unknown")

        # 提取状态消息文本
        status_parts = status.get("message", {}).get("parts", [])
        status_text = " | ".join(
            p.get("text", "")
            for p in status_parts
            if isinstance(p, dict) and p.get("text")
        )
        if status_text:
            logger.debug(
                "[Poller] 任务 %s 状态=%s 消息: %s",
                task_id, state, status_text,
            )

        # 收集所有 artifacts：{name → 第一个 text part}
        artifacts: dict[str, str] = {}
        for artifact in result.get("artifacts", []):
            name = artifact.get("name", "")
            if not name:
                continue
            for p in artifact.get("parts", []):
                # A2A SDK 有两种序列化：{text: ...} 或 {root: {text: ...}}
                text = p.get("text") or (
                    p.get("root", {}).get("text")
                    if isinstance(p.get("root"), dict)
                    else None
                )
                if text:
                    artifacts[name] = text
                    break

        return state, artifacts, status_text

    async def _inject_completion(
        self,
        meta: _PendingTask,
        artifacts: dict[str, str],
    ) -> None:
        """任务完成后：落盘产出文件，注入携带绝对路径的系统通知。

        产出文件写入 {artifacts_dir}/{task_id}/{filename}，
        系统通知中携带完整绝对路径，确保主 Agent 精准命中文件。

        输入:
            meta: _PendingTask 元数据。
            artifacts: 产出文件映射 {name: text_content}。

        输出:
            None。
        """
        # 落盘产出文件到 task 专属子目录
        task_dir = self._artifacts_dir / meta.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: dict[str, str] = {}
        for name, text in artifacts.items():
            # 文件名安全化：防止路径穿越
            safe_name = Path(name).name or name
            file_path = task_dir / safe_name
            file_path.write_text(text, encoding="utf-8")
            artifact_paths[name] = str(file_path)
            logger.info(
                "[Poller] 产出已落盘: %s (%d chars)",
                file_path, len(text),
            )

        artifact_lines = "\n".join(
            f"  - {name}: {path}"
            for name, path in artifact_paths.items()
        ) or "  （无产出文件）"
        text = (
            f"[系统通知] 后台任务已完成。\n"
            f"执行的任务：{meta.goal}\n"
            f"执行者：{meta.agent_name}\n"
            f"产出文件：\n{artifact_lines}\n\n"
            f"请根据产出内容向用户汇报结果。"
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=meta.channel,
                sender="system",
                chat_id=meta.chat_id,
                content=text,
                metadata={
                    "system_injected": True,
                    "task_id": meta.task_id,
                },
            )
        )

    async def _inject_failure(
        self,
        meta: _PendingTask,
        reason: str,
    ) -> None:
        """任务失败后向 MessageBus 注入失败通知。

        输入:
            meta: _PendingTask 元数据。
            reason: 失败原因文本。

        输出:
            None。
        """
        text = (
            f"[系统通知] 后台任务未能完成：{reason}。\n"
            f"执行的任务：{meta.goal}\n"
            f"执行者：{meta.agent_name}\n"
            f"请告知用户，并建议他们稍后重试。"
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=meta.channel,
                sender="system",
                chat_id=meta.chat_id,
                content=text,
                metadata={
                    "system_injected": True,
                    "task_id": meta.task_id,
                },
            )
        )