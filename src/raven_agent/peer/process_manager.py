"""
peer/process_manager.py —— Peer Agent 子进程生命周期管理。

管理所有 Peer Agent 子进程的启动、健康检查和终止。
采用冷启动策略：只在第一次提交任务时拉起子进程，完成后销毁。
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from pathlib import Path

import httpx

from raven_agent.peer.models import PeerProcessConfig

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 2.0
_SPAWN_POLL_INTERVAL_S = 1.0


class PeerProcessManager:
    """管理 Peer Agent 子进程的生命周期。

    参数:
        configs: PeerProcessConfig 列表。
        client: httpx.AsyncClient 实例（用于健康检查和与 peer 通信）。
        log_dir: 子进程日志输出目录。
    """

    def __init__(
        self,
        configs: list[PeerProcessConfig],
        client: httpx.AsyncClient,
        log_dir: Path,
    ) -> None:
        self._configs: dict[str, PeerProcessConfig] = {
            c.name: c for c in configs
        }
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._client = client
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # 每个 agent 一把锁——同一 agent 的 ensure_ready 不会被并发执行
        self._locks: dict[str, asyncio.Lock] = {
            c.name: asyncio.Lock() for c in configs
        }

    # ── 公共 API ──────────────────────────────────────────────────

    async def ensure_ready(self, name: str) -> None:
        """确保指定 agent 已启动且通过健康检查。未启动则冷启动。

        输入:
            name: Agent 名称。

        输出:
            None。

        异常:
            ValueError: 未知 agent 名称。
            RuntimeError: 启动失败或健康检查超时。
        """
        cfg = self._configs.get(name)
        if cfg is None:
            raise ValueError(f"未知 peer agent: {name!r}")

        async with self._locks[name]:
            if await self._is_healthy(cfg):
                logger.debug("[PeerProcess] %s 已在线", name)
                return
            logger.info("[PeerProcess] %s 未运行，开始冷启动", name)
            await self._spawn(cfg)
            logger.info("[PeerProcess] %s 启动成功", name)

    async def terminate(self, name: str) -> None:
        """销毁指定 peer agent 子进程。

        任务完成后由 Poller 调用。先发 SIGTERM，超时后 SIGKILL。

        输入:
            name: Agent 名称。

        输出:
            None。幂等——agent 未被管理时静默返回。
        """
        lock = self._locks.get(name)
        if lock is None:
            return
        async with lock:
            proc = self._procs.pop(name, None)
            if proc is None:
                return
            logger.info("[PeerProcess] 终止 %s (pid=%s)", name, proc.pid)
            await self._kill(proc, self._configs[name].shutdown_timeout_s)

    async def shutdown_all(self) -> None:
        """raven-agent 退出时批量终止所有子进程。

        输出:
            None。单个 agent 终止失败不影响其他 agent 的终止。
        """
        names = list(self._procs.keys())
        if names:
            logger.info("[PeerProcess] 关闭所有子进程: %s", names)
        await asyncio.gather(
            *(self.terminate(name) for name in names),
            return_exceptions=True,
        )

    # ── 内部方法 ──────────────────────────────────────────────────

    async def _is_healthy(self, cfg: PeerProcessConfig) -> bool:
        """检查 agent 是否在线。

        输入:
            cfg: PeerProcessConfig。

        输出:
            True 表示 /health 返回 200。
        """
        try:
            response = await self._client.get(
                cfg.base_url.rstrip("/") + cfg.health_path,
                timeout=_HEALTH_TIMEOUT_S,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def _spawn(self, cfg: PeerProcessConfig) -> None:
        """冷启动一个 peer agent 子进程，等待健康检查通过。

        输入:
            cfg: PeerProcessConfig。

        输出:
            None。

        异常:
            RuntimeError: 子进程启动后立即退出、或健康检查超时。
        """
        log_path = self._log_dir / f"{cfg.name.replace(' ', '_')}.log"
        log_fp = log_path.open("wb")

        # 进程组隔离：确保孙子进程也能被一键清理
        _spawn_kwargs: dict[str, object] = {}
        if os.name == "nt":
            _spawn_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            _spawn_kwargs["start_new_session"] = True

        proc = await asyncio.create_subprocess_exec(
            *cfg.launcher,
            stdout=log_fp,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cfg.cwd,
            **_spawn_kwargs,
        )
        self._procs[cfg.name] = proc
        logger.info(
            "[PeerProcess] 已启动 %s pid=%d 日志=%s",
            cfg.name, proc.pid, log_path,
        )

        # 轮询等待健康检查通过
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cfg.startup_timeout_s
        while loop.time() < deadline:
            await asyncio.sleep(_SPAWN_POLL_INTERVAL_S)
            if proc.returncode is not None:
                raise RuntimeError(
                    f"{cfg.name} 启动后立即退出 (rc={proc.returncode})"
                )
            if await self._is_healthy(cfg):
                return

        # 超时：终止整个进程组
        if os.name == "nt":
            proc.terminate()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
        self._procs.pop(cfg.name, None)
        raise RuntimeError(
            f"{cfg.name} 启动超时（{cfg.startup_timeout_s}s）"
        )

    @staticmethod
    async def _kill(
        proc: asyncio.subprocess.Process,
        timeout_s: int,
    ) -> None:
        """终止子进程及其所有后代。

        Linux/macOS: SIGTERM 进程组 → 超时 → SIGKILL 进程组。
        Windows: terminate() → 超时 → kill()。

        输入:
            proc: 子进程对象。
            timeout_s: 等待进程退出的秒数。

        输出:
            None。
        """
        if proc.returncode is not None:
            return
        if os.name == "nt":
            # Windows：无进程组概念，直接 terminate/kill
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=float(timeout_s))
            except asyncio.TimeoutError:
                logger.warning(
                    "[PeerProcess] terminate 超时，强制 kill pid=%d", proc.pid,
                )
                proc.kill()
                await proc.wait()
        else:
            # Unix：优先向整个进程组发信号（清理孙子进程）
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=float(timeout_s))
            except asyncio.TimeoutError:
                logger.warning(
                    "[PeerProcess] SIGTERM 超时，强制 SIGKILL pgid=%d", proc.pid,
                )
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                await proc.wait()