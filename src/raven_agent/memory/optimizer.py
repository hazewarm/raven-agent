from __future__ import annotations

import asyncio
from datetime import datetime
from collections.abc import Callable
import logging

from raven_agent.llm import LLMProvider
from raven_agent.memory.markdown import DEFAULT_SELF_MD, MarkdownMemoryStore
from raven_agent.messages import system_message, user_message

logger = logging.getLogger(__name__)


class MemoryOptimizerBusy(RuntimeError):
    """MemoryOptimizer 已在运行时抛出。"""

_MERGE_SYSTEM = "你是 Raven 的用户长期记忆整理器，只输出完整 MEMORY.md。"

_MERGE_PROMPT = """今天日期：{today}

请把【现有长期记忆】和【待归档事实】合并为新的 MEMORY.md。

规则：
- 只保留跨对话仍有长期价值的信息。
- 合并重复事实。
- correction 要反映为最终事实，不保留旧值。
- 删除短期状态、一次性事件、普通寒暄和对话过程总结。
- 保持 Markdown 格式。
- 直接输出完整 MEMORY.md，不要代码块，不要解释。

【现有长期记忆】
{memory}

【待归档事实】
{pending}
"""

_SELF_SYSTEM = "你是 Raven 的自我认知整理器，只输出完整 SELF.md。"

_SELF_PROMPT = """请根据当前 SELF.md 和本次待归档事实，整理新的 SELF.md。

规则：
- 保留 Raven 的自我定位、协作方式和对用户的稳定理解。
- 不要把用户资料清单机械写进 SELF.md。
- 不要写短期事件、工具过程或对话流水账。
- 如果待归档事实与 SELF.md 无关，可以基本保持原文。
- 直接输出完整 SELF.md，不要代码块，不要解释。

【当前 SELF.md】
{self_content}

【待归档事实】
{pending}
"""

class MemoryOptimizer:
    """把 PENDING.md 合并进 MEMORY.md / SELF.md 的优化器。

    参数:
        store: MarkdownMemoryStore。
        provider: LLMProvider。
    """

    def __init__(self, store: MarkdownMemoryStore, provider: LLMProvider) -> None:
        self._store = store
        self._provider = provider
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """返回 optimizer 当前是否正在运行。

        返回:
            正在运行返回 True。
        """

        return self._lock.locked()

    async def optimize(self) -> None:
        """执行一次 PENDING.md 归档。

        返回:
            None。

        异常:
            MemoryOptimizerBusy: 当已有 optimize 正在运行时抛出。
        """

        if self._lock.locked():
            raise MemoryOptimizerBusy("memory optimizer is already running")
        async with self._lock:
            await self._optimize_unlocked()
    
    async def _optimize_unlocked(self) -> None:
        """在持锁状态下执行 optimizer。

        返回:
            None。
        """

        pending = self._store.snapshot_pending().strip()
        current_memory = self._store.read_long_term().strip()
        if not pending:
            self._store.commit_pending_snapshot()
            return

        merged_memory = await self._merge_memory(current_memory, pending)
        if not merged_memory:
            self._store.rollback_pending_snapshot()
            return

        self._store.write_long_term(merged_memory)
        updated_self = await self._merge_self(pending)
        if updated_self:
            self._store.write_self(updated_self)
        self._store.commit_pending_snapshot()
    
    async def _merge_memory(self, memory: str, pending: str) -> str:
        """调用 LLM 合并 MEMORY.md。

        参数:
            memory: 当前 MEMORY.md。
            pending: 本次 snapshot 的 pending 内容。

        返回:
            新 MEMORY.md 文本；失败时返回空字符串。
        """

        prompt = _MERGE_PROMPT.format(
            today=datetime.now().strftime("%Y-%m-%d"),
            memory=memory or "（空）",
            pending=pending,
        )
        response = await self._provider.chat(
            messages=[
                system_message(_MERGE_SYSTEM),
                user_message(prompt),
            ],
            tools=[],
            tool_choice="none",
        )
        return response.content.strip()

    async def _merge_self(self, pending: str) -> str:
        """调用 LLM 更新 SELF.md。

        参数:
            pending: 本次 snapshot 的 pending 内容。

        返回:
            新 SELF.md 文本；失败时返回空字符串。
        """

        current_self = self._store.read_self().strip() or DEFAULT_SELF_MD.strip()
        prompt = _SELF_PROMPT.format(
            self_content=current_self,
            pending=pending or "（无）",
        )
        response = await self._provider.chat(
            messages=[
                system_message(_SELF_SYSTEM),
                user_message(prompt),
            ],
            tools=[],
            tool_choice="none",
        )
        return response.content.strip()


# ── MemoryOptimizerLoop ───────────────────────────────────────────

_DEFAULT_INTERVAL_SECONDS = 64800  # 默认每 18 小时（对齐整点）


class MemoryOptimizerLoop:
    """自动定期执行 memory optimizer 的后台循环。

    每 interval_seconds 对齐整点执行一次 optimizer.optimize()。
    原理很简单——不是 cron 解析器：

    1. 计算距下一个对齐整点的秒数
    2. sleep 到整点
    3. 调用 optimizer.optimize()
    4. 回到 1

    参数:
        optimizer: MemoryOptimizer 实例；为 None 时 run() 是空操作。
        interval_seconds: 执行间隔（秒），默认 64800（18 小时）。
                          对齐到 epoch 的整数倍（例如 64800 秒 = 18 小时，
                          每天在 UTC 0:00 → 18:00 → 12:00 → 6:00 循环）。
        _now_fn: 可选的 `() -> datetime` 函数，用于确定性测试。
    """

    def __init__(
        self,
        optimizer: MemoryOptimizer | None,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._interval = max(60, interval_seconds)
        self._now_fn = _now_fn or datetime.now
        self._running = False

    async def run(self) -> None:
        """启动优化循环（阻塞当前协程直到 stop() 被调用）。

        输入:
            无。

        输出:
            None。循环持续到 stop() 被调用。
        """
        self._running = True
        logger.info(
            "[memory_optimizer] 优化循环已启动，间隔=%ds (%.1fh)，对齐 %ds 倍整点",
            self._interval,
            self._interval / 3600,
            self._interval,
        )
        while self._running:
            secs = self._seconds_until_next_tick()
            logger.info(
                "[memory_optimizer] 距下次优化 %.0f 秒 (%.1f 小时)",
                secs,
                secs / 3600,
            )
            # 分段 sleep，每秒检查 _running，确保 stop() 延迟 ≤ 1 秒
            elapsed = 0.0
            step = 1.0
            while elapsed < secs and self._running:
                await asyncio.sleep(min(step, secs - elapsed))
                elapsed += step
            if not self._running:
                break
            try:
                if self._optimizer:
                    await self._optimizer.optimize()
            except MemoryOptimizerBusy:
                logger.info("[memory_optimizer] 上一次 optimize 尚未结束，跳过本轮")
            except Exception:
                logger.exception("[memory_optimizer] 优化异常")

    def start(self) -> asyncio.Task[None]:
        """在后台启动优化循环并返回 Task。

        输入:
            无。

        输出:
            运行 MemoryOptimizerLoop.run() 的 asyncio.Task。
        """
        return asyncio.create_task(self.run(), name="memory_optimizer_loop")

    def stop(self) -> None:
        """停止优化循环。

        输入:
            无。

        输出:
            None。幂等——重复调用无副作用。
        """
        self._running = False

    def _seconds_until_next_tick(self) -> float:
        """计算距下一个对齐整点的秒数。

        epoch 起算，对齐到 interval_seconds 的整数倍。
        例如 interval_seconds=64800（18h）：
        - 当前 epoch 秒数为 12345 → 下一个整点是 64800
        - 当前 epoch 秒数为 70000 → 下一个整点是 129600

        输出:
            距下一个整点的秒数（≥ 1.0）。
        """
        now = self._now_fn()
        now_ts = now.timestamp()
        next_ts = (now_ts // self._interval + 1) * self._interval
        return max(1.0, next_ts - now_ts)