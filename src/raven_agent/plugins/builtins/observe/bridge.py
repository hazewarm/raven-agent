"""Observe 桥接器 — 连接 Memory2 操作与 observe 轨迹记录。

桥接器不直接依赖 Memory2 的内部实现，而是依赖 EventBus 上的
RetrievalCompleted 和 MemoryWritten 事件。调用方（prompt.py 的
retrieve 步骤、memory2 memorizer）在完成操作后 emit 对应事件即可。

ObserveBridge 监听这些事件，将结构化的业务事件转换为 observe
的 trace 事件（RagQueryLog / MemoryWriteTrace），然后通过 TraceWriter
的 emit() 入队异步写入。

输入:
    writer: TraceWriter 实例（来自 ObservePlugin）。
    event_bus: EventBus 实例（来自 CoreRuntime）。

输出:
    ObserveBridge 实例。调用 start() 开始监听，stop() 取消监听。
"""

from __future__ import annotations

import logging
from typing import Any

from raven_agent.event_bus import EventBus
from raven_agent.events import MemoryWritten, RetrievalCompleted
from raven_agent.plugins.builtins.observe.events import (
    MemoryWriteTrace,
    RagHitLog,
    RagQueryLog,
)
from raven_agent.plugins.builtins.observe.writer import TraceWriter

logger = logging.getLogger("observe.bridge")


class ObserveBridge:
    """将 Memory2 操作事件桥接到 Observe 轨迹记录。

    输入:
        writer: TraceWriter 实例。为 None 时桥接器静默丢弃所有事件
                （observe 插件未启用时）。
        event_bus: EventBus 实例，用于订阅 RetrievalCompleted 和
                   MemoryWritten 事件。
    """

    def __init__(
        self,
        writer: TraceWriter | None,
        event_bus: EventBus,
    ) -> None:
        self._writer = writer
        self._event_bus = event_bus
        self._started = False

    def start(self) -> None:
        """开始监听 EventBus 事件。

        输入:
            无。

        输出:
            None。幂等——重复调用不重复注册。
        """
        if self._started:
            return
        self._event_bus.on(RetrievalCompleted, self._on_retrieval_completed)
        self._event_bus.on(MemoryWritten, self._on_memory_written)
        self._started = True
        logger.info("ObserveBridge started")

    def stop(self) -> None:
        """停止监听 EventBus 事件。

        输入:
            无。

        输出:
            None。幂等。
        """
        if not self._started:
            return
        self._event_bus.off(RetrievalCompleted, self._on_retrieval_completed)
        self._event_bus.off(MemoryWritten, self._on_memory_written)
        self._started = False
        logger.info("ObserveBridge stopped")

    # ── EventBus handlers ──────────────────────────────────────────

    def _on_retrieval_completed(self, event: RetrievalCompleted) -> None:
        """收到 RetrievalCompleted → 转换为 RagQueryLog → emit。

        输入:
            event: RetrievalCompleted 事件。

        输出:
            None。writer 为 None 时静默丢弃。
        """
        if self._writer is None:
            return
        hits = [
            RagHitLog(
                item_id=str(h.get("item_id", "")),
                memory_type=str(h.get("memory_type", "")),
                score=float(h.get("score", 0.0)),
                summary=str(h.get("summary", ""))[:120],
                injected=bool(h.get("injected", False)),
                forced=bool(h.get("forced", False)),
            )
            for h in event.hits
        ]
        self._writer.emit(
            RagQueryLog(
                caller=event.caller,
                session_key=event.session_key,
                query=event.query,
                orig_query=event.orig_query,
                aux_queries=list(event.aux_queries),
                hits=hits,
                injected_count=event.injected_count,
                route_decision=event.route_decision,
                error=event.error,
            )
        )

    def _on_memory_written(self, event: MemoryWritten) -> None:
        """收到 MemoryWritten → 转换为 MemoryWriteTrace → emit。

        输入:
            event: MemoryWritten 事件。

        输出:
            None。writer 为 None 时静默丢弃。
        """
        if self._writer is None:
            return
        self._writer.emit(
            MemoryWriteTrace(
                session_key=event.session_key,
                source_ref=event.source_ref,
                action=event.action,
                memory_type=event.memory_type,
                item_id=event.item_id,
                summary=event.summary,
                superseded_ids=list(event.superseded_ids),
                error=event.error,
            )
        )


async def emit_retrieval_event(
    event_bus: EventBus | None,
    *,
    caller: str,
    session_key: str,
    query: str,
    orig_query: str | None = None,
    aux_queries: list[str] | None = None,
    hits: list[dict[str, Any]] | None = None,
    injected_count: int = 0,
    route_decision: str | None = None,
    error: str | None = None,
) -> None:
    """便捷函数：在 Memory2 检索完成后 emit RetrievalCompleted 事件。

    输入:
        event_bus: EventBus 实例；为 None 时静默跳过（测试/禁用场景）。
        caller: 调用来源。
        session_key: 会话 key。
        query: 检索 query。
        orig_query: 原始 query。
        aux_queries: HyDE 假想条目。
        hits: 命中列表。
        injected_count: 注入条目数。
        route_decision: 路由决策。
        error: 错误信息。

    输出:
        None。
    """
    if event_bus is None:
        return
    await event_bus.emit(
        RetrievalCompleted(
            caller=caller,
            session_key=session_key,
            query=query,
            orig_query=orig_query,
            aux_queries=list(aux_queries or []),
            hits=list(hits or []),
            injected_count=injected_count,
            route_decision=route_decision,
            error=error,
        )
    )


async def emit_memory_write_event(
    event_bus: EventBus | None,
    *,
    session_key: str,
    source_ref: str = "",
    action: str = "",
    memory_type: str | None = None,
    item_id: str | None = None,
    summary: str | None = None,
    superseded_ids: list[str] | None = None,
    error: str | None = None,
) -> None:
    """便捷函数：在 Memory2 写入完成后 emit MemoryWritten 事件。

    输入:
        event_bus: EventBus 实例；为 None 时静默跳过。
        session_key: 会话 key。
        source_ref: 来源引用。
        action: "write" | "supersede"。
        memory_type: 记忆类型。
        item_id: 条目 id。
        summary: 摘要。
        superseded_ids: 被替换 id 列表。
        error: 错误信息。

    输出:
        None。
    """
    if event_bus is None:
        return
    await event_bus.emit(
        MemoryWritten(
            session_key=session_key,
            source_ref=source_ref,
            action=action,
            memory_type=memory_type,
            item_id=item_id,
            summary=summary,
            superseded_ids=list(superseded_ids or []),
            error=error,
        )
    )