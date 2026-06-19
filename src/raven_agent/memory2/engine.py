from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import cast

from raven_agent.memory import (
    EngineProfile,
    EvidenceRef,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
    MemoryToolProfile,
    MemoryToolSpec,
)
from raven_agent.memory2.embedder import EmbeddingProvider
from raven_agent.memory2.models import MemoryItem, normalize_memory_type
from raven_agent.memory2.store import MemoryStore2
from raven_agent.memory2.retriever import Retriever
from raven_agent.memory2.memorizer import Memorizer
from raven_agent.memory2.rule_schema import build_procedure_rule_schema

logger = logging.getLogger(__name__)

# 类的定义
class Memory2Engine:
    """基于 SQLite store 和 embedding provider 的 MemoryEngine 实现。

    参数:
        store: MemoryStore2，用于 SQLite 持久化。
        embedder: EmbeddingProvider，用于生成文本向量。

    返回:
        Memory2Engine 实例。
    """

    DESCRIPTOR = MemoryEngineDescriptor(
        name="memory2",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        # 能摄入文本
        # 能做语义检索
        # 能返回结构化命中数据
        # 能更新记忆
        # 能删除（废弃）记忆
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_TEXT,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.MANAGE_UPDATE,
                MemoryCapability.MANAGE_DELETE,
            }
        ),
        # 备注：底层用 sqlite，检索方案是混合搜索
        notes={"store": "sqlite", "retrieval": "sqlite_vec_numpy_fts_rrf"},
    )

    def __init__(
        self,
        *,
        store: MemoryStore2,
        embedder: EmbeddingProvider,
        retriever: Retriever | None = None,
        memorizer: Memorizer | None = None,
    ) -> None:
        """初始化 Memory2Engine。

        参数:
            store: MemoryStore2，用于 SQLite 持久化。
            embedder: EmbeddingProvider，用于生成文本向量。
            retriever: 可选 Retriever；不传则使用默认 Retriever。
            memorizer: 可选 Memorizer；不传则使用默认 Memorizer。

        返回:
            None。
        """

        self._store = store
        self._embedder = embedder
        self._retriever = retriever or Retriever(store=store, embedder=embedder)
        self._memorizer = memorizer or Memorizer(store=store, embedder=embedder)

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        """摄入文本或 consolidation event。

        参数:
            request: 摄入请求。content 可以是字符串或包含 summary/memory_type 的 dict。

        返回:
            MemoryIngestResult。
        """

        if request.source_kind not in {"text", "consolidation_event"}:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported source_kind for memory2",
                raw={"source_kind": request.source_kind},
            )

        if isinstance(request.content, dict):
            summary = str(request.content.get("summary", "")).strip()
            memory_type = normalize_memory_type(str(request.content.get("memory_type", "event")))
            happened_at = str(request.content.get("happened_at", "")).strip() or None
            extra_json = dict(request.content.get("extra_json", {})) if isinstance(request.content.get("extra_json"), dict) else {}
            emotional_weight = int(request.content.get("emotional_weight", 0) or 0)
        else:
            summary = str(request.content).strip()
            memory_type = "event" if request.source_kind == "consolidation_event" else "preference"
            happened_at = None
            extra_json = {}
            emotional_weight = 0

        if not summary:
            return MemoryIngestResult(accepted=False, summary="empty content")

        source_ref = str(request.metadata.get("source_ref") or "").strip() or None
        if request.source_kind == "consolidation_event" and source_ref:
            result = await self._memorizer.save_from_consolidation(
                history_entry=summary,
                source_ref=source_ref,
                scope_channel=request.scope.channel,
                scope_chat_id=request.scope.chat_id,
                emotional_weight=emotional_weight,
            )
        else:
            result = await self._memorizer.save_item_with_supersede(
                summary=summary,
                memory_type=memory_type,
                extra_json=extra_json,
                source_ref=source_ref,
                happened_at=happened_at,
                emotional_weight=emotional_weight,
            )

        status, item_id = _split_store_result(result)
        return MemoryIngestResult(
            accepted=status in {"new", "reinforced", "merged", "semantic_dedup"},
            created_ids=[item_id] if item_id and status in {"new", "merged"} else [],
            summary=result,
            raw={"engine": self.DESCRIPTOR.name, "status": status},
        )

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        """查询结构化语义记忆。

        参数:
            request: MemoryQuery，包含查询文本、intent、filters 和 limit。

        返回:
            MemoryQueryResult，包含 MemoryRecord 列表和可选 text_block。
        """

        if request.intent == "timeline":
            return self._query_timeline(request)
        if request.intent == "interest":
            return await self._query_interest(request)
        if request.intent in {"context", "procedure"}:
            return await self._query_context(request)
        return await self._query_answer(request)
    
    async def _query_answer(self, request: MemoryQuery) -> MemoryQueryResult:
        """执行 answer intent 查询。

        参数:
            request: MemoryQuery。

        返回:
            MemoryQueryResult。
        """

        memory_types = _resolve_memory_types(request)
        aux_queries = _resolve_aux_queries(request)
        hits = await self._retriever.retrieve(
            request.text,
            memory_types=memory_types,
            top_k=max(1, request.limit),
            aux_queries=aux_queries,
            score_threshold=float(request.filters.hints.get("score_threshold", 0.35)),
            time_start=request.filters.time_start,
            time_end=request.filters.time_end,
            keyword_enabled=True,
        )
        records = [_record_from_hit(hit) for hit in hits[: request.limit]]
        return MemoryQueryResult(
            records=records,
            trace={
                "engine": self.DESCRIPTOR.name,
                "intent": request.intent,
                "hit_count": len(records),
                "aux_queries": aux_queries,
            },
            raw={"items": hits[: request.limit]},
        )
    
    async def _query_context(self, request: MemoryQuery) -> MemoryQueryResult:
        """执行 context / procedure intent 查询并生成注入块。

        参数:
            request: MemoryQuery。

        返回:
            MemoryQueryResult。
        """

        memory_types = _resolve_memory_types(request)
        hits = await self._retriever.retrieve(
            request.text,
            memory_types=memory_types,
            top_k=max(1, request.limit),
            aux_queries=_resolve_aux_queries(request),
            score_threshold=float(request.filters.hints.get("score_threshold", 0.35)),
            time_start=request.filters.time_start,
            time_end=request.filters.time_end,
            keyword_enabled=True,
        )
        text_block, injected_ids = self._retriever.build_injection_block(hits)
        records = [_record_from_hit(hit, injected_ids=injected_ids) for hit in hits[: request.limit]]
        return MemoryQueryResult(
            text_block=text_block,
            records=records,
            trace={
                "engine": self.DESCRIPTOR.name,
                "intent": request.intent,
                "hit_count": len(records),
                "injected_count": len(injected_ids),
            },
            raw={"items": hits[: request.limit]},
        )
    
    async def _query_interest(self, request: MemoryQuery) -> MemoryQueryResult:
        """执行 interest intent 查询。

        参数:
            request: MemoryQuery。

        返回:
            MemoryQueryResult。
        """

        hits = await self._retriever.retrieve(
            request.text,
            memory_types=["preference", "profile"],
            top_k=max(1, request.limit),
            score_threshold=float(request.filters.hints.get("score_threshold", 0.35)),
            keyword_enabled=True,
        )
        records = [_record_from_hit(hit) for hit in hits[: request.limit]]
        return MemoryQueryResult(
            text_block="\n---\n".join(record.summary for record in records),
            records=records,
            trace={"engine": self.DESCRIPTOR.name, "intent": request.intent, "hit_count": len(records)},
            raw={"items": hits[: request.limit]},
        )
    
    def _query_timeline(self, request: MemoryQuery) -> MemoryQueryResult:
        """执行 timeline intent 查询。

        参数:
            request: MemoryQuery。

        返回:
            MemoryQueryResult。
        """

        if request.filters.time_start is None or request.filters.time_end is None:
            return MemoryQueryResult(
                trace={"engine": self.DESCRIPTOR.name, "intent": "timeline_missing_time"}
            )
        hits = self._store.list_events_by_time_range(
            request.filters.time_start,
            request.filters.time_end,
            limit=request.limit,
        )
        records = [_record_from_hit(hit) for hit in hits]
        return MemoryQueryResult(
            records=records,
            text_block="\n".join(f"- {record.summary}" for record in records),
            trace={"engine": self.DESCRIPTOR.name, "intent": "timeline", "hit_count": len(records)},
            raw={"items": hits},
        )
    

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        """执行 remember / update / forget / restore。

        参数:
            request: 语义记忆变更请求。

        返回:
            MemoryMutationResult。
        """

        if request.kind == "forget":
            return self._forget(request)
        if request.kind == "restore":
            return self._restore(request)
        if request.kind == "update":
            return await self._update(request)
        return await self._remember(request)

    async def _remember(self, request: MemoryMutation) -> MemoryMutationResult:
        """写入一条显式记忆。

        参数:
            request: kind=remember 的 MemoryMutation。

        返回:
            MemoryMutationResult。
        """

        summary = request.summary.strip()
        if not summary:
            return MemoryMutationResult(accepted=False, status="empty")
        memory_type = normalize_memory_type(request.memory_kind)
        result = await self._memorizer.save_item_with_supersede(
            summary=summary,
            memory_type=memory_type,
            extra_json=dict(request.metadata),
            source_ref=request.source_ref or "memory_mutation",
        )
        status, item_id = _split_store_result(result)
        return MemoryMutationResult(
            accepted=status in {"new", "reinforced", "merged", "semantic_dedup"},
            item_id=item_id,
            actual_kind=memory_type,
            status=status,
        )

    async def _update(self, request: MemoryMutation) -> MemoryMutationResult:
        """用新记忆替换旧 ids。

        参数:
            request: kind=update 的 MemoryMutation。

        返回:
            MemoryMutationResult。
        """
        # 1. 找出要被替换的旧记忆
        old_items = self._store.get_items_by_ids(list(request.ids))
        missing = [item_id for item_id in request.ids if item_id not in {item.id for item in old_items}]
        
        # 2. 先把新记忆存进去
        remember_result = await self._remember(request)
        if not remember_result.item_id:
            return remember_result
        new_item = self._store.get_item(remember_result.item_id)
        if new_item is None:
            return remember_result
        
        # 3. 如果成功存了新记忆，且确实有旧记忆存在，执行更迭逻辑
        if old_items:
            # 3.1 批量软删除旧记忆 (把状态改成 superseded)
            self._store.mark_superseded_batch([item.id for item in old_items])
            # 3.2 记录更迭轨迹 (建立新老记忆的血缘链接)
            for old_item in old_items:
                self._store.record_replacement(
                    old_item=old_item,
                    new_item=new_item,
                    relation_type="supersede",
                    source_ref=request.source_ref or "memory_update",
                )
        # 4. 组装报告返回
        remember_result.affected_ids = [item.id for item in old_items]
        remember_result.missing_ids = missing
        remember_result.status = "updated" if old_items else remember_result.status
        return remember_result

    def _forget(self, request: MemoryMutation) -> MemoryMutationResult:
        """把指定 memory ids 标记为 superseded。

        参数:
            request: kind=forget 的 MemoryMutation。

        返回:
            MemoryMutationResult。
        """

        items = self._store.get_items_by_ids(list(request.ids))
        found_ids = [item.id for item in items]
        missing = [item_id for item_id in request.ids if item_id not in set(found_ids)]
        affected = self._store.mark_superseded_batch(found_ids)
        return MemoryMutationResult(
            accepted=affected > 0,
            status="superseded",
            affected_ids=found_ids,
            missing_ids=missing,
            items=[_item_summary(item) for item in items],
        )

    def _restore(self, request: MemoryMutation) -> MemoryMutationResult:
        """把指定 memory ids 恢复为 active。

        参数:
            request: kind=restore 的 MemoryMutation。

        返回:
            MemoryMutationResult。
        """

        items = self._store.get_items_by_ids(list(request.ids))
        found_ids = [item.id for item in items]
        missing = [item_id for item_id in request.ids if item_id not in set(found_ids)]
        affected = self._store.restore_items_batch(found_ids)
        return MemoryMutationResult(
            accepted=affected > 0,
            status="restored",
            affected_ids=found_ids,
            missing_ids=missing,
            items=[_item_summary(item) for item in items],
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        """批量强化被使用过的记忆条目。

        参数:
            ids: 要强化的 memory ids。

        返回:
            None。
        """

        self._store.reinforce_items_batch(ids)

    
    # === 以下是为后台管理 Dashboard 预留的接口，MVP 阶段暂未实现 ===
    def describe(self) -> MemoryEngineDescriptor:
        """返回 Memory2Engine 描述。

        参数:
            无。

        返回:
            MemoryEngineDescriptor。
        """

        return self.DESCRIPTOR

    def tool_profile(self) -> MemoryToolProfile:
        """返回 memory tools profile。

        参数:
            无。

        返回:
            当前引擎 MemoryToolProfile。
        """

        return _memory2_tool_profile()

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict[str, object]]:
        """按关键词匹配 procedure。

        参数:
            action_tokens: 动作词列表。

        返回:
            命中的 procedure 简表列表。
        """

        return self._store.keyword_match_procedures(action_tokens)

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """列出时间范围内的事件。

        参数:
            time_start: 时间范围起点。
            time_end: 时间范围终点。
            limit: 最多返回多少条。

        返回:
            event 记忆简表列表。
        """

        return self._store.list_events_by_time_range(time_start, time_end, limit=limit)

    def list_items_for_dashboard(self, **kwargs: object) -> tuple[list[dict[str, object]], int]:
        """给 Dashboard 列出 memory items。

        参数:
            kwargs: Dashboard 查询参数。

        返回:
            本章暂不实现 Dashboard 查询，返回空列表和 0。
        """

        return [], 0

    def get_item_for_dashboard(self, item_id: str, *, include_embedding: bool = False) -> dict[str, object] | None:
        """给 Dashboard 读取单条 memory item。

        参数:
            item_id: memory item id。
            include_embedding: 是否包含 embedding。

        返回:
            本章暂不实现 Dashboard 查询，返回 None。
        """

        return None

    def update_item_for_dashboard(self, item_id: str, **kwargs: object) -> dict[str, object] | None:
        """给 Dashboard 更新单条 memory item。

        参数:
            item_id: memory item id。
            kwargs: 更新字段。

        返回:
            本章暂不实现 Dashboard 更新，返回 None。
        """

        return None

    def delete_item(self, item_id: str) -> bool:
        """物理删除单条 memory item。

        参数:
            item_id: memory item id。

        返回:
            本章不开放物理删除，返回 False。
        """

        return False

    def delete_items_batch(self, ids: list[str]) -> int:
        """批量物理删除 memory items。

        参数:
            ids: memory item id 列表。

        返回:
            本章不开放物理删除，返回 0。
        """

        return 0

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        """给 Dashboard 查找相似 memory items。

        参数:
            item_id: 基准 memory item id。
            top_k: 最多返回多少条。
            memory_type: 类型过滤。
            score_threshold: 最低相似度。
            include_superseded: 是否包含 superseded。

        返回:
            本章暂不实现 Dashboard 相似查询，返回空列表。
        """

        return []

    async def close(self) -> None:
        """关闭 Memory2Engine 资源。

        参数:
            无。

        返回:
            None。
        """

        await self._embedder.close()
        self._store.close()


def _split_store_result(result: str) -> tuple[str, str]:
    """拆分 store 返回的 status:id 字符串。

    参数:
        result: store 返回值。

    返回:
        二元组：(status, item_id)。
    """

    if ":" not in result:
        return result, ""
    status, item_id = result.split(":", 1)
    return status, item_id


def _evidence_from_source_ref(source_ref: str) -> list[EvidenceRef]:
    """从 source_ref 构造 EvidenceRef。

    参数:
        source_ref: memory item 来源引用。

    返回:
        EvidenceRef 列表；source_ref 为空时返回空列表。
    """

    raw = source_ref.strip()
    if not raw:
        return []
    base = raw.split("#", 1)[0].strip()
    try:
        parsed = json.loads(base)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        refs = [str(item).strip() for item in parsed if str(item).strip()]
        if refs:
            return [
                EvidenceRef(
                    kind="message_range",
                    refs=refs,
                    resolver="session",
                    source_ref=raw,
                )
            ]
    return [EvidenceRef(kind="external", refs=[raw], source_ref=raw)]


def _record_from_hit(
    hit: dict[str, object],
    *,
    injected_ids: list[str] | None = None,
) -> MemoryRecord:
    """把 store hit 转换为 MemoryRecord。

    参数:
        hit: Retriever 返回的一条命中。
        injected_ids: 已注入 prompt 的 memory id 列表。

    返回:
        MemoryRecord。
    """

    source_ref = str(hit.get("source_ref", "") or "")
    score = hit.get("score", 0.0)
    item_id = str(hit.get("id", "") or "")
    extra = hit.get("extra_json")
    signals = dict(cast(dict[str, object], extra)) if isinstance(extra, dict) else {}
    signals.update(
        {
            "reinforcement": hit.get("reinforcement", 0),
            "emotional_weight": hit.get("emotional_weight", 0),
            "rrf_score": hit.get("rrf_score", 0.0),
            "keyword_score": hit.get("keyword_score", 0.0),
            "score_debug": hit.get("_score_debug", {}),
        }
    )
    return MemoryRecord(
        id=item_id,
        kind=str(hit.get("memory_type", "") or ""),
        summary=str(hit.get("summary", "") or ""),
        score=float(score) if isinstance(score, int | float) else 0.0,
        engine_kind="memory2",
        evidence=_evidence_from_source_ref(source_ref),
        signals=signals,
        injected=item_id in set(injected_ids or []),
    )


def _records_to_text_block(records: list[MemoryRecord]) -> str:
    """把 MemoryRecord 列表渲染成文本块。

    参数:
        records: 查询命中的 MemoryRecord 列表。

    返回:
        可注入 prompt 或工具结果的文本块。
    """

    if not records:
        return ""
    lines: list[str] = []
    for record in records:
        lines.append(f"- [{record.kind}] {record.summary} (id={record.id}, score={record.score:.4f})")
    return "\n".join(lines)


def _item_summary(item: MemoryItem) -> dict[str, object]:
    """把 MemoryItem 转换成 mutation result 简表。

    参数:
        item: MemoryItem。

    返回:
        包含 id、memory_type、summary、status 的字典。
    """

    return {
        "id": item.id,
        "memory_type": item.memory_type,
        "summary": item.summary,
        "status": item.status,
    }


def _resolve_memory_types(request: MemoryQuery) -> list[str] | None:
    """根据 MemoryQuery 推导 memory_types 过滤条件。

    参数:
        request: MemoryQuery。

    返回:
        memory type 列表；不限制时返回 None。
    """

    if request.filters.kinds:
        return [str(item) for item in request.filters.kinds if str(item).strip()]
    if request.intent == "procedure":
        return ["procedure", "preference"]
    return None


def _resolve_aux_queries(request: MemoryQuery) -> list[str]:
    """从 filters.hints 中读取辅助查询。

    参数:
        request: MemoryQuery。

    返回:
        辅助 query 列表。
    """

    raw_queries = request.filters.hints.get("queries")
    if isinstance(raw_queries, list):
        return [str(item).strip() for item in raw_queries if str(item).strip()]
    return []


def _memory2_tool_profile() -> MemoryToolProfile:
    """构造 Memory2 暴露给模型的 memory tools profile。

    参数:
        无。

    返回:
        包含 recall / memorize / forget 三个 MemoryToolSpec 的 MemoryToolProfile。
    """

    return MemoryToolProfile(
        recall=MemoryToolSpec(
            description=(
                "检索长期记忆中的事实、偏好、流程与历史事件线索。"
                "query 写成陈述句；intent=answer 做主题检索，intent=timeline 配合 time_filter 做时间线回顾。"
                "返回记忆摘要和 evidence；evidence.refs 是来源消息 id，可用于后续回查原文。"
                "若回复使用了任何返回的记忆条目，必须在正文末尾输出 §cited:[实际使用的id列表]§。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查找的记忆主题，推荐写成陈述句"},
                    "intent": {
                        "type": "string",
                        "enum": ["answer", "timeline"],
                        "description": "answer=主题检索；timeline=按 time_filter 列出历史事件",
                        "default": "answer",
                    },
                    "memory_kind": {
                        "type": "string",
                        "enum": ["event", "profile", "preference", "procedure", ""],
                        "description": "限定记忆类型，留空表示不限",
                        "default": "",
                    },
                    "time_filter": {
                        "type": "string",
                        "description": "today / yesterday / recent_3d / recent_7d / recent_30d / YYYY-MM-DD / YYYY-MM-DD~YYYY-MM-DD",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回条数",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
            risk="read-only",
            search_hint="记得 以前 历史 做过什么 有没有 记忆查询 回忆",
        ),
        memorize=MemoryToolSpec(
            description=(
                "将用户明确要求长期保留的信息写入记忆。"
                "memory_kind 可选 event/profile/preference/procedure，engine 会按类型做去重、合并与 supersede。"
                "只在用户明确表达要长期记住，或给出长期行为规则时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "一句话描述要记住的内容"},
                    "memory_kind": {
                        "type": "string",
                        "enum": ["procedure", "preference", "event", "profile", ""],
                        "description": "记忆类型，留空由 engine 决定",
                        "default": "",
                    },
                    "tool_requirement": {
                        "type": "string",
                        "description": "该 procedure 规则要求必须调用的工具名（可选）",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "procedure 执行步骤（可选）",
                    },
                },
                "required": ["summary"],
            },
            risk="write",
            search_hint="记住 记下 保存偏好 长期规则 以后都这样",
        ),
        forget=MemoryToolSpec(
            description="将已确认错误的记忆条目标记为失效（supersede，可恢复，不物理删除）。",
            parameters={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要失效的 memory item id 列表",
                    }
                },
                "required": ["ids"],
            },
            risk="write",
            search_hint="记错了 删除记忆 撤销错误记忆 失效记忆 忘掉",
        ),
    )