from __future__ import annotations

import logging
import re
from typing import Any

from raven_agent.memory2.embedder import EmbeddingProvider
from raven_agent.memory2.models import content_hash
from raven_agent.memory2.procedure_tagger import ProcedureTagger
from raven_agent.memory2.rule_schema import build_procedure_rule_schema
from raven_agent.memory2.store import MemoryStore2

logger = logging.getLogger(__name__)

_TIME_PREFIX_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?\]"
)


class Memorizer:
    """Memory2 结构化写入器。

    参数:
        store: MemoryStore2，用于持久化 memory items。
        embedder: EmbeddingProvider，用于生成 summary embedding。
        procedure_tagger: 可选 ProcedureTagger，用于给 procedure 添加 trigger_tags。

    返回:
        Memorizer 实例。
    """

    def __init__(
        self,
        *,
        store: MemoryStore2,
        embedder: EmbeddingProvider,
        procedure_tagger: ProcedureTagger | None = None,
        event_bus: object | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._procedure_tagger = procedure_tagger
        self._event_bus = event_bus

    async def save_item(
        self,
        *,
        summary: str,
        memory_type: str,
        extra_json: dict[str, object] | None,
        source_ref: str | None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """直接写入或强化一条 memory item。

        参数:
            summary: 记忆摘要。
            memory_type: 记忆类型。
            extra_json: 类型专用扩展字段。
            source_ref: 来源引用。
            happened_at: 事件发生时间。
            emotional_weight: 情绪权重。

        返回:
            store 写入结果，例如 new:{id} 或 reinforced:{id}。
        """

        clean_summary = summary.strip()
        if not clean_summary:
            return "skipped:empty"
        enriched_extra = await self._enrich_extra(
            summary=clean_summary,
            memory_type=memory_type,
            extra_json=extra_json or {},
        )
        embedding = await self._embedder.embed_text(clean_summary)
        return self._store.upsert_item(
            memory_type=memory_type,
            summary=clean_summary,
            embedding=embedding or None,
            source_ref=source_ref,
            extra_json=enriched_extra,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def save_item_with_supersede(
        self,
        *,
        summary: str,
        memory_type: str,
        extra_json: dict[str, object] | None,
        source_ref: str | None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
        merge_threshold: float = 0.70,
        supersede_threshold: float = 0.90,
    ) -> str:
        """带语义去重和 supersede 策略地写入 memory item。

        参数:
            summary: 记忆摘要。
            memory_type: 记忆类型。
            extra_json: 类型专用扩展字段。
            source_ref: 来源引用。
            happened_at: 事件发生时间。
            emotional_weight: 情绪权重。
            merge_threshold: procedure merge 最低相似度。
            supersede_threshold: 旧条目 supersede 最低相似度。

        返回:
            写入结果，例如 new:{id}、reinforced:{id} 或 merged:{id}。
        """

        clean_summary = summary.strip()
        if not clean_summary:
            return "skipped:empty"
        enriched_extra = await self._enrich_extra(
            summary=clean_summary,
            memory_type=memory_type,
            extra_json=extra_json or {},
        )
        embedding = await self._embedder.embed_text(clean_summary)

        if memory_type in {"procedure", "preference"} and embedding:
            similar = self._store.vector_search(
                query_embedding=embedding,
                top_k=5,
                memory_types=[memory_type],
                score_threshold=min(merge_threshold, supersede_threshold),
            )
            if memory_type == "procedure":
                merge_target = self._pick_explicit_merge_target(similar, enriched_extra, merge_threshold)
                if merge_target is not None:
                    merged_summary = self._merge_summary_text(
                        str(merge_target.get("summary", "")),
                        clean_summary,
                    )
                    await self.merge_item(
                        str(merge_target["id"]),
                        merged_summary,
                        extra_patch=enriched_extra,
                    )
                    return f"merged:{merge_target['id']}"
            supersede_ids = [
                str(item["id"])
                for item in similar
                if isinstance(item.get("score"), int | float)
                and float(item["score"]) >= supersede_threshold
            ]
            if supersede_ids:
                self._store.mark_superseded_batch(supersede_ids)

        elif memory_type == "profile" and embedding:
            category = str((enriched_extra or {}).get("category") or "")
            if category in {"status", "purchase"}:
                similar = self._store.vector_search(
                    query_embedding=embedding,
                    top_k=5,
                    memory_types=["profile"],
                    score_threshold=supersede_threshold,
                )
                same_category_ids = [
                    str(item["id"])
                    for item in similar
                    if isinstance(item.get("extra_json"), dict)
                    and item["extra_json"].get("category") == category
                    and isinstance(item.get("score"), int | float)
                    and float(item["score"]) >= supersede_threshold
                ]
                if same_category_ids:
                    self._store.mark_superseded_batch(same_category_ids)

        return self._store.upsert_item(
            memory_type=memory_type,
            summary=clean_summary,
            embedding=embedding or None,
            source_ref=source_ref,
            extra_json=enriched_extra,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def save_from_consolidation(
        self,
        *,
        history_entry: str,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> str:
        """把 Markdown consolidation 产生的 history entry 写入 Memory2 event。

        参数:
            history_entry: HISTORY.md 风格事件摘要。
            source_ref: JSON message id list 来源引用。
            scope_channel: 消息渠道。
            scope_chat_id: 渠道内聊天标识。
            emotional_weight: 情绪权重。

        返回:
            写入结果；语义去重时返回 semantic_dedup:{id}。
        """

        text = history_entry.strip()
        if not text:
            return "skipped:empty"
        if self._store.has_consolidation_source_ref(source_ref):
            return f"skipped:{source_ref}"

        embedding = await self._embedder.embed_text(text)
        if embedding and self._should_semantic_dedup_event(embedding, emotional_weight=emotional_weight):
            similar_ids = self._store.find_similar_recent_events(embedding, threshold=0.92, days_back=7)
            item_id = similar_ids[0] if similar_ids else ""
            return f"semantic_dedup:{item_id}"

        item_id = self._store.upsert_consolidation_event(
            source_ref=source_ref,
            summary=text,
            embedding=embedding or None,
            extra_json={
                "scope_channel": scope_channel,
                "scope_chat_id": scope_chat_id,
            },
            happened_at=parse_history_entry_happened_at(text),
            emotional_weight=emotional_weight,
        )

        # ── observe: emit 记忆写入事件 ──
        if self._event_bus is not None:
            from raven_agent.plugins.builtins.observe.bridge import (
                emit_memory_write_event,
            )
            try:
                session_key = f"{scope_channel}:{scope_chat_id}"
                emit_memory_write_event(
                    self._event_bus,
                    session_key=session_key,
                    source_ref=source_ref,
                    action="write",
                    memory_type="event",
                    item_id=item_id,
                    summary=text,
                )
            except Exception:
                pass  # observe 失败不应影响记忆写入

        return item_id

    def _should_semantic_dedup_event(
        self,
        embedding: list[float],
        *,
        emotional_weight: int = 0,
    ) -> bool:
        """判断新 event 是否与近期 event 语义重复。

        参数:
            embedding: 新 event embedding。
            emotional_weight: 新 event 情绪权重。

        返回:
            重复并已强化旧 event 时返回 True，否则返回 False。
        """

        similar_ids = self._store.find_similar_recent_events(
            embedding,
            threshold=0.92,
            days_back=7,
        )
        if not similar_ids:
            return False
        self._store.reinforce_items_batch(similar_ids[:1], emotional_weight=emotional_weight)
        return True

    def supersede_batch(self, ids: list[str]) -> None:
        """批量 supersede memory items。

        参数:
            ids: 要标记为 superseded 的 item ids。

        返回:
            None。
        """

        self._store.mark_superseded_batch(ids)

    def reinforce_items_batch(self, ids: list[str]) -> None:
        """批量强化 memory items。

        参数:
            ids: 要强化的 item ids。

        返回:
            None。
        """

        self._store.reinforce_items_batch(ids)

    async def merge_item(
        self,
        item_id: str,
        merged_summary: str,
        *,
        extra_patch: dict[str, object] | None = None,
    ) -> None:
        """原地合并更新一条 memory item。

        参数:
            item_id: 目标 memory item id。
            merged_summary: 合并后的 summary。
            extra_patch: 新写入候选带来的 extra_json 补丁。

        返回:
            None。
        """

        item = self._store.get_item(item_id)
        if item is None:
            return
        new_extra = dict(item.extra_json)
        new_extra["_merge_note"] = merged_summary.strip()
        if extra_patch:
            if extra_patch.get("tool_requirement"):
                new_extra["tool_requirement"] = extra_patch.get("tool_requirement")
            if extra_patch.get("steps"):
                new_extra["steps"] = _merge_steps(item.extra_json.get("steps"), extra_patch.get("steps"))

        if item.memory_type == "procedure":
            new_extra["rule_schema"] = build_procedure_rule_schema(
                summary=merged_summary,
                tool_requirement=str(new_extra.get("tool_requirement") or "") or None,
                steps=[str(step) for step in new_extra.get("steps", [])] if isinstance(new_extra.get("steps"), list) else [],
                rule_schema=new_extra.get("rule_schema") if isinstance(new_extra.get("rule_schema"), dict) else None,
            )
            new_extra.pop("trigger_tags", None)
            if self._procedure_tagger is not None:
                trigger_tags = await self._procedure_tagger.tag(merged_summary)
                if trigger_tags is not None:
                    new_extra["trigger_tags"] = trigger_tags

        embedding = await self._embedder.embed_text(merged_summary)
        self._store.merge_item_raw(
            item_id=item_id,
            new_summary=merged_summary,
            new_hash=content_hash(merged_summary, item.memory_type),
            new_embedding=embedding or [],
            new_extra_json=new_extra,
        )

    async def _enrich_extra(
        self,
        *,
        summary: str,
        memory_type: str,
        extra_json: dict[str, object],
    ) -> dict[str, object]:
        """按 memory_type 补全 extra_json。

        参数:
            summary: 记忆摘要。
            memory_type: 记忆类型。
            extra_json: 原始 extra_json。

        返回:
            补全后的 extra_json。
        """

        extra = dict(extra_json)
        if memory_type == "procedure":
            steps = [str(step) for step in extra.get("steps", [])] if isinstance(extra.get("steps"), list) else []
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=summary,
                tool_requirement=str(extra.get("tool_requirement") or "") or None,
                steps=steps,
                rule_schema=extra.get("rule_schema") if isinstance(extra.get("rule_schema"), dict) else None,
            )
            if "trigger_tags" not in extra and self._procedure_tagger is not None:
                trigger_tags = await self._procedure_tagger.tag(summary)
                if trigger_tags is not None:
                    extra["trigger_tags"] = trigger_tags
        return extra

    @staticmethod
    def _merge_summary_text(old_summary: str, new_summary: str) -> str:
        """合并两段 summary 文本。

        参数:
            old_summary: 旧 summary。
            new_summary: 新 summary。

        返回:
            合并后的 summary。
        """

        old = old_summary.strip()
        new = new_summary.strip()
        if not old:
            return new
        if not new:
            return old
        if new in old:
            return old
        if old in new:
            return new
        return f"{old.rstrip('。；;，, ')}；{new}"

    @staticmethod
    def _pick_explicit_merge_target(
        similar: list[dict[str, object]],
        extra: dict[str, object],
        merge_threshold: float,
    ) -> dict[str, object] | None:
        """选择同 tool_requirement 的 procedure merge 目标。

        参数:
            similar: 相似 procedure hits。
            extra: 新 procedure extra_json。
            merge_threshold: merge 最低相似度。

        返回:
            merge 目标 hit；没有合适目标时返回 None。
        """

        wanted_tool = str(extra.get("tool_requirement") or "").strip()
        if not wanted_tool:
            return None
        for item in similar:
            if float(item.get("score", 0.0)) < merge_threshold:
                continue
            item_extra = item.get("extra_json")
            if not isinstance(item_extra, dict):
                continue
            item_tool = str(item_extra.get("tool_requirement") or "").strip()
            if item_tool == wanted_tool:
                return item
        return None


def parse_history_entry_happened_at(summary: str) -> str | None:
    """从 HISTORY entry 前缀解析 happened_at。

    参数:
        summary: 形如 [YYYY-MM-DD HH:MM] 事件摘要 的文本。

    返回:
        ISO 风格时间字符串；无前缀时返回 None。
    """

    match = _TIME_PREFIX_RE.match((summary or "").strip())
    if not match:
        return None
    date = match.group("date")
    hour = match.group("hour") or "00"
    minute = match.group("minute") or "00"
    second = match.group("second") or "00"
    return f"{date}T{hour}:{minute}:{second}"


def _merge_steps(old_steps: object, new_steps: object) -> list[str]:
    """合并 procedure steps。

    参数:
        old_steps: 旧 steps 字段。
        new_steps: 新 steps 字段。

    返回:
        去重后的步骤列表。
    """

    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*(old_steps if isinstance(old_steps, list) else []), *(new_steps if isinstance(new_steps, list) else [])]:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged