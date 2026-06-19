from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import cast

from raven_agent.memory2.embedder import EmbeddingProvider
from raven_agent.memory2.store import MemoryStore2
from raven_agent.memory2.tokenizer import extract_terms

logger = logging.getLogger(__name__)

_RRF_K = 60
_KEYWORD_RRF_WEIGHT = 0.5
_KEYWORD_LIMIT_FLOOR = 30
_KEYWORD_LIMIT_MULTIPLIER = 2
_EMBED_TIMEOUT_S = 8.0


class Retriever:
    """Memory2 多路检索器。

    参数:
        store: MemoryStore2，负责 SQLite / sqlite-vec / FTS5 检索。
        embedder: EmbeddingProvider，负责把查询文本转成向量。
        top_k: 默认返回条数。
        score_threshold: 默认向量相似度阈值。
        score_thresholds: 按 memory_type 定制阈值。
        inject_max_chars: 注入块最大字符数。
        inject_max_forced: 强制约束最多注入几条。
        inject_max_procedure_preference: procedure/preference 最多注入几条。
        inject_max_event_profile: event/profile 最多注入几条。
        high_inject_delta: 分数低于阈值附近时标注不确定。
        hotness_alpha: 热度融合权重。
        hotness_half_life_days: 热度半衰期。

    返回:
        Retriever 实例。
    """

    def __init__(
        self,
        *,
        store: MemoryStore2,
        embedder: EmbeddingProvider,
        top_k: int = 8,
        score_threshold: float = 0.35,
        score_thresholds: dict[str, float] | None = None,
        inject_max_chars: int = 1200,
        inject_max_forced: int = 3,
        inject_max_procedure_preference: int = 4,
        inject_max_event_profile: int = 2,
        high_inject_delta: float = 0.15,
        hotness_alpha: float = 0.20,
        hotness_half_life_days: float = 14.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = max(1, int(top_k))
        self._score_threshold = float(score_threshold)
        thresholds = score_thresholds or {}
        self._score_thresholds = {
            "procedure": float(thresholds.get("procedure", score_threshold)),
            "preference": float(thresholds.get("preference", score_threshold)),
            "event": float(thresholds.get("event", score_threshold)),
            "profile": float(thresholds.get("profile", score_threshold)),
        }
        self._inject_max_chars = max(200, int(inject_max_chars))
        self._inject_max_forced = max(1, int(inject_max_forced))
        self._inject_max_procedure_preference = max(1, int(inject_max_procedure_preference))
        self._inject_max_event_profile = max(0, int(inject_max_event_profile))
        self._high_inject_delta = max(0.0, float(high_inject_delta))
        self._hotness_alpha = max(0.0, min(1.0, float(hotness_alpha)))
        self._hotness_half_life_days = max(1.0, float(hotness_half_life_days))

    async def retrieve(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        aux_queries: list[str] | None = None,
        score_threshold: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        keyword_enabled: bool = True,
    ) -> list[dict[str, object]]:
        """执行 vector lane、keyword lane 并用 RRF 融合。

        参数:
            query: 原始查询文本。
            memory_types: 可选记忆类型过滤。
            top_k: 最多返回多少条；None 使用默认值。
            aux_queries: HyDE 或 rewrite 生成的辅助查询。
            score_threshold: 向量相似度阈值；None 使用默认值。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。
            keyword_enabled: 是否启用关键词 lane。

        返回:
            融合后的 hit 字典列表。
        """

        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        actual_threshold = self._score_threshold if score_threshold is None else float(score_threshold)
        query_texts = _dedupe_texts([query, *(aux_queries or [])])
        vector_items = await self._retrieve_vector_lanes(
            query_texts,
            actual_top_k=actual_top_k,
            memory_types=memory_types,
            score_threshold=actual_threshold,
            time_start=time_start,
            time_end=time_end,
        )
        keyword_items: list[dict[str, object]] = []
        if keyword_enabled:
            keyword_items = self._retrieve_keyword_lane(
                query,
                actual_top_k=actual_top_k,
                memory_types=memory_types,
                time_start=time_start,
                time_end=time_end,
            )
        return _rrf_merge(vector_items, keyword_items, top_n=actual_top_k)

    async def embed(self, query: str) -> list[float]:
        """只执行 embedding，不检索。

        参数:
            query: 查询文本。

        返回:
            查询向量；失败时返回空列表。
        """

        try:
            return await asyncio.wait_for(self._embedder.embed_text(query), timeout=_EMBED_TIMEOUT_S)
        except Exception as exc:
            logger.warning("memory2 embed failed: %s", exc)
            return []

    async def retrieve_with_embedding(
        self,
        query_embedding: list[float],
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        """复用已有 query embedding 执行向量检索。

        参数:
            query_embedding: 已生成的查询向量。
            memory_types: 可选记忆类型过滤。
            top_k: 最多返回多少条。
            score_threshold: 最低相似度。

        返回:
            hit 字典列表。
        """

        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        actual_threshold = self._score_threshold if score_threshold is None else float(score_threshold)
        return self._store.vector_search(
            query_embedding=query_embedding,
            top_k=actual_top_k,
            memory_types=memory_types,
            score_threshold=actual_threshold,
            hotness_alpha=self._hotness_alpha,
            hotness_half_life_days=self._hotness_half_life_days,
        )

    async def _retrieve_vector_lanes(
        self,
        query_texts: list[str],
        *,
        actual_top_k: int,
        memory_types: list[str] | None,
        score_threshold: float,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[dict[str, object]]:
        """对 raw query 和 aux queries 执行向量检索。

        参数:
            query_texts: 查询文本列表。
            actual_top_k: 每路最多返回多少条。
            memory_types: 可选记忆类型过滤。
            score_threshold: 最低相似度。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            去重后的 vector hits。
        """

        if not query_texts:
            return []
        vectors = await asyncio.gather(*(self.embed(text) for text in query_texts))
        clean_vectors = [vector for vector in vectors if vector]
        if not clean_vectors:
            return []
        groups = self._store.vector_search_batch(
            clean_vectors,
            top_k=actual_top_k,
            memory_types=memory_types,
            score_threshold=score_threshold,
            hotness_alpha=self._hotness_alpha,
            hotness_half_life_days=self._hotness_half_life_days,
            time_start=time_start,
            time_end=time_end,
        )
        seen: dict[str, dict[str, object]] = {}
        for hits in groups:
            for hit in hits:
                _remember_better_hit(seen, hit)
        return list(seen.values())

    def _retrieve_keyword_lane(
        self,
        query: str,
        *,
        actual_top_k: int,
        memory_types: list[str] | None,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[dict[str, object]]:
        """执行关键词检索 lane。

        参数:
            query: 原始查询文本。
            actual_top_k: 最终 top_k。
            memory_types: 可选记忆类型过滤。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            keyword hits。
        """

        terms = _extract_terms(query)
        if not terms:
            return []
        return self._store.keyword_search_summary(
            terms,
            memory_types=memory_types,
            limit=max(_KEYWORD_LIMIT_FLOOR, actual_top_k * _KEYWORD_LIMIT_MULTIPLIER),
            time_start=time_start,
            time_end=time_end,
        )

    def build_injection_block(self, items: list[dict[str, object]]) -> tuple[str, list[str]]:
        """把检索结果渲染成 prompt 注入块。

        参数:
            items: Retriever 返回的 hit 字典列表。

        返回:
            二元组：(注入文本块, 实际注入的 memory ids)。
        """

        selected, forced, norms, events = self._select_injection_sections(items)
        if not selected:
            return "", []
        parts: list[tuple[str, list[str]]] = []
        if forced:
            parts.append(("## 【强制约束】记忆规则\n" + "\n".join(line for _, line in forced), [item_id for item_id, _ in forced]))
        if norms:
            parts.append(("## 【流程规范】用户偏好与规则\n" + "\n".join(line for _, line in norms), [item_id for item_id, _ in norms]))
        if events:
            parts.append(("## 【相关历史】长期记忆检索结果\n" + "\n".join(line for _, line in events), [item_id for item_id, _ in events]))
        return _apply_char_budget(parts, max_chars=self._inject_max_chars)

    def _select_injection_sections(
        self,
        items: list[dict[str, object]],
    ) -> tuple[
        list[dict[str, object]],
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[tuple[str, str]],
    ]:
        """选择适合 prompt 注入的条目并分段。

        参数:
            items: 检索 hit 列表。

        返回:
            四元组：(selected, forced lines, norm lines, event lines)。
        """

        sorted_items = sorted(items, key=_hit_score, reverse=True)
        selected: list[dict[str, object]] = []
        forced: list[tuple[str, str]] = []
        norms: list[tuple[str, str]] = []
        events: list[tuple[str, str]] = []
        forced_count = 0
        norm_count = 0
        event_count = 0
        for item in sorted_items:
            memory_type = str(item.get("memory_type", "") or "")
            item_id = str(item.get("id", "") or "")
            summary = str(item.get("summary", "") or "").strip()
            if not item_id or not summary:
                continue
            extra = item.get("extra_json") if isinstance(item.get("extra_json"), dict) else {}
            score = _hit_score(item)
            if memory_type == "procedure" and cast(dict[str, object], extra).get("tool_requirement"):
                if forced_count >= self._inject_max_forced:
                    continue
                forced_count += 1
                selected.append(item)
                forced.append((item_id, f"- [{item_id}] {summary}（必须调用工具：{extra.get('tool_requirement')}）"))
                continue
            threshold = self._score_thresholds.get(memory_type, self._score_threshold)
            if score < threshold:
                continue
            confidence = ""
            if score < threshold + self._high_inject_delta:
                confidence = "；有印象，不确定"
            if memory_type in {"procedure", "preference"}:
                if norm_count >= self._inject_max_procedure_preference:
                    continue
                norm_count += 1
                selected.append(item)
                norms.append((item_id, f"- [{item_id}] {summary}{_format_source_meta(item)}{confidence}"))
            elif memory_type in {"event", "profile"}:
                if event_count >= self._inject_max_event_profile:
                    continue
                event_count += 1
                selected.append(item)
                prefix = f"[{item.get('happened_at')}] " if item.get("happened_at") else ""
                events.append((item_id, f"- [{item_id}] {prefix}{summary}{_format_source_meta(item)}{confidence}"))
        return selected, forced, norms, events


def _dedupe_texts(texts: list[str]) -> list[str]:
    """按原顺序去重查询文本。

    参数:
        texts: 原始查询文本列表。

    返回:
        去重后的非空文本列表。
    """

    seen: set[str] = set()
    result: list[str] = []
    for text in texts:
        normalized = str(text or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _hit_id(item: dict[str, object]) -> str:
    """读取 hit id。

    参数:
        item: hit 字典。

    返回:
        hit id；缺失时返回空字符串。
    """

    return str(item.get("id", "") or "")


def _hit_score(item: dict[str, object]) -> float:
    """读取 hit 排序分数。

    参数:
        item: hit 字典。

    返回:
        分数；缺失时返回 0.0。
    """

    raw = item.get("score")
    if isinstance(raw, int | float):
        return float(raw)
    raw_keyword = item.get("keyword_score")
    return float(raw_keyword) if isinstance(raw_keyword, int | float) else 0.0


def _remember_better_hit(seen: dict[str, dict[str, object]], hit: dict[str, object]) -> None:
    """按 id 保留分数更高的 hit。

    参数:
        seen: 已收集 hit 的字典。
        hit: 新 hit。

    返回:
        None。
    """

    item_id = _hit_id(hit)
    if not item_id:
        return
    if item_id not in seen or _hit_score(hit) > _hit_score(seen[item_id]):
        seen[item_id] = hit



def _extract_terms(query: str) -> list[str]:
    """从 query 中提取关键词。

    参数:
        query: 原始查询文本。

    返回:
        最多 20 个关键词。
    """

    return extract_terms(query)


def _rrf_merge(
    vector_items: list[dict[str, object]],
    keyword_items: list[dict[str, object]],
    *,
    top_n: int,
    k: int = _RRF_K,
) -> list[dict[str, object]]:
    """用 Reciprocal Rank Fusion 融合 vector 和 keyword 结果。

    参数:
        vector_items: 向量检索结果。
        keyword_items: 关键词检索结果。
        top_n: 最多返回多少条。
        k: RRF 平滑常数。

    返回:
        融合后的 hit 列表。
    """

    vector_rank: dict[str, int] = {}
    for index, item in enumerate(sorted(vector_items, key=_hit_score, reverse=True)):
        item_id = _hit_id(item)
        if item_id and item_id not in vector_rank:
            vector_rank[item_id] = index + 1

    keyword_rank: dict[str, int] = {}
    for index, item in enumerate(keyword_items):
        item_id = _hit_id(item)
        if item_id and item_id not in keyword_rank:
            keyword_rank[item_id] = index + 1

    id_to_item: dict[str, dict[str, object]] = {}
    for item in keyword_items:
        item_id = _hit_id(item)
        if item_id:
            id_to_item[item_id] = dict(item)
    for item in vector_items:
        item_id = _hit_id(item)
        if item_id:
            id_to_item[item_id] = dict(item)

    scored: list[tuple[str, float, float]] = []
    for item_id in set(vector_rank) | set(keyword_rank):
        rrf_score = 0.0
        if item_id in vector_rank:
            rrf_score += 1.0 / (k + vector_rank[item_id])
        if item_id in keyword_rank:
            rrf_score += _KEYWORD_RRF_WEIGHT / (k + keyword_rank[item_id])
        scored.append((item_id, rrf_score, _hit_score(id_to_item[item_id])))

    scored.sort(key=lambda value: (value[1], value[2]), reverse=True)
    result: list[dict[str, object]] = []
    for item_id, rrf_score, _score in scored[: max(1, int(top_n))]:
        item = dict(id_to_item[item_id])
        item["rrf_score"] = rrf_score
        result.append(item)
    return result


def _format_source_meta(item: dict[str, object]) -> str:
    """为注入行格式化来源摘要。

    参数:
        item: hit 字典。

    返回:
        形如 （证据: xxx） 的短文本；无来源时返回空字符串。
    """

    source_ref = str(item.get("source_ref", "") or "").strip()
    if not source_ref:
        return ""
    raw = source_ref.split("#", 1)[0]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        ids = [str(value) for value in parsed if str(value).strip()]
    else:
        ids = [raw]
    if not ids:
        return ""
    return f"（证据: {', '.join(ids[:2])}）"


def _apply_char_budget(
    parts: list[tuple[str, list[str]]],
    *,
    max_chars: int,
) -> tuple[str, list[str]]:
    """按字符预算拼接注入块。

    参数:
        parts: 分段文本和对应 item ids。
        max_chars: 最大字符数。

    返回:
        二元组：(文本块, 注入 ids)。
    """

    final_parts: list[str] = []
    injected_ids: list[str] = []
    seen_ids: set[str] = set()
    total = 0
    for part, part_ids in parts:
        add_len = len(part) + (2 if final_parts else 0)
        if final_parts and total + add_len > max_chars:
            continue
        final_parts.append(part)
        total += add_len
        for item_id in part_ids:
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                injected_ids.append(item_id)
    return "\n\n".join(final_parts), injected_ids