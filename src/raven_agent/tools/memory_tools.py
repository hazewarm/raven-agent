from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from raven_agent.memory import (
    EvidenceRef,
    MemoryMutation,
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryIntent,
    MemoryRecord,
    MemoryScope,
    MemoryToolSpec,
)
from raven_agent.memory.engine import MemoryRetrievalApi, MemoryWriteApi
from raven_agent.tools.base import Tool

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_RECENT_PRESETS = {"recent_3d": 3, "recent_7d": 7, "recent_30d": 30}


class MemorizeTool(Tool):
    """让模型把信息写入长期记忆的工具。

    参数:
        memory: 实现 MemoryWriteApi 的记忆引擎，用于执行 remember mutation。
        spec: MemoryToolSpec，提供工具描述与参数 schema。

    返回:
        MemorizeTool 实例。
    """

    name = "memorize"
    description = "由 memory engine 的 tool_profile 注入。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def __init__(self, memory: MemoryWriteApi, spec: MemoryToolSpec, event_bus: object | None = None,) -> None:
        self._memory = memory
        self._spec = spec
        self.description = spec.description
        self.parameters = spec.parameters
        self._event_bus = event_bus

    async def execute(
        self,
        summary: str,
        memory_kind: str = "",
        tool_requirement: str | None = None,
        steps: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        current_user_source_ref: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **extra_kwargs: Any,
    ) -> str:
        """写入一条显式记忆。

        参数:
            summary: 要记住的内容（一句话）。
            memory_kind: 建议记忆类型；留空由 engine 决定。
            tool_requirement: procedure 要求的必需工具名（可选）。
            steps: procedure 执行步骤（可选）。
            metadata: 额外 extra_json 字段（可选）。
            current_user_source_ref: 当前用户消息来源引用，由运行时注入。
            channel: 当前渠道，由运行时注入。
            chat_id: 当前聊天标识，由运行时注入。
            **extra_kwargs: 预留扩展参数，会并入 metadata。

        返回:
            人类可读的写入结果字符串。
        """

        clean_summary = str(summary or "").strip()
        if not clean_summary:
            return "未记住：summary 为空。"

        extra = dict(metadata or {})
        extra.update(extra_kwargs)
        if tool_requirement is not None:
            extra["tool_requirement"] = tool_requirement
        if steps is not None:
            extra["steps"] = steps

        result = await self._memory.mutate(
            MemoryMutation(
                kind="remember",
                summary=clean_summary,
                memory_kind=str(memory_kind or "").strip(),
                source_ref=str(current_user_source_ref or "").strip(),
                scope=_build_scope(channel, chat_id),
                metadata=extra,
            )
        )
        # ── observe: emit 记忆写入事件 ──
        if self._event_bus is not None:
            from raven_agent.plugins.builtins.observe.bridge import (
                emit_memory_write_event,
            )
            try:
                session_key = f"{channel}:{chat_id}" if (channel and chat_id) else ""
                await emit_memory_write_event(
                    self._event_bus,
                    session_key=session_key,
                    source_ref=str(current_user_source_ref or "").strip(),
                    action="write",
                    memory_type=str(result.actual_kind or "").strip() or None,
                    item_id=str(result.item_id or "").strip(),
                    summary=clean_summary,
                )
            except Exception:
                pass  # observe 失败不应影响 Agent 主循环
        return _format_memorize_result(result.item_id, result.status, result.actual_kind, clean_summary)


class RecallMemoryTool(Tool):
    """让模型检索长期记忆的工具。

    参数:
        memory: 实现 MemoryRetrievalApi 的记忆引擎，用于执行 query。
        spec: MemoryToolSpec，提供工具描述与参数 schema。

    返回:
        RecallMemoryTool 实例。
    """

    name = "recall_memory"
    description = "由 memory engine 的 tool_profile 注入。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self, memory: MemoryRetrievalApi, spec: MemoryToolSpec, event_bus: object | None = None,) -> None:
        self._memory = memory
        self._spec = spec
        self.description = spec.description
        self.parameters = spec.parameters
        self._event_bus = event_bus

    async def execute(
        self,
        query: str,
        intent: str = "answer",
        memory_kind: str = "",
        time_filter: str = "",
        limit: int = 8,
        channel: str | None = None,
        chat_id: str | None = None,
        **extra: Any,
    ) -> str:
        """检索长期记忆并返回带证据的结构化结果。

        参数:
            query: 检索主题，推荐写成陈述句。
            intent: answer 或 timeline。
            memory_kind: 限定记忆类型，留空不限。
            time_filter: 时间过滤预设或日期/区间。
            limit: 最多返回条数（1-200）。
            channel: 当前渠道，由运行时注入。
            chat_id: 当前聊天标识，由运行时注入。
            **extra: 预留扩展参数，并入 query context。

        返回:
            JSON 字符串，包含 count / items / trace / citation 协议。
        """

        text = str(query or "").strip()
        if not text:
            return _render_records([], trace={})

        time_window = _parse_time_filter(time_filter)
        if time_filter and time_window is None:
            return json.dumps(
                {"count": 0, "items": [], "error": "invalid_time_filter"},
                ensure_ascii=False,
            )

        result = await self._memory.query(
            MemoryQuery(
                text=text,
                intent=_normalize_intent(intent),
                scope=_build_scope(channel, chat_id),
                filters=MemoryQueryFilters(
                    kinds=_memory_kinds(memory_kind),
                    time_start=time_window[0] if time_window else None,
                    time_end=time_window[1] if time_window else None,
                ),
                limit=max(1, min(int(limit), 200)),
                context=dict(extra),
            )
        )
        # ── observe: emit 检索完成事件 ──
        if self._event_bus is not None:
            from raven_agent.plugins.builtins.observe.bridge import emit_retrieval_event
            try:
                hits_list = [
                    {
                        "item_id": rec.item_id,
                        "memory_type": rec.memory_type,
                        "score": rec.score,
                        "summary": getattr(rec, "summary", ""),
                        "injected": True,  # recall_memory 工具的结果全部注入
                        "forced": False,
                    }
                    for rec in (result.records if result else [])
                ]
                # session_key 可从 extra 参数或工具上下文获取
                sk = str(extra.get("session_key", "")) if extra else ""
                await emit_retrieval_event(
                    self._event_bus,
                    caller="explicit",     # recall_memory 是显式工具调用
                    session_key=sk,
                    query=text,
                    hits=hits_list,
                    injected_count=len(hits_list),
                )
            except Exception:
                pass  # observe 失败不应影响 Agent 主循环
        return _render_records(result.records, trace=result.trace)


class ForgetMemoryTool(Tool):
    """让模型把错误记忆标记为失效的工具。

    参数:
        memory: 实现 MemoryWriteApi 的记忆引擎，用于执行 forget mutation。
        spec: MemoryToolSpec，提供工具描述与参数 schema。

    返回:
        ForgetMemoryTool 实例。
    """

    name = "forget_memory"
    description = "由 memory engine 的 tool_profile 注入。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["ids"],
    }

    def __init__(
        self,
        memory: MemoryWriteApi,
        spec: MemoryToolSpec,
        event_bus: object | None = None,
    ) -> None:
        self._memory = memory
        self._spec = spec
        self._event_bus = event_bus
        self.description = spec.description
        self.parameters = spec.parameters

    async def execute(self, ids: list[str], **_: Any) -> str:
        """把指定 memory id 标记为 superseded。

        参数:
            ids: 要失效的 memory item id 列表。
            **_: 忽略运行时注入的其他上下文。

        返回:
            JSON 字符串，包含 requested / superseded / missing 报告。
        """

        clean_ids = _clean_ids(ids)
        if not clean_ids:
            return _render_forget_result([], [], [], [])

        result = await self._memory.mutate(
            MemoryMutation(kind="forget", ids=tuple(clean_ids))
        )
        return _render_forget_result(
            clean_ids,
            result.affected_ids,
            result.missing_ids,
            result.items,
        )


def _build_scope(channel: str | None, chat_id: str | None) -> MemoryScope:
    """根据 channel / chat_id 构造 MemoryScope。

    参数:
        channel: 渠道名；可能为 None。
        chat_id: 聊天标识；可能为 None。

    返回:
        MemoryScope；channel 和 chat_id 都有时填充 session_key。
    """

    ch = str(channel or "").strip()
    cid = str(chat_id or "").strip()
    return MemoryScope(
        session_key=f"{ch}:{cid}" if ch and cid else "",
        channel=ch,
        chat_id=cid,
    )


def _format_memorize_result(item_id: str, status: str, actual_kind: str, summary: str) -> str:
    """格式化 memorize 写入结果。

    参数:
        item_id: 写入或命中的 memory id。
        status: 写入状态，例如 new / reinforced / merged。
        actual_kind: engine 最终采用的记忆类型。
        summary: 写入的内容摘要。

    返回:
        人类可读结果字符串。
    """

    value = (item_id or "").strip()
    write_status = (status or "new").strip()
    kind = (actual_kind or "").strip()
    if kind:
        return f"已记住（item_id={value}；kind={kind}；status={write_status}）：{summary}"
    return f"已记住（item_id={value}；status={write_status}）：{summary}"


def _render_records(records: list[MemoryRecord], *, trace: dict[str, object]) -> str:
    """把检索记录渲染成带 citation 协议的 JSON。

    参数:
        records: 查询命中的 MemoryRecord 列表。
        trace: engine 查询 trace。

    返回:
        JSON 字符串。
    """

    items: list[dict[str, object]] = []
    for record in records:
        evidence = _render_evidence(record.evidence)
        source_ref = _first_source_ref(evidence)
        item: dict[str, object] = {
            "id": record.id,
            "memory_type": record.kind,
            "summary": record.summary,
            "score": round(record.score, 4),
            "evidence": evidence,
            "signals": record.signals,
        }
        if source_ref:
            item["source_ref"] = source_ref
        items.append(item)
    cited_item_ids = [str(item["id"]) for item in items if str(item.get("id", "")).strip()]
    return json.dumps(
        {
            "count": len(items),
            "items": items,
            "trace": trace,
            "citation_required": True,
            "citation_format": "§cited:[id1,id2,...]§",
            "cited_item_ids": cited_item_ids,
            "citation_rule": (
                "若最终回复使用了本工具返回的任何记忆条目，"
                "必须在正文末尾输出 §cited:[实际使用的id列表]§"
            ),
        },
        ensure_ascii=False,
    )


def _render_evidence(evidence: list[EvidenceRef]) -> list[dict[str, object]]:
    """把 EvidenceRef 列表渲染成可序列化字典。

    参数:
        evidence: MemoryRecord 的证据引用列表。

    返回:
        字典列表。
    """

    return [
        {
            "kind": item.kind,
            "refs": item.refs,
            "resolver": item.resolver,
            "source_ref": item.source_ref,
            "metadata": item.metadata,
        }
        for item in evidence
    ]


def _first_source_ref(evidence: list[dict[str, object]]) -> str:
    """从 evidence 列表里取第一个可用 source_ref / ref。

    参数:
        evidence: 已渲染的证据字典列表。

    返回:
        第一个非空 source_ref 或 ref；都没有时返回空字符串。
    """

    for item in evidence:
        source_ref = str(item.get("source_ref") or "").strip()
        if source_ref:
            return source_ref
        refs = item.get("refs")
        if isinstance(refs, list):
            for ref in cast(list[object], refs):
                text = str(ref).strip() if isinstance(ref, str) else ""
                if text:
                    return text
    return ""


def _render_forget_result(
    requested_ids: list[str],
    affected_ids: list[str],
    missing_ids: list[str],
    items: list[dict[str, object]],
) -> str:
    """渲染 forget 结果 JSON。

    参数:
        requested_ids: 去重后的请求 id 列表。
        affected_ids: 实际被 supersede 的 id 列表。
        missing_ids: 未找到的 id 列表。
        items: 受影响条目简表。

    返回:
        JSON 字符串。
    """

    return json.dumps(
        {
            "requested_ids": requested_ids,
            "superseded_ids": affected_ids,
            "missing_ids": missing_ids,
            "count": len(affected_ids),
            "items": items,
        },
        ensure_ascii=False,
    )


def _clean_ids(ids: list[str]) -> list[str]:
    """清洗 forget 的 id 列表。

    参数:
        ids: 原始 id 列表。

    返回:
        去空、去重、保序后的 id 列表。
    """

    clean: list[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        item_id = str(raw).strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            clean.append(item_id)
    return clean


def _normalize_intent(value: str) -> MemoryQueryIntent:
    """把工具入参 intent 归一化为合法 MemoryQueryIntent。

    参数:
        value: 模型传入的 intent 文本。

    返回:
        合法 MemoryQueryIntent；非法值降级为 answer。
    """

    intents: dict[str, MemoryQueryIntent] = {
        "context": "context",
        "answer": "answer",
        "timeline": "timeline",
        "interest": "interest",
        "procedure": "procedure",
    }
    return intents.get(str(value or "").strip(), "answer")


def _memory_kinds(memory_kind: str) -> tuple[str, ...]:
    """把 memory_kind 入参转换为 filters.kinds。

    参数:
        memory_kind: 单个记忆类型或空字符串。

    返回:
        包含该类型的单元素元组；留空时返回空元组。
    """

    value = str(memory_kind or "").strip()
    return (value,) if value else ()


def _now_local() -> datetime:
    """返回当前本地时间（Asia/Shanghai）。

    参数:
        无。

    返回:
        带时区的当前 datetime。
    """

    return datetime.now(_LOCAL_TZ)


def _parse_day(value: str) -> datetime | None:
    """把 YYYY-MM-DD 解析为当天零点 datetime。

    参数:
        value: 日期字符串。

    返回:
        datetime；解析失败时返回 None。
    """

    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=_LOCAL_TZ)
    except ValueError:
        return None


def _parse_time_filter(value: str) -> tuple[datetime, datetime] | None:
    """解析 recall 的 time_filter。

    参数:
        value: 预设词、单日或区间字符串。

    返回:
        (time_start, time_end) 二元组；空字符串返回 None；非法格式返回 None。
    """

    text = str(value or "").strip()
    if not text:
        return None

    now = _now_local()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "today":
        return today, today + timedelta(days=1)
    if text == "yesterday":
        return today - timedelta(days=1), today
    if text in _RECENT_PRESETS:
        return now - timedelta(days=_RECENT_PRESETS[text]), now

    if "~" in text:
        left, right = [part.strip() for part in text.split("~", 1)]
        start = _parse_day(left)
        end_day = _parse_day(right)
        if start is None or end_day is None:
            return None
        return start, end_day + timedelta(days=1)

    day = _parse_day(text)
    if day is None:
        return None
    return day, day + timedelta(days=1)