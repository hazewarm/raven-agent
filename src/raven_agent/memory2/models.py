from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


_ALLOWED_MEMORY_TYPES = {"event", "profile", "preference", "procedure"}
_ALLOWED_STATUS = {"active", "superseded"}


@dataclass(frozen=True)
class MemoryItem:
    """SQLite memory_items 表对应的领域对象。

    参数:
        id: memory item 的稳定 id。
        memory_type: 记忆类型，支持 event/profile/preference/procedure。
        summary: 可展示、可检索、可注入 prompt 的摘要。
        content_hash: summary 和 memory_type 归一化后的短 hash。
        embedding: 该摘要的 embedding；未生成时为 None。
        reinforcement: 条目被重复写入或使用的强化次数。
        extra_json: 类型专用扩展字段。
        source_ref: 来源引用，例如 session 窗口或手动工具来源。
        happened_at: 事件发生时间；非 event 可为空。
        status: 条目状态，active 或 superseded。
        created_at: 创建时间 ISO 字符串。
        updated_at: 更新时间 ISO 字符串。
        emotional_weight: 情绪权重，范围 0-10。

    返回:
        MemoryItem 实例。
    """

    id: str
    memory_type: str
    summary: str
    content_hash: str
    embedding: list[float] | None
    reinforcement: int
    extra_json: dict[str, object] = field(default_factory=dict)
    source_ref: str | None = None
    happened_at: str | None = None
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    emotional_weight: int = 0


def now_iso() -> str:
    """生成当前 UTC 时间的 ISO 字符串。

    参数:
        无。

    返回:
        当前时间的 ISO 字符串。
    """

    return datetime.now(timezone.utc).isoformat()


def normalize_memory_type(memory_type: str) -> str:
    """归一化 memory_type。

    参数:
        memory_type: 外部传入的记忆类型文本。

    返回:
        合法记忆类型；非法或空值返回 preference。
    """

    value = memory_type.strip().lower()
    if value in _ALLOWED_MEMORY_TYPES:
        return value
    return "preference"


def normalize_status(status: str) -> str:
    """归一化 memory item 状态。

    参数:
        status: 外部传入的状态文本。

    返回:
        active 或 superseded。
    """

    value = status.strip().lower()
    if value in _ALLOWED_STATUS:
        return value
    return "active"


def clamp_emotional_weight(value: int) -> int:
    """限制 emotional_weight 的范围。

    参数:
        value: 外部传入的情绪权重。

    返回:
        0 到 10 之间的整数。
    """

    return max(0, min(10, int(value)))


def content_hash(summary: str, memory_type: str) -> str:
    """生成 memory item 的短内容 hash。

    参数:
        summary: 记忆摘要。
        memory_type: 记忆类型。

    返回:
        SHA256 前 16 位十六进制字符串。
    """

    normalized_summary = re.sub(r"\s+", " ", summary.strip().lower())
    normalized_type = normalize_memory_type(memory_type)
    raw = f"{normalized_type}:{normalized_summary}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]