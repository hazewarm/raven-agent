"""
proactive/contracts.py —— Alert / Content / Context 标准合同。

外部数据源（MCP）返回的数据格式千差万别。合同层负责将它们
归一化为统一的 dataclass，包含 Judge prompt 需要的标准字段。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_PROMPT_TEXT_LEN = 300
MAX_METRICS_KEYS = 8
MAX_METRICS_VALUE_STR_LEN = 60


def _trim(text: str, limit: int) -> str:
    """截断字符串到指定长度。

    输入:
        text: 原始字符串。
        limit: 最大字符数。

    输出:
        截断后的字符串；超出时末尾添加 "..."。
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_metrics(metrics: Any) -> dict[str, Any] | None:
    """规范化 metrics 字段：限制 key 数量和 value 长度。

    输入:
        metrics: 原始 metrics（期望 dict）。

    输出:
        规范化后的 dict；无效输入返回 None。
    """
    if not isinstance(metrics, dict) or not metrics:
        return None
    normalized: dict[str, Any] = {}
    items = list(metrics.items())
    for key, value in items[:MAX_METRICS_KEYS]:
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, str):
            normalized[key_text] = _trim(value, MAX_METRICS_VALUE_STR_LEN)
        elif isinstance(value, (int, float, bool)) or value is None:
            normalized[key_text] = value
        else:
            text = json.dumps(value, ensure_ascii=False)
            normalized[key_text] = _trim(text, MAX_METRICS_VALUE_STR_LEN)
    truncated = len(items) - MAX_METRICS_KEYS
    if truncated > 0:
        normalized["_truncated_keys"] = truncated
    return normalized or None


# ── AlertContract ────────────────────────────────────────────────────


@dataclass(slots=True)
class AlertContract:
    """告警类主动推送事件。

    字段:
        item_id: 全局唯一标识（格式 "{ack_server}:{event_id}"）。
        title: 告警标题。
        content: 告警详情。
        severity: 严重程度（"high" / "medium" / "low"）。
        suggested_tone: 建议通知语气（"direct" / "neutral" / "gentle"）。
        metrics: 可选的规范化指标字典。
        raw: 原始事件 dict，供链路下游使用。
    """

    item_id: str
    title: str
    content: str
    severity: str
    suggested_tone: str
    metrics: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_prompt_line(self, index: int) -> str:
        """格式化为 Judge prompt 中的单行展示。

        输入:
            index: 候选序号（从 1 开始）。

        输出:
            格式化字符串。
        """
        severity_part = f"  severity={self.severity}" if self.severity else ""
        line = f"  [{index}] id={self.item_id}{severity_part}\n       title={self.title}"
        if self.content:
            line += f"\n       内容：{self.content}"
        if self.metrics:
            line += f"\n       metrics：{json.dumps(self.metrics, ensure_ascii=False)}"
        if self.suggested_tone:
            line += f"\n       建议语气：{self.suggested_tone}"
        return line


def normalize_alert(event: dict[str, Any]) -> AlertContract:
    """从 MCP source 返回的原始 dict 构建 AlertContract。

    输入:
        event: MCP source 返回的原始事件 dict。

    输出:
        AlertContract 实例。缺失字段使用空字符串。
    """
    ack_server = str(event.get("ack_server") or "?").strip() or "?"
    event_id = str(event.get("event_id") or event.get("id") or "?").strip() or "?"
    title = str(event.get("title") or "").strip()
    content = str(event.get("content") or event.get("body") or "").strip()
    severity = str(event.get("severity") or "").strip()
    tone = str(event.get("suggested_tone") or "").strip()
    return AlertContract(
        item_id=f"{ack_server}:{event_id}",
        title=title,
        content=content,
        severity=severity,
        suggested_tone=tone,
        metrics=_normalize_metrics(event.get("metrics")),
        raw=event,
    )


# ── ContentContract ──────────────────────────────────────────────────


@dataclass(slots=True)
class ContentContract:
    """可阅读内容类主动推送事件。

    字段:
        item_id: 全局唯一标识。
        title: 内容标题。
        source: 来源名称（如 "python.org" / "GitHub"）。
        url: 可选的原文链接。
        raw: 原始事件 dict。
    """

    item_id: str
    title: str
    source: str
    url: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid_url(self) -> bool:
        """是否有可用的原文链接。"""
        return bool(self.url)

    def to_prompt_line(self, index: int, has_content: bool) -> str:
        """格式化为 Judge prompt 中的单行展示。

        输入:
            index: 候选序号。
            has_content: LLM 是否已通过 web_fetch 获取正文。

        输出:
            格式化字符串。
        """
        status = "✓" if has_content else "✗(预取失败)"
        url_part = f"\n       url={self.url}" if self.has_valid_url else ""
        return (
            f"  [{index}] id={self.item_id}\n"
            f"       title={self.title}\n"
            f"       source={self.source}  正文:{status}"
            f"{url_part}"
        )


def normalize_content(item: dict[str, Any]) -> ContentContract:
    """从 MCP source 返回的原始 dict 构建 ContentContract。

    item_id 格式为 "{ack_server}:{event_id}"，与 normalize_alert 保持一致。
    这是 ACK 去重和 seen_items 去重正确工作的前提——
    ack_events() 通过 item_id.split(":", 1)[0] 反查 server，
    缺少 ack_server 前缀会导致 mark_read 永远不被调用，
    进而 rss.get_posts(unread_only=true) 每次 tick 都返回相同条目，
    最终阻塞 drift 入口。

    输入:
        item: MCP source 返回的原始内容 dict。

    输出:
        ContentContract 实例。
    """
    ack_server = str(item.get("ack_server") or "").strip()
    event_id = str(item.get("id") or "").strip()
    if ack_server and event_id:
        item_id = f"{ack_server}:{event_id}"
    else:
        item_id = event_id or ""
    return ContentContract(
        item_id=item_id,
        title=str(item.get("title") or "").strip(),
        source=str(item.get("source") or item.get("source_name") or "").strip(),
        url=str(item.get("url") or "").strip(),
        raw=item,
    )


# ── ContextContract ──────────────────────────────────────────────────


@dataclass(slots=True)
class ContextContract:
    """环境上下文类数据。

    字段:
        available: 源是否可用。None 表示不确定。
        source: 来源名称。
        raw: 原始 context dict。
    """

    available: bool | None
    source: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_prompt_item(self) -> dict[str, Any]:
        """将 context 转为注入 Judge prompt 的 payload。

        输出:
            扁平化的 dict，时间字段有本地化标注。
        """
        payload = dict(self.raw)
        if self.available is not None:
            payload["available"] = self.available
        if self.source:
            payload["_source"] = self.source
        # 睡眠概率 → 清醒概率（便于 LLM 直接判断）
        if "sleep_prob" in payload and payload["sleep_prob"] is not None:
            payload["awake_prob"] = round(1.0 - float(payload["sleep_prob"]), 3)
        return payload


def normalize_context(
    item: dict[str, Any],
    *,
    source: str = "",
) -> ContextContract:
    """从 MCP source 返回的原始 dict 构建 ContextContract。

    输入:
        item: MCP source 返回的原始 context dict。
        source: 来源名称。

    输出:
        ContextContract 实例。
    """
    available_raw = item.get("available")
    available = None if available_raw is None else bool(available_raw)
    return ContextContract(
        available=available,
        source=source or str(item.get("_source", "")),
        raw=item,
    )