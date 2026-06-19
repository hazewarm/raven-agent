from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from raven_agent.tools.base import Tool, ToolResult
from raven_agent.tools.registry import ToolRegistry


class ToolSearchTool(Tool):
    """搜索并解锁 deferred tools 的元工具。

    参数:
        registry: 当前进程中的 ToolRegistry。tool_search 只搜索该注册表，不直接执行其他工具。
    """

    name = "tool_search"
    description = (
        "在工具目录中搜索可用工具。搜索结果中的工具会被当前 turn 解锁，"
        "之后可以直接调用。已知工具名但当前不可见时，请使用 select:工具名 精确加载。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "搜索查询。支持两种形式："
                    "1. select:工具名 精确加载；"
                    "2. 用自然语言描述需要的能力，例如 读取文件、发送消息、定时提醒。"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "关键词搜索最多返回多少个工具，默认 5，最大 10。",
                "default": 5,
                "minimum": 1,
            },
            "allowed_risk": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["read-only", "write", "external-side-effect"],
                },
                "description": "允许返回的工具风险等级；不传则不过滤风险等级。",
            },
        },
        "required": ["query"],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._excluded_names: set[str] = set()

    def set_excluded_names(self, names: set[str] | None) -> None:
        """设置本次搜索需要排除的已可见工具名。

        参数:
            names: 当前 turn 已经暴露给模型的工具名；为 None 时表示不排除额外工具。

        返回:
            None。
        """

        self._excluded_names = set(names or set())

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行工具搜索或精确加载。

        参数:
            query: 搜索词。以 select: 开头时按工具名精确加载，否则走关键词搜索。
            top_k: 关键词搜索最多返回多少个工具，最终限制在 1 到 10。
            allowed_risk: 可选风险等级过滤列表。
            **kwargs: 预留参数，当前不会使用。

        返回:
            ToolResult。text 是 JSON 字符串；metadata.unlocked 保存本次解锁的工具名。
        """

        excluded_names = set(self._excluded_names)
        self._excluded_names.clear()

        clean_query = (query or "").strip()
        if not clean_query:
            return _json_tool_result(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "query 不能为空，请描述你需要的工具能力。",
                },
                ok=False,
                unlocked=[],
            )

        if clean_query.lower().startswith("select:"):
            return self._handle_select(
                clean_query[7:],
                allowed_risk=allowed_risk,
                excluded_names=excluded_names,
            )

        results = self._registry.search(
            query=clean_query,
            top_k=top_k,
            allowed_risk=allowed_risk,
            excluded_names=excluded_names,
        )
        unlocked = [
            item["name"]
            for item in results
            if isinstance(item.get("name"), str) and item["name"]
        ]

        if not results:
            return _json_tool_result(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "没有找到匹配工具，请换个关键词重试。",
                },
                ok=True,
                unlocked=[],
            )

        return _json_tool_result(
            {
                "matched": results,
                "unlocked": unlocked,
                "already_loaded": [],
                "next_action": "unlocked 中的工具已加载。下一步请直接调用需要的工具，不要再次 tool_search。",
            },
            ok=True,
            unlocked=unlocked,
        )

    def _handle_select(
        self,
        names_text: str,
        *,
        allowed_risk: list[str] | None,
        excluded_names: set[str],
    ) -> ToolResult:
        """处理 select: 工具名精确加载。

        参数:
            names_text: select: 后面的原始工具名文本，支持逗号分隔多个工具名。
            allowed_risk: 可选风险等级过滤列表。
            excluded_names: 当前 turn 已经可见的工具名。

        返回:
            ToolResult。text 是 JSON 字符串；metadata.unlocked 保存成功解锁的工具名。
        """

        requested = [name.strip() for name in names_text.split(",") if name.strip()]
        if not requested:
            return _json_tool_result(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "select: 后面需要提供至少一个工具名。",
                },
                ok=False,
                unlocked=[],
            )

        risk_filter = set(allowed_risk) if allowed_risk else None
        already_loaded: list[str] = []
        unlocked: list[str] = []
        missing: list[str] = []
        risk_blocked: list[str] = []

        for name in requested:
            if name in excluded_names:
                already_loaded.append(name)
                continue

            document = self._registry.get_document(name)
            if document is None:
                missing.append(name)
                continue

            if risk_filter is not None and document.risk not in risk_filter:
                risk_blocked.append(name)
                continue

            unlocked.append(name)

        result: dict[str, Any] = {
            "matched": self._registry.get_search_results_for_names(unlocked),
            "unlocked": unlocked,
            "already_loaded": already_loaded,
            "missing": missing,
            "risk_blocked": risk_blocked,
        }

        tips: list[str] = []
        if unlocked:
            result["next_action"] = "unlocked 中的工具已加载。下一步请直接调用需要的工具，不要再次 tool_search。"
        if already_loaded:
            tips.append(f"已加载可直接调用: {', '.join(already_loaded)}")
        if missing:
            tips.append(f"未找到工具: {', '.join(missing)}，请换关键词搜索确认正确名称。")
        if risk_blocked:
            tips.append(f"风险等级不符合 allowed_risk: {', '.join(risk_blocked)}")
        if tips:
            result["tip"] = "; ".join(tips)

        return _json_tool_result(result, ok=True, unlocked=unlocked)


@dataclass
class ToolDiscoveryState:
    """按 session 保存最近解锁工具名的 LRU 状态。

    参数:
        capacity: 每个 session 最多保留多少个最近解锁工具。
        _unlocked: 内部状态，key 为 session_key，value 为保持顺序的工具名集合。
    """

    capacity: int = 5
    _unlocked: dict[str, OrderedDict[str, None]] = field(default_factory=dict)

    def get_preloaded(self, session_key: str) -> set[str]:
        """读取某个 session 的预加载工具名集合。

        参数:
            session_key: 会话 key。

        返回:
            工具名集合。没有记录时返回空集合。
        """

        return set(self._unlocked.get(_normalize_session_key(session_key), {}).keys())

    def get_preloaded_ordered(self, session_key: str) -> list[str]:
        """读取某个 session 的预加载工具名列表。

        参数:
            session_key: 会话 key。

        返回:
            按最近使用顺序排列的工具名列表。
        """

        return list(self._unlocked.get(_normalize_session_key(session_key), {}).keys())

    def unlock_names_from_result(self, result_json: str) -> list[str]:
        """从 tool_search 的 JSON 输出中提取 unlocked 工具名。

        参数:
            result_json: tool_search 返回给模型的 JSON 字符串。

        返回:
            去重后的工具名列表。解析失败或字段不存在时返回空列表。
        """

        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return []

        raw_unlocked = payload.get("unlocked")
        if not isinstance(raw_unlocked, list):
            return []

        names: list[str] = []
        seen: set[str] = set()
        for item in raw_unlocked:
            if isinstance(item, str) and item.strip() and item not in seen:
                names.append(item)
                seen.add(item)
        return names

    def update(
        self,
        session_key: str,
        tool_names: Iterable[str],
        always_on: set[str],
    ) -> None:
        """更新某个 session 的最近工具 LRU。

        参数:
            session_key: 会话 key。
            tool_names: 本轮解锁或成功使用过的工具名。
            always_on: always-on 工具名集合，这些工具不会写入 LRU。

        返回:
            None。
        """

        key = _normalize_session_key(session_key)
        skip = set(always_on) | {"tool_search"}
        lru = self._unlocked.setdefault(key, OrderedDict())

        for name in tool_names:
            clean_name = name.strip()
            if not clean_name or clean_name in skip:
                continue
            if clean_name in lru:
                lru.move_to_end(clean_name)
            else:
                lru[clean_name] = None
            while len(lru) > self.capacity:
                lru.popitem(last=False)


def _json_tool_result(
    payload: dict[str, Any],
    *,
    ok: bool,
    unlocked: Iterable[str],
) -> ToolResult:
    """创建 JSON 文本形式的 ToolResult。

    参数:
        payload: 要返回给模型的 JSON 对象。
        ok: 本次 tool_search 调用是否成功完成。
        unlocked: 本次成功解锁的工具名。

    返回:
        ToolResult，其中 text 是格式化 JSON，metadata 包含 ok 与 unlocked。
    """

    unlocked_list = list(unlocked)
    return ToolResult(
        text=json.dumps(payload, ensure_ascii=False, indent=2),
        metadata={"ok": ok, "unlocked": unlocked_list},
    )


def _normalize_session_key(session_key: str) -> str:
    """归一化 session key。

    参数:
        session_key: 原始会话 key。

    返回:
        去掉首尾空白后的 key；空 key 会返回 __default__。
    """

    return session_key.strip() or "__default__"