from __future__ import annotations

import re
from collections.abc import Iterable, Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from raven_agent.tools.base import Tool, ToolResult, normalize_tool_result


META_TOOL_NAMES: frozenset[str] = frozenset({"tool_search"})
_VALID_RISKS: frozenset[str] = frozenset(
    {"read-only", "write", "external-side-effect"}
)
_VALID_SOURCE_TYPES: frozenset[str] = frozenset({"builtin", "mcp", "plugin"})


@dataclass(frozen=True)
class ToolMeta:
    """工具搜索与可见性元数据。

    参数:
        risk: 工具风险等级，可选 read-only / write / external-side-effect。
        always_on: True 表示每轮默认把该工具 schema 暴露给模型。
        search_hint: 额外搜索提示词，用于补充工具名和描述没有覆盖的表达。
    """

    risk: str = "read-only"
    always_on: bool = False
    search_hint: str = ""


@dataclass(frozen=True)
class ToolDocument:
    """工具目录中的搜索文档。

    参数:
        name: 工具名称。
        description: 工具描述。
        risk: 工具风险等级。
        always_on: 工具是否默认可见。
        search_hint: 额外搜索提示词。
        source_type: 工具来源类型，当前支持 builtin / mcp。
        source_name: 工具来源名称；builtin 工具为空字符串，mcp 工具为 server 名。
    """

    name: str
    description: str
    risk: str
    always_on: bool
    search_hint: str = ""
    source_type: str = "builtin"
    source_name: str = ""

    @classmethod
    def from_tool(
        cls,
        tool: Tool,
        meta: ToolMeta,
        *,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> ToolDocument:
        """从 Tool 与 ToolMeta 创建搜索文档。

        参数:
            tool: 已注册工具实例。
            meta: 工具元数据。
            source_type: 工具来源类型。
            source_name: 工具来源名称。

        返回:
            ToolDocument，用于关键词搜索和 select 精确加载结果展示。
        """

        return cls(
            name=tool.name.strip(),
            description=tool.description.strip(),
            risk=meta.risk,
            always_on=meta.always_on,
            search_hint=meta.search_hint.strip(),
            source_type=source_type,
            source_name=source_name.strip(),
        )


class ToolRegistry:
    """管理当前 Agent 可用工具的注册表。

    参数:
        无。创建后通过 register() 添加工具。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._documents: dict[str, ToolDocument] = {}

    def register(
        self,
        tool: Tool,
        *,
        risk: str = "read-only",
        always_on: bool = False,
        search_hint: str | None = None,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> None:
        """注册一个工具。

        参数:
            tool: 要注册的 Tool 实例。
            risk: 工具风险等级，可选 read-only / write / external-side-effect。
            always_on: True 表示该工具每轮默认暴露给模型。
            search_hint: 额外搜索提示词，用于提升关键词召回。
            source_type: 工具来源类型，当前支持 builtin / mcp。
            source_name: 工具来源名称；builtin 工具通常为空字符串。

        返回:
            None。

        异常:
            ValueError: 当工具名称为空、工具重复注册或元数据非法时抛出。
        """

        name = tool.name.strip()
        if not name:
            raise ValueError("工具名称不能为空")
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        if risk not in _VALID_RISKS:
            raise ValueError(f"非法工具风险等级: {risk}")
        if source_type not in _VALID_SOURCE_TYPES:
            raise ValueError(f"非法工具来源类型: {source_type}")

        meta = ToolMeta(
            risk=risk,
            always_on=always_on,
            search_hint=search_hint or "",
        )
        self._tools[name] = tool
        self._metadata[name] = meta
        self._documents[name] = ToolDocument.from_tool(
            tool,
            meta,
            source_type=source_type,
            source_name=source_name,
        )
    
    def unregister(self, name: str) -> None:
        """注销一个已注册工具。

        输入:
            name: 工具名称。

        输出:
            None。工具不存在时直接返回，不抛错。
        """

        self._tools.pop(name, None)
        self._metadata.pop(name, None)
        self._documents.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """按名称获取工具。

        参数:
            name: 工具名称。

        返回:
            找到时返回 Tool，否则返回 None。
        """

        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """判断工具是否已经注册。

        参数:
            name: 工具名称。

        返回:
            True 表示工具存在，False 表示工具不存在。
        """

        return name in self._tools

    def get_document(self, name: str) -> ToolDocument | None:
        """按名称获取工具搜索文档。

        参数:
            name: 工具名称。

        返回:
            找到时返回 ToolDocument，否则返回 None。
        """

        return self._documents.get(name)

    def list_names(self) -> list[str]:
        """列出所有已注册工具名。

        返回:
            按注册顺序排列的工具名列表。
        """

        return list(self._tools.keys())

    def get_schemas(
        self,
        names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """返回工具的 OpenAI tool schema。

        参数:
            names: 为 None 时返回全部工具 schema；传入集合时按注册顺序过滤；传入列表或元组时按调用方顺序返回。

        返回:
            schema 字典列表，可直接传给 Chat Completions API 的 tools 参数。
        """

        if names is None:
            selected_names = list(self._tools.keys())
        elif isinstance(names, AbstractSet):
            selected_names = [name for name in self._tools.keys() if name in names]
        else:
            selected_names = [name for name in names if name in self._tools]

        return [self._tools[name].to_schema() for name in selected_names]

    def get_always_on_names(self) -> set[str]:
        """获取 always-on 工具名称集合。

        返回:
            所有标记为 always_on=True 的已注册工具名。
        """

        return {
            name
            for name, meta in self._metadata.items()
            if meta.always_on and name in self._tools
        }

    def get_visible_names(
        self,
        loaded_names: Iterable[str] | None = None,
    ) -> set[str]:
        """计算当前 turn 对模型可见的工具名。

        参数:
            loaded_names: 本轮已经通过 tool_search 解锁或由 LRU preload 预加载的工具名。

        返回:
            always-on 工具、已注册 meta tool、loaded_names 的并集；不存在的工具名会被忽略。
        """

        visible = self.get_always_on_names()
        visible.update(name for name in META_TOOL_NAMES if name in self._tools)
        if loaded_names is not None:
            visible.update(name for name in loaded_names if name in self._tools)
        return visible

    def get_visible_schemas(
        self,
        loaded_names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """返回当前 turn 应暴露给模型的工具 schema。

        参数:
            loaded_names: 本轮已经解锁或预加载的工具名。

        返回:
            按注册顺序排列的可见工具 schema 列表。
        """

        return self.get_schemas(self.get_visible_names(loaded_names))

    def get_deferred_names(
        self,
        loaded_names: Iterable[str] | None = None,
    ) -> dict[str, object]:
        """列出当前 turn 仍处于 deferred 状态的工具名。

        参数:
            loaded_names: 本轮已经解锁或预加载的工具名。

        返回:
            按来源分组的 deferred 工具名。格式为 {"builtin": [...], "mcp": {server: [...]}}。
        """

        visible = self.get_visible_names(loaded_names)
        builtin: list[str] = []
        mcp: dict[str, list[str]] = {}

        for name, document in self._documents.items():
            if name in visible:
                continue
            if document.source_type == "mcp":
                mcp.setdefault(document.source_name, []).append(name)
            else:
                builtin.append(name)

        return {
            "builtin": sorted(builtin),
            "mcp": {key: sorted(value) for key, value in sorted(mcp.items())},
        }

    def get_execution_guard_message(
        self,
        name: str,
        loaded_names: Iterable[str] | None = None,
    ) -> str | None:
        """判断一次工具调用是否应该被运行时拦截。

        参数:
            name: 模型请求调用的工具名。
            loaded_names: 当前 turn 已经解锁或预加载的工具名。

        返回:
            None 表示可以继续执行；字符串表示应该拦截并把该提示作为工具结果返回给模型。
        """

        if name not in self._tools:
            return (
                f"工具不存在: {name}。请调用 tool_search 搜索相关能力；"
                f"如果你认为工具名正确，可以尝试 query=\"select:{name}\"。"
            )

        if name not in self.get_visible_names(loaded_names):
            return (
                f"工具 {name} 已注册但当前未加载。"
                f"请先调用 tool_search，query 使用 \"select:{name}\"，"
                "成功解锁后再直接调用该工具。"
            )

        return None

    def get_tool_names_by_source(
        self,
        source_type: str,
        source_name: str = "",
    ) -> set[str]:
        """返回指定来源的所有工具名称集合。

        输入:
            source_type: 工具来源类型，如 "mcp"。
            source_name: 可选来源名称；传空字符串匹配该 source_type 下所有工具。

        输出:
            set[str]。无匹配结果时返回空集合。
        """
        result: set[str] = set()
        for name, document in self._documents.items():
            if document.source_type != source_type:
                continue
            if source_name and document.source_name != source_name:
                continue
            result.add(name)
        return result


    def get_mcp_server_names(self) -> set[str]:
        """返回当前已注册的所有 MCP server 名称集合。

        输出:
            set[str]。无已注册 MCP server 时返回空集合。
        """
        servers: set[str] = set()
        for document in self._documents.values():
            if document.source_type == "mcp" and document.source_name:
                servers.add(document.source_name)
        return servers
    
    
    
    def get_search_results_for_names(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """把工具名列表转换成 tool_search 返回的 matched 项。

        参数:
            names: 要转换的工具名列表。

        返回:
            与 search() 相同结构的结果列表，未注册工具会被跳过。
        """

        results: list[dict[str, Any]] = []
        for name in names:
            document = self._documents.get(name)
            if document is not None:
                results.append(
                    _document_to_search_result(
                        document,
                        why_matched=["名称:精确匹配"],
                    )
                )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """关键词搜索工具目录。

        参数:
            query: 用户或模型输入的搜索词。
            top_k: 最多返回多少个工具，最终会限制在 1 到 10 之间。
            allowed_risk: 允许返回的风险等级；为 None 时不过滤风险等级。
            excluded_names: 已经对模型可见的工具名，搜索结果会排除这些工具。

        返回:
            搜索结果列表，每项包含 name / summary / why_matched / risk / always_on / source_type / source_name。
        """

        clean_query = query.strip()
        limited_top_k = min(max(1, int(top_k)), 10)
        risk_filter = set(allowed_risk) if allowed_risk else None
        excluded = set(excluded_names or set()) | META_TOOL_NAMES

        if clean_query in self._documents and clean_query not in excluded:
            document = self._documents[clean_query]
            if risk_filter is None or document.risk in risk_filter:
                return [
                    _document_to_search_result(
                        document,
                        why_matched=["名称:精确匹配"],
                    )
                ]

        keywords = _normalize_query(clean_query)
        if not keywords:
            return []

        scored_results: list[tuple[int, str, dict[str, Any]]] = []
        for name, document in self._documents.items():
            if name in excluded:
                continue
            if risk_filter is not None and document.risk not in risk_filter:
                continue

            score = _score_document(document, keywords)
            if score <= 0:
                continue

            scored_results.append(
                (
                    score,
                    name,
                    _document_to_search_result(
                        document,
                        why_matched=_explain_document(document, keywords),
                    ),
                )
            )

        scored_results.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored_results[:limited_top_k]]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行指定工具。

        参数:
            name: 要执行的工具名称。
            arguments: 工具参数字典。

        返回:
            ToolResult。即使工具不存在或执行失败，也返回受控的 ToolResult。
        """

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                text=f"工具不存在: {name}",
                metadata={"ok": False, "error": "tool_not_found"},
            )

        try:
            result = await tool.execute(**arguments)
            normalized = normalize_tool_result(result)
            return ToolResult(
                text=normalized.text,
                metadata={"ok": True, **normalized.metadata},
            )
        except TypeError as exc:
            return ToolResult(
                text=f"工具参数错误: {exc}",
                metadata={"ok": False, "error": "invalid_arguments"},
            )
        except Exception as exc:
            return ToolResult(
                text=f"工具执行失败: {exc}",
                metadata={"ok": False, "error": "execution_failed"},
            )


def _normalize_query(query: str) -> set[str]:
    """把搜索 query 归一化为关键词集合。

    参数:
        query: 原始搜索词。

    返回:
        关键词集合。英文按空格切分，中文额外加入单字和 bigram，不依赖外部分词库。
    """

    lowered = query.lower().strip()
    tokens: set[str] = set()
    if lowered:
        tokens.add(lowered)

    tokens.update(part for part in lowered.split() if part)

    for segment in re.split(r"([一-鿿]+)", lowered):
        clean_segment = segment.strip()
        if clean_segment:
            tokens.add(clean_segment)

    cjk_chars = [char for char in lowered if "一" <= char <= "鿿"]
    tokens.update(cjk_chars)
    for index in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[index] + cjk_chars[index + 1])

    tokens.discard("")
    return tokens


def _score_document(document: ToolDocument, keywords: set[str]) -> int:
    """计算一个工具文档与搜索词的匹配分数。

    参数:
        document: 工具搜索文档。
        keywords: 归一化后的搜索关键词集合。

    返回:
        整数分数。0 表示不匹配，分数越高排序越靠前。
    """

    name_lower = document.name.lower()
    name_parts = [part for part in name_lower.split("_") if part]
    hint_lower = document.search_hint.lower()
    description_lower = document.description.lower()

    score = 0
    for keyword in keywords:
        if keyword in name_parts:
            score += 10
        elif any(keyword in part or part in keyword for part in name_parts):
            score += 5
        elif keyword in name_lower:
            score += 3

        if hint_lower and keyword in hint_lower:
            score += 4
        if keyword in description_lower:
            score += 2

    return score


def _explain_document(document: ToolDocument, keywords: set[str]) -> list[str]:
    """生成工具搜索命中的解释文本。

    参数:
        document: 工具搜索文档。
        keywords: 归一化后的搜索关键词集合。

    返回:
        命中原因列表，用于帮助模型判断为什么该工具被返回。
    """

    name_lower = document.name.lower()
    name_parts = [part for part in name_lower.split("_") if part]
    hint_lower = document.search_hint.lower()
    description_lower = document.description.lower()

    reasons: list[str] = []
    seen: set[str] = set()

    for keyword in keywords:
        if keyword in name_parts:
            _append_unique(reasons, seen, f"名称精确:{keyword}")
        elif any(keyword in part or part in keyword for part in name_parts):
            _append_unique(reasons, seen, f"名称部分:{keyword}")
        elif keyword in name_lower:
            _append_unique(reasons, seen, f"名称:{keyword}")

        if hint_lower and keyword in hint_lower:
            _append_unique(reasons, seen, f"提示:{keyword}")
        if keyword in description_lower:
            _append_unique(reasons, seen, f"描述:{keyword}")

    return reasons


def _append_unique(items: list[str], seen: set[str], value: str) -> None:
    """向列表追加一个不重复的字符串。

    参数:
        items: 要追加内容的列表。
        seen: 已出现内容集合。
        value: 准备追加的字符串。

    返回:
        None。
    """

    if value in seen:
        return
    seen.add(value)
    items.append(value)


def _document_to_search_result(
    document: ToolDocument,
    why_matched: list[str],
) -> dict[str, Any]:
    """把 ToolDocument 转换为 tool_search 的 matched 结果。

    参数:
        document: 工具搜索文档。
        why_matched: 命中原因列表。

    返回:
        可 JSON 序列化的搜索结果字典。
    """

    return {
        "name": document.name,
        "summary": document.description[:120],
        "why_matched": why_matched,
        "risk": document.risk,
        "always_on": document.always_on,
        "source_type": document.source_type,
        "source_name": document.source_name,
    }