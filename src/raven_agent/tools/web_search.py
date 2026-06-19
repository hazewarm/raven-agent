from __future__ import annotations

import asyncio
import json
from typing import Any

from raven_agent.tools.base import Tool, ToolResult

_DEFAULT_NUM_RESULTS = 5
_MAX_NUM_RESULTS = 10


class WebSearchTool(Tool):
    """基于 SerpAPI 的网页搜索工具。

    输入:
        api_key: 构造函数参数，来自 config.toml 的 SerpAPI API Key。
        default_gl: 构造函数参数，默认 Google 搜索国家代码。
        default_hl: 构造函数参数，默认 Google 搜索语言。

    输出:
        一个 Tool 实例。执行 execute() 后返回搜索结果 ToolResult。
    """

    name = "web_search"
    description = (
        "使用 SerpAPI 执行 Google 网页搜索，返回直接答案、知识图谱和自然搜索结果。"
        "适合查询时效信息、公开网页资料和模型知识库中没有的信息。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询关键词。"},
            "num_results": {
                "type": "integer",
                "description": "最多返回多少条自然搜索结果，默认 5，最大 10。",
                "minimum": 1,
                "maximum": _MAX_NUM_RESULTS,
                "default": _DEFAULT_NUM_RESULTS,
            },
            "gl": {
                "type": "string",
                "description": "Google 搜索国家代码，默认 cn。",
                "default": "cn",
            },
            "hl": {
                "type": "string",
                "description": "Google 搜索界面语言，默认 zh-cn。",
                "default": "zh-cn",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: str,
        default_gl: str = "cn",
        default_hl: str = "zh-cn",
    ) -> None:
        """初始化 WebSearchTool。

        输入:
            api_key: SerpAPI API Key，由 config.toml 加载后注入。
            default_gl: 默认 Google 搜索国家代码。
            default_hl: 默认 Google 搜索语言。

        输出:
            None。初始化后的状态保存在 self._api_key / self._default_gl / self._default_hl。
        """

        self._api_key = api_key.strip()
        self._default_gl = default_gl.strip() or "cn"
        self._default_hl = default_hl.strip() or "zh-cn"

    async def execute(
        self,
        query: str,
        num_results: int = _DEFAULT_NUM_RESULTS,
        gl: str | None = None,
        hl: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行网页搜索。

        输入:
            query: 搜索关键词。
            num_results: 最多返回多少条自然搜索结果。
            gl: Google 搜索国家代码。
            hl: Google 搜索语言。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。text 是 JSON 字符串，包含 answer、knowledge_graph 和 organic_results。
        """

        clean_query = query.strip()
        if not clean_query:
            return _json_result({"error": "query 不能为空"}, ok=False)
        if not self._api_key:
            return _json_result({"query": clean_query, "error": "config.toml 未配置 tools.web_search.api_key"}, ok=False)

        search_gl = (gl or self._default_gl).strip() or "cn"
        search_hl = (hl or self._default_hl).strip() or "zh-cn"
        limit = min(max(1, int(num_results)), _MAX_NUM_RESULTS)
        params = {
            "engine": "google",
            "q": clean_query,
            "api_key": self._api_key,
            "gl": search_gl,
            "hl": search_hl,
            "num": limit,
        }

        try:
            results = await asyncio.to_thread(_serpapi_search, params)
        except ModuleNotFoundError:
            return _json_result({"query": clean_query, "error": "缺少依赖 google-search-results，请先安装"}, ok=False)
        except Exception as exc:
            return _json_result({"query": clean_query, "error": f"搜索失败: {exc}"}, ok=False)

        payload = _parse_serpapi_results(clean_query, results, limit)
        return _json_result(payload, ok=True)


def _serpapi_search(params: dict[str, Any]) -> dict[str, Any]:
    """调用同步 SerpAPI SDK。

    输入:
        params: 传给 SerpApiClient 的参数字典。

    输出:
        SerpAPI 返回的结果字典。

    异常:
        ModuleNotFoundError: 未安装 google-search-results 时抛出。
        Exception: SerpAPI SDK 请求失败时抛出。
    """

    from serpapi import SerpApiClient

    client = SerpApiClient(params)
    return client.get_dict()


def _parse_serpapi_results(query: str, results: dict[str, Any], limit: int) -> dict[str, Any]:
    """解析 SerpAPI 搜索结果。

    输入:
        query: 原始搜索关键词。
        results: SerpAPI 返回的原始字典。
        limit: 最多返回多少条 organic results。

    输出:
        结构化搜索结果字典。
    """

    answer_box = results.get("answer_box") if isinstance(results.get("answer_box"), dict) else {}
    knowledge_graph = results.get("knowledge_graph") if isinstance(results.get("knowledge_graph"), dict) else {}
    organic_results = results.get("organic_results") if isinstance(results.get("organic_results"), list) else []

    return {
        "query": query,
        "answer": _extract_answer(answer_box),
        "knowledge_graph": _extract_knowledge_graph(knowledge_graph),
        "organic_results": [_extract_organic_result(item) for item in organic_results[:limit] if isinstance(item, dict)],
    }


def _extract_answer(answer_box: dict[str, Any]) -> str:
    """从 answer_box 提取直接答案。

    输入:
        answer_box: SerpAPI answer_box 字典。

    输出:
        直接答案字符串；没有可用答案时返回空字符串。
    """

    for key in ("answer", "snippet", "snippet_highlighted_words", "title"):
        value = answer_box.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return " ".join(str(item) for item in value if str(item).strip())
    return ""


def _extract_knowledge_graph(knowledge_graph: dict[str, Any]) -> dict[str, str]:
    """提取知识图谱摘要。

    输入:
        knowledge_graph: SerpAPI knowledge_graph 字典。

    输出:
        包含 title、type、description 的字典；缺失字段返回空字符串。
    """

    return {
        "title": str(knowledge_graph.get("title") or ""),
        "type": str(knowledge_graph.get("type") or ""),
        "description": str(knowledge_graph.get("description") or ""),
    }


def _extract_organic_result(item: dict[str, Any]) -> dict[str, str]:
    """提取单条自然搜索结果。

    输入:
        item: SerpAPI organic_results 中的一项。

    输出:
        包含 title、snippet、link 的字典。
    """

    return {
        "title": str(item.get("title") or ""),
        "snippet": str(item.get("snippet") or ""),
        "link": str(item.get("link") or ""),
    }


def _json_result(payload: dict[str, Any], *, ok: bool) -> ToolResult:
    """把搜索结果字典包装为 ToolResult。

    输入:
        payload: 要返回给模型的结构化搜索结果。
        ok: 搜索调用是否成功。

    输出:
        ToolResult。text 是格式化 JSON，metadata.ok 等于 ok。
    """

    return ToolResult(text=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"ok": ok})