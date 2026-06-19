from __future__ import annotations

import asyncio
import json

from raven_agent.tools import WebSearchTool


def _run(coro):
    """同步运行异步测试调用。

    输入:
        coro: 要运行的 coroutine。

    输出:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def test_web_search_requires_api_key() -> None:
    """测试 web_search 在没有 config.toml API key 注入时返回受控错误。

    输入:
        无。

    输出:
        None。通过 assert 验证 metadata.ok=False。
    """

    tool = WebSearchTool(api_key="")

    result = _run(tool.execute(query="raven agent"))

    assert result.metadata["ok"] is False
    assert "tools.web_search.api_key" in result.text


def test_web_search_parses_serpapi_results(monkeypatch) -> None:
    """测试 web_search 可以解析 SerpAPI 返回结构。

    输入:
        monkeypatch: pytest fixture，用于替换内部同步搜索函数。

    输出:
        None。通过 assert 验证 answer、knowledge_graph 和 organic_results。
    """

    import raven_agent.tools.web_search as web_search_module

    def fake_search(params: dict[str, object]) -> dict[str, object]:
        """模拟 SerpAPI SDK 返回。

        输入:
            params: WebSearchTool 传给 SerpAPI 的参数。

        输出:
            模拟的 SerpAPI 搜索结果字典。
        """

        assert params["q"] == "raven agent"
        return {
            "answer_box": {"answer": "direct answer"},
            "knowledge_graph": {
                "title": "Raven Agent",
                "type": "Software",
                "description": "Agent project",
            },
            "organic_results": [
                {"title": "A", "snippet": "Snippet A", "link": "https://example.com/a"},
                {"title": "B", "snippet": "Snippet B", "link": "https://example.com/b"},
            ],
        }

    monkeypatch.setattr(web_search_module, "_serpapi_search", fake_search)
    tool = WebSearchTool(api_key="test-key")

    result = _run(tool.execute(query="raven agent", num_results=1))
    payload = json.loads(result.text)

    assert result.metadata["ok"] is True
    assert payload["answer"] == "direct answer"
    assert payload["knowledge_graph"]["title"] == "Raven Agent"
    assert len(payload["organic_results"]) == 1
    assert payload["organic_results"][0]["link"] == "https://example.com/a"