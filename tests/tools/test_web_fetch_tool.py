from __future__ import annotations

import asyncio
import json

import httpx

from raven_agent.tools import WebFetchTool


def _run(coro):
    """同步运行异步测试调用。

    输入:
        coro: 要运行的 coroutine。

    输出:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def test_web_fetch_converts_html_to_markdown() -> None:
    """测试 web_fetch 可以把 HTML 转成 Markdown。

    输入:
        无。测试内部使用 httpx.MockTransport 构造响应。

    输出:
        None。通过 assert 验证 Markdown 内容和脚本清理。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """MockTransport 回调。

        输入:
            request: httpx 传入的请求对象。

        输出:
            httpx.Response，模拟 HTML 响应。
        """

        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><h1>Hello</h1><script>bad()</script><p>Raven</p></body></html>",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = _run(WebFetchTool(client=client).execute(url="https://example.com", format="markdown"))
    finally:
        _run(client.aclose())

    payload = json.loads(result.text)
    assert result.metadata["ok"] is True
    assert "# Hello" in payload["text"]
    assert "Raven" in payload["text"]
    assert "bad()" not in payload["text"]


def test_web_fetch_rejects_local_ip() -> None:
    """测试 web_fetch 拒绝本地 IP。

    输入:
        无。

    输出:
        None。通过 assert 验证 metadata.ok=False 和错误文本。
    """

    result = _run(WebFetchTool().execute(url="http://127.0.0.1:8000"))

    assert result.metadata["ok"] is False
    assert "禁止访问内网/本地地址" in result.text


def test_web_fetch_validates_redirect_target() -> None:
    """测试 web_fetch 会校验重定向目标。

    输入:
        无。测试内部使用 httpx.MockTransport 构造跳转到本地地址的响应。

    输出:
        None。通过 assert 验证跳转被拒绝。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """MockTransport 回调。

        输入:
            request: httpx 传入的请求对象。

        输出:
            httpx.Response，模拟 302 重定向响应。
        """

        return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/private"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = _run(WebFetchTool(client=client).execute(url="https://example.com"))
    finally:
        _run(client.aclose())

    assert result.metadata["ok"] is False
    assert "禁止访问内网/本地地址" in result.text