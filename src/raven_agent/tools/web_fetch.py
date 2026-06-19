from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

from raven_agent.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT = 20
_MAX_TIMEOUT = 60
_DEFAULT_MAX_CHARS = 30_000
_MAX_BODY_BYTES = 5 * 1024 * 1024
_MAX_REDIRECTS = 5
_BINARY_MARKERS = (
    "application/octet-stream",
    "application/pdf",
    "image/",
    "audio/",
    "video/",
)


class WebFetchTool(Tool):
    """抓取网页内容的工具。

    输入:
        client: 构造函数参数，可选 httpx.AsyncClient；测试时可注入 MockTransport。

    输出:
        一个 Tool 实例。执行 execute() 后返回包含网页内容的 ToolResult。
    """

    name = "web_fetch"
    description = (
        "抓取 HTTP/HTTPS URL 内容，支持 markdown、text、html 三种输出。"
        "会校验重定向目标，拒绝本地/内网地址和明显二进制内容。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头。"},
            "format": {
                "type": "string",
                "enum": ["markdown", "text", "html"],
                "description": "返回格式，默认 markdown。",
                "default": "markdown",
            },
            "timeout": {
                "type": "integer",
                "description": "请求超时秒数，默认 20，最大 60。",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT,
            },
            "max_chars": {
                "type": "integer",
                "description": "最多返回多少字符，默认 30000。",
                "minimum": 1,
                "default": _DEFAULT_MAX_CHARS,
            },
            "follow_redirects": {
                "type": "boolean",
                "description": "是否跟随重定向，默认 true。每次跳转都会重新做 URL 安全校验。",
                "default": True,
            },
        },
        "required": ["url"],
    }

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """初始化 WebFetchTool。

        输入:
            client: 可选 httpx.AsyncClient。传入时复用该 client；不传时每次请求临时创建。

        输出:
            None。初始化后的状态保存在 self._client。
        """

        self._client = client

    async def execute(
        self,
        url: str,
        format: str = "markdown",
        timeout: int = _DEFAULT_TIMEOUT,
        max_chars: int = _DEFAULT_MAX_CHARS,
        follow_redirects: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """抓取 URL 并返回指定格式文本。

        输入:
            url: 要抓取的 HTTP/HTTPS URL。
            format: 返回格式，可选 markdown、text、html。
            timeout: 请求超时秒数。
            max_chars: 返回文本最大字符数。
            follow_redirects: 是否跟随重定向。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。text 为 JSON 字符串，metadata.ok 表示是否成功。
        """

        fmt = format if format in {"markdown", "text", "html"} else "markdown"
        timeout_s = min(max(1, int(timeout)), _MAX_TIMEOUT)
        char_limit = min(max(1, int(max_chars)), _DEFAULT_MAX_CHARS)
        client, should_close = self._get_client(timeout_s)

        try:
            response, redirects = await _fetch_with_safe_redirects(
                client=client,
                url=url,
                follow_redirects=follow_redirects,
            )
        except ValueError as exc:
            return _json_result({"url": url, "error": str(exc)}, ok=False)
        except httpx.TimeoutException:
            return _json_result({"url": url, "error": f"请求超时（>{timeout_s}s）"}, ok=False)
        except httpx.RequestError as exc:
            return _json_result({"url": url, "error": f"请求失败: {exc}"}, ok=False)
        finally:
            if should_close:
                await client.aclose()

        status = response.status_code
        if status >= 400:
            return _json_result({"url": url, "final_url": str(response.url), "status": status, "error": "HTTP 请求失败"}, ok=False)

        content_type = response.headers.get("content-type", "")
        if _is_binary_content(content_type):
            return _json_result({"url": url, "final_url": str(response.url), "content_type": content_type, "error": "不支持二进制内容"}, ok=False)

        body = response.content
        if len(body) > _MAX_BODY_BYTES:
            return _json_result({"url": url, "final_url": str(response.url), "error": f"响应过大，超过 {_MAX_BODY_BYTES} 字节"}, ok=False)

        raw_text = response.text
        text = _convert_text(raw_text, content_type=content_type, fmt=fmt)
        truncated = len(text) > char_limit
        visible_text = text[:char_limit] if truncated else text

        return _json_result(
            {
                "url": url,
                "final_url": str(response.url),
                "status": status,
                "content_type": content_type,
                "format": fmt,
                "redirects": redirects,
                "length": len(text),
                "truncated": truncated,
                "text": visible_text,
            },
            ok=True,
        )

    def _get_client(self, timeout_s: int) -> tuple[httpx.AsyncClient, bool]:
        """获取 HTTP client。

        输入:
            timeout_s: 临时 client 的超时秒数。

        输出:
            二元组 `(client, should_close)`。should_close=True 表示调用方用完后需要关闭 client。
        """

        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": "raven-agent/0.1"}), True


async def _fetch_with_safe_redirects(
    *,
    client: httpx.AsyncClient,
    url: str,
    follow_redirects: bool,
) -> tuple[httpx.Response, list[str]]:
    """手动执行带安全校验的重定向抓取。

    输入:
        client: 用于发送 HTTP 请求的 httpx.AsyncClient。
        url: 初始 URL。
        follow_redirects: 是否跟随重定向。

    输出:
        二元组 `(response, redirects)`。response 是最终响应，redirects 是经过校验的跳转 URL 列表。

    异常:
        ValueError: URL 非法、重定向目标非法或重定向次数过多时抛出。
        httpx.RequestError: HTTP 请求失败时由 httpx 抛出。
    """

    current_url = url
    redirects: list[str] = []

    for _ in range(_MAX_REDIRECTS + 1):
        _validate_public_http_url(current_url)
        response = await client.get(current_url, follow_redirects=False)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response, redirects

        location = response.headers.get("location")
        if not follow_redirects or not location:
            return response, redirects

        next_url = urljoin(str(response.url), location)
        _validate_public_http_url(next_url)
        redirects.append(next_url)
        current_url = next_url

    raise ValueError(f"重定向次数超过 {_MAX_REDIRECTS}")


def _validate_public_http_url(url: str) -> None:
    """校验 URL 是否为允许访问的公网 HTTP/HTTPS 地址。

    输入:
        url: 待校验 URL。

    输出:
        None。校验通过时不返回值。

    异常:
        ValueError: URL scheme 非 HTTP/HTTPS，或目标为本地/内网/保留地址时抛出。
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL 必须以 http:// 或 https:// 开头")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL 缺少主机名")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host.endswith(".localhost") or host.endswith(".local"):
            raise ValueError(f"禁止访问本地域名: {host}")
        return

    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"禁止访问内网/本地地址: {host}")


def _is_binary_content(content_type: str) -> bool:
    """判断响应 Content-Type 是否明显是二进制内容。

    输入:
        content_type: HTTP 响应的 Content-Type header。

    输出:
        bool。True 表示应该拒绝作为文本处理。
    """

    lowered = content_type.lower()
    return any(marker in lowered for marker in _BINARY_MARKERS)


def _convert_text(raw_html: str, *, content_type: str, fmt: str) -> str:
    """根据输出格式转换响应文本。

    输入:
        raw_html: 原始响应文本。
        content_type: HTTP Content-Type。
        fmt: 输出格式，可选 markdown、text、html。

    输出:
        转换后的文本字符串。
    """

    if "html" not in content_type.lower():
        return raw_html

    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "object", "embed", "svg"]):
        tag.decompose()

    if fmt == "html":
        return str(soup)
    if fmt == "text":
        return " ".join(soup.get_text(" ").split())
    return html_to_markdown(str(soup), heading_style="ATX").strip()


def _json_result(payload: dict[str, Any], *, ok: bool) -> ToolResult:
    """把字典包装为 JSON 文本 ToolResult。

    输入:
        payload: 要序列化给模型阅读的结构化结果。
        ok: 工具调用是否成功。

    输出:
        ToolResult。text 是格式化 JSON，metadata.ok 等于 ok。
    """

    return ToolResult(text=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"ok": ok})