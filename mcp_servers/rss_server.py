"""
rss_server.py —— rss-mcp (0xquinto) 的薄封装 MCP Server。

设计原则：
  1. 保留 rss-mcp 所有原始工具名和参数签名 → 被动通道零感知切换
  2. 对主动通道需要的工具做字段名映射 → normalize_content() 正确消费
  3. 每个参数用 Annotated + Field 完整描述 → LLM 能精确传参
  4. 透传类工具不做修改 → 减少维护成本

架构：
  raven-agent ←→ rss_server.py (FastMCP stdio)
                     │
                     │ fastmcp.Client (子进程)
                     │
                     ▼
               npx @0xquinto/rss-mcp → ~/.rss-mcp/rss.db

字段映射（仅 get_posts）：
  rss-mcp 原始:  {id, title, feed_title, url, summary, published_at, is_read, ...}
  adapter 映射后: {kind, id, title, source_name,   url, summary, published_at, ...}
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Optional
from pathlib import Path

from pydantic import Field
from mcp.server.fastmcp import FastMCP
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

logger = logging.getLogger(__name__)

# ── MCP 实例 ──────────────────────────────────────────────────────────

mcp = FastMCP("rss-adapter")

# ── 内部：管理 rss-mcp 子进程 ────────────────────────────────────────

_rss_client: Client | None = None


async def _get_client() -> Client:
    """懒加载连接 rss-mcp 子进程。全局复用同一个连接。"""
    global _rss_client
    if _rss_client is None:
        _rss_client = await _connect_rss()
    return _rss_client


async def _connect_rss() -> Client:
    """创建并连接一个新的 rss-mcp 子进程 Client。"""
    client = Client(
        StdioTransport(command="npx", args=["-y", "@0xquinto/rss-mcp"])
    )
    await client.__aenter__()
    logger.info("[rss-adapter] 已连接到 rss-mcp")
    return client


async def _call_rss(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    """调用 rss-mcp 的远端工具，返回 Python 对象。

    rss-mcp 返回的 MCP content 统一为 JSON 文本，这里做 JSON.parse。
    解析失败时返回原始文本，保证不丢数据。

    npx 子进程崩溃后自动重连并重试一次。
    """
    global _rss_client

    async def _do_call(client: Client) -> Any:
        result = await client.call_tool(tool_name, arguments or {})
        # fastmcp ToolResult: 单 text block → 取其 text
        if hasattr(result, "content") and result.content:
            block = result.content[0]
            text = block.text if hasattr(block, "text") else str(block)
        elif isinstance(result, str):
            text = result
        else:
            text = "[]"
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    client = await _get_client()
    try:
        return await _do_call(client)
    except RuntimeError as e:
        if "not connected" not in str(e).lower():
            raise
        logger.warning(
            "[rss-adapter] rss-mcp 连接丢失，重连并重试 %s", tool_name,
        )
        # 清理旧连接
        try:
            await _rss_client.__aexit__(None, None, None)
        except Exception:
            pass
        _rss_client = None
        # 建立新连接并重试
        _rss_client = await _connect_rss()
        return await _do_call(_rss_client)


# ── 字段映射 ─────────────────────────────────────────────────────────

def _map_post_for_proactive(post: dict[str, Any]) -> dict[str, Any]:
    """将 rss-mcp 的 post 映射为 proactive ContentContract 可消费的格式。

    同时保留原始字段以便 raw 透传，不影响被动通道使用。
    """
    return {
        # ── 主动通道路由 + normalize_content 消费 ──
        "kind": "content",
        "id": str(post.get("id", "")),
        "title": post.get("title") or "",
        "source_name": post.get("feed_title") or post.get("feed", "") or "RSS",
        "url": post.get("url") or "",
        # ── 以下字段进入 raw，供 Judge LLM 参考 ──
        "summary": post.get("summary") or "",
        "published_at": post.get("published_at") or "",
        "is_read": post.get("is_read", 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════════════

# ── 透传类工具（参数与 rss-mcp 完全一致，零修改）─────────────────────


@mcp.tool()
async def list_feeds() -> str:
    """列出所有已订阅的 RSS/Atom 源。

    返回 JSON 数组，每个 feed 包含：
    - id: 订阅源 ID（可用于 get_posts/remove_feed/refresh_feeds）
    - url: RSS/Atom 订阅地址
    - title: 订阅源名称
    - site_url: 站点主页
    - last_fetched: 最近抓取时间
    - created_at: 添加时间
    """
    result = await _call_rss("list_feeds")
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def add_feed(
    url: Annotated[
        str,
        Field(description="RSS/Atom feed URL to subscribe to. Example: 'https://rsshub.app/zhihu/hotlist'."),
    ],
) -> str:
    """订阅一个新的 RSS/Atom 源。重复订阅同一 URL 会报错。

    订阅后需调用 refresh_feeds 抓取最新文章。
    """
    result = await _call_rss("add_feed", {"url": url})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def remove_feed(
    feed_id: Annotated[
        int,
        Field(description="ID of the feed to remove. Use list_feeds to find the ID."),
    ],
) -> str:
    """删除一个已订阅的 RSS 源及其所有文章。操作不可逆。"""
    result = await _call_rss("remove_feed", {"feed_id": feed_id})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def import_opml(
    file_path: Annotated[
        str,
        Field(description="Absolute path to an OPML file on the local filesystem. Example: '/home/user/feeds.opml'."),
    ],
) -> str:
    """从 OPML 文件批量导入 RSS 订阅。重复的 URL 会自动跳过。

    返回 {imported: N, total_in_file: M}，imported 是实际新增数。
    """
    result = await _call_rss("import_opml", {"file_path": file_path})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def refresh_feeds(
    feed_id: Annotated[
        Optional[int],
        Field(default=None, description="Optional specific feed ID to refresh. Omit to refresh all feeds."),
    ] = None,
) -> str:
    """抓取全部（或指定）订阅源的最新文章。内置 15 分钟最小刷新间隔，
    支持 ETag 条件请求，不会重复抓取未更新的源。

    返回 {refreshed: N, new_posts: N, skipped: N, errors: [...]}。
    """
    args: dict[str, Any] = {}
    if feed_id is not None:
        args["feed_id"] = feed_id
    result = await _call_rss("refresh_feeds", args)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_post_content(
    post_id: Annotated[
        int,
        Field(description="ID of the post to fetch full content for."),
    ],
    max_length: Annotated[
        int,
        Field(default=5000, ge=100, le=100000, description="Maximum characters to return per chunk (100-100000). Use a large value like 100000 for full content."),
    ] = 5000,
    offset: Annotated[
        int,
        Field(default=0, ge=0, description="Character offset to start reading from (0-based). Use with max_length to paginate through long content."),
    ] = 0,
) -> str:
    """获取单篇文章的正文。首次调用时自动从原文 URL 抓取并提取正文
    （使用 Mozilla Readability），之后从本地缓存读取。

    返回 JSON 包含：content（内容文本）、truncated（是否截断）、total_length（原始总长度）、
    offset（当前偏移）、chunk_length（本段长度）。
    """
    result = await _call_rss("get_post_content", {
        "post_id": post_id,
        "max_length": max_length,
        "offset": offset,
    })
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_daily_digest(
    hours: Annotated[
        int,
        Field(default=24, ge=1, le=168, description="Hours to look back (1-168, i.e. up to 7 days)."),
    ] = 24,
    max_summary_length: Annotated[
        int,
        Field(default=300, ge=50, le=2000, description="Maximum characters per summary before truncation."),
    ] = 300,
) -> str:
    """获取近期文章摘要日报。不区分已读/未读，按发布时间倒序排列。

    返回 JSON 包含 period（时间范围）、total_posts（文章总数）、feeds（来源数）、
    posts（文章列表，含 id / feed / title / summary / url / published_at）。
    """
    result = await _call_rss("get_daily_digest", {
        "hours": hours,
        "max_summary_length": max_summary_length,
    })
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def mark_unread(
    post_ids: Annotated[
        list[int],
        Field(min_length=1, max_length=1000, description="Array of post IDs to mark as unread. Minimum 1, maximum 1000 per call."),
    ],
) -> str:
    """将指定的文章标记为未读。用于撤销 mark_read 操作。"""
    result = await _call_rss("mark_unread", {"post_ids": post_ids})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_popular_posts(
    days: Annotated[
        int,
        Field(default=7, ge=1, le=30, description="Days to look back (1-30)."),
    ] = 7,
    limit: Annotated[
        int,
        Field(default=10, ge=1, le=100, description="Maximum posts to return (1-100)."),
    ] = 10,
) -> str:
    """按 Hacker News 评分排名获取近期热门文章。会并发查询每篇文章的 HN 评分，
    按 score 降序排列后取前 N 条。

    返回 JSON：period（时间范围）、total_checked（检查的文章总数）、
    posts（排名后的文章列表，含 id / feed / title / url / published_at / hn_score / hn_comments / hn_url）。
    """
    result = await _call_rss("get_popular_posts", {"days": days, "limit": limit})
    return json.dumps(result, ensure_ascii=False)



@mcp.tool()
async def get_posts(
    feed_id: Annotated[
        Optional[int],
        Field(default=None, description="Filter posts by feed ID. Omit to get posts from all subscribed feeds. Use list_feeds to find feed IDs."),
    ] = None,
    limit: Annotated[
        int,
        Field(default=50, ge=1, le=500, description="Maximum number of posts to return (1-500)."),
    ] = 50,
    offset: Annotated[
        int,
        Field(default=0, ge=0, description="Pagination offset (0-based). Combine with limit: offset=0,limit=50 for page 1; offset=50,limit=50 for page 2."),
    ] = 0,
    unread_only: Annotated[
        bool,
        Field(default=False, description="When true, return only unread posts. Useful for proactive fetching of new content."),
    ] = False,
    search: Annotated[
        Optional[str],
        Field(default=None, description="FTS5 full-text search query. Supports: simple terms ('python'), boolean operators ('python AND tutorial'), prefix matching ('prog*'), and phrase queries ('\"machine learning\"'). Searches across title, summary, and content."),
    ] = None,
    since: Annotated[
        Optional[str],
        Field(default=None, description="ISO 8601 date/time string. Only return posts published after this time. Example: '2026-06-01T00:00:00Z' or '2026-06-01'."),
    ] = None,
) -> str:
    """查询文章列表。支持按订阅源过滤、全文搜索、已读状态过滤、时间范围过滤和分页。

    返回 JSON 数组，每篇文章包含：
    - kind: "content"（固定值）
    - id: 文章 ID
    - title: 标题
    - source_name: 来源名称（如 "HLTV.org"、"GitHub Trending"）
    - url: 原文链接
    - summary: 摘要
    - published_at: 发布时间（ISO 8601）
    - is_read: 已读标记（0=未读, 1=已读）

    **注意**：此工具返回的字段名已针对 raven-agent 主动通道规范化，
    原始 rss-mcp 的 feed_title 字段已映射为 source_name。
    """
    # 构建参数（只传有值的，避免覆盖远端默认值）
    args: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "unread_only": unread_only,
    }
    if feed_id is not None:
        args["feed_id"] = feed_id
    if search is not None:
        args["search"] = search
    if since is not None:
        args["since"] = since

    raw = await _call_rss("get_posts", args)

    # 如果不是列表（可能是错误对象），原样返回
    if not isinstance(raw, list):
        return json.dumps(raw, ensure_ascii=False)

    # 逐条字段映射
    mapped = [_map_post_for_proactive(p) for p in raw if isinstance(p, dict)]
    return json.dumps(mapped, ensure_ascii=False)


# ── ACK 工具 ────────────────────────────────────────────────────────
#
# 参数名 event_ids 适配框架硬编码的 {"event_ids": [...]}，
# 内部转为 post_ids 调 rss-mcp——与返回值字段映射同样的数据映射思路。


@mcp.tool()
async def mark_read(
    event_ids: Annotated[
        list[int | str],
        Field(min_length=1, max_length=1000, description="Array of post IDs to mark as read. Both passive and proactive channels use this parameter."),
    ],
) -> str:
    """将文章标记为已读。已读文章在 unread_only=true 的查询中不再出现。"""
    # 统一 int() 转换：LLM 传 int 直通，框架 ACK 传 str 也能转
    ids = [int(eid) for eid in event_ids]
    result = await _call_rss("mark_read", {"post_ids": ids})
    return json.dumps(result, ensure_ascii=False)


# ── 启动入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()