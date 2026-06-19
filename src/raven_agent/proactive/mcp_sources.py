"""
proactive/mcp_sources.py —— proactive_sources.json 管理 + MCP 源拉取。

三层架构：
  1. SourceStore           — proactive_sources.json 读写
  2. McpSourceFetcher      — 通过 McpServerRegistry.call_tool() 拉取数据
  3. 模块级便利入口        — fetch_alerts / fetch_content / fetch_context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from raven_agent.persistence import load_json, save_json
from raven_agent.proactive.contracts import (
    AlertContract,
    ContentContract,
    ContextContract,
    normalize_alert,
    normalize_content,
    normalize_context,
)

logger = logging.getLogger(__name__)

_POLL_TOOL_TIMEOUT = 180.0


# ── SourceStore ──────────────────────────────────────────────────────


class SourceStore:
    """管理 proactive_sources.json 的读写。

    参数:
        config_path: proactive_sources.json 的完整路径。
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = Path(config_path)

    def load_sources(self) -> list[dict[str, Any]]:
        """加载所有启用的 source 配置。

        输出:
            启用的 source 配置字典列表。文件不存在/损坏时返回空列表。
        """
        raw = load_json(self._config_path, default={})
        if not isinstance(raw, dict):
            logger.warning("[mcp_sources] proactive_sources.json 不是 dict，返回空")
            return []
        sources = raw.get("sources", [])
        if not isinstance(sources, list):
            return []
        enabled = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
        logger.debug(
            "[mcp_sources] 已加载 %d/%d 个启用的 source",
            len(enabled), len(sources),
        )
        return enabled

    def save_sources(self, sources: list[dict[str, Any]]) -> None:
        """持久化 source 列表。

        输入:
            sources: 完整 source 配置列表。

        输出:
            None。
        """
        save_json(self._config_path, {"version": 1, "sources": sources})

    def add_source(self, source_cfg: dict[str, Any]) -> str:
        """添加或更新一个 source。

        输入:
            source_cfg: source 配置 dict（至少含 name）。

        输出:
            "ok" 或错误描述字符串。
        """
        name = str(source_cfg.get("name", "")).strip()
        if not name:
            return "错误：name 不能为空"
        server = str(source_cfg.get("server", "")).strip()
        if not server:
            return "错误：server 不能为空"
        sources = self.load_sources()
        for i, s in enumerate(sources):
            if s.get("name") == name:
                sources[i] = source_cfg
                self.save_sources(sources)
                logger.info("[mcp_sources] 更新 source=%s server=%s", name, server)
                return "ok"
        sources.append(source_cfg)
        self.save_sources(sources)
        logger.info("[mcp_sources] 新增 source=%s server=%s", name, server)
        return "ok"

    def remove_source(self, name: str) -> str:
        """移除一个 source。

        输入:
            name: source 名称。

        输出:
            "ok" 或错误描述字符串。
        """
        name = str(name).strip()
        sources = self.load_sources()
        new_sources = [s for s in sources if s.get("name") != name]
        if len(new_sources) == len(sources):
            names = [s.get("name", "?") for s in sources]
            return f"未找到 source {name!r}，当前已有：{names}"
        self.save_sources(new_sources)
        logger.info("[mcp_sources] 移除 source=%s", name)
        return "ok"

    def list_sources(self) -> str:
        """列出所有 source。

        输出:
            格式化的 source 列表字符串。
        """
        sources = self.load_sources()
        if not sources:
            return "当前没有已配置的 proactive source。"
        lines = []
        for s in sources:
            name = s.get("name", "?")
            server = s.get("server", "?")
            channel = s.get("channel", "?")
            get_tool = s.get("get_tool", "")
            lines.append(
                f"- {name}  [{channel}]  server={server}"
                f"  get={get_tool or 'default'}"
            )
        return "\n".join(lines)


# ── McpSourceFetcher ────────────────────────────────────────────────


class McpSourceFetcher:
    """通过 McpServerRegistry 拉取三代数据：alert / content / context。

    不创建新连接——复用第 29 章的 McpServerRegistry 常驻连接。
    每个 source 声明其 server 和 get_tool，fetcher 负责 call 和 normalize。

    参数:
        mcp_registry: McpServerRegistry 实例。
        source_store: SourceStore 实例。
    """

    def __init__(self, mcp_registry: Any, source_store: SourceStore) -> None:
        self._registry = mcp_registry
        self._source_store = source_store

    # ── 公共 API ────────────────────────────────────────────────────

    async def fetch_alerts(
        self,
        sources: list[dict[str, Any]] | None = None,
    ) -> list[AlertContract]:
        """拉取所有 alert 源的数据。

        输入:
            sources: 可选的预加载 source 列表。调用方如果已经在外部读过了
                proactive_sources.json，可以直接传入以节省一次磁盘 I/O；
                传 None 则内部自动调用 load_sources()。

        输出:
            AlertContract 列表。
        """
        return await self._fetch_by_channel("alert", sources=sources)

    async def fetch_content(
        self,
        sources: list[dict[str, Any]] | None = None,
    ) -> list[ContentContract]:
        """拉取所有 content 源的数据。

        输入:
            sources: 可选的预加载 source 列表。

        输出:
            ContentContract 列表。
        """
        return await self._fetch_by_channel("content", sources=sources)

    async def fetch_context(
        self,
        sources: list[dict[str, Any]] | None = None,
    ) -> list[ContextContract]:
        """拉取所有 context 源的数据。

        输入:
            sources: 可选的预加载 source 列表。

        输出:
            ContextContract 列表。
        """
        return await self._fetch_by_channel("context", sources=sources)

    async def poll_feeds(self) -> None:
        """对 content 源执行预轮询（poll_tool）。

        每个 content source 可配置 poll_tool，在 proactive tick 之前
        触发远端抓取/更新，这样随后的 fetch_content 能拿到最新数据。

        输出:
            None。
        """
        for src in self._source_store.load_sources():
            poll_tool = str(src.get("poll_tool", "")).strip()
            if not poll_tool:
                continue
            server = str(src.get("server", "")).strip()
            try:
                await self._registry.call_tool(
                    server, poll_tool, {}, timeout=_POLL_TOOL_TIMEOUT,
                )
                logger.info("[mcp_sources] poll 完成: %s.%s", server, poll_tool)
            except Exception as e:
                logger.warning(
                    "[mcp_sources] poll 失败 %s.%s: %s", server, poll_tool, e,
                )

    async def ack_events(self, events: list) -> None:
        """对已处理的事件执行 ack。

        对有 ack_tool 的 source，收集其事件 ID，批量确认。

        输入:
            events: 已处理的 AlertContract / ContentContract 列表。

        输出:
            None。
        """
        from raven_agent.proactive.contracts import AlertContract, ContentContract

        ack_map: dict[str, tuple[str, list[str]]] = {}
        for src in self._source_store.load_sources():
            ack_tool = str(src.get("ack_tool", "")).strip()
            if ack_tool:
                ack_map[src["server"]] = (ack_tool, [])

        for event in events:
            if isinstance(event, (AlertContract, ContentContract)):
                parts = event.item_id.split(":", 1)
                if len(parts) == 2 and parts[0] in ack_map:
                    ack_map[parts[0]][1].append(parts[1])

        for server, (ack_tool, ids) in ack_map.items():
            if not ids:
                continue
            try:
                await self._registry.call_tool(
                    server, ack_tool, {"event_ids": ids},
                )
                logger.info(
                    "[mcp_sources] ack %d 事件 via %s.%s",
                    len(ids), server, ack_tool,
                )
            except Exception as e:
                logger.warning(
                    "[mcp_sources] ack 失败 %s.%s: %s", server, ack_tool, e,
                )

    # ── 内部 ────────────────────────────────────────────────────────

    async def _fetch_by_channel(
        self,
        channel: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> list:
        """按 channel 拉取并归一化数据。

        输入:
            channel: "alert" / "content" / "context"。
            sources: 预加载的 source 列表；传 None 则内部调用 load_sources()。
                当 collect_external() 并发拉取 alert + content + context 时，
                Sensor 先统一读一次磁盘，然后把同一份 sources 传给三个 fetch，
                避免并发 I/O 争抢。

        输出:
            对应合同类型的列表。
        """
        result: list = []
        for src in (sources if sources is not None else self._source_store.load_sources()):
            src_channel = str(src.get("channel", "")).strip().lower()
            # 路由过滤
            if channel == "context":
                if src_channel != "context":
                    continue
            else:
                if src_channel == "context":
                    continue
                if channel == "alert" and src_channel == "content":
                    continue
                if channel == "content" and src_channel == "alert":
                    continue

            server = str(src.get("server", "")).strip()
            if server not in self._registry.connected_server_names():
                logger.debug(
                    "[mcp_sources] server %r 未连接，跳过 source %s",
                    server, src.get("name"),
                )
                continue

            get_tool = str(src.get("get_tool", "")) or (
                "get_context" if channel == "context" else "get_proactive_events"
            )
            args = dict(src.get("args", {}) or {})

            try:
                data = await self._registry.call_tool(server, get_tool, args)
            except Exception as e:
                logger.warning(
                    "[mcp_sources] fetch 失败 %s.%s: %s", server, get_tool, e,
                )
                continue

            if channel == "context":
                result.extend(self._extract_context_items(data, server=server))
            else:
                result.extend(
                    self._extract_proactive_events(data, server=server, kind=channel)
                )

        return result

    @staticmethod
    def _extract_proactive_events(
        data: Any,
        *,
        server: str,
        kind: str,
    ) -> list:
        """从 MCP 返回数据中提取指定 kind 的事件并规范化。

        兼容两种 MCP 工具返回格式：
        1. list[dict] — 直接可迭代（内存 / 原生 Python MCP 工具）
        2. JSON 字符串 — 需要先 json.loads（stdio MCP 工具如 rss_server）

        输入:
            data: MCP call_tool 返回值（list / str / 其他）。
            server: 来源 server 名称。
            kind: 期望的事件 kind ("alert" / "content")。

        输出:
            AlertContract 或 ContentContract 列表。
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "[mcp_sources] %s.%s 返回非法 JSON，已丢弃",
                    server, kind,
                )
                return []
        if not isinstance(data, list):
            return []
        result: list = []
        for event in data:
            if not isinstance(event, dict):
                continue
            enriched = dict(event)
            enriched.setdefault("ack_server", server)
            if kind == "alert":
                result.append(normalize_alert(enriched))
            else:
                result.append(normalize_content(enriched))
        return result

    @staticmethod
    def _extract_context_items(data: Any, *, server: str) -> list[ContextContract]:
        """从 MCP 返回数据中提取 context 条目。

        兼容四种返回形态：
        1. 单个 dict → 长度为 1 的列表
        2. list[dict] → 逐条包装
        3. JSON 字符串 → 先 json.loads 再按 1/2 处理
        4. 纯文本字符串 → 包装为 {"text": data} 的 context 条目

        输入:
            data: MCP call_tool 返回值（dict / list / str / 其他）。
            server: 来源 server 名称。

        输出:
            ContextContract 列表。
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                # 非 JSON 文本（如 health 的纯文本摘要）→ 包装为 text 条目
                return [normalize_context({"text": data}, source=server)]
        if isinstance(data, dict):
            return [normalize_context(data, source=server)]
        if isinstance(data, list):
            result: list[ContextContract] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                result.append(normalize_context(item, source=server))
            return result
        return []