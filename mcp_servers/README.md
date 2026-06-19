# 添加 MCP 工具

MCP (Model Context Protocol) 允许你接入外部工具服务，扩展 Agent 的能力。

## 两种添加方式

### 方式一：直接编辑配置文件

编辑 `.raven/mcp_servers.json`。有两种连接方式：本地子进程（`command`）和远程 HTTP 服务（`url`），二选一：

```json
{
  "servers": {
    "amap-maps": {
      "command": null,
      "env": {},
      "cwd": null,
      "url": "https://mcp.amap.com/mcp?key=<Your API Key>"
    },
    "my-local-server": {
      "command": ["python", "mcp_servers/my_server.py"],
      "env": {},
      "cwd": null,
      "url": null
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `command` | 启动命令及参数列表，或 null（以 `url` 方式连接时） |
| `env` | 环境变量 |
| `cwd` | 工作目录，或 null |
| `url` | MCP Server 的 HTTP/SSE 地址，或 null（以 `command` 方式启动时） |

编辑 `config.toml` 将名称加入 `auto_connect`，即可启动时自动连接：

```toml
[mcp]
enabled = true
auto_connect = ["amap-maps"]
```

### 方式二：和 Agent 对话添加

直接告诉 Agent 你想添加什么 MCP 服务，它会调用 `mcp_add` 工具帮你完成。例如：

> "帮我添加一个高德地图的 MCP 工具，名字叫 amap-maps，URL 是 https://mcp.amap.com/mcp?key=<你的高德Key>"

Agent 会自动写入配置文件并连接。

## 内置 MCP Server

### 苹果健康数据（health_server.py）

`mcp_servers/health_server.py` 是一个定制化的健康数据 MCP 工具，数据流为：

```
iPhone 健康 App → 快捷指令自动导出 → Cloudflare Worker → 本工具拉取
```

提供两个工具：

| 工具 | 用途 |
|------|------|
| `get_health_context` | 获取当日步数、睡眠、心率、HRV 等健康快照 |
| `get_proactive_events` | 返回健康异常告警（心率偏高、睡眠不足、暴晒），供 Proactive 主动推送 |

配置文件 `mcp_servers/config.toml`：

```toml
[cloudflare]
worker_url = "https://your-worker.xxx.workers.dev"

[network]
proxy_url = "http://127.0.0.1:7890"     # 可选，国内访问 Cloudflare 可能需要 VPN
```

如果你使用的其他品牌设备或者不想配置CF，也可以自行替换成开源的 Health MCP 等方案。

## 添加后效果

连接成功后，MCP server 提供的工具会自动注册到 Agent 的工具列表中，LLM 可以直接调用。比如高德地图会注册 `maps_weather`、`maps_search`、`maps_geo` 等工具。

## 添加 Proactive 数据源

将已连接的 MCP 工具注册为 Proactive 数据源后，Agent 才会定时拉取并决策是否推送。同样有两种方式：

### 方式一：直接编辑配置文件

编辑 `.raven/proactive_sources.json`：

```json
{
  "version": 1,
  "sources": [
    {
      "name": "health-alert",
      "server": "health",
      "channel": "alert",
      "get_tool": "get_proactive_events",
      "args": {},
      "enabled": true
    },
    {
      "name": "health-context",
      "server": "health",
      "channel": "context",
      "get_tool": "get_health_context",
      "args": {},
      "enabled": true
    },
    {
      "name": "rss-content",
      "server": "rss",
      "channel": "content",
      "get_tool": "get_posts",
      "poll_tool": "refresh_feeds",
      "ack_tool": "mark_read",
      "args": { "unread_only": true, "limit": 50 },
      "enabled": true
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `name` | 数据源名称，随意起 |
| `server` | 已连接的 MCP server 名称 |
| `channel` | 数据通道：`alert` / `content` / `context` |
| `get_tool` | 获取数据的 MCP 工具名 |
| `poll_tool` | 预抓取工具名（content 通道推荐配，用于提前拉取） |
| `ack_tool` | 已读标记工具名（content 通道推荐配，用于去重） |
| `args` | 调用 get_tool 时的默认参数 |
| `enabled` | 是否启用 |

### 方式二：和 Agent 对话添加

直接告诉 Agent，它会调用 `proactive_source_add` 工具：

> "帮我把 health server 的 get_proactive_events 工具注册为 alert 数据源，名字叫 health-alert"

### 快速添加示例

以下是基于项目已内置的 `health_server.py` 的完整示例。它同时提供两个工具：`get_proactive_events`（alert）和 `get_health_context`（context），一个 server 对应两个数据源。

**1. 在对话中直接告诉 Agent：**

> "帮我添加一个 MCP server，名字叫 health，用本地进程启动，命令是 python mcp_servers/health_server.py"
>
> "把 health 的 get_proactive_events 注册为 alert 数据源，名字叫 health-alert"
>
> "把 health 的 get_health_context 注册为 context 数据源，名字叫 health-context"

**2. 或编辑配置文件一步到位：**

`mcp_servers.json`：
```json
{
  "servers": {
    "health": {
      "command": ["python", "mcp_servers/health_server.py"],
      "env": {},
      "cwd": null,
      "url": null
    }
  }
}
```

`proactive_sources.json`：
```json
{
  "version": 1,
  "sources": [
    { "name": "health-alert",  "server": "health", "channel": "alert",   "get_tool": "get_proactive_events" },
    { "name": "health-context","server": "health", "channel": "context", "get_tool": "get_health_context" }
  ]
}
```

同理，其他如 RSS、天气、高德地图等均可按此模式添加。

### 三种数据通道

| 通道 | 用途 | 触发时机 |
|------|------|----------|
| `alert` | 高优先级告警 | 每次 tick 都拉取，有内容直接推送 |
| `content` | 内容流（RSS、新闻等） | tick 时拉取，LLM 逐条评分后选择性推送 |
| `context` | 背景上下文（天气、健康等） | tick 时拉取，注入 Judge prompt 辅助决策，不直接推送 |

### 返回值格式要求

Proactive 数据源工具必须返回 JSON 数组。每个事件的字段会被 `contracts.py` 中的 `normalize_*` 函数消费——你只需要按以下字段返回即可。

**alert 通道**：对应 `normalize_alert()`（[源码](../src/raven_agent/proactive/contracts.py)）

| 字段 | 必填 | 说明 |
|------|------|------|
| `event_id` | ✅ | 事件唯一 ID，也接受 `id` |
| `title` | ✅ | 告警标题 |
| `content` | ✅ | 告警详情，也接受 `body` |
| `severity` | ✅ | `high` / `medium` / `low` |
| `suggested_tone` | | 建议推送语气：`direct` / `neutral` / `gentle` |
| `ack_server` | | 用于 ACK 路由的 server 名，不配则 item_id 前缀为 `?`（alert 通常不需要 ACK） |
| `metrics` | | 指标字典，最多 8 个 key，value 超过 60 字符会被截断 |

```json
[
  {
    "event_id": "health_rhr_75",
    "title": "静息心率偏高",
    "content": "今日静息心率 75 次/分，高于个人基线...",
    "severity": "medium",
    "suggested_tone": "gentle",
    "metrics": { "current_bpm": 75, "baseline_bpm": 62 }
  }
]
```

**content 通道**：对应 `normalize_content()`（[源码](../src/raven_agent/proactive/contracts.py)）

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | ✅ | 内容唯一 ID |
| `title` | ✅ | 内容标题 |
| `source` | ✅ | 来源名称，也接受 `source_name` |
| `url` | | 原文链接 |
| `ack_server` | ⚠️ | 如果该 source 配了 `ack_tool`，此处必须填对应 server 名，否则 ACK 无法路由，`mark_read` 不会被调用，每次 tick 都会重复返回相同条目 |

```json
[
  {
    "id": "12345",
    "title": "Python 3.13 发布",
    "source": "Python.org",
    "url": "https://python.org/..."
  }
]
```

> `ack_server` 对 content 通道至关重要：`ack_events()` 通过 `item_id.split(":", 1)[0]` 找到 server → 调 `mark_read`。如果返回的事件不带 `ack_server`，`item_id` 只有数字，ACK 永远不触发，RSS 源会反复推送已读内容。

**context 通道**：格式宽松，返回扁平 dict 即可，各字段会注入 Judge prompt 辅助推送决策。

## 自建 MCP Server

用 `fastmcp` 几行即可写一个简易mcp：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tool")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

mcp.run()
```

更完整示例见 `mcp_servers/echo_server.py`。

## 桥接外部 MCP：让第三方工具适配 Proactive

当使用外部开源 MCP（如 `@0xquinto/rss-mcp`）作为 proactive 数据源时，其返回格式通常不符合本系统的字段规范。此时需要写一个薄封装层（wrapper MCP server），在内部透传原始工具的同时，对主动通道用的工具做字段映射。

`mcp_servers/rss_server.py` 是一个完整示例：

```
raven-agent ←→ rss_server.py (FastMCP stdio，薄封装)
                    │
                    │ fastmcp.Client (子进程)
                    │
                    ▼
          npx @0xquinto/rss-mcp (外部 MCP)
```

核心做法很简洁——只映射主动通道关心的 `get_posts`、`ack_tool`、`poll_tool`，其余工具全透传：

```python
def _map_post_for_proactive(post: dict) -> dict:
    """rss-mcp 原始字段 → proactive ContentContract 字段"""
    return {
        "kind": "content",
        "id": str(post.get("id", "")),
        "title": post.get("title") or "",
        "source_name": post.get("feed_title") or "",   # feed_title → source_name
        "url": post.get("url") or "",
        "summary": post.get("summary") or "",
        "published_at": post.get("published_at") or "",
        "is_read": post.get("is_read", 0),
    }
```

- 保留了原始 `id`、`title`、`url` 等字段，透传给 `ContentContract`
- 将 `feed_title` 映射为 `source_name`（RSS 源名称）
- 其余工具（`add_feed`、`remove_feed`、`import_opml` 等）直接透传，零改动
- `mark_read` 也做了参数名桥接：框架 ACK 传 `event_ids`，内部转 `post_ids` 调原始 MCP

当你想用外部 MCP 作为 proactive 源时，照这个模式写一个 wrapper 即可——工具透传保被动通道兼容，字段映射让主动通道正确消费。

## 参考资料

- [PulseMCP](https://www.pulsemcp.com/) — MCP Server 目录
- [MCP Hub](https://github.com/modelcontextprotocol/servers) — 官方推荐
