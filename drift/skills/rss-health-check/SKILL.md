---
name: rss-health-check
description: 定期检查 RSS 订阅源健康状态，发现异常源时推送 Telegram 通知
---

## 目标

后台定期检查所有 RSS 订阅源是否正常工作。如果某个 feed URL 不可达、源站返回错误或 feed 解析失败，通过 Telegram 推送通知用户。如果之前异常的 feed 恢复了，也通知用户。

## 工作文件

- skills/rss-health-check/state.md：上次检查结果记录（异常 feed 列表和检查时间）

## 工作流程

### 1. 挂载 RSS MCP 工具

mount_server server=rss

### 2. 获取订阅源列表

调用 mcp_rss__list_feeds() 获取所有已订阅的 feed 列表。解析 JSON，提取每个 feed 的 id、title、url、last_fetched。

### 3. 刷新所有订阅源

调用 mcp_rss__refresh_feeds()（不传 feed_id，刷新全部）。解析返回的 JSON，提取：

- `refreshed`：成功刷新的数量
- `skipped`：跳过的数量（正常，可能是 15 分钟内刚刷新过）
- `new_posts`：新文章数
- `errors`：失败的 feed 列表（关键字段）

errors 中每个元素可能包含 feed 的 name/url 和错误原因（连接超时、HTTP 错误、DNS 失败、解析错误等）。

### 4. 读取上次状态

read_file skills/rss-health-check/state.md。如果不存在，说明是首次运行，视作之前没有异常 feed。

state.md 格式（自行维护）：
```
last_check: 2026-06-27T10:00:00
total_feeds: 5
broken_feeds:
  - title: 某订阅源
    url: https://example.com/feed.xml
    error: 连接超时
    since: 2026-06-25T08:00:00
```

### 5. 分析与决策

将本次 errors 中的 feed 与 state.md 中上次记录的 broken_feeds 对比：

**需要推送通知的情况：**
- 本次发现新的异常 feed（上次正常的，这次坏了）
- 之前异常的 feed 本次恢复了（在 errors 中消失了）

**不需要推送的情况：**
- 一切正常（errors 为空，之前也没有 broken_feeds）
- 异常 feed 跟上次完全一样，没有新变化（避免重复骚扰）

### 6. 更新状态文件

write_text_file 将本次检查结果写回 skills/rss-health-check/state.md，包含最新的检查时间、feed 总数和 broken_feeds 列表。

### 7. 推送通知（仅在需要时）

如果第 5 步判定需要推送，调用 message_push 发送通知。

消息用中文，简洁清晰，格式参考：
```
🔍 RSS 健康检查

⚠ 新增异常源：
• 知乎热榜 — 连接超时
  https://rsshub.app/zhihu/hotlist

✅ 已恢复：
• GitHub Trending（之前连接超时）

5 个源中 1 个异常。
```

如果本次无需推送，不要调用 message_push。

### 8. 收尾

如果推送了消息：
  finish_drift(skill_used="rss-health-check", one_line="发现 N 个异常源，已推送通知", next="下次检查时对比异常是否恢复", message_result="sent")

如果未推送：
  finish_drift(skill_used="rss-health-check", one_line="所有 RSS 源正常", next="继续定期检查", message_result="silent")

## 要求

- 不要对已经正常的 feeds 做额外操作（不调用 remove_feed、add_feed 等修改性工具）
- 如果 refresh_feeds 因为最小刷新间隔被跳过且无 errors，视为正常
- 优先使用 refresh_feeds 返回的 errors 字段判断异常
- 只在本轮服务，不要等待或确认——每次 drift 运行是独立的
- 消息必须用中文，格式简洁清晰
- 只在出现新异常或旧异常恢复时才推送，重复的异常不要反复推送
