---
name: rss-manage
description: 管理 RSS 订阅源
---

## 目标

帮助用户增删 RSS 订阅源，查询和筛选文章。

## 工作流程

1. 理解用户意图（添加/删除/列出/搜索）
2. 如需添加 → 调用 rss MCP server 的 add_feed 工具
3. 如需删除 → 先 list_feeds，用户确认后 remove_feed
4. 如需列出 → list_feeds 返回所有已订阅源
5. 如需搜索 → get_posts 带 search 参数

## 要求

- 删除操作必须先让用户确认
- 添加前告知用户即将添加的 URL