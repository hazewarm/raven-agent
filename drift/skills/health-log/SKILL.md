---
name: health-log
description: 定期查询健康数据（health MCP），记录时间戳和关键指标到日志。
---

## 目标

利用已连接的 health MCP server 的 `get_health_context` 工具，获取实时健康数据并记录到日志。为健身追踪提供数据点。

## 工作文件

- skills/health-log/log.md：健康数据日志

## 工作流程

1. mount_server health（确保 health MCP 工具可用）
2. 调用 mcp_health__get_health_context 获取当前健康数据
3. write_text_file 追加一条带时间戳的记录到 log.md（不覆盖已有内容）
4. 如果获取到值得关注的数据（如异常指标），可考虑 message_push
5. finish_drift（默认 silent）
