# 编写 Drift Skill

Drift skill 是 Proactive 系统在空闲时执行的后台任务。用 Markdown 写操作指南，LLM 按照步骤自动执行。

## 1. 目录结构

```
drift/skills/<skill-name>/SKILL.md
```

## 2. SKILL.md 格式

```markdown
---
name: my-drift-skill              # 必填，与目录名一致
description: <一句话描述>          # 必填
---

## 目标

这个 skill 要完成什么任务。

## 工作流程

1. 第一步操作
2. 第二步操作
3. finish_drift

## 要求

- 约束规则
- 注意事项
```

## 3. 快速添加示例

以项目内置的 `drift/skills/health-log/` 为例——空闲时通过 `health` MCP server 获取健康数据并记录日志：

```bash
mkdir -p drift/skills/health-log
touch drift/skills/health-log/SKILL.md
```

**SKILL.md：**

```markdown
---
name: health-log
description: 定期查询健康数据（health MCP），记录时间戳和关键指标到日志。
---

## 目标

利用已连接的 health MCP server 获取实时健康数据并记录到日志。

## 工作文件

- skills/health-log/log.md：健康数据日志

## 工作流程

1. 调用 mcp_health__get_health_context 获取当前健康数据
2. write_text_file 追加一条带时间戳的记录到 log.md
3. 如果获取到异常指标，可 message_push 通知用户
4. finish_drift（默认 silent）
```

## 4. 关键规则

- Skill 被注入为 LLM 的 system context
- `always: true` 的 skill 每轮对话都注入，不要放太长的内容
- 结束时必须调用 `finish_drift`
- 放在 `drift/skills/` 下，由 Proactive 在空闲时触发

参考示例：`drift/skills/hello-world/SKILL.md`、`drift/skills/explore-curiosity/SKILL.md`
