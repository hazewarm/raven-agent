# 编写 Skill

Skill 是给 LLM 看的场景指南——告诉它在特定对话场景下该怎么处理。不是后台任务（后台任务见 [`drift/README.md`](../drift/README.md)）。

## 1. 目录结构

```
skills/<skill-name>/SKILL.md
```

## 2. SKILL.md 格式

```markdown
---
name: my-skill                    # 必填，与目录名一致
description: <一句话描述>          # 必填
always: false                     # true 表示每轮对话都注入，建议 false
---

## 何时触发

什么时候应该执行这个 skill。用具体例子说明。

## 工作流程

1. 第一步
2. 第二步
...

## 要求

- 约束和注意事项
```

## 3. 快速添加示例

以项目内置的 `skills/travel-plan/` 为例——告诉 LLM 用户想规划旅行时应该调用 `delegate_travel_planner` 工具：

```bash
mkdir -p skills/travel-plan
touch skills/travel-plan/SKILL.md
```

**SKILL.md：**

```markdown
---
name: travel-plan
description: 生成旅行规划 HTML 网页
---

## 何时触发

用户是否要一份完整的旅行规划。例如：
- "帮我规划北京3日游" → 需要
- "杭州有什么好吃的？" → 不需要，普通问答

## 工作流程

1. 确认目的地、日期、偏好（信息不全时追问补齐）
2. 调用 delegate_travel_planner(goal="<用户原始请求>")
3. 告知用户："规划已提交，3-5 分钟后通知您"
```

## 4. 和 Drift Skill 的区别

| | skills/ | drift/skills/ |
|---|---|---|
| **谁触发** | 用户对话触发 | Proactive 空闲时自动触发 |
| **典型用途** | 对话场景指南、工具使用规范 | 后台巡检、数据记录、定时任务 |
| **示例** | travel-plan、rss-manage | health-log、explore-curiosity |

参考示例：`skills/example/SKILL.md`、`skills/rss-manage/SKILL.md`
