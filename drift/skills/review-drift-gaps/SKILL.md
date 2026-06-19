---
name: review-drift-gaps
description: 定期回顾 drift 全局行动历史，发现停滞方向，维护有界 backlog
---

## 目标

防止 drift skills 烂尾。定期扫荡所有 skill 的运行状态和 drift.json 历史，
找出长期无进展、阶段停滞、待办未推进的方向，维护最多 10 项的优先级 backlog。

## 工作文件

- skills/review-drift-gaps/backlog.md：按优先级排列的待推进方向列表
- skills/review-drift-gaps/last_snapshot.md：上轮快照，用于对比变化

## 工作流程

1. read_file skills/review-drift-gaps/backlog.md
2. read_file skills/review-drift-gaps/last_snapshot.md
3. read_file ../../../drift.json（绝对路径，获取最近 10 条运行记录）
4. 对 drift/skills/ 下每个 skill：
   a. read_file skills/<skill-name>/state.json 获取 last_run_at 和 next
   b. 对比上轮快照中同 skill 的状态
   c. 判断停滞：超过 24h 未运行且 next 非"等待用户回复" → 标记停滞
   d. 判断进展：run_count 增加或 next 变化 → 标记活跃
5. 生成新的 backlog：
   - 显式跳过自身（review-drift-gaps）和 create-drift-skill
   - 停滞轮数越多的方向排越前
   - 连续恢复 3 轮的方向自动移除
   - 新增停滞加入队列
   - 总项数不超过 10
6. write_text_file 写回 backlog.md
7. write_text_file 写回 last_snapshot.md（保存本轮快照）
8. finish_drift(skill_used="review-drift-gaps", one_line="扫描 N 个 skill，发现 M 个停滞方向", next="backlog 共 K 项", message_result="silent")

## 要求

- 不调用 message_push（纯后台审计）
- 不在 backlog 中加入自己
- 如果没有任何变化，也要更新 last_snapshot.md 的时间戳
- finish_drift.message_result 必须是 "silent"
- 至少间隔 24h 才值得跑一次（频繁跑没有新信息）
