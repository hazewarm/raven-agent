---
name: explore-curiosity
description: 在空闲时像朋友随口一问，了解用户偏好和生活化信息
---

## 目标

在没有 RSS 新闻可推送的空闲时段，基于长期记忆提出一个轻量自然的问题，
慢慢补全用户画像中的生活化空白。

## 工作文件

- skills/explore-curiosity/queue.md：待问的问题列表，每行一个问题

## 工作流程

1. read_file skills/explore-curiosity/queue.md
2. 如果文件不存在或为空：
   a. recall_memory 检索用户长期记忆（搜索"兴趣""日常""偏好"等词）
   b. 基于记忆中的空白，生成 5 个轻量问题，write_text_file 写入 queue.md
   c. 取第一行作为本轮问题
3. 如果 queue.md 非空，取第一行
4. message_push 发送该问题
5. edit_file 删除 queue.md 中已发送的那一行（或 write_text_file 写回剩余行）
6. finish_drift(skill_used="explore-curiosity", one_line="推送了：{问题摘要}", next="队列剩余 N 题", message_result="sent")

## 要求

- 问题必须轻量、自然、像朋友随口一问，不超过 30 字
- 优先问：音乐偏好、开源项目、运动习惯、食物口味、日常消遣、最近在看什么
- 禁止问太大太虚的问题（如"你的人生目标是什么"）
- 避开长期记忆里已经明确有答案的信息
- 队列为空时先生成问题再发送，不要空手 finish_drift
- finish_drift.message_result 必须是 "sent"（本轮一定发了消息）
