---
name: hello-world
description: 一个示例 Drift skill，练习文件读写和 finish_drift
---

## 目标

这是一个测试用 skill。每次 drift 时读自己的工作文件，追加一行时间戳，然后 finish_drift。

## 工作文件

- skills/hello-world/log.md：时间戳日志

## 工作流程

1. read_file skills/hello-world/log.md（如果不存在会返回空）
2. write_file 追加一行当前任务记录
3. finish_drift（message_result=silent，因为不需要通知用户）
