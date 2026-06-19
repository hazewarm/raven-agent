# 添加 Peer Agent

Peer Agent 是一个独立的 AI Agent，有自己的 LLM 和配置，通过 A2A 协议与主 Agent 通信。

## 1. 快速添加示例

以项目内置的 `travel-planner` 为例，编辑 `peer_agents.toml`：

```toml
[[peer_agents]]
name = "travel-planner"
base_url = "http://127.0.0.1:9100"
description = "智能旅行规划助手，支持多日深度规划"
skills = [
  { id = "plan_trip", name = "旅行规划", description = "根据目的地、日期和偏好自动规划完整行程", tags = ["旅行", "规划"], examples = ["帮我规划北京3日游"] }
]
launcher = ["uv", "run", "python", "-m", "server"]
cwd = "peerAgent/travel-planner"
startup_timeout_s = 120
```

添加后重启 Agent，LLM 即可调用 `delegate_travel_planner` 工具委托旅行规划任务。同理，其他 Peer Agent 按此格式添加即可。

## 2. 配置字段说明

| 字段 | 说明 |
|------|------|
| `name` | Agent 名称，生成的工具名为 `delegate_<name>` |
| `description` | 用途描述，会注入工具描述帮助 LLM 路由 |
| `base_url` | Peer Agent 的 HTTP 地址 |
| `launcher` | 冷启动命令列表 |
| `cwd` | 工作目录（相对/绝对路径） |
| `startup_timeout_s` | 冷启动健康检查超时（秒） |
| `skills` | 技能列表，每条含 id / name / description / tags / examples |

## 3. 自建 Peer Agent

Peer Agent 需要实现 A2A 接口：

**最小实现（FastAPI）：**

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/")
async def a2a_endpoint(request: dict):
    # 处理 message/send
    return {"jsonrpc": "2.0", "result": {"id": "task-xxx", "status": "submitted"}}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

完整实现参考 `peerAgent/travel-planner/server.py`。

## 4. 工作流程

1. LLM 调用 `delegate_<name>` 工具 → 框架通过 `launcher` 冷启动子进程
2. 等待 `/health` 返回 200
3. POST `message/send`（A2A 协议）提交任务
4. Poller 每 10 秒轮询 `tasks/get` 直到完成
5. 完成后将结果注入 MessageBus，通知用户
6. 自动终止子进程

Peer Agent 和主 Agent 使用不同的 LLM 和配置，互不干扰。
