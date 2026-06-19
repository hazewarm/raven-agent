"""
peerAgent/travel-planner/server.py —— A2A JSON-RPC HTTP Server。

基于 FastAPI 实现 Google A2A (Agent-to-Agent) 协议的一个最小子集:
  - POST /message/send  → 接收任务，启动后台流水线
  - POST /tasks/get     → 查询任务状态和结果
  - GET  /health        → 健康检查（ProcessManager 用它判断是否启动成功）

任务生命周期:
  submitted → processing → completed / failed

设计要点:
  - 使用内存字典存储任务（冷启动架构，用完即销）
  - 每个任务有唯一 task_id（UUID）
  - ★ JSON 在校验通过后直接落盘，A2A artifact 只返回文件路径 + 渲染指令
  - 不依赖数据库，进程重启后任务丢失（符合冷启动设计）
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from config import load_config
from planner import TripPlanner

logger = logging.getLogger(__name__)

# ── FastAPI 应用 ──
app = FastAPI(
    title="Travel Planner Peer Agent",
    description="5-Agent 旅行规划流水线 —— A2A 兼容 HTTP Server",
    version="1.0.0",
)

# ── 任务存储 ──
# 内存字典: task_id → {status, result, error, created_at}
# 任务完成后保留 1 小时（由后台清理协程处理）
_TASK_RETENTION_SECONDS = 3600  # 1 小时

_tasks: dict[str, dict[str, Any]] = {}

# ── TripPlanner 实例（懒加载） ──
_planner: TripPlanner | None = None


async def _get_planner() -> TripPlanner:
    """获取或创建 TripPlanner 单例。

    懒加载 + 单例模式，避免每次请求都重新创建。
    配置从 Peer Agent 自身的 config.toml 读取。

    输入:
        无。

    输出:
        TripPlanner 实例。
    """
    global _planner
    if _planner is None:
        config = load_config()
        _planner = TripPlanner(config)
    return _planner


# ═══════════════════════════════════════════════════════════════════
# ★ 根路由 —— raven-agent 的 PeerAgentTool/Poller 都 POST 到这里
#   使用 JSON body 中的 "method" 字段区分操作
# ═══════════════════════════════════════════════════════════════════

@app.post("/")
async def jsonrpc_root(request: Request) -> dict[str, Any]:
    """JSON-RPC 根路由 —— 根据 body.method 分发。

    raven-agent 的 PeerAgentTool 和 Poller 都是 POST 到根 URL，
    不携带路径，通过 JSON-RPC method 字段区分操作。
    """
    body = await request.json()
    method = body.get("method", "")

    if method == "message/send":
        return await _handle_message_send(body)
    elif method == "tasks/get":
        return await _handle_tasks_get(body)
    else:
        raise HTTPException(status_code=400, detail=f"未知 method: {method}")


# ═══════════════════════════════════════════════════════════════════
# A2A 处理函数
# ═══════════════════════════════════════════════════════════════════

def _extract_goal(body: dict[str, Any]) -> str:
    """从 A2A message/send 请求中提取用户目标文本。

    输入:
        body: 完整的 JSON-RPC 请求 dict。

    输出:
        goal 字符串。

    异常:
        ValueError: 无法提取 goal 时抛出。
    """
    parts: list[dict[str, Any]] = (
        body.get("params", {}).get("message", {}).get("parts", [])
    )
    for part in parts:
        if part.get("kind") == "text" and part.get("text"):
            return part["text"]
    raise ValueError("缺少 message.parts[].text")


async def _handle_message_send(body: dict[str, Any]) -> dict[str, Any]:
    """处理 message/send: 接收任务，启动后台流水线。

    输入:
        body: 完整的 JSON-RPC 请求 dict。

    输出:
        JSON-RPC 响应 dict。
    """
    try:
        goal = _extract_goal(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析请求: {exc}") from exc

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "submitted",
        "json_path": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "goal": goal,
    }
    asyncio.create_task(_run_planner(task_id, goal))

    logger.info("[Server] 任务已接收 task_id=%s goal=%.80s...", task_id, goal)

    return {
        "jsonrpc": "2.0",
        "id": body.get("id", ""),
        "result": {"id": task_id, "status": "submitted"},
    }


async def _handle_tasks_get(body: dict[str, Any]) -> dict[str, Any]:
    """处理 tasks/get: 查询任务状态。

    输入:
        body: 完整的 JSON-RPC 请求 dict。

    输出:
        JSON-RPC 响应 dict。
    """
    task_id = body.get("params", {}).get("id", "")
    if not task_id:
        raise HTTPException(status_code=400, detail="缺少 params.id")

    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    state = task["status"]
    result: dict[str, Any] = {
        "id": task_id,
        "status": {
            "state": state,
            "message": {
                "parts": [{"text": (
                    "规划完成" if state == "completed" else
                    f"规划失败: {task.get('error', '未知错误')}" if state == "failed" else
                    "正在规划中..."
                )}]
            },
        },
    }

    # ★ 完成时返回 JSON 文件路径 + 渲染指令
    if state == "completed" and task.get("json_path"):
        jp = task["json_path"]
        result["artifacts"] = [{
            "name": "render_instruction.txt",
            "parts": [{"text": (
                f"TripPlan JSON 已生成并校验通过。\n"
                f"文件路径: {jp}\n"
                f"---\n"
                f"→ 请先简要告知用户「行程规划已完成！」，然后列出全部 29 种风格让用户选择：\n"
                f"  01 极简主义(minimalist)  02 大胆现代(bold_modern)  03 优雅复古(elegant_vintage)\n"
                f"  04 未来科技(futuristic_tech)  05 斯堪的纳维亚(scandinavian)  06 艺术装饰(art_deco)\n"
                f"  07 日式极简(japanese_minimalism)  08 后现代解构(postmodern_deconstruct)\n"
                f"  09 朋克(punk)  10 英伦摇滚(british_rock)  11 黑金属(black_metal)\n"
                f"  12 孟菲斯(memphis)  13 赛博朋克(cyberpunk)  14 波普艺术(pop_art)\n"
                f"  15 瑞士解构(deconstructed_swiss)  16 蒸汽波(vaporwave)\n"
                f"  17 新表现主义(neo_expressionism)  18 极端极简(extreme_minimalism)\n"
                f"  19 新未来主义(neo_futurism)  20 超现实拼贴(surrealist_collage)\n"
                f"  21 新巴洛克(neo_baroque)  22 液态数字(liquid_digital)\n"
                f"  23 超感官极简(hypersensory_minimalism)\n"
                f"  24 新表现数据(neo_expressionist_data)  25 维多利亚(victorian)\n"
                f"  26 包豪斯(bauhaus)  27 构成主义(constructivism)\n"
                f"  28 孟菲斯设计(memphis_design)  29 德国表现主义(german_expressionism)\n"
                f"→ 用户选择后，调用 render_trip_html 工具：\n"
                f"  参数 trip_json_path=\"{jp}\"，style=\"<用户选的风格 slug>\"\n"
                f"→ ★ 务必等待用户选择风格后再渲染，不要在当前轮自动渲染。\n"
                f"→ 只有用户明确说「随便」「都行」「默认」时，才用 art_deco 风格渲染。\n"
                f"→ 渲染完成后告知用户 HTML 文件路径。"
            )}],
        }]

    return {"jsonrpc": "2.0", "id": body.get("id", ""), "result": result}


# ═══════════════════════════════════════════════════════════════════
# 保留原有路径路由（方便手动调试）
# ═══════════════════════════════════════════════════════════════════

@app.post("/message/send")
async def message_send(request: Request) -> dict[str, Any]:
    """[已废弃，保留用于手动调试] POST /message/send"""
    body = await request.json()
    return await _handle_message_send(body)


@app.post("/tasks/get")
async def tasks_get(request: Request) -> dict[str, Any]:
    """[调试用] POST /tasks/get"""
    body = await request.json()
    return await _handle_tasks_get(body)


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查端点。

    由 raven-agent 的 PeerProcessManager 在启动时轮询此端点，
    返回 200 即表示 Peer Agent 已就绪。

    输入:
        无。

    输出:
        {"status": "ok"}
    """
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════
# 后台任务与清理
# ═══════════════════════════════════════════════════════════════════

async def _run_planner(task_id: str, goal: str) -> None:
    """后台执行 5-Agent 流水线，JSON 校验通过后直接落盘。

    ★ 核心优化：Plan JSON 在校验通过后立即写入本地文件，
    不经过 A2A artifact → Poller → 系统通知 → LLM 上下文这一长链路。
    LLM 只收到文件路径（一行字符串），零数据损失。

    输入:
        task_id: 任务 UUID。
        goal: 用户原始请求。

    输出:
        None。结果写入 _tasks[task_id]，JSON 写入本地文件。
    """
    try:
        _tasks[task_id]["status"] = "processing"

        planner = await _get_planner()
        plan_dict = await planner.plan(goal)  # dict，已通过 Pydantic 校验

        # ── ★ 直接落盘 JSON ──
        # 输出到 peerAgent/travel-planner/outputs/ 目录
        output_dir = planner._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"trip_{task_id}.json"
        json_path = output_dir / json_filename

        json_path.write_text(
            json.dumps(plan_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "[Server] JSON 已落盘 task_id=%s path=%s size=%d",
            task_id, json_path, json_path.stat().st_size,
        )

        # ── 只存储文件路径，不存储 JSON 内容 ──
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["json_path"] = str(json_path.resolve())

        logger.info("[Server] 任务完成 task_id=%s", task_id)

    except Exception as exc:
        logger.error("[Server] 任务失败 task_id=%s: %s", task_id, exc)
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(exc)


async def _cleanup_old_tasks() -> None:
    """后台协程：定期清理超过保留时间的已完成/失败任务。

    每 10 分钟检查一次，删除超过 1 小时的任务记录。
    冷启动架构下任务量通常很小，此操作非常轻量。

    输入:
        无。

    输出:
        None。
    """
    while True:
        await asyncio.sleep(600)  # 10 分钟
        now = datetime.now(timezone.utc)
        to_remove = []
        for tid, task in _tasks.items():
            created = datetime.fromisoformat(task["created_at"])
            if (now - created).total_seconds() > _TASK_RETENTION_SECONDS:
                to_remove.append(tid)
        for tid in to_remove:
            del _tasks[tid]
        if to_remove:
            logger.info("[Server] 清理了 %d 个过期任务", len(to_remove))


# 启动时注册清理协程
@app.on_event("startup")
async def startup() -> None:
    """FastAPI 启动事件：启动后台清理协程。

    输入:
        无。

    输出:
        None。
    """
    asyncio.create_task(_cleanup_old_tasks())
    logger.info("[Server] Travel Planner Peer Agent 已启动")


# ═══════════════════════════════════════════════════════════════════
# 入口点（uv run python -m server）
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=9100,
        log_level="info",
    )
