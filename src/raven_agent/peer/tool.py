"""
peer/tool.py —— PeerAgentTool：将 Peer Agent 包装为 raven-agent 工具。

行为：
  - 调用前通过 ProcessManager 确保 peer agent 在线
  - 通过 A2A JSON-RPC message/send 提交异步任务
  - 将 task_id 注册到 Poller 进行后台跟踪
  - 立即返回（fire & forget），不阻塞主 Agent 的 ReAct 循环
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

import httpx

from raven_agent.peer.models import AgentCard
from raven_agent.tools.base import Tool

logger = logging.getLogger(__name__)

_SUBMIT_TIMEOUT_S = 15.0


def _slugify(name: str) -> str:
    """将 Peer Agent 名称转换为合法的工具名后缀。

    输入:
        name: Agent 名称（如 "Deep Research"）。

    输出:
        slug 字符串（如 "deep_research"）。

    示例:
        _slugify("Deep Research")     → "deep_research"
        _slugify("Code-Review Bot")   → "code_review_bot"
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class PeerAgentTool(Tool):
    """委托远程 A2A Peer Agent 执行深度任务。fire & forget 模式。

    参数:
        card: AgentCard 实例（工具名和描述由此生成）。
        process_manager: PeerProcessManager 实例。
        poller: PeerAgentPoller 实例。
        client: httpx.AsyncClient 实例。
    """

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "用户的原始请求，原封不动传入，"
                    "不要改写、扩展或补充细节"
                ),
            },
            "breadth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "并行子问题数，控制调研广度，默认 1",
            },
            "rounds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "最大调研轮次，控制调研深度，默认 2",
            },
        },
        "required": ["goal"],
    }

    def __init__(
        self,
        card: AgentCard,
        process_manager: Any,
        poller: Any,
        client: httpx.AsyncClient,
    ) -> None:
        self._card = card
        self._pm = process_manager
        self._poller = poller
        self._client = client
        self._name = f"delegate_{_slugify(card.name)}"

        # 构建工具描述
        skill = card.primary_skill
        base_desc = (
            (skill.description if skill else card.description)
            or (
                f"委托 {card.name} 执行深度调研任务，生成结构化长报告。"
                "适合：技术选型、论文调研、行业分析、竞品对比、综合报告。"
            )
        )
        self._description = (
            base_desc
            + "\n注意：任务异步执行，完成后系统会自动通知并总结报告给用户。"
        )

    @property
    def name(self) -> str:
        """返回工具名称（如 delegate_deep_research）。

        输出:
            str。
        """
        return self._name

    @property
    def description(self) -> str:
        """返回工具描述（供 LLM 路由决策）。

        输出:
            str。
        """
        return self._description

    async def execute(self, **kwargs: Any) -> str:
        """提交一个 A2A 异步任务。

        输入:
            goal: 用户原始请求（必填）。
            breadth: 并行子问题数，1-3，默认 1。
            rounds: 最大调研轮次，1-3，默认 2。
            channel: 发起任务的 channel（由 ToolRegistry 注入）。
            chat_id: 发起任务的 chat_id（由 ToolRegistry 注入）。

        输出:
            JSON 字符串 {"status": "submitted", "task_id": "...", ...}。
        """
        goal: str = kwargs["goal"]
        breadth: int = int(kwargs.get("breadth", 1))
        rounds: int = int(kwargs.get("rounds", 2))
        channel: str = kwargs.get("channel", "unknown")
        chat_id: str = kwargs.get("chat_id", "unknown")

        # 1. 冷启动 peer agent
        try:
            await self._pm.ensure_ready(self._card.name)
        except Exception as exc:
            logger.error(
                "[PeerAgentTool] 启动 %s 失败: %s", self._card.name, exc,
            )
            return json.dumps(
                {
                    "error": f"peer agent 启动失败：{exc}",
                    "agent": self._card.name,
                },
                ensure_ascii=False,
            )

        # 2. 提交 A2A 任务
        try:
            task_id = await self._submit_task(goal, breadth, rounds)
        except Exception as exc:
            logger.error("[PeerAgentTool] 提交任务失败: %s", exc)
            return json.dumps(
                {
                    "error": f"任务提交失败：{exc}",
                    "agent": self._card.name,
                },
                ensure_ascii=False,
            )

        # 3. 注册到 Poller
        self._poller.register(
            task_id=task_id,
            agent_name=self._card.name,
            agent_url=self._card.url,
            channel=channel,
            chat_id=chat_id,
            goal=goal,
        )

        logger.info(
            "[PeerAgentTool] 任务已提交 task_id=%s agent=%s "
            "channel=%s chat_id=%s",
            task_id, self._card.name, channel, chat_id,
        )

        return json.dumps(
            {
                "status": "submitted",
                "task_id": task_id,
                "agent": self._card.name,
                "message": (
                    "深度调研任务已在后台启动，通常需要 3-10 分钟。"
                    "完成后系统会自动通知你，"
                    "届时请读取报告文件并向用户总结。"
                ),
            },
            ensure_ascii=False,
        )

    async def _submit_task(
        self,
        goal: str,
        breadth: int,
        rounds: int,
    ) -> str:
        """通过 A2A JSON-RPC message/send 提交任务。

        输入:
            goal: 用户原始请求。
            breadth: 并行子问题数。
            rounds: 最大调研轮次。

        输出:
            A2A 服务端返回的 task_id 字符串。
        """
        task_id = str(uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": "submit-1",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": goal}],
                    "metadata": {
                        "breadth": breadth,
                        "max_rounds": rounds,
                    },
                },
                "configuration": {"blocking": False},
            },
        }
        response = await self._client.post(
            self._card.url,
            json=payload,
            timeout=_SUBMIT_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"A2A 错误: {data['error']}")

        # 优先使用服务端生成的 task_id，否则用本地生成的
        server_id = data.get("result", {}).get("id")
        return server_id or task_id