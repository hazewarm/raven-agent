from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from raven_agent.llm import LLMProvider
from raven_agent.messages import user_message


@dataclass(frozen=True)
class GateDecision:
    """Query Rewrite 的检索决策结果。

    参数:
        needs_episodic: 是否需要检索 episodic / profile / event 记忆。
        episodic_query: 用于历史事实检索的查询文本。
        latency_ms: 本次决策耗时毫秒数。
        procedure_query: 用于 procedure / preference 检索的查询文本。

    返回:
        GateDecision 实例。
    """

    needs_episodic: bool
    episodic_query: str
    latency_ms: int
    procedure_query: str = ""


class QueryRewriter:
    """基于轻量 LLM 的 Memory2 查询改写器。

    参数:
        provider: LLMProvider。
        timeout_ms: 总超时时间毫秒。

    返回:
        QueryRewriter 实例。
    """

    def __init__(self, *, provider: LLMProvider, timeout_ms: int = 800) -> None:
        self._provider = provider
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)

    async def decide(self, user_msg: str, recent_history: str = "") -> GateDecision:
        """判断是否需要查记忆，并生成改写查询。

        参数:
            user_msg: 当前用户消息。
            recent_history: 近期对话文本。

        返回:
            GateDecision。失败时 fail-open，使用原始消息作为 episodic_query。
        """

        started = time.perf_counter()

        # 【防御性编程核心】：在进行任何网络请求前，先构建好 fallback (兜底) 方案。
        # 默认放行 (needs_episodic=True)，且直接拿用户原话 (user_msg) 作为检索词。
        fallback = self._build_decision(
            started=started,
            user_msg=user_msg,
            needs_episodic=True,
            episodic_query=user_msg,
        )
        # 创建两个并发的异步任务：
        # 1. 决策是否需要查历史，并重写历史 Query
        main_task = asyncio.create_task(self._call_llm(self._build_prompt(user_msg, recent_history)))
        # 2. 独立重写 Procedure (规范/偏好) Query
        procedure_task = asyncio.create_task(self._rewrite_procedure_query(user_msg))
        done, pending = await asyncio.wait({main_task, procedure_task}, timeout=self._timeout_s)
        for task in pending:
            task.cancel()
        if not done:
            return fallback

        raw_output = ""
        procedure_query = ""
        if main_task in done:
            try:
                raw_output = main_task.result()
            except Exception:
                raw_output = ""
        if procedure_task in done:
            try:
                procedure_query = procedure_task.result()
            except Exception:
                procedure_query = ""

        parsed = self._parse_output(raw_output)
        if parsed is None:
            return self._build_decision(
                started=started,
                user_msg=user_msg,
                needs_episodic=True,
                episodic_query=user_msg,
                procedure_query=procedure_query,
            )
        parsed["procedure_query"] = procedure_query
        return self._build_decision(started=started, user_msg=user_msg, **parsed)

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM 并返回文本。

        参数:
            prompt: 用户消息 prompt。

        返回:
            LLM 文本输出。
        """

        response = await self._provider.chat(
            messages=[user_message(prompt)],
            tools=[],
            tool_choice="none",
        )
        return response.content.strip()

    async def _rewrite_procedure_query(self, user_msg: str) -> str:
        """把用户消息改写成 procedure/preference 检索 query。

        参数:
            user_msg: 当前用户消息。

        返回:
            改写后的查询文本；失败或无意义时返回空字符串。
        """

        try:
            output = await self._call_llm(self._build_procedure_prompt(user_msg))
        except Exception:
            return ""
        return self._clean_procedure_query(output)

    def _parse_output(self, raw_output: str) -> dict[str, object] | None:
        """解析 LLM XML 输出。

        参数:
            raw_output: LLM 原始输出。

        返回:
            包含 needs_episodic 和 episodic_query 的字典；解析失败返回 None。
        """

        decision_text = self._extract_tag(raw_output, "decision").upper()
        if decision_text not in {"RETRIEVE", "NO_RETRIEVE"}:
            return None
        return {
            "needs_episodic": decision_text == "RETRIEVE",
            "episodic_query": self._extract_tag(raw_output, "history_query"),
        }

    def _build_decision(
        self,
        *,
        started: float,
        user_msg: str,
        needs_episodic: bool,
        episodic_query: str,
        procedure_query: str = "",
    ) -> GateDecision:
        """构造 GateDecision 并补齐 fallback query。

        参数:
            started: 开始时间 perf_counter。
            user_msg: 原始用户消息。
            needs_episodic: 是否需要查历史。
            episodic_query: 历史检索 query。
            procedure_query: procedure 检索 query。

        返回:
            GateDecision。
        """

        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        fallback_query = user_msg.strip()
        return GateDecision(
            needs_episodic=needs_episodic,
            episodic_query=episodic_query.strip() or fallback_query,
            procedure_query=procedure_query.strip(),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_tag(raw_output: str, tag: str) -> str:
        """提取 XML 标签内容。

        参数:
            raw_output: LLM 原始输出。
            tag: 标签名。

        返回:
            标签内容；不存在时返回空字符串。
        """

        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_output or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _build_prompt(user_msg: str, recent_history: str) -> str:
        """构造 episodic retrieval gate prompt。

        参数:
            user_msg: 当前用户消息。
            recent_history: 近期对话文本。

        返回:
            prompt 字符串。
        """

        history_block = recent_history.strip() or "（无）"
        return f"""你是记忆检索决策器。根据近期对话和当前用户消息，判断是否需要检索用户个人事实或历史事件，并输出查询。

近期对话：
{history_block}

当前用户消息：
{user_msg}

规则：
- NO_RETRIEVE：打招呼、闲聊、确认当前轮内容、通用知识问答、简单回应、用户提出新的服务偏好或执行规则
- RETRIEVE：询问过去发生的事、用户个人信息、用户是否告诉过某事、最近购买/健康/设备/偏好等历史信息
- 用户提出新的偏好或规则时，decision 仍是 NO_RETRIEVE；这类内容只交给 procedure_query 处理
- 出现“都有哪些/列举/所有/一共/总共/历史上”这类聚合问题 → RETRIEVE，并改写成覆盖主题的宽泛 query
- “你还记得吗/你知道我的/你记不记得/我跟你说过”等元问题是在查事实本身，history_query 要贴近记忆 summary
- 提到快递、物流、包裹、到货时，若语境指向用户最近购买行为，应查购买历史
- 提到身体症状、药、复查时，若语境指向用户健康状态，应查健康档案或历史记录

只输出 XML，不要解释：
<decision>RETRIEVE|NO_RETRIEVE</decision>
<history_query>...</history_query>
"""

    @staticmethod
    def _build_procedure_prompt(user_msg: str) -> str:
        """构造 procedure/preference query rewrite prompt。

        参数:
            user_msg: 当前用户消息。

        返回:
            prompt 字符串。
        """

        return f"""只输出一行检索 query，不要解释。

把用户消息改写成 preference/procedure 库能命中的 summary 风格查询：
- 用户希望 agent 怎样服务、讲解、推荐
- agent 在某类请求下必须怎么做、用什么工具
- 用户发来某类外部资源、文件、图片、链接时 agent 应如何处理
不要抽一次性标题词，要写可复用场景。

示例：
用户消息：以后讲复杂问题先给我一个贯穿始终的例子
输出：用户希望 agent 讲解复杂问题时先给贯穿始终的例子

用户消息：【视频-哔哩哔哩】 https://example.test/item
输出：用户发送哔哩哔哩视频链接时 agent 应如何处理

用户消息：{user_msg}
输出：
"""

    @staticmethod
    def _clean_procedure_query(raw_output: str) -> str:
        """清洗 procedure query 输出。

        参数:
            raw_output: LLM 输出。

        返回:
            清洗后的 query；哨兵空值返回空字符串。
        """

        text = re.sub(r"\s+", " ", str(raw_output or "")).strip("。 .")
        if text.lower() in {"", "空", "无", "none", "null", "n/a", "not applicable", "(empty)"}:
            return ""
        return text