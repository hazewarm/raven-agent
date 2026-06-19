from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from raven_agent.llm import LLMProvider
from raven_agent.messages import user_message


@dataclass(frozen=True)
class ProfileFact:
    """从对话中提取出的 profile 事实。

    参数:
        summary: 可写入 Memory2 的事实摘要。
        category: profile 子类别，支持 purchase、decision、status、personal_fact。
        happened_at: 事实发生日期；personal_fact 固定为 None。

    返回:
        ProfileFact 实例。
    """

    summary: str
    category: str
    happened_at: str | None


class ProfileFactExtractor:
    """USER-first profile 事实提取器。

    参数:
        provider: LLMProvider，用于从对话中提取 profile facts。
        timeout_ms: 单次抽取超时时间。

    返回:
        ProfileFactExtractor 实例。
    """

    def __init__(self, *, provider: LLMProvider, timeout_ms: int = 5000) -> None:
        self._provider = provider
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)

    async def extract(
        self,
        conversation: str,
        *,
        existing_profile: str = "",
    ) -> list[ProfileFact]:
        """从一段对话中提取 profile facts。

        参数:
            conversation: 格式化后的对话文本。
            existing_profile: 已有 profile 文本，用于让模型避免重复输出。

        返回:
            ProfileFact 列表；LLM 失败或无事实时返回空列表。
        """

        if not conversation.strip():
            return []
        prompt = self._build_prompt(
            conversation=conversation,
            existing_profile=existing_profile,
        )
        facts = await self._extract_with_prompt(prompt, timeout_s=self._timeout_s)
        if facts:
            return facts

        clause_facts: list[ProfileFact] = []
        for clause in self._split_user_clauses(conversation):
            clause_items = await self.extract_from_exchange(
                clause,
                "",
                existing_profile=existing_profile,
            )
            clause_facts.extend(clause_items)
        return _dedupe_facts(clause_facts)

    async def extract_from_exchange(
        self,
        user_msg: str,
        agent_response: str,
        *,
        existing_profile: str = "",
    ) -> list[ProfileFact]:
        """从单轮 user/assistant 交换中提取 profile facts。

        参数:
            user_msg: 当前轮用户消息。
            agent_response: 当前轮助手回复；只作背景，不能作为事实证据。
            existing_profile: 已有 profile 文本。

        返回:
            ProfileFact 列表；仅保留 purchase/status/personal_fact。
        """

        if not (user_msg.strip() or agent_response.strip()):
            return []
        prompt = self._build_exchange_prompt(
            user_msg=user_msg,
            agent_response=agent_response,
            existing_profile=existing_profile,
        )
        facts = await self._extract_with_prompt(prompt, timeout_s=min(self._timeout_s, 1.5))
        allowed = {"purchase", "status", "personal_fact"}
        return [fact for fact in facts if fact.category in allowed]

    async def _extract_with_prompt(self, prompt: str, *, timeout_s: float) -> list[ProfileFact]:
        """调用 LLM 并解析 XML facts。

        参数:
            prompt: 已构造的提取 prompt。
            timeout_s: 超时时间秒数。

        返回:
            ProfileFact 列表；异常时返回空列表。
        """

        try:
            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=[user_message(prompt)],
                    tools=[],
                    tool_choice="none",
                ),
                timeout=timeout_s,
            )
        except Exception:
            return []
        return self._parse_facts(response.content)

    @staticmethod
    def _build_prompt(*, conversation: str, existing_profile: str) -> str:
        """构造多轮 profile extraction prompt。

        参数:
            conversation: 待处理对话。
            existing_profile: 已有 profile 文本。

        返回:
            prompt 字符串。
        """

        return f"""你是 profile 事实提取器。请只从对话里提取用户长期可检索的 profile 事实，并输出 XML。

profile 的语义是：关于用户本人或其客观处境的事实。
不是“用户希望怎样被服务/怎样被讲解/怎样被推荐”。

仅允许以下 4 类：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划，或重要宣布
- status：用户某件事的状态变化，例如等待 / 完成 / 放弃 / 里程碑达成
- personal_fact：用户关于自身的事实性披露，包括身份、背景、拥有物、家庭成员、健康状况、技能、兴趣、长期习惯

必须遵守：
- 纯技术讨论、闲聊、打招呼，不输出
- 只有当 USER 原话中明确陈述自己的事实时，才允许提取
- ASSISTANT 的回复只作为背景参考，不能作为提取证据
- 用户提问、追问、反问、记忆测试句不算事实披露
- 用户在举例 / 假设 / 如果 / 比如 / 设想 / 虚构场景中使用第一人称，不算事实披露
- 用户希望助手怎样服务、怎样讲解、怎样推荐，这是 preference，不是 profile
- 工程操作过程，例如安装依赖、配置环境、调试步骤、更新工具版本，不属于 profile
- personal_fact 默认不写 happened_at；purchase / status / decision 可以写 happened_at
- 每条 summary 只表达一条完整事实，写具体内容，不要概括成“用户购买了多件商品”

当前已有 profile（用于查重）：
{existing_profile or "（空）"}

待处理对话：
{conversation}

只输出 XML：
<facts>
<fact>
  <summary>...</summary>
  <category>purchase|decision|status|personal_fact</category>
  <happened_at>YYYY-MM-DD</happened_at>
</fact>
</facts>"""

    @staticmethod
    def _build_exchange_prompt(
        *,
        user_msg: str,
        agent_response: str,
        existing_profile: str,
    ) -> str:
        """构造单轮 profile extraction prompt。

        参数:
            user_msg: 用户消息。
            agent_response: 助手回复；只作背景。
            existing_profile: 已有 profile 文本。

        返回:
            prompt 字符串。
        """

        return f"""你是单轮 profile 事实提取器。只看这一轮对话，不要推断、不要联想。

只允许提取以下 3 类：
- purchase：用户刚购买/下单了什么
- status：用户某件事的状态变化，或里程碑达成
- personal_fact：用户关于自身的事实性披露

禁止输出：
- decision
- preference
- 纯闲聊、打招呼
- 纯技术讨论
- 用户提问、追问、记忆测试句
- 用户在举例、假设、类比、虚构场景里用第一人称说的话
- ASSISTANT 确认或复述的内容；ASSISTANT 不算用户陈述，不得作为事实来源
- 工程操作，例如安装依赖、更新工具版本、配置环境

当前已有 profile（用于查重）：
{existing_profile or "（空）"}

本轮对话：
USER: {user_msg}
ASSISTANT: {agent_response}

只输出 XML：
<facts>
<fact>
  <summary>...</summary>
  <category>purchase|status|personal_fact</category>
  <happened_at>YYYY-MM-DD</happened_at>
</fact>
</facts>"""

    @staticmethod
    def _parse_facts(raw_output: str) -> list[ProfileFact]:
        """解析 LLM 输出的 XML facts。

        参数:
            raw_output: LLM 原始文本。

        返回:
            去重后的 ProfileFact 列表。
        """

        allowed = {"purchase", "decision", "status", "personal_fact"}
        matches = re.findall(r"<fact>\s*(.*?)\s*</fact>", raw_output or "", re.DOTALL)
        facts: list[ProfileFact] = []
        seen: set[tuple[str, str]] = set()
        for block in matches:
            summary = _extract_tag(block, "summary")
            category = _extract_tag(block, "category").lower()
            happened_at = _extract_tag(block, "happened_at") or None
            if not summary or category not in allowed:
                continue
            if category == "personal_fact":
                happened_at = None
            key = (summary, category)
            if key in seen:
                continue
            seen.add(key)
            facts.append(ProfileFact(summary=summary, category=category, happened_at=happened_at))
        return facts

    @staticmethod
    def _split_user_clauses(conversation: str) -> list[str]:
        """从格式化对话中拆出 USER 子句。

        参数:
            conversation: 多轮对话文本。

        返回:
            去重后的 USER 子句列表。
        """

        clauses: list[str] = []
        seen: set[str] = set()
        for line in str(conversation or "").splitlines():
            match = re.search(r"\bUSER:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
            if not match:
                continue
            text = match.group(1).strip()
            parts = re.split(r"[。！？；;.!?，,]\s*", text)
            for part in parts:
                clause = part.strip()
                if len(clause) < 4 or clause in seen:
                    continue
                seen.add(clause)
                clauses.append(clause)
        return clauses


def _extract_tag(raw_output: str, tag: str) -> str:
    """提取 XML 标签内容。

    参数:
        raw_output: XML 片段。
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


def _dedupe_facts(facts: list[ProfileFact]) -> list[ProfileFact]:
    """按 summary + category 去重 profile facts。

    参数:
        facts: 候选 ProfileFact 列表。

    返回:
        去重后的 ProfileFact 列表。
    """

    deduped: list[ProfileFact] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        key = (fact.summary.strip(), fact.category.strip())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped