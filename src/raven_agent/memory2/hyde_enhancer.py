from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from raven_agent.llm import LLMProvider
from raven_agent.messages import user_message

logger = logging.getLogger(__name__)

RetrieveFn = Callable[..., Awaitable[list[dict[str, object]]]]


@dataclass(frozen=True)
class HyDEAugmentResult:
    """HyDE 增强检索结果。

    参数:
        items: raw hits 和 HyDE 新增 hits 合并后的结果。
        used_hyde: HyDE 是否实际追加了新条目。
        hypothesis: LLM 生成的假想记忆条目；失败时为 None。
        raw_hits: 原始 query 的检索结果。

    返回:
        HyDEAugmentResult 实例。
    """

    items: list[dict[str, object]]
    used_hyde: bool
    hypothesis: str | None
    raw_hits: list[dict[str, object]] = field(default_factory=list)

    def __iter__(self):
        """兼容二元解包。

        参数:
            无。

        返回:
            依次 yield items 和 used_hyde。
        """

        yield self.items
        yield self.used_hyde


class HyDEEnhancer:
    """HyDE 假想记忆条目增强器。

    参数:
        provider: 用于生成 hypothesis 的 LLMProvider。
        timeout_s: hypothesis 生成超时时间。

    返回:
        HyDEEnhancer 实例。
    """

    def __init__(self, *, provider: LLMProvider, timeout_s: float = 2.0) -> None:
        self._provider = provider
        self._timeout_s = max(0.5, float(timeout_s))

    async def generate_hypothesis(self, query: str, context: str = "") -> str | None:
        """生成一条假想记忆摘要。

        参数:
            query: 用户原始查询。
            context: 可选近期对话上下文。

        返回:
            假想记忆摘要；失败或超时时返回 None。
        """

        prompt = self._build_prompt(query=query, context=context)
        try:
            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=[user_message(prompt)],
                    tools=[],
                    tool_choice="none",
                ),
                timeout=self._timeout_s,
            )
        except Exception as exc:
            logger.debug("hyde hypothesis generation failed: %s", exc)
            return None
        text = response.content.strip()
        return text if text else None

    async def augment(
        self,
        *,
        raw_query: str,
        context: str,
        retrieve_fn: RetrieveFn,
        top_k: int,
        **retrieve_kwargs: object,
    ) -> HyDEAugmentResult:
        """执行 raw 检索 + HyDE 检索并合并结果。

        参数:
            raw_query: 用户原始查询。
            context: 可选近期上下文。
            retrieve_fn: 可调用的检索函数。
            top_k: 最多返回多少条。
            retrieve_kwargs: 透传给 retrieve_fn 的参数。

        返回:
            HyDEAugmentResult。
        """

        raw_task = asyncio.create_task(
            retrieve_fn(raw_query, top_k=top_k, **retrieve_kwargs)
        )
        hypothesis_task = asyncio.create_task(self.generate_hypothesis(raw_query, context))
        raw_hits, hypothesis = await asyncio.gather(raw_task, hypothesis_task)
        if not hypothesis:
            return HyDEAugmentResult(
                items=raw_hits,
                used_hyde=False,
                hypothesis=None,
                raw_hits=raw_hits,
            )
        try:
            hyde_hits = await retrieve_fn(hypothesis, top_k=top_k, **retrieve_kwargs)
        except Exception as exc:
            logger.debug("hyde retrieve failed: %s", exc)
            return HyDEAugmentResult(
                items=raw_hits,
                used_hyde=False,
                hypothesis=hypothesis,
                raw_hits=raw_hits,
            )
        merged = _union_dedup(raw_hits, hyde_hits)
        return HyDEAugmentResult(
            items=merged,
            used_hyde=len(merged) > len(raw_hits),
            hypothesis=hypothesis,
            raw_hits=raw_hits,
        )

    @staticmethod
    def _build_prompt(*, query: str, context: str) -> str:
        """构造 HyDE prompt。

        参数:
            query: 用户原始查询。
            context: 可选近期上下文。

        返回:
            prompt 字符串。
        """

        context_section = f"\n近期对话背景：\n{context}\n" if context.strip() else ""
        return (
            "你是个人助手的记忆系统。根据用户提问，生成一条"
            "如果该信息存在于记忆数据库中会长什么样的假想条目。\n"
            f"{context_section}"
            "规则：\n"
            "- 第三人称，使用“用户...”句式\n"
            "- 简洁事实陈述\n"
            "- 不要回答问题本身\n"
            "- 只输出一条文本，不要解释\n\n"
            f"用户提问：{query}\n"
            "假想记忆条目："
        )


def _union_dedup(
    raw: list[dict[str, object]],
    hyde: list[dict[str, object]],
) -> list[dict[str, object]]:
    """保留 raw 全部结果，并追加 HyDE 新条目。

    参数:
        raw: 原始 query 检索结果。
        hyde: hypothesis query 检索结果。

    返回:
        union dedup 后的结果列表。
    """

    seen_ids: set[str] = set()
    result: list[dict[str, object]] = []
    for item in raw:
        item_id = str(item.get("id", "") or "")
        if item_id:
            seen_ids.add(item_id)
        result.append(item)
    for item in hyde:
        item_id = str(item.get("id", "") or "")
        if item_id and item_id in seen_ids:
            continue
        result.append(item)
        if item_id:
            seen_ids.add(item_id)
    return result