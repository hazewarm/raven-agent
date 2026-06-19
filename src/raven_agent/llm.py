from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from raven_agent.config import LLMConfig
from raven_agent.messages import ChatMessage, ToolCall
from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class LLMResponse:
    """模型回复结果。

    参数:
        content: 模型返回的文本内容。
        tool_calls: 模型请求执行的工具调用列表。
        reasoning_content: thinking 模式下供应商返回的推理内容；当本次响应包含 tool_calls 时，下一次请求必须随 assistant tool_call 消息回传。
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """解析模型返回的工具参数 JSON。

    参数:
        raw_arguments: 模型返回的 function.arguments 字符串。

    返回:
        参数字典。解析失败或不是对象时返回空字典。
    """

    if not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


class LLMProvider:
    """OpenAI 兼容大模型 Provider。

    参数:
        config: LLMConfig，包含 provider、model、api_key、base_url、max_tokens。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        model: str | None = None,
        max_tokens: int | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None
    ) -> LLMResponse:
        """向模型发送消息列表并返回回复。

        参数:
            messages: 已由 PromptBuilder 或 ReAct 循环构造好的 ChatMessage 列表。
            tools: 可供模型调用的 OpenAI tool schema 列表。
            tool_choice: 工具选择策略，默认 auto，让模型自行决定是否调用工具。

        返回:
            LLMResponse，包含文本内容和可能的工具调用。
        """

        request: dict[str, Any] = {
            "model": model or self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "messages": [message.to_openai_dict() for message in messages],
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        if on_content_delta is not None:
            return await self._chat_streaming(request, on_content_delta)

        response = await self._client.chat.completions.create(**request)
        message = response.choices[0].message
        content = message.content or ""
        reasoning_content = str(getattr(message, "reasoning_content", "") or "")
        tool_calls: list[ToolCall] = []

        for raw_tool_call in message.tool_calls or []:
            if raw_tool_call.type != "function":
                continue
            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.id,
                    name=raw_tool_call.function.name,
                    arguments=_parse_tool_arguments(raw_tool_call.function.arguments or "{}"),
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
    
    async def chat_dicts(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """使用原始 dict 消息列表向模型发送请求。

        与 chat() 的区别：此方法直接接受 OpenAI API 格式的 dict 消息，
        不做 ChatMessage.to_openai_dict() 转换。专供需要发送非文本内容
        （如 image_url content block）的工具使用。

        输入:
            messages: 原始 OpenAI API 格式的消息 dict 列表。
                每个 dict 必须包含 "role" 字段，content 可以是字符串
                或 content block 列表。
            tools: 可供模型调用的 OpenAI tool schema 列表。
            tool_choice: 工具选择策略，默认 auto。
            model: 覆盖默认模型名。
            max_tokens: 覆盖默认 max_tokens。

        输出:
            LLMResponse，包含文本内容和可能的工具调用。
        """

        request: dict[str, Any] = {
            "model": model or self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**request)
        message = response.choices[0].message
        content = message.content or ""
        reasoning_content = str(getattr(message, "reasoning_content", "") or "")
        tool_calls: list[ToolCall] = []

        for raw_tool_call in message.tool_calls or []:
            if raw_tool_call.type != "function":
                continue
            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.id,
                    name=raw_tool_call.function.name,
                    arguments=_parse_tool_arguments(
                        raw_tool_call.function.arguments or "{}"
                    ),
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
    
    
    
    async def _chat_streaming(
        self,
        request: dict[str, Any],
        on_content_delta: Callable[[str], Awaitable[None]],
    ) -> LLMResponse:
        """流式调用模型并累积完整响应。

        参数:
            request: 已构造好的 API 请求参数字典。
            on_content_delta: 每个 content delta 到达时调用的异步回调。

        返回:
            LLMResponse——与非流式 chat() 相同的完整响应结构。
        """

        stream = await self._client.chat.completions.create(
            **request, stream=True
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, str]] = {}
        tool_call_seen = False

        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            # —— thinking / reasoning ——
            reasoning_piece = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning_piece, str) and reasoning_piece:
                reasoning_parts.append(reasoning_piece)

            # —— tool_calls 增量 ——
            for tc_delta in getattr(delta, "tool_calls", None) or []:
                tool_call_seen = True
                idx = getattr(tc_delta, "index", 0)
                slot = tool_call_chunks.setdefault(idx, {})
                tc_id = getattr(tc_delta, "id", "") or ""
                fn = getattr(tc_delta, "function", None)
                tc_name = getattr(fn, "name", "") or "" if fn else ""
                tc_args = getattr(fn, "arguments", "") or "" if fn else ""
                if tc_id:
                    slot["id"] = slot.get("id", "") + tc_id
                if tc_name:
                    slot["name"] = slot.get("name", "") + tc_name
                if tc_args:
                    slot["arguments"] = slot.get("arguments", "") + tc_args

            # —— content delta ——
            content_piece = getattr(delta, "content", None)
            if isinstance(content_piece, str) and content_piece:
                content_parts.append(content_piece)
                if not tool_call_seen:
                    # 只有非工具调用轮次才回调给用户
                    await on_content_delta(content_piece)

        # —— 组装完整响应 ——
        content = "".join(content_parts)
        reasoning_content = "".join(reasoning_parts)

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_chunks):
            item = tool_call_chunks[idx]
            tool_calls.append(
                ToolCall(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=_parse_tool_arguments(
                        item.get("arguments", "{}")
                    ),
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )