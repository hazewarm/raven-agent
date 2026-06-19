from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """模型请求执行的工具调用。

    参数:
        id: 模型生成的 tool call id，用于把工具结果对应回本次调用。
        name: 工具名称。
        arguments: 工具参数字典。
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        """转换为 OpenAI assistant tool_call 格式。

        返回:
            OpenAI Chat Completions API 接收的 tool call 字典。
        """

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

@dataclass(frozen=True)
class MediaItem:
    """消息中携带的媒体附件。

    用 type 区分媒体种类，避免以后每加一种类型就要在 ChatMessage
    上新增字段。to_openai_dict() 根据 type 决定 OpenAI content block
    的格式。

    参数:
        type: 媒体类型——"image" | "audio" | "video" | "document" 等。
        uri: 数据 URI（如 "data:image/jpeg;base64,..."），或本地文件路径。
        mime: 可选 MIME 类型字符串。uri 是文件路径时用于标识实际格式。
    """

    type: str
    uri: str
    mime: str = ""

@dataclass(frozen=True)
class ChatMessage:
    """发送给大模型的基础消息，同时携带本地 session message 身份。

    参数:
        role: 消息角色，可以是 system、user、assistant、tool。
        content: 消息文本内容。
        tool_calls: assistant 消息携带的工具调用列表。
        tool_call_id: tool 消息对应的工具调用 id。
        reasoning_content: thinking 模式下供应商返回的推理内容。
        id: 本地持久化 message id。
        seq: session 内单调递增序号。
        media_items: 媒体附件列表。非空时 to_openai_dict() 输出 content list
            而非 content string。仅 user 消息使用此字段。
            每个 MediaItem.type 决定其 OpenAI content block 类型
            （当前支持 "image"，可后续扩展 "audio" 等）。
        timestamp: 消息时间戳字符串。

    返回:
        ChatMessage 实例。
    """

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    reasoning_content: str = ""
    id: str = ""
    seq: int = -1
    media_items: list[MediaItem] = field(default_factory=list)
    timestamp: str = ""

    def to_openai_dict(self) -> dict[str, Any]:
        """转换为 OpenAI Chat Completions API 接收的 dict 格式。

        当 self.media_items 非空时（仅 user 消息），content 输出为
        [{type: "text", text: ...}, {type: "image_url", ...}, ...] 的
        content list 格式；否则 content 为纯字符串。

        返回:
            OpenAI 消息字典。

        异常:
            ValueError: 当 tool 消息缺少 tool_call_id 时抛出。
        """

        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool 消息必须包含 tool_call_id")
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }

        # ── 当 media_items 非空时，构造 content list ──
        if self.role == "user" and self.media_items:
            content_parts: list[dict[str, Any]] = []
            if self.content.strip():
                content_parts.append({"type": "text", "text": self.content})
            for item in self.media_items:
                if item.type == "image":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": item.uri, "detail": "auto"},
                    })
                # future: elif item.type == "audio":
                #     content_parts.append({
                #         "type": "input_audio",
                #         "input_audio": {"data": item.uri, "format": "ogg"},
                #     })
            payload: dict[str, Any] = {
                "role": self.role,
                "content": content_parts,
            }
        else:
            payload = {
                "role": self.role,
                "content": self.content,
            }

        if self.role == "assistant" and self.reasoning_content:
            payload["reasoning_content"] = self.reasoning_content
        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = [
                tool_call.to_openai_dict() for tool_call in self.tool_calls
            ]
        return payload


def system_message(content: str) -> ChatMessage:
    """创建 system 消息。

    参数:
        content: 系统提示词内容。

    返回:
        role 为 system 的 ChatMessage。
    """

    return ChatMessage(role="system", content=content)


def user_message(
    content: str,
    *,
    id: str = "",
    seq: int = -1,
    media_items: list[MediaItem] | None = None,
    timestamp: str = "",
) -> ChatMessage:
    """创建 user 消息。

    参数:
        content: 用户输入内容。
        id: 可选本地持久化 message id。
        seq: 可选 session 内序号。
        media_items: 可选媒体附件列表，用于多模态输入。
        timestamp: 消息时间戳字符串。

    返回:
        role 为 user 的 ChatMessage。
    """

    return ChatMessage(
        role="user",
        content=content,
        id=id,
        seq=seq,
        media_items=list(media_items or []),
        timestamp=timestamp,
    )


def assistant_message(
    content: str,
    tool_calls: list[ToolCall] | None = None,
    reasoning_content: str = "",
    *,
    id: str = "",
    seq: int = -1,
    timestamp: str = "",
) -> ChatMessage:
    """创建 assistant 消息。

    参数:
        content: 助手回复内容；当助手只请求工具时可以为空字符串。
        tool_calls: 助手请求执行的工具调用列表。
        reasoning_content: thinking 模式下供应商返回的推理内容；仅用于后续 API 回传。
        id: 可选本地持久化 message id；未持久化时传空字符串。
        seq: 可选 session 内序号；未持久化时传 -1。
        timestamp: 消息时间戳字符串。

    返回:
        role 为 assistant 的 ChatMessage。
    """

    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning_content,
        id=id,
        seq=seq,
        timestamp=timestamp,
    )


def tool_message(
    tool_call_id: str,
    content: str,
    *,
    id: str = "",
    seq: int = -1,
) -> ChatMessage:
    """创建 tool 结果消息。

    参数:
        tool_call_id: 对应 assistant tool call 的 id。
        content: 工具执行结果文本。
        id: 可选本地持久化 message id；未持久化时传空字符串。
        seq: 可选 session 内序号；未持久化时传 -1。

    返回:
        role 为 tool 的 ChatMessage。
    """

    return ChatMessage(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
        id=id,
        seq=seq,
    )