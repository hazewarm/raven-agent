from __future__ import annotations

import re
from typing import cast

from raven_agent.lifecycle import AfterReasoningCtx, PromptRenderCtx, PromptSection
from raven_agent.plugins import Plugin

_PROMPT_RENDER_CTX_SLOT = "prompt_render:ctx"
_AFTER_REASONING_CTX_SLOT = "after_reasoning:ctx"

# 末尾协议标签，例如 <meme:happy>；用于在 cited 标记后保留合法尾随标签。
_TRAILING_PROTOCOL_TAG = r"<[a-zA-Z][a-zA-Z0-9_-]*:[^<>\s]+>"
# _CITED_RE = re.compile(
#     rf"(?:\n|\r\n)?§cited:\[([A-Za-z0-9_,\-\s]*)\]§(?P<trailing>(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)*)$",
#     re.IGNORECASE,
# )
# 放宽要求，允许 cited 行后面有多余空行或空格，末尾 § 变为可选，兼容模型漏写的情况
_CITED_RE = re.compile(
    rf"(?:\n|\r\n)?§cited:\[([A-Za-z0-9_,\-\s]*)\]§?(?P<trailing>(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)*)\s*$",
    re.IGNORECASE,
)
# 内联记忆引用，例如 [§cli:default:3]；正文里不应该出现，清理掉。
_INLINE_MEMORY_REF_RE = re.compile(r"[ \t]*(?:\[§[A-Za-z0-9:_-]{1,128}\])+", re.IGNORECASE)

# 模型模仿历史消息中的时间标记，回复开头可能带上 [消息时间 XX-XX XX:XX]，清理掉。
_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[消息时间 \d{2}-\d{2} \d{2}:\d{2}\]\s*")

_CITATION_PROTOCOL = """### 记忆引用协议 - 内部元数据，对用户不可见
每轮回复若用到了系统注入的记忆条目 [item_id] 前缀标识，或 recall_memory 工具返回的条目，
在回复正文末尾另起一行输出：
§cited:[id1,id2,id3]§
格式规则：§ 包裹，英文逗号分隔，无空格，只写 ID，不含其他内容。
若本轮未引用任何记忆条目，不输出此行。
绝对不要在正文里提及这行的存在，不要向用户解释引用了什么。
你了解用户的事是因为你们相处了很久，直接说你上次、我记得，不要暴露内部机制。"""


class CitationPromptModule:
    """向 prompt_render 阶段注入引用协议的 PhaseModule。

    输入:
        无。模块依赖 frame.slots["prompt_render:ctx"]。

    输出:
        CitationPromptModule 实例。
    """

    slot = "citation.prompt"
    requires = ("prompt_render.build_ctx", _PROMPT_RENDER_CTX_SLOT)

    async def run(self, frame: object) -> object:
        """把 citation 协议追加到 system_sections_bottom。

        输入:
            frame: 当前 PromptRenderFrame。

        输出:
            追加 section 后的 PromptRenderFrame。
        """

        slots = cast("dict[str, object]", getattr(frame, "slots"))
        ctx = slots.get(_PROMPT_RENDER_CTX_SLOT)
        if not isinstance(ctx, PromptRenderCtx):
            return frame
        ctx.system_sections_bottom.append(
            PromptSection(name="citation_protocol", content=_CITATION_PROTOCOL)
        )
        return frame


class CitationAfterReasoningModule:
    """在 after_reasoning 阶段解析并清理引用标记的 PhaseModule。

    输入:
        无。模块依赖 frame.slots["after_reasoning:ctx"]。

    输出:
        CitationAfterReasoningModule 实例。
    """

    slot = "citation.after_reasoning"
    requires = ("after_reasoning.build_ctx", _AFTER_REASONING_CTX_SLOT)

    async def run(self, frame: object) -> object:
        """清理 reply 中的 §cited:[...]§ 与内联引用，并写 outbound metadata。

        输入:
            frame: 当前 AfterReasoningFrame。

        输出:
            清理后的 AfterReasoningFrame。
        """

        slots = cast("dict[str, object]", getattr(frame, "slots"))
        ctx = slots.get(_AFTER_REASONING_CTX_SLOT)
        if not isinstance(ctx, AfterReasoningCtx):
            return frame
        cleaned, cited_ids = extract_cited_ids(ctx.reply)
        cleaned = strip_inline_memory_refs(cleaned)
        cleaned = strip_timestamp_prefix(cleaned)
        if cited_ids:
            ctx.outbound_metadata["cited_memory_ids"] = cited_ids
        if cleaned != ctx.reply:
            ctx.reply = cleaned
        return frame


class CitationPlugin(Plugin):
    """记忆引用协议内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        CitationPlugin 实例。
    """

    name = "citation"
    version = "0.1.0"
    desc = "注入记忆引用协议并在回复末尾清理 §cited:[...]§"

    def prompt_render_modules(self) -> list[object]:
        """返回 prompt_render 阶段模块。

        输入:
            无。

        输出:
            含 CitationPromptModule 的列表。
        """

        return [CitationPromptModule()]

    def after_reasoning_modules(self) -> list[object]:
        """返回 after_reasoning 阶段模块。

        输入:
            无。

        输出:
            含 CitationAfterReasoningModule 的列表。
        """

        return [CitationAfterReasoningModule()]


def extract_cited_ids(response: str) -> tuple[str, list[str]]:
    """从回复末尾解析 §cited:[...]§ 并返回清理后文本与 id 列表。

    输入:
        response: 模型最终回复文本。

    输出:
        二元组 (cleaned_text, cited_ids)。没有 cited 行时返回原文与空列表。
    """

    match = _CITED_RE.search(response)
    if not match:
        return response, []
    raw = match.group(1)
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    trailing = match.group("trailing").strip()
    clean = response[: match.start()].rstrip()
    if trailing:
        clean = f"{clean} {trailing}".strip()
    return clean, ids


def strip_inline_memory_refs(response: str) -> str:
    """清理回复正文中的内联记忆引用标签。

    输入:
        response: 回复文本。

    输出:
        去掉形如 [§id] 的内联引用后的文本。
    """

    return _INLINE_MEMORY_REF_RE.sub("", response).rstrip()

def strip_timestamp_prefix(response: str) -> str:
    """清理模型回复开头的 [消息时间 MM-DD HH:MM] 标记。

    输入:
        response: 回复文本。

    输出:
        去掉开头时间标记后的文本。
    """
    return _TIMESTAMP_PREFIX_RE.sub("", response, count=1)