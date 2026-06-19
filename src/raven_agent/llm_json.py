from __future__ import annotations

from typing import Any

from json_repair import repair_json


def load_json_object_loose(text: str) -> dict[str, Any] | None:
    """从 LLM 文本中尽量解析 JSON object。

    参数:
        text: LLM 返回的原始文本。

    返回:
        解析出的 dict；无法解析或解析结果不是 object 时返回 None。
    """

    stripped = _strip_code_fence(text.strip())
    if not stripped:
        return None

    parsed = _repair_object(stripped)
    if parsed is not None:
        return parsed

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return _repair_object(stripped[start : end + 1])


def _strip_code_fence(text: str) -> str:
    """去掉 Markdown fenced code block 包裹。

    参数:
        text: 可能包含 fenced JSON 的文本。

    返回:
        去掉首尾 fence 后的文本。
    """

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _repair_object(text: str) -> dict[str, Any] | None:
    """用 json-repair 修复并解析 JSON object。

    参数:
        text: 可能不严格的 JSON 文本。

    返回:
        dict；无法修复或不是 dict 时返回 None。
    """

    try:
        parsed = repair_json(text, return_objects=True)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None