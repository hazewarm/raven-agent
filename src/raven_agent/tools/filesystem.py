from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from raven_agent.tools.base import Tool, ToolResult
from raven_agent.tools.paths import resolve_tool_path


class WriteTextFileTool(Tool):
    """写入 UTF-8 文本文件的工具。

    输入:
        allowed_dir: 构造函数参数，限制本工具只能写入该目录内的路径。

    输出:
        一个 Tool 实例。执行 execute() 后返回 ToolResult。
    """

    name = "write_text_file"
    description = "写入 UTF-8 文本文件。默认不覆盖已有文件；覆盖必须显式传 overwrite=true。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要写入的文件路径。"},
            "content": {"type": "string", "description": "要写入的文本内容。"},
            "overwrite": {
                "type": "boolean",
                "description": "目标文件存在时是否覆盖，默认 false。",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, allowed_dir: Path | None = None) -> None:
        """初始化写文件工具。

        输入:
            allowed_dir: 允许写入的根目录；为 None 时不做目录限制。

        输出:
            None。初始化后的状态保存在 self._allowed_dir。
        """

        self._allowed_dir = allowed_dir

    async def execute(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """执行文本写入。

        输入:
            path: 要写入的文件路径。
            content: 要写入的 UTF-8 文本。
            overwrite: 目标文件存在时是否允许覆盖。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。成功时 metadata.ok=True；失败时 metadata.ok=False 并包含 error。
        """

        try:
            file_path = resolve_tool_path(path, self._allowed_dir)
            if file_path.exists() and file_path.is_dir():
                return ToolResult(text=f"写入失败，目标是目录: {path}", metadata={"ok": False, "error": "target_is_directory"})
            if file_path.exists() and not overwrite:
                return ToolResult(text=f"写入失败，文件已存在: {path}。如需覆盖请设置 overwrite=true。", metadata={"ok": False, "error": "file_exists"})

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(
                text=f"已写入 {len(content)} 个字符到 {path}",
                metadata={"ok": True, "path": str(file_path), "chars": len(content), "overwritten": overwrite},
            )
        except PermissionError as exc:
            return ToolResult(text=str(exc), metadata={"ok": False, "error": "permission_denied"})
        except OSError as exc:
            return ToolResult(text=f"写入失败: {exc}", metadata={"ok": False, "error": "write_failed"})


class EditFileTool(Tool):
    """精确替换 UTF-8 文本文件内容的工具。

    输入:
        allowed_dir: 构造函数参数，限制本工具只能编辑该目录内的文件。

    输出:
        一个 Tool 实例。执行 execute() 后返回包含 diff 的 ToolResult。
    """

    name = "edit_file"
    description = (
        "精确替换 UTF-8 文本文件中的 old_text。old_text 必须与文件内容完全一致；"
        "默认只替换一处，多处替换必须显式传 replace_all=true。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要编辑的文件路径。"},
            "old_text": {"type": "string", "description": "要查找的原始文本，必须完全匹配。"},
            "new_text": {"type": "string", "description": "替换后的新文本。"},
            "replace_all": {
                "type": "boolean",
                "description": "是否替换所有匹配项，默认 false。",
                "default": False,
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, allowed_dir: Path | None = None) -> None:
        """初始化编辑工具。

        输入:
            allowed_dir: 允许编辑的根目录；为 None 时不做目录限制。

        输出:
            None。初始化后的状态保存在 self._allowed_dir。
        """

        self._allowed_dir = allowed_dir

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """执行精确文本替换。

        输入:
            path: 要编辑的文件路径。
            old_text: 要查找的原始文本，必须与文件内容完全一致。
            new_text: 替换后的新文本。
            replace_all: 是否替换所有匹配项。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。成功时 text 包含 unified diff；失败时 metadata.ok=False。
        """

        try:
            file_path = resolve_tool_path(path, self._allowed_dir)
            if not file_path.exists():
                return ToolResult(text=f"编辑失败，文件不存在: {path}", metadata={"ok": False, "error": "not_found"})
            if not file_path.is_file():
                return ToolResult(text=f"编辑失败，路径不是文件: {path}", metadata={"ok": False, "error": "not_file"})

            raw_content = file_path.read_text(encoding="utf-8")
            content, has_bom = _strip_utf8_bom(raw_content)
            matched_old_text = old_text
            replacement_text = new_text

            if matched_old_text not in content and _supports_crlf_compat(content):
                crlf_old_text = old_text.replace("\n", "\r\n")
                if crlf_old_text in content:
                    matched_old_text = crlf_old_text
                    replacement_text = new_text.replace("\n", "\r\n")

            if matched_old_text not in content:
                return ToolResult(text="编辑失败，未找到 old_text，请确保文本完全一致。", metadata={"ok": False, "error": "old_text_not_found"})

            count = content.count(matched_old_text)
            if count > 1 and not replace_all:
                return ToolResult(
                    text=f"编辑失败，old_text 出现 {count} 次。请提供更多上下文，或设置 replace_all=true。",
                    metadata={"ok": False, "error": "multiple_matches", "count": count},
                )

            new_content = content.replace(matched_old_text, replacement_text) if replace_all else content.replace(matched_old_text, replacement_text, 1)
            replaced = count if replace_all else 1
            diff_text = _build_diff(content, new_content, path)
            file_path.write_text(_restore_utf8_bom(new_content, has_bom), encoding="utf-8", newline="")
            return ToolResult(
                text=f"已编辑 {path}，替换 {replaced} 处。\n\n```diff\n{diff_text}\n```",
                metadata={"ok": True, "path": str(file_path), "replaced": replaced},
            )
        except UnicodeDecodeError:
            return ToolResult(text=f"编辑失败，文件不是有效 UTF-8 文本: {path}", metadata={"ok": False, "error": "decode_failed"})
        except PermissionError as exc:
            return ToolResult(text=str(exc), metadata={"ok": False, "error": "permission_denied"})
        except OSError as exc:
            return ToolResult(text=f"编辑失败: {exc}", metadata={"ok": False, "error": "edit_failed"})


class ReadImageInfoTool(Tool):
    """读取图片文件元信息的工具。

    输入:
        allowed_dir: 构造函数参数，限制本工具只能读取该目录内的图片。

    输出:
        一个 Tool 实例。执行 execute() 后返回图片格式、尺寸、颜色模式和文件大小。
    """

    name = "read_image_info"
    description = "读取图片文件的格式、尺寸、颜色模式和文件大小。当前只返回元信息，不把图片内容发送给模型。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取元信息的图片路径。"},
        },
        "required": ["path"],
    }

    def __init__(self, allowed_dir: Path | None = None) -> None:
        """初始化图片元信息工具。

        输入:
            allowed_dir: 允许读取的根目录；为 None 时不做目录限制。

        输出:
            None。初始化后的状态保存在 self._allowed_dir。
        """

        self._allowed_dir = allowed_dir

    async def execute(self, path: str, **kwargs: Any) -> ToolResult:
        """读取图片元信息。

        输入:
            path: 图片文件路径。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。成功时 text 为 JSON 字符串，metadata 包含图片元信息。
        """

        try:
            file_path = resolve_tool_path(path, self._allowed_dir)
            if not file_path.exists():
                return ToolResult(text=f"图片不存在: {path}", metadata={"ok": False, "error": "not_found"})
            if not file_path.is_file():
                return ToolResult(text=f"路径不是文件: {path}", metadata={"ok": False, "error": "not_file"})

            with Image.open(file_path) as image:
                transposed = ImageOps.exif_transpose(image)
                width, height = transposed.size
                payload = {
                    "path": str(file_path),
                    "format": image.format,
                    "width": width,
                    "height": height,
                    "mode": transposed.mode,
                    "bytes": file_path.stat().st_size,
                    "note": "当前章节只读取图片元信息；真正多模态图片理解会在后续模型消息格式支持后实现。",
                }
            return ToolResult(text=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"ok": True, **payload})
        except PermissionError as exc:
            return ToolResult(text=str(exc), metadata={"ok": False, "error": "permission_denied"})
        except Exception as exc:
            return ToolResult(text=f"读取图片信息失败: {exc}", metadata={"ok": False, "error": "image_failed"})


def _strip_utf8_bom(text: str) -> tuple[str, bool]:
    """移除 UTF-8 BOM 并记录是否存在 BOM。

    输入:
        text: 从文件读取出的原始文本。

    输出:
        二元组 `(clean_text, has_bom)`；clean_text 为去掉 BOM 后的文本，has_bom 表示原文是否包含 BOM。
    """

    if text.startswith("﻿"):
        return text[1:], True
    return text, False


def _restore_utf8_bom(text: str, has_bom: bool) -> str:
    """按需恢复 UTF-8 BOM。

    输入:
        text: 准备写回文件的文本。
        has_bom: 原文件是否带 BOM。

    输出:
        如果 has_bom=True，返回带 BOM 的文本；否则返回原文本。
    """

    return "﻿" + text if has_bom else text


def _supports_crlf_compat(text: str) -> bool:
    """判断文本是否适合把 LF 匹配自动兼容为 CRLF 匹配。

    输入:
        text: 文件内容。

    输出:
        bool。True 表示文件主要使用 CRLF 且没有混合换行，可以尝试兼容匹配。
    """

    if "\r\n" not in text:
        return False
    without_crlf = text.replace("\r\n", "")
    return "\r" not in without_crlf and "\n" not in without_crlf


def _build_diff(old_text: str, new_text: str, path: str) -> str:
    """构造 unified diff 文本。

    输入:
        old_text: 修改前文件内容。
        new_text: 修改后文件内容。
        path: 文件路径，用于 diff header。

    输出:
        unified diff 字符串。
    """

    return "\n".join(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            lineterm="",
            n=2,
        )
    )