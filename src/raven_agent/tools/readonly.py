from __future__ import annotations

from pathlib import Path
from typing import Any

from raven_agent.tools.base import Tool, ToolResult
from raven_agent.tools.paths import resolve_tool_path


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """兼容旧调用的路径解析包装函数。

    输入:
        path: 用户或模型传入的文件路径。
        allowed_dir: 允许访问的根目录；为 None 时不做目录限制。

    输出:
        解析后的绝对 Path。
    """

    return resolve_tool_path(path, allowed_dir)


class ReadTextFileTool(Tool):
    """读取 UTF-8 文本文件的只读工具。

    参数:
        allowed_dir: 允许读取的根目录；为 None 时不限制。
        default_max_chars: 默认最多返回的字符数。
    """

    name = "read_text_file"
    description = "读取 UTF-8 文本文件内容。适合查看配置、源码和普通文本。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文本文件路径。",
            },
            "max_chars": {
                "type": "integer",
                "description": "最多返回多少个字符，默认 4000。",
                "minimum": 1,
                "default": 4000,
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        allowed_dir: Path | None = None,
        default_max_chars: int = 4000,
    ) -> None:
        self._allowed_dir = allowed_dir
        self._default_max_chars = max(1, int(default_max_chars))

    async def execute(
        self,
        path: str,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """读取指定文本文件。

        参数:
            path: 要读取的文件路径。
            max_chars: 最多返回多少个字符；为 None 时使用默认值。
            **kwargs: 预留参数，本工具当前不会使用。

        返回:
            ToolResult，包含文件文本或错误说明。
        """

        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return ToolResult(
                    text=f"文件不存在: {path}",
                    metadata={"path": str(file_path), "ok": False},
                )
            if not file_path.is_file():
                return ToolResult(
                    text=f"路径不是文件: {path}",
                    metadata={"path": str(file_path), "ok": False},
                )

            limit = self._default_max_chars if max_chars is None else max(1, int(max_chars))
            content = file_path.read_text(encoding="utf-8")
            truncated = len(content) > limit
            visible = content[:limit]
            if truncated:
                visible += f"\n\n[已截断，原文件 {len(content)} 字符，本次返回前 {limit} 字符]"

            return ToolResult(
                text=visible,
                metadata={
                    "path": str(file_path),
                    "ok": True,
                    "truncated": truncated,
                    "chars": len(content),
                },
            )
        except UnicodeDecodeError:
            return ToolResult(
                text=f"文件不是有效的 UTF-8 文本: {path}",
                metadata={"ok": False, "error": "decode_failed"},
            )
        except PermissionError as exc:
            return ToolResult(
                text=str(exc),
                metadata={"ok": False, "error": "permission_denied"},
            )


class ListDirectoryTool(Tool):
    """列出目录内容的只读工具。

    参数:
        allowed_dir: 允许访问的根目录；为 None 时不限制。
        default_max_items: 默认最多返回的目录项数量。
    """

    name = "list_directory"
    description = "列出目录下的文件和子目录。适合查看项目结构。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录路径。",
                "default": ".",
            },
            "max_items": {
                "type": "integer",
                "description": "最多返回多少个目录项，默认 100。",
                "minimum": 1,
                "default": 100,
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        allowed_dir: Path | None = None,
        default_max_items: int = 100,
    ) -> None:
        self._allowed_dir = allowed_dir
        self._default_max_items = max(1, int(default_max_items))

    async def execute(
        self,
        path: str = ".",
        max_items: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """列出目录内容。

        参数:
            path: 要列出的目录路径。
            max_items: 最多返回多少个目录项；为 None 时使用默认值。
            **kwargs: 预留参数，本工具当前不会使用。

        返回:
            ToolResult，包含目录项列表或错误说明。
        """

        try:
            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return ToolResult(
                    text=f"目录不存在: {path}",
                    metadata={"path": str(dir_path), "ok": False},
                )
            if not dir_path.is_dir():
                return ToolResult(
                    text=f"路径不是目录: {path}",
                    metadata={"path": str(dir_path), "ok": False},
                )

            limit = self._default_max_items if max_items is None else max(1, int(max_items))
            entries = sorted(
                dir_path.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
            visible_entries = entries[:limit]
            lines = [
                f"dir\t{item.name}" if item.is_dir() else f"file\t{item.name}"
                for item in visible_entries
            ]
            if len(entries) > limit:
                lines.append(f"[已截断，目录共 {len(entries)} 项，本次返回前 {limit} 项]")
            if not lines:
                lines.append("[空目录]")

            return ToolResult(
                text="\n".join(lines),
                metadata={
                    "path": str(dir_path),
                    "ok": True,
                    "count": len(entries),
                    "truncated": len(entries) > limit,
                },
            )
        except PermissionError as exc:
            return ToolResult(
                text=str(exc),
                metadata={"ok": False, "error": "permission_denied"},
            )
