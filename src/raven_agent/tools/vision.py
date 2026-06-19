"""视觉工具：使用独立的 VL 模型分析图片，返回文本描述。"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from raven_agent.llm import LLMProvider
from raven_agent.tools.base import Tool, ToolResult
from raven_agent.media import detect_image_mime, encode_image_to_data_uri
from raven_agent.tools.paths import resolve_tool_path


class ReadImageVisionTool(Tool):
    """使用 VL 模型分析图片，返回视觉理解结果。

    适用场景：主模型不支持多模态（Path B），需要单独调用视觉模型
    来识别图片内容。

    输入:
        vl_provider: 用于调用 VL 模型的 LLMProvider 实例。
        vl_model: VL 模型名称，例如 "qwen-vl-max"。
        allowed_dir: 可选，限制本工具只能读取该目录内的图片。

    输出:
        一个 Tool 实例。执行 execute() 后返回 VL 模型的文本描述。
    """

    name = "read_image_vision"
    description = (
        "使用独立的视觉模型分析图片内容。主模型无法直接查看图片时使用此工具。"
        "你需要提供一个 prompt 来说明你想从图片中了解什么。\n\n"
        "参数说明：\n"
        "- path：图片文件的路径\n"
        "- prompt：描述你想从这张图片中了解什么内容，越具体越好。"
        "例如 '图中有什么文字？'、'描述这张图片中的物体和场景'、"
        "'这张表格中第3行的数据是什么？'\n\n"
        "限制：原始文件不超过20MB，超限图片会自动缩放至最宽/最高4096像素并压缩。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "图片文件的路径",
            },
            "prompt": {
                "type": "string",
                "description": "描述你想从图片中了解什么内容，越具体越好",
            },
        },
        "required": ["path", "prompt"],
    }

    def __init__(
        self,
        vl_provider: LLMProvider,
        vl_model: str,
        allowed_dir: Path | None = None,
    ) -> None:
        """初始化 VL 视觉工具。

        输入:
            vl_provider: 用于调用 VL 模型的 LLMProvider。
            vl_model: VL 模型名称。
            allowed_dir: 允许读取图片的根目录；为 None 时不限制。

        输出:
            None。
        """

        self._provider = vl_provider
        self._model = vl_model
        self._allowed_dir = allowed_dir

    async def execute(self, path: str, prompt: str, **kwargs: Any) -> ToolResult:
        """执行图片视觉分析。

        输入:
            path: 图片文件路径。
            prompt: 描述想从图片中了解什么，越具体越好。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。成功时 text 为 VL 模型的文本描述。
        """

        # ── 路径校验 ──
        try:
            file_path = resolve_tool_path(path, self._allowed_dir)
            if not file_path.exists():
                return ToolResult(
                    text=f"错误：文件不存在：{path}",
                    metadata={"ok": False, "error": "not_found"},
                )
            if not file_path.is_file():
                return ToolResult(
                    text=f"错误：路径不是文件：{path}",
                    metadata={"ok": False, "error": "not_file"},
                )
        except PermissionError as e:
            return ToolResult(
                text=str(e),
                metadata={"ok": False, "error": "permission_denied"},
            )

        # ── 图片编码（使用共享的 encode_image_to_data_uri）──
        try:
            data_uri = encode_image_to_data_uri(file_path)
        except ValueError as e:
            return ToolResult(
                text=f"图片处理失败：{e}",
                metadata={"ok": False, "error": "encoding_failed"},
            )
        except Exception as e:
            return ToolResult(
                text=f"读取图片文件失败：{e}",
                metadata={"ok": False, "error": "read_failed"},
            )

        # ── 构造 VL 多模态消息 ──
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            }
        ]

        # ── 调用 VL 模型 ──
        try:
            response = await self._provider.chat_dicts(
                messages=messages,
                tools=None,
                model=self._model,
                max_tokens=2048,
            )
            if response.content:
                return ToolResult(
                    text=response.content,
                    metadata={"ok": True},
                )
            if response.reasoning_content:
                return ToolResult(
                    text=f"[VL 模型思考过程]\n{response.reasoning_content}",
                    metadata={"ok": True, "thinking_only": True},
                )
            return ToolResult(
                text="视觉模型未返回任何内容，请尝试调整 prompt 后重试。",
                metadata={"ok": False, "error": "empty_response"},
            )
        except Exception as e:
            return ToolResult(
                text=f"调用视觉模型失败：{e}",
                metadata={"ok": False, "error": "vl_call_failed"},
            )