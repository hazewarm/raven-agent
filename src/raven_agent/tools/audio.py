"""音频工具：使用本地 Whisper 模型将语音转录为文本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raven_agent.tools.base import Tool, ToolResult
from raven_agent.tools.paths import resolve_tool_path
from raven_agent.media import transcribe_audio


class TranscribeAudioTool(Tool):
    """使用本地 Whisper 模型将语音文件转录为文本。

    适用场景：主模型不支持多模态音频，用户发送语音消息后，
    通过此工具将其转录为文本，主模型基于文本进行理解。

    输入:
        allowed_dir: 可选，限制本工具只能读取该目录内的音频文件。

    输出:
        一个 Tool 实例。执行 execute() 后返回转录文本。
    """

    name = "transcribe_audio"
    description = (
        "使用本地 Whisper 模型将语音文件转录为文本。"
        "将音频文件转写文字，供后续理解处理。\n\n"
        "参数说明：\n"
        "- path：音频文件的路径\n\n"
        "支持格式：WAV、MP3、OGG、FLAC、M4A。"
        "自动检测语言（包括中文、英文等 99 种语言）。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "音频文件的路径",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        allowed_dir: Path | None = None,
        model: str = "small",
    ) -> None:
        """初始化音频转录工具。

        输入:
            allowed_dir: 允许读取音频文件的根目录；为 None 时不限制。
            model: Whisper 模型名，如 "small"、"medium"。
                对应 config.toml 的 [audio] model 字段。

        输出:
            None。
        """

        self._allowed_dir = allowed_dir
        self._model = model

    async def execute(self, path: str, **kwargs: Any) -> ToolResult:
        """执行语音转录。

        输入:
            path: 音频文件路径。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。成功时 text 为转录文本；失败时 metadata.ok=False。
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

        # ── 转录 ──
        try:
            text = transcribe_audio(file_path, model=self._model)
            if not text.strip():
                return ToolResult(
                    text="音频中没有检测到语音内容（可能是静音或无有效语音段）。",
                    metadata={"ok": True, "empty": True},
                )
            return ToolResult(
                text=text,
                metadata={"ok": True, "length": len(text)},
            )
        except ImportError as e:
            return ToolResult(
                text=f"转录失败：{e}",
                metadata={"ok": False, "error": "whisper_not_installed"},
            )
        except RuntimeError as e:
            return ToolResult(
                text=f"转录模型加载失败：{e}",
                metadata={"ok": False, "error": "model_load_failed"},
            )
        except Exception as e:
            return ToolResult(
                text=f"转录失败：{e}",
                metadata={"ok": False, "error": "transcription_failed"},
            )