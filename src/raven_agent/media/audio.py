"""音频处理基础设施：语音转文字（STT）。

本模块提供基于 faster-whisper 的本地语音转录功能。
Whisper 模型在首次调用时自动下载并缓存，后续调用复用已加载的模型。
"""

from __future__ import annotations

import logging
from pathlib import Path
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ── 模型配置 ─────────────────────────────────────────────────

_model: object | None = None
_loaded_model: str | None = None     # 缓存当前已加载的模型名


def _load_model(model: str = "small") -> object:
    """加载 Whisper 模型（模块级单例）。

    同规格复用缓存；更换规格时重建模型。

    输入:
        model: Whisper 模型名，如 "small"、"medium"。

    输出:
        faster_whisper.WhisperModel 实例。
    """

    global _model, _loaded_model
    if _model is not None and _loaded_model == model:
        return _model

    try:
        logger.info(
            "加载 Whisper 模型  size=%s  device=auto  compute=auto",
            model,
        )
        _model = WhisperModel(model, device="auto", compute_type="auto")
    except Exception:
        logger.warning("GPU 不可用，降级为 CPU 推理")
        _model = WhisperModel(model, device="cpu", compute_type="int8")
    _loaded_model = model
    return _model

def transcribe_audio(file_path: str | Path, model: str = "small") -> str:
    """将音频文件转录为文本。

    输入:
        file_path: 音频文件路径。支持 WAV、MP3、OGG、FLAC、M4A 等
            Whisper 原生支持的格式。
        model: Whisper 模型名，如 "small"、"medium"。

    输出:
        转录后的文本字符串。空音频返回空字符串。

    异常:
        RuntimeError: 模型加载失败时抛出。
        ValueError: 文件不存在或格式不支持时抛出。
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"音频文件不存在: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    try:
        model = _load_model(model)
    except ImportError as e:
        raise ImportError(
            "faster-whisper 未安装。请运行: uv pip install faster-whisper"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Whisper 模型加载失败: {e}") from e

    segments, _info = model.transcribe(
        str(file_path),
        beam_size=5,
        language=None,          # 自动检测语言
        vad_filter=True,        # 过滤静音段
    )

    text_parts: list[str] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            text_parts.append(text)

    result = " ".join(text_parts).strip()
    word_count = len(result) if result else 0

    logger.info(
        "音频转录完成  path=%s  words=%d  preview=%s",
        file_path, word_count,
        result[:80] + "..." if len(result) > 80 else result,
    )
    return result