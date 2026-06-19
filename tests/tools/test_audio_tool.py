"""测试音频工具：transcribe_audio() + TranscribeAudioTool。"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _create_test_wav(path: Path, duration_seconds: float = 1.0) -> None:
    """在指定路径创建一个包含静音的测试 WAV 文件。

    输入:
        path: 目标文件路径。
        duration_seconds: 音频时长（秒），默认 1 秒。

    输出:
        None。文件写入 path。
    """

    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    silence = b"\x00\x00" * num_samples

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)


class TestTranscribeAudio:
    """测试 transcribe_audio() 转录引擎。"""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """不存在的文件应抛出 ValueError。"""
        from raven_agent.media import transcribe_audio

        with pytest.raises(ValueError, match="音频文件不存在"):
            transcribe_audio(tmp_path / "nope.wav")

    def test_directory_raises(self, tmp_path: Path) -> None:
        """路径是目录时应抛出 ValueError。"""
        from raven_agent.media import transcribe_audio

        with pytest.raises(ValueError, match="路径不是文件"):
            transcribe_audio(tmp_path)

    def test_model_loads_on_first_call(self, tmp_path: Path) -> None:
        """首次调用应触发模型加载，第二次应复用缓存。"""
        from raven_agent.media import audio

        p = tmp_path / "test.wav"
        _create_test_wav(p)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], {"language": "en"})
        audio._model = None

        def _fake_load(model=None):
            """模拟 _load_model()：加载后写入模块级单例缓存。"""
            audio._model = mock_model
            audio._loaded_model = model
            return mock_model

        with patch.object(audio, "_load_model", side_effect=_fake_load) as mock_load:
            # 第一次调用 → 触发 _load_model（mock）
            audio.transcribe_audio(p)
            assert mock_load.call_count == 1

        # ── patch 退出，_load_model 恢复为真实函数 ──
        # 此时 audio._model 已被 _fake_load 写入 mock_model，
        # 真实 _load_model 看到缓存非空，直接 return，不创建新 WhisperModel
        # 第二次调用应成功完成（验证缓存短路不会报错或挂起）
        audio.transcribe_audio(p)

    def test_transcribe_returns_text(self, tmp_path: Path) -> None:
        """转录应返回文本——测试 mock 路径，不加载真实模型。"""
        from raven_agent.media import audio

        p = tmp_path / "test.wav"
        _create_test_wav(p)

        # 构造一个 mock segment
        mock_segment = MagicMock()
        mock_segment.text = "这是一段测试语音"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], {"language": "zh"})

        with patch.object(audio, "_load_model", return_value=mock_model):
            # 必须重置 _model 单例，否则前一个测试可能已缓存
            audio._model = None
            result = audio.transcribe_audio(p)
            assert "测试语音" in result


class TestTranscribeAudioTool:
    """测试 TranscribeAudioTool 的路径校验和错误处理。"""

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path: Path) -> None:
        """不存在的文件应返回错误。"""
        from raven_agent.tools.audio import TranscribeAudioTool

        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(tmp_path / "nope.ogg"), prompt="")
        assert "文件不存在" in result.text
        assert result.metadata["ok"] is False

    @pytest.mark.asyncio
    async def test_path_is_directory(self, tmp_path: Path) -> None:
        """目标路径是目录时应返回错误。"""
        from raven_agent.tools.audio import TranscribeAudioTool

        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(tmp_path), prompt="")
        assert "不是文件" in result.text
        assert result.metadata["ok"] is False

    @pytest.mark.asyncio
    async def test_path_outside_allowed_dir(self, tmp_path: Path) -> None:
        """越权访问应返回 PermissionError。"""
        from raven_agent.tools.audio import TranscribeAudioTool

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        tool = TranscribeAudioTool(allowed_dir=allowed)

        outside = tmp_path / "outside.ogg"
        outside.write_bytes(b"fake ogg data")
        result = await tool.execute(path=str(outside), prompt="")
        assert result.metadata["ok"] is False
        assert result.metadata.get("error") == "permission_denied"

    @pytest.mark.asyncio
    async def test_transcribe_success(self, tmp_path: Path) -> None:
        """正常转录应返回 transcribed 文本。"""
        import wave
        from raven_agent.tools.audio import TranscribeAudioTool
        from raven_agent.media import audio as audio_mod

        p = tmp_path / "test.wav"
        _create_test_wav(p)

        mock_segment = MagicMock()
        mock_segment.text = "用户说了一段话"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], {"language": "zh"})

        with patch.object(audio_mod, "_load_model", return_value=mock_model):
            audio_mod._model = None
            tool = TranscribeAudioTool()
            result = await tool.execute(path=str(p))
            assert result.metadata["ok"] is True
            assert "用户说了一段话" in result.text

    @pytest.mark.asyncio
    async def test_transcribe_failure_is_caught(self, tmp_path: Path) -> None:
        """转录异常应返回错误信息而非抛出。"""
        from raven_agent.tools.audio import TranscribeAudioTool

        p = tmp_path / "test.wav"
        _create_test_wav(p)

        # patch 目标必须是 Tool 实际 import 的位置，而非 media/audio.py 模块
        with patch(
            "raven_agent.tools.audio.transcribe_audio",
            side_effect=RuntimeError("Whisper 模型加载失败"),
        ):
            tool = TranscribeAudioTool()
            result = await tool.execute(path=str(p))
            assert result.metadata["ok"] is False
            assert "Whisper 模型加载失败" in result.text