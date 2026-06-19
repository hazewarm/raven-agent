"""测试 VL 视觉工具：Path A 消息格式 + 共享基础设施 + Path B 工具执行。"""

from __future__ import annotations

import base64
import io
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raven_agent.media import detect_image_mime


def _create_test_image(path: Path, fmt: str, size: tuple[int, int] = (100, 100)) -> None:
    """在指定路径创建一张测试图片。

    输入:
        path: 目标文件路径。根据后缀名决定图片格式。
        fmt: 图片格式字符串，例如 "PNG"、"JPEG"。
        size: 图片尺寸 (width, height)，默认 100×100。

    输出:
        None。文件写入 path。
    """

    from PIL import Image

    img = Image.new("RGB", size, color=(255, 0, 0))
    img.save(path, format=fmt)


class TestDetectImageMime:
    """测试 detect_image_mime() 的各种格式检测。"""

    def test_detect_png(self, tmp_path: Path) -> None:
        """PNG 图片应返回 image/png。"""
        p = tmp_path / "test.png"
        _create_test_image(p, "PNG")
        assert detect_image_mime(p) == "image/png"

    def test_detect_jpeg(self, tmp_path: Path) -> None:
        """JPEG 图片应返回 image/jpeg。"""
        p = tmp_path / "test.jpg"
        _create_test_image(p, "JPEG")
        assert detect_image_mime(p) == "image/jpeg"

    def test_detect_gif(self, tmp_path: Path) -> None:
        """GIF 图片应返回 image/gif。"""
        from PIL import Image
        p = tmp_path / "test.gif"
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        img.save(p, format="GIF")
        assert detect_image_mime(p) == "image/gif"

    def test_detect_bmp(self, tmp_path: Path) -> None:
        """BMP 图片应返回 image/bmp。"""
        p = tmp_path / "test.bmp"
        _create_test_image(p, "BMP")
        assert detect_image_mime(p) == "image/bmp"

    def test_detect_webp(self, tmp_path: Path) -> None:
        """WebP 图片应返回 image/webp。"""
        p = tmp_path / "test.webp"
        _create_test_image(p, "WEBP")
        assert detect_image_mime(p) == "image/webp"

    def test_detect_non_image(self, tmp_path: Path) -> None:
        """非图片文件应返回 None。"""
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        assert detect_image_mime(p) is None

    def test_detect_missing_file(self, tmp_path: Path) -> None:
        """不存在的文件应返回 None。"""
        p = tmp_path / "does_not_exist.png"
        assert detect_image_mime(p) is None

    def test_detect_magic_bytes_only(self, tmp_path: Path) -> None:
        """Magic bytes fallback：手动构造有效文件头的文件应被正确识别。"""
        p = tmp_path / "fake.png"
        # 写入最小有效 PNG 文件头（Pillow 可能拒绝打开，但 magic bytes 检测应工作）
        p.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        )
        # 这个残缺文件 Pillow 可能无法打开，但 magic bytes 应能识别
        # 先尝试 Pillow，失败后走 magic bytes
        result = detect_image_mime(p)
        assert result == "image/png"


from raven_agent.media import encode_image_to_data_uri


class TestEncodeImageToDataUri:
    """测试 encode_image_to_data_uri() 图片编码。"""

    def test_encode_small_png(self, tmp_path: Path) -> None:
        """小图片应正确编码为 data URI。"""
        p = tmp_path / "small.png"
        _create_test_image(p, "PNG", (50, 50))
        uri = encode_image_to_data_uri(p)
        assert uri.startswith("data:image/")
        assert ";base64," in uri
        # 验证 base64 解码
        b64_part = uri.split(";base64,", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

    def test_encode_jpeg(self, tmp_path: Path) -> None:
        """JPEG 图片应编码为 image/jpeg MIME。"""
        p = tmp_path / "test.jpg"
        _create_test_image(p, "JPEG", (50, 50))
        uri = encode_image_to_data_uri(p)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_encode_oversized_triggers_compression(self, tmp_path: Path) -> None:
        """超大尺寸图片应触发缩放。"""
        p = tmp_path / "large.png"
        _create_test_image(p, "PNG", (5000, 5000))
        uri = encode_image_to_data_uri(p)
        assert uri.startswith("data:image/")
        b64_part = uri.split(";base64,", 1)[1]
        decoded = base64.b64decode(b64_part)
        # 压缩后应远小于原始尺寸
        from PIL import Image
        img = Image.open(io.BytesIO(decoded))
        assert max(img.size) <= 4096

    def test_encode_rgba_converts_to_rgb(self, tmp_path: Path) -> None:
        """RGBA 模式图片应正确转换为 RGB。"""
        from PIL import Image
        p = tmp_path / "rgba.png"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(p, format="PNG")
        uri = encode_image_to_data_uri(p)
        assert uri.startswith("data:image/")

    def test_encode_unsupported_format_raises(self, tmp_path: Path) -> None:
        """不支持的文件应抛出 ValueError。"""
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            encode_image_to_data_uri(p)



from raven_agent.messages import ChatMessage, MediaItem, user_message, ToolCall


class TestChatMessageWithMediaItems:
    """测试 Path A：ChatMessage 携带 media_items 时的 OpenAI 格式输出。"""

    def test_user_message_without_media_uses_string_content(self) -> None:
        """无媒体附件时 content 应为纯字符串。"""
        msg = user_message("描述这张图片")
        d = msg.to_openai_dict()
        assert d["role"] == "user"
        assert isinstance(d["content"], str)
        assert d["content"] == "描述这张图片"

    def test_user_message_with_image_uses_content_list(self) -> None:
        """有图片时 content 应为 content list 格式。"""
        msg = user_message(
            "看看这张图",
            media_items=[MediaItem(type="image", uri="data:image/png;base64,abc123")],
        )
        d = msg.to_openai_dict()
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2  # text + image_url

        text_block = d["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "看看这张图"

        image_block = d["content"][1]
        assert image_block["type"] == "image_url"
        assert image_block["image_url"]["url"] == "data:image/png;base64,abc123"
        assert image_block["image_url"]["detail"] == "auto"

    def test_user_message_empty_text_with_image(self) -> None:
        """纯图片消息（无文本）也正确输出。"""
        msg = user_message(
            "",
            media_items=[MediaItem(type="image", uri="data:image/jpeg;base64,xyz")],
        )
        d = msg.to_openai_dict()
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 1  # 只有 image_url，无 text block
        assert d["content"][0]["type"] == "image_url"

    def test_assistant_message_ignores_media_items(self) -> None:
        """assistant 消息不应该使用 media_items 字段。"""
        msg = ChatMessage(
            role="assistant",
            content="我看到了图片",
            media_items=[MediaItem(type="image", uri="data:image/png;base64,ignored")],
        )
        d = msg.to_openai_dict()
        # assistant 消息的 media_items 被忽略（content 仍是字符串）
        assert isinstance(d["content"], str)

    def test_multiple_media_items(self) -> None:
        """多个媒体附件应全部出现在 content list 中。"""
        msg = user_message(
            "比较这两张图",
            media_items=[
                MediaItem(type="image", uri="data:image/png;base64,img1"),
                MediaItem(type="image", uri="data:image/png;base64,img2"),
            ],
        )
        d = msg.to_openai_dict()
        assert len(d["content"]) == 3  # 1 text + 2 images

    def test_system_message_ignores_media_items(self) -> None:
        """system 消息在 to_openai_dict 时 media_items 字段被忽略。"""
        msg = ChatMessage(
            role="system",
            content="你是一个助手",
            media_items=[MediaItem(type="image", uri="data:image/png;base64,ignored")],
        )
        d = msg.to_openai_dict()
        assert isinstance(d["content"], str)
        assert d["content"] == "你是一个助手"


from raven_agent.tools.vision import ReadImageVisionTool


class TestReadImageVisionTool:
    """测试 ReadImageVisionTool 的路径校验和错误处理。"""

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path: Path) -> None:
        """不存在的文件应返回错误。"""
        mock_provider = MagicMock()
        tool = ReadImageVisionTool(
            vl_provider=mock_provider,
            vl_model="qwen-vl-max",
        )
        result = await tool.execute(path=str(tmp_path / "nope.png"), prompt="描述图片")
        assert "文件不存在" in result.text
        assert result.metadata["ok"] is False

    @pytest.mark.asyncio
    async def test_path_is_directory(self, tmp_path: Path) -> None:
        """目标路径是目录时应返回错误。"""
        mock_provider = MagicMock()
        tool = ReadImageVisionTool(
            vl_provider=mock_provider,
            vl_model="qwen-vl-max",
        )
        result = await tool.execute(path=str(tmp_path), prompt="描述图片")
        assert "不是文件" in result.text
        assert result.metadata["ok"] is False

    @pytest.mark.asyncio
    async def test_path_outside_allowed_dir(self, tmp_path: Path) -> None:
        """越权访问应返回 PermissionError。"""
        mock_provider = MagicMock()
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        tool = ReadImageVisionTool(
            vl_provider=mock_provider,
            vl_model="qwen-vl-max",
            allowed_dir=allowed,
        )
        # 访问 allowed 目录外的文件
        outside = tmp_path / "outside.txt"
        outside.write_text("test")
        result = await tool.execute(path=str(outside), prompt="描述图片")
        assert result.metadata["ok"] is False
        assert result.metadata.get("error") == "permission_denied"

    @pytest.mark.asyncio
    async def test_execute_calls_vl_provider(self, tmp_path: Path) -> None:
        """正常执行时应调用 VL provider 的 chat_dicts()，并返回其内容。"""
        from PIL import Image

        p = tmp_path / "test.png"
        img = Image.new("RGB", (50, 50), color=(0, 0, 255))
        img.save(p, format="PNG")

        mock_provider = MagicMock()
        mock_provider.chat_dicts = AsyncMock()
        mock_provider.chat_dicts.return_value = MagicMock(
            content="图片描述：蓝色正方形",
            reasoning_content="",
        )

        tool = ReadImageVisionTool(
            vl_provider=mock_provider,
            vl_model="qwen-vl-max",
        )
        result = await tool.execute(path=str(p), prompt="描述这张图片")

        # 验证 chat_dicts 被调用
        mock_provider.chat_dicts.assert_called_once()
        call_args = mock_provider.chat_dicts.call_args
        assert call_args.kwargs["model"] == "qwen-vl-max"
        # 验证返回内容
        assert "蓝色正方形" in result.text
        assert result.metadata["ok"] is True

    @pytest.mark.asyncio
    async def test_execute_vl_call_failure(self, tmp_path: Path) -> None:
        """VL 模型调用失败时应返回错误信息。"""
        from PIL import Image

        p = tmp_path / "test.png"
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))
        img.save(p, format="PNG")

        mock_provider = MagicMock()
        mock_provider.chat_dicts = AsyncMock(
            side_effect=RuntimeError("VL 模型暂时不可用")
        )

        tool = ReadImageVisionTool(
            vl_provider=mock_provider,
            vl_model="qwen-vl-max",
        )
        result = await tool.execute(path=str(p), prompt="描述图片")

        assert result.metadata["ok"] is False
        assert "调用视觉模型失败" in result.text
        assert "VL 模型暂时不可用" in result.text