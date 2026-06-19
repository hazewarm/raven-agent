"""图片处理基础设施：格式检测、编码、压缩。

本模块是 Path A（主线注入）和 Path B（VL 工具）的共享层。
任何需要将磁盘上的图片文件转换为 data URI 的代码都应该从这里导入。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

# ── 图片处理常量 ──────────────────────────────────────────────

IMAGE_MAX_DATA_URI_BYTES = 8 * 1024 * 1024     # 8 MB data URI 上限（base64 编码后）
IMAGE_MAX_EDGE = 4096                          # 最长边像素上限，超限自动缩放


def detect_image_mime(file_path: str | Path) -> str | None:
    """检测图片文件的真实 MIME 类型。

    优先使用 Pillow 检测（更可靠），失败时回退到 magic bytes 检测。

    输入:
        file_path: 图片文件路径（字符串或 Path 对象）。

    输出:
        MIME 类型字符串（如 "image/png"、"image/jpeg"），
        无法识别时返回 None。
    """

    file_path = Path(file_path)

    # —— 方案 A：Pillow 检测（优先） ——
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            fmt = (img.format or "").upper()
        if fmt in ("PNG", "JPEG", "GIF", "BMP", "WEBP"):
            return f"image/{fmt.lower()}"
        if fmt == "MPO":
            return "image/jpeg"
    except Exception:
        pass

    # —— 方案 B：Magic bytes 检测（fallback） ——
    try:
        head = file_path.read_bytes()[:12]
    except OSError:
        return None

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"

    return None


def encode_image_to_data_uri(file_path: str | Path) -> str:
    """将图片文件编码为 data URI，大图自动缩放压缩。

    不管原始文件多大，系统都会自动缩放到最长边 {IMAGE_MAX_EDGE}px 以内
    并逐级降低 JPEG quality 直到满足 {IMAGE_MAX_DATA_URI_BYTES / 1024 / 1024:.0f}MB
    的 data URI 上限。只在极限情况下（压缩到 quality=45 仍超限）才报错。

    输入:
        file_path: 图片文件路径（字符串或 Path 对象）。

    输出:
        形如 "data:image/jpeg;base64,<base64>" 的 data URI 字符串。

    异常:
        ValueError: 格式不支持、解码失败或压缩到最低质量后仍超限时抛出。
    """

    file_path = Path(file_path)

    # ── 格式检测 ──
    raw = file_path.read_bytes()
    mime = detect_image_mime(file_path)
    if mime is None:
        raise ValueError("不支持的图片格式。仅支持 PNG、JPEG、GIF、BMP、WebP。")

    # ── Pillow 校验：图片可解码且无损坏 ──
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError("图片文件无法解码或已损坏。请确认这是有效图片。") from e

    # ── 打开、EXIF 方向修正、RGBA→RGB 转换 ──
    from PIL import Image, ImageOps
    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            canvas = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A") if "A" in img.getbands() else None
            canvas.paste(img.convert("RGB"), mask=alpha)
            img = canvas
        elif img.mode == "L":
            img = img.convert("RGB")

        # ── 尺寸和大小控制 ──
        raw_b64_len = len(base64.b64encode(raw).decode())
        if max(img.size) > IMAGE_MAX_EDGE or raw_b64_len > IMAGE_MAX_DATA_URI_BYTES:
            img.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE))

        # ── 尝试无重编码直接输出 ──
        if raw_b64_len <= IMAGE_MAX_DATA_URI_BYTES and max(img.size) <= IMAGE_MAX_EDGE:
            buf = io.BytesIO()
            if mime == "image/jpeg":
                img.save(buf, format="JPEG", quality=95, optimize=True)
                clean_mime = "image/jpeg"
            else:
                img.save(buf, format="PNG", optimize=True)
                clean_mime = "image/png"
            clean_b64 = base64.b64encode(buf.getvalue()).decode()
            if len(clean_b64) <= IMAGE_MAX_DATA_URI_BYTES:
                return f"data:{clean_mime};base64,{clean_b64}"

        # ── 降级压缩：逐级降低 JPEG quality ──
        best: bytes | None = None
        for quality in (85, 75, 65, 55, 45):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = buf.getvalue()
            candidate_b64 = base64.b64encode(candidate).decode()
            best = candidate
            if len(candidate_b64) <= IMAGE_MAX_DATA_URI_BYTES:
                return f"data:image/jpeg;base64,{candidate_b64}"

    # 循环未 return → quality=45 仍超限，报错
    raise ValueError(
        f"图片过大，压缩到最低质量后仍无法满足 {IMAGE_MAX_DATA_URI_BYTES / 1024 / 1024:.0f}MB "
        "data URI 上限。请尝试缩小图片尺寸后重发。"
    )