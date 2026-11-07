"""图片输入处理：data URL/Base64 解析、魔数嗅探、大小与像素限制、方向矫正。

对应 SPEC §3.1 字段说明与 §2 图像预处理：
- 格式以魔数嗅探校验，不仅信 data URL 前缀（415）；
- 解码后 ≤ MAX_IMAGE_MB（413），编码长度同步预检；
- 最长边 > 8192 px 自动等比缩放并标记 resized=True。
"""

from __future__ import annotations

import base64
import binascii
import io

from PIL import Image, ImageOps

from grader.core.config import Settings
from grader.core.errors import (
    ImageTooLargeError,
    InvalidArgumentError,
    UnsupportedImageTypeError,
)

MAX_SIDE_PX = 8192

# 魔数表：(起始字节, 名称)
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),  # webp 为 RIFF 容器，需二次确认 WEBP 标记
]


def strip_data_url(raw: str) -> tuple[bytes, str | None]:
    """拆 data URL，返回 (二进制, 声明的 mime)；裸 Base64 时 mime 为 None。"""
    raw = raw.strip()
    if raw.lower().startswith("data:"):
        head, _, payload = raw.partition(",")
        if not _:
            raise InvalidArgumentError("image_base64 的 data URL 缺少逗号分隔的负载部分")
        declared = head.partition(":")[2].partition(";")[0].strip().lower() or None
        return payload.strip(), declared
    return raw, None


def decode_base64(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidArgumentError(f"image_base64 不是合法 Base64: {exc}") from exc


def sniff_format(data: bytes) -> str:
    """魔数嗅探，返回归一化格式名；不认识则抛 415。"""
    if len(data) < 12:
        raise UnsupportedImageTypeError("图片内容过短，无法识别格式")
    for magic, name in _MAGIC:
        if data.startswith(magic):
            if name == "webp" and data[8:12] != b"WEBP":
                continue
            return name
    raise UnsupportedImageTypeError("图片格式不支持或魔数不符（仅支持 JPEG/PNG/WebP）")


def check_declared_format(declared: str | None, actual: str) -> None:
    """data URL 声明的类型与魔数不符时拒绝（防伪造前缀）。"""
    if declared is None:
        return
    alias = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    if declared.split("/")[0] == "image":
        declared = declared.split("/", 1)[1]
    if declared in alias and alias[declared] != actual:
        raise UnsupportedImageTypeError(f"声明的图片类型 {declared} 与实际魔数 {actual} 不符")


class PreparedImage:
    """预处理后的图像及其元数据。"""

    __slots__ = ("image", "format", "resized", "decoded_bytes")

    def __init__(self, image: Image.Image, format: str, resized: bool, decoded_bytes: int) -> None:
        self.image = image
        self.format = format
        self.resized = resized
        self.decoded_bytes = decoded_bytes


def prepare_image(raw_b64: str, settings: Settings) -> PreparedImage:
    """完整预处理管线：解析 → 魔数 → 大小 → 解码 → 方向 → 超限缩放。"""
    payload, declared = strip_data_url(raw_b64)

    # 编码长度预检（Base64 膨胀约 4/3）：超出即拒绝，避免无谓解码
    if len(payload) > settings.max_image_bytes * 4.2:
        raise ImageTooLargeError(
            f"图片编码长度超过限制（解码后上限 {settings.max_image_mb:g} MB）"
        )

    data = decode_base64(payload)
    if len(data) > settings.max_image_bytes:
        raise ImageTooLargeError(
            f"图片解码后 {len(data)} 字节，超过限制 {settings.max_image_bytes} 字节"
        )

    fmt = sniff_format(data)
    check_declared_format(declared, fmt)

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # PIL 解码失败一律视为非法输入
        raise InvalidArgumentError(f"图片解码失败: {exc}") from exc

    # EXIF 方向矫正（拍摄件常见）
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    resized = False
    if max(image.size) > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)
        resized = True

    return PreparedImage(image=image, format=fmt, resized=resized, decoded_bytes=len(data))
