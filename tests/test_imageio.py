"""图像处理测试：魔数嗅探、大小限制、超限缩放（AC10 依据）。"""

import base64

import pytest
from PIL import Image

from grader.core.config import Settings
from grader.core.errors import (
    ImageTooLargeError,
    InvalidArgumentError,
    UnsupportedImageTypeError,
)
from grader.core.image import prepare_image
from tests.conftest import png_data_url


def _settings(**over):
    base = dict(max_image_mb=10.0)
    base.update(over)
    return Settings(**base)


def test_accepts_png_data_url():
    prepared = prepare_image(png_data_url(Image.new("L", (50, 40), 255)), _settings())
    assert prepared.format == "png" and not prepared.resized


def test_accepts_raw_base64_without_prefix():
    raw = base64.b64encode(_png_bytes(30, 20)).decode()
    prepared = prepare_image(raw, _settings())
    assert prepared.format == "png"


def _png_bytes(w, h):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue()


def test_wrong_magic_rejected_415():
    data = b"NOT-AN-IMAGE-FORMAT-1234567890"
    b64 = base64.b64encode(data).decode()
    with pytest.raises(UnsupportedImageTypeError):
        prepare_image(b64, _settings())


def test_declared_prefix_mismatch_rejected_415():
    """data URL 声明 jpeg 但魔数是 PNG → 415（魔数嗅探，不仅信前缀）。"""
    from tests.conftest import png_data_url

    payload = png_data_url(Image.new("L", (20, 20), 255)).split(",", 1)[1]
    with pytest.raises(UnsupportedImageTypeError):
        prepare_image("data:image/jpeg;base64," + payload, _settings())


def test_invalid_base64_rejected_400():
    with pytest.raises(InvalidArgumentError):
        prepare_image("data:image/png;base64,@@@not-base64@@@", _settings())


def test_oversize_rejected_413():
    """编码长度预检 + 解码后限制，双保险。"""
    big = Image.new("L", (3000, 3000), 255)
    url = png_data_url(big)
    with pytest.raises(ImageTooLargeError):
        prepare_image(url, _settings(max_image_mb=0.001))


def test_oversize_encoded_preflight_413():
    # 15M 字符 Base64 ≈ 11.25MB 解码后 > 10MB 上限
    url = "data:image/png;base64," + "A" * 15_000_000
    with pytest.raises(ImageTooLargeError):
        prepare_image(url, _settings(max_image_mb=10.0))


def test_huge_dimension_downscaled_and_flagged():
    from PIL import Image

    img = Image.new("L", (9000, 100), 255)
    prepared = prepare_image(png_data_url(img), _settings())
    assert prepared.resized is True
    assert max(prepared.image.size) <= 8192
