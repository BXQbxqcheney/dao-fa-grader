"""RapidOCR 适配器（onnxruntime 轻量引擎，检测关闭、仅对已切分行做识别）。

置信度口径（SPEC §4.0）：引擎自带行 score，截取到 0~1；
与 OFA 非同一标定分布，但阈值同语义使用（低于阈值即转人工）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from grader.core.errors import OCRError
from grader.ocr.base import OCRLine

logger = logging.getLogger("grader.ocr.rapidocr")


class RapidOCRAdapter:
    """懒加载适配器；对行图 use_det=False 仅做识别，保证与管线行切分口径一致。"""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._ready = False

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except Exception as exc:
            raise OCRError(f"RapidOCR 引擎加载失败: {exc}") from exc
        self._ready = True
        logger.info("RapidOCR 引擎加载成功")

    def recognize_lines(self, line_images: list[Image.Image]) -> list[OCRLine]:
        self._ensure_loaded()
        return [self._recognize_one(img) for img in line_images]

    def _recognize_one(self, image: Image.Image) -> OCRLine:
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()  # BGR
        try:
            result, _ = self._engine(arr, use_det=False, use_cls=False, use_rec=True)
        except Exception as exc:
            raise OCRError(f"RapidOCR 识别失败: {exc}") from exc
        text, score = self._parse_result(result)
        return OCRLine(text=text.strip(), confidence=max(0.0, min(1.0, score)))

    @staticmethod
    def _parse_result(result: Any) -> tuple[str, float]:
        """兼容不同版本的 rec-only 返回结构：[(text, score)] 或 [[text, score]]。"""
        if not result:
            return "", 1.0
        first = result[0]
        if isinstance(first, (list, tuple)):
            if len(first) >= 2:
                text, score = first[0], first[1]
            else:
                return "", 1.0
        else:
            return "", 1.0
        try:
            return str(text), float(score)
        except (TypeError, ValueError):
            return str(text), 0.0
