"""OCR 适配器协议：所有引擎（OFA / RapidOCR / 测试 Fake）实现同一接口。

置信度统一口径（SPEC §4.0）：
- 每行置信度 ∈ (0, 1]；
- OFA = 该行生成序列 token 概率几何平均 exp(平均对数概率)；
- RapidOCR = 引擎自带行 score（非同一标定分布，但阈值同语义使用）；
- 整题置信度 = 各行最小值（宁严勿松），空行不参与最小值计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from PIL import Image


@dataclass(frozen=True)
class OCRLine:
    """单行识别结果。text 为空表示该行无可识别内容。"""

    text: str
    confidence: float


@dataclass
class OCRResult:
    lines: list[OCRLine] = field(default_factory=list)
    removed_marks: int = 0
    resized: bool = False

    @property
    def text(self) -> str:
        """按阅读顺序拼接，保留换行；空行在拼接中体现为空行。"""
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float | None:
        """整题置信度：各行最小值；无有效行时返回 None。"""
        values = [line.confidence for line in self.lines if line.text.strip()]
        return min(values) if values else None


class OCRAdapter(Protocol):
    """逐行识别协议：输入已切分并完成涂改剔除的行图，输出逐行结果。"""

    name: str

    def recognize_lines(self, line_images: Sequence[Image.Image]) -> list[OCRLine]:
        """按顺序识别每张行图；实现必须自行处理超时与异常（抛 OCRError/OCRTimeoutError）。"""
        ...
