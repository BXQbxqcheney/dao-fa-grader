"""行切分（SPEC §4.1）：灰度化 + 自适应二值化 + 水平投影 + 连通域合并。

仅依赖 numpy / Pillow，不引入 opencv 硬依赖。
切分失败（无法分出任何行）抛 LineSplitFailure，由管线转 needs_human: illegible_line_split。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


class LineSplitFailure(Exception):
    """无法可靠分行。"""


def _to_gray(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    return np.asarray(gray, dtype=np.float32)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """自适应二值化：与局部均值比较，亮底深字为墨迹。

    局部均值用盒滤波近似（累积和实现，避免引入 scipy/opencv）。
    """
    h, w = gray.shape
    # 局部窗口约为行高尺度的 1/8，最小 15px
    k = max(15, min(h, w) // 8) | 1  # 奇数化
    padded = np.pad(gray, k // 2, mode="edge")
    # 累积和做积分图，计算窗口均值
    integral = padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    win = k
    y0 = np.clip(np.arange(h) - win // 2, 0, padded.shape[0] - 1)
    y1 = np.clip(np.arange(h) + win // 2 + 1, 1, padded.shape[0])
    x0 = np.clip(np.arange(w) - win // 2, 0, padded.shape[1] - 1)
    x1 = np.clip(np.arange(w) + win // 2 + 1, 1, padded.shape[1])
    local_mean = (
        integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)] - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)]
    ) / (win * win)
    # 墨迹显著深于局部背景（留 10 个灰度余量）；
    # 另加绝对暗度判据：实心涂抹块的内部「局部均值≈墨色」，自适应判据会漏检其内部，用 <60 兜底
    adaptive = (gray < local_mean - 10.0) & (gray < 200.0)
    return adaptive | (gray < 60.0)


def _trim_blank(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """裁掉四周全空白边缘，返回 (裁剪后数组, 行偏移, 列偏移)；全空返回空数组。"""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask[:0, :0], 0, 0
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return mask[r0:r1, c0:c1], r0, c0


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _split_deep_valleys(proj: np.ndarray, s: int, e: int, min_line: int, depth: int = 0) -> list[tuple[int, int]]:
    """递归谷点切分：band 内部存在足够深的谷即切（处理紧排/粘连多行）。

    - 平滑窗口随最小行高缩放：字内小空隙被抹平，行间谷保持深度；
    - 谷深显著性：谷行墨量 ≤ 峰值 40% 才切；
    - 两半都必须 ≥ min_line（防止把单行从中腰切开）。
    """
    height = e - s
    if height < 2 * min_line or depth > 8:
        return [(s, e)]
    window = max(3, min_line // 3) | 1
    seg = _smooth(proj[s:e], window=window)
    peak = float(seg.max())
    if peak <= 0:
        return [(s, e)]
    margin = max(1, height // 6)
    lo, hi = margin, height - margin
    if hi <= lo:
        return [(s, e)]
    v = lo + int(np.argmin(seg[lo:hi]))
    if float(seg[v]) <= 0.40 * peak:
        left = _split_deep_valleys(proj, s, s + v, min_line, depth + 1)
        right = _split_deep_valleys(proj, s + v, e, min_line, depth + 1)
        return left + right
    return [(s, e)]


def split_lines(image: Image.Image, min_band_px: int = 8) -> list[Image.Image]:
    """把整题作答图切为按阅读顺序的行图列表。

    - 全空白 / 无有效墨迹 → LineSplitFailure；
    - 行距过密导致粘连的 band 会尝试谷点再切。
    """
    gray = _to_gray(image)
    mask, row_off, col_off = _trim_blank(_ink_mask(gray))
    if mask.size == 0:
        raise LineSplitFailure("图片中没有检测到任何墨迹")

    row_ink = mask.sum(axis=1).astype(np.float64)
    active = row_ink > max(1.0, 0.02 * row_ink.max())

    # 连续 active 行 → 候选 band
    bands: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(active)))

    # 过滤噪声 band（高度过小且墨量极少）
    bands = [(s, e) for s, e in bands if e - s >= min_band_px]
    if not bands:
        raise LineSplitFailure("无法从图片中分出任何文本行")

    # 递归深谷切分：紧排/粘连的多行 band 在谷点处逐层切开（谷深 0.4 峰值 + 两半 ≥ 最小行高）
    min_line = max(12, mask.shape[0] // 15)
    final: list[tuple[int, int]] = []
    for s, e in bands:
        final.extend(_split_deep_valleys(row_ink, s, e, min_line))

    # 细条带过滤：墨迹总量远小于中位 band 的（污渍/噪点）不作为作答行，
    # 避免其低置信度拖垮整题最小值（非作答行，不违反"所有作答行都参与识别"）
    if len(final) >= 2:
        band_ink = [float(row_ink[s:e].sum()) for s, e in final]
        median_ink = float(np.median([v for v in band_ink if v > 0])) if any(v > 0 for v in band_ink) else 0.0
        final = [band for band, ink in zip(final, band_ink) if ink >= 0.15 * median_ink]
    if not final:
        raise LineSplitFailure("切分行均为噪声条带")

    # 裁切行图：band 坐标位于裁剪后坐标系，加回偏移得到原图坐标；上下左右各留 4px 余量
    pad = 4
    lines: list[Image.Image] = []
    for s, e in final:
        top = row_off + max(0, s - pad)
        bottom = row_off + min(mask.shape[0], e + pad)
        left = max(0, col_off - pad)
        right = min(image.width, col_off + mask.shape[1] + pad)
        lines.append(image.crop((left, top, right, bottom)))
    return lines
