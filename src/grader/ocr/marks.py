"""涂改 / 圈划检测与剔除（SPEC §4.3）。

启发式规则，仅在标记明确时剔除：
1. 贯穿删除线：行内存在横向墨迹长 ≥ 55% 行有效宽度、且位于行高中部的连续水平笔画；
2. 高密度涂抹块：列段墨迹密度 ≥ 0.6（绝对）且 ≥ 2 倍文本列基线、宽度 ≥ 12% 行宽的浓墨区。

被判定为「明确标记」的列段在行图上涂白后再送 OCR，不参与识别；
removed_marks 按剔除区域估算的字符数累计。无法确认的标记一律保留，由置信度分流兜底。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from grader.ocr.linesplit import _ink_mask, _to_gray

STRIKE_RATIO = 0.55      # 删除线需覆盖行有效宽度的比例
DENSITY_RATIO = 2.0      # 涂抹块密度阈值（相对文本列基线；另有绝对密度 0.6 下限）
BLOB_MIN_RATIO = 0.12    # 涂抹块最小宽度（相对行宽）


@dataclass(frozen=True)
class MaskResult:
    image: Image.Image      # 剔除后的行图
    removed_est_chars: int  # 估算剔除字符数
    marked_regions: int     # 明确标记的区域数


def _long_horizontal_runs(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """找出足够长的水平墨迹段，返回 (row, col_start, col_end, run_len)。"""
    runs: list[tuple[int, int, int, int]] = []
    h, w = mask.shape
    for r in range(h):
        row = mask[r]
        # 找连续 True 段
        d = np.diff(np.concatenate(([False], row, [False])).astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 0.5 * w:  # 先过滤超半宽的段，减少候选量
                runs.append((r, int(s), int(e), int(e - s)))
    return runs


def detect_and_mask(line_image: Image.Image) -> MaskResult:
    """检测行内明确涂改标记并涂白；无标记时原样返回（removed=0）。"""
    gray = _to_gray(line_image)
    mask = _ink_mask(gray)
    h, w = mask.shape
    if h == 0 or w == 0:
        return MaskResult(line_image, 0, 0)

    cols_with_ink = np.where(mask.any(axis=0))[0]
    if cols_with_ink.size == 0:
        return MaskResult(line_image, 0, 0)
    x_start, x_end = int(cols_with_ink[0]), int(cols_with_ink[-1])
    ink_width = x_end - x_start + 1

    marked_cols = np.zeros(w, dtype=bool)
    regions = 0

    # 规则 1：贯穿删除线 —— 位于行高中部（20%~80%）且长度 ≥ 55% 有效宽度
    for r, s, e, run_len in _long_horizontal_runs(mask):
        if r < 0.2 * h or r > 0.8 * h:
            continue
        if run_len >= STRIKE_RATIO * ink_width:
            # 删除线覆盖的列段：向两侧扩 2px
            marked_cols[max(0, s - 2) : min(w, e + 2)] = True
            regions += 1

    # 规则 2：高密度涂抹块 —— 基线取「文本列」（密度 0.03~0.6）的中位数，
    # 涂抹块需同时满足绝对密度 ≥ 0.6 与 ≥ 2 倍基线，避免把笔画密集的正常字误判为涂抹。
    col_density = mask.mean(axis=0)
    window = col_density[x_start : x_end + 1]
    text_like = window[(window > 0.03) & (window < 0.6)]
    if text_like.size:
        baseline = float(np.median(text_like))
        # 绝对密度 ≥ 0.6 且 ≥ DENSITY_RATIO 倍基线，双条件防误判
        hot = (col_density >= 0.6) & (col_density >= DENSITY_RATIO * baseline)
        # 连续 hot 段
        d = np.diff(np.concatenate(([False], hot, [False])).astype(np.int8))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= BLOB_MIN_RATIO * w:
                # 排除已被删除线覆盖的区域重复计数
                if marked_cols[s:e].any():
                    continue
                marked_cols[s:e] = True
                regions += 1

    if not marked_cols.any():
        return MaskResult(line_image, 0, 0)

    # 估算字符宽度：手写体经验值——有效墨迹宽度约容纳 24 个字符位，下限 6px 防极端行
    est_char_w = max(6.0, ink_width / 24.0)

    # 统计被标记列段宽度 → 估算字符数
    removed_est = 0
    d = np.diff(np.concatenate(([False], marked_cols, [False])).astype(np.int8))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        removed_est += max(1, round((e - s) / est_char_w))

    masked = line_image.copy()
    draw = ImageDraw.Draw(masked)
    d2 = np.diff(np.concatenate(([False], marked_cols, [False])).astype(np.int8))
    for s, e in zip(np.where(d2 == 1)[0], np.where(d2 == -1)[0]):
        draw.rectangle([max(0, s - 1), 0, min(w - 1, e), h], fill=255)

    return MaskResult(masked, removed_est, regions)
