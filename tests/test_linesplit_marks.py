"""行切分与涂改剔除测试（合成图像，确定性验收 AC3 前提）。"""

import pytest
from PIL import Image, ImageDraw

from grader.ocr.linesplit import LineSplitFailure, split_lines
from grader.ocr.marks import detect_and_mask
from tests.conftest import make_text_image


def test_split_three_lines():
    img = make_text_image(["第一行：树立远大理想", "第二行：参加志愿服务", "第三行：努力学习科学文化"])
    lines = split_lines(img)
    assert len(lines) == 3, f"期望 3 行，实际 {len(lines)}"


def test_split_single_line():
    img = make_text_image(["只有一行文字的内容"])
    lines = split_lines(img)
    assert len(lines) == 1


def test_blank_image_fails():
    blank = Image.new("L", (300, 200), 255)
    with pytest.raises(LineSplitFailure):
        split_lines(blank)


def test_marks_strike_through_masked():
    """贯穿删除线：行中部一条横贯删除线 → 明确标记，剔除并计数。"""
    img = make_text_image(["这段话被划掉了必须剔除"], font_size=30)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # 行高中部的横向粗线（覆盖行宽 60% 以上）
    draw.rectangle([40, int(h * 0.45), int(w * 0.75), int(h * 0.55)], fill=0)
    result = detect_and_mask(img)
    assert result.removed_est_chars >= 1
    assert result.marked_regions >= 1
    # 涂白后图像必须与原图不同
    import numpy as np

    diff = (np.asarray(img) != np.asarray(result.image)).any()
    assert diff, "标记区域未被涂白"


def test_marks_clean_line_untouched():
    img = make_text_image(["正常书写没有任何涂改标记的一行"])
    result = detect_and_mask(img)
    assert result.removed_est_chars == 0
    assert result.marked_regions == 0


def test_marks_scribble_blob_masked():
    """高密度涂抹块 → 明确标记剔除。"""
    img = make_text_image(["前面正常，后面全部涂抹废掉不可见"], font_size=30)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # 右侧 1/3 浓墨涂抹块
    draw.rectangle([int(w * 0.62), 6, int(w * 0.9), h - 6], fill=0)
    result = detect_and_mask(img)
    assert result.marked_regions >= 1
