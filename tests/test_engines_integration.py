"""真实引擎集成测试：RapidOCR（AC13）+ OFA（需环境变量 RUN_OFA_TESTS=1）。

RapidOCR / OFA 均为懒加载真实引擎；本文件用例需要字体可用，
不满足条件时自动跳过，保证单测套件在任何机器可跑。
"""

import pytest

from tests.conftest import (
    CLEANUP_OK,
    FakeLLM,
    grading_json,
    load_cjk_font,
    make_text_image,
    png_data_url,
    sample_criteria,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _full_image():
    img = make_text_image([
        "我们要树立远大理想，把个人梦想和国家发展结合起来。",
        "在学校认真学习，积极参加志愿服务活动。",
        "努力学习科学文化知识，锻炼强健体魄。",
    ])
    return png_data_url(img)


@pytest.fixture()
def rapidocr_adapter():
    pytest.importorskip("rapidocr_onnxruntime")
    from grader.ocr.rapidocr import RapidOCRAdapter

    return RapidOCRAdapter()


def test_ac13_rapidocr_full_flow(app_client_factory, rapidocr_adapter):
    """AC13：OCR_PROVIDER=rapidocr 全流程可完成判分（真实行切分 + 真实识别）。"""
    if load_cjk_font() is None:
        pytest.skip("本机无中文字体，无法构造识别样本")
    llm = FakeLLM([CLEANUP_OK, grading_json({"P1": 2, "P2": 4}, [])])
    client = app_client_factory(ocr=rapidocr_adapter, llm=llm)
    r = client.post("/api/v1/grade", json={
        "image_base64": _full_image(),
        "criteria": sample_criteria(),
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "graded"
    assert data["ocr"]["confidence"] is not None and data["ocr"]["confidence"] > 0
    # 印刷体识别置信度应远高于阈值
    assert data["ocr"]["confidence"] >= 0.5
    assert data["score"] == 6


def test_rapidocr_linesplit_integrated(rapidocr_adapter):
    """行切分 + RapidOCR 逐行识别：行数与顺序保持。"""
    if load_cjk_font() is None:
        pytest.skip("本机无中文字体，无法构造识别样本")
    from grader.ocr.linesplit import split_lines
    from grader.ocr.marks import detect_and_mask

    img = make_text_image(["第一行测试文字内容", "第二行测试文字内容", "第三行测试文字内容"])
    lines = split_lines(img)
    assert len(lines) == 3
    masked = [detect_and_mask(x).image for x in lines]
    results = rapidocr_adapter.recognize_lines(masked)
    assert len(results) == 3
    for line in results:
        assert "测试文字内容" in line.text
        assert 0 < line.confidence <= 1


# ---------- OFA 实测（M2a：按环境变量显式触发，避免常规跑测加载 700MB 权重） ----------


@pytest.mark.skipif(
    __import__("os").getenv("RUN_OFA_TESTS", "") != "1",
    reason="OFA 实测需设置 RUN_OFA_TESTS=1（加载 700MB 权重，耗时长）",
)
def test_ofa_confidence_pipeline(app_client_factory):
    """M2a 实测：OFA 权重加载 + generate 置信度改造 + 全流程判分。"""
    if load_cjk_font() is None:
        pytest.skip("本机无中文字体")
    from grader.core.config import Settings
    from grader.ocr.ofa import OFAAdapter

    adapter = OFAAdapter(Settings().ofa_model_path)
    llm = FakeLLM([CLEANUP_OK, grading_json({"P1": 2, "P2": 4}, [])])
    client = app_client_factory(ocr=adapter, llm=llm)
    r = client.post("/api/v1/grade", json={
        "image_base64": _full_image(),
        "criteria": sample_criteria(),
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] in {"graded", "needs_human"}  # 置信度分流由实测结果决定
    if data["status"] == "graded":
        assert data["ocr"]["confidence"] is not None
