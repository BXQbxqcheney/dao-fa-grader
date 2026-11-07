"""测试夹具与 Fake 实现：OCR / LLM 均可依赖注入，单测不依赖外部服务（SPEC §8）。

FakeLLM 直接注入 pipeline，OPENAI_API_KEY 留空即可（不会创建真实适配器），
测试夹具因此不携带任何凭据形态字符串。
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from grader.core.config import Settings
from grader.ocr.base import OCRLine

# ---------- Fake OCR / LLM ----------


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, lines: list[tuple[str, float]] | None = None, *, delay: float = 0.0,
                 fail: Exception | None = None) -> None:
        self.lines = lines if lines is not None else [("我们要树立远大理想", 0.95)]
        self.delay = delay
        self.fail = fail
        self.calls: list[int] = []

    def recognize_lines(self, line_images):
        import time

        self.calls.append(len(line_images))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return [OCRLine(text=t, confidence=c) for t, c in self.lines]


class FakeLLM:
    """按脚本顺序返回响应；记录每次 (system, user) 调用供断言。

    支持并发逐点判分路由：user 含 <scoring_point> → 按配置返回该点 JSON；
    含 <deduction_rules> → 返回配置的扣分数组。这两类调用不消耗脚本。
    """

    name = "fake-llm"

    def __init__(self, responses: list[str], *, delay: float = 0.0,
                 point_awarded: dict[str, float] | None = None,
                 deductions: list[dict] | None = None,
                 full_score: bool = True,
                 judge_fail_pids: set[str] | None = None) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.delay = delay
        self.point_awarded = point_awarded   # pid → awarded（未列出的点给满分/一半）
        self.deductions = deductions or []
        self.full_score = full_score         # True: 未配置的点给满分；False: 给一半
        self.judge_fail_pids = judge_fail_pids or set()  # 这些点始终返回坏 JSON

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        import json as _j
        import re as _re
        import time

        self.calls.append((system, user))
        if self.delay:
            time.sleep(self.delay)
        if "<scoring_point>" in user:
            m = _re.search(r'"id": "(P\d+)"', user)
            pid = m.group(1) if m else "P1"
            if pid in self.judge_fail_pids:
                return "不是 JSON"
            score_m = _re.search(r'"score": ([0-9.]+)', user)
            full = float(score_m.group(1)) if score_m else 3.0
            if self.point_awarded and pid in self.point_awarded:
                awarded = self.point_awarded[pid]
            else:
                awarded = full if self.full_score else full / 2
            hit = awarded > 0
            return _j.dumps({"hit": hit, "awarded_score": awarded,
                             "evidence": ("“学生原文”" if hit else "")}, ensure_ascii=False)
        if "<deduction_rules>" in user:
            return _j.dumps({"deductions": self.deductions}, ensure_ascii=False)
        if not self._responses:
            raise AssertionError("FakeLLM 脚本已耗尽")
        return self._responses.pop(0)


# ---------- 公共构造器 ----------

CJK_FONTS = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/Deng.ttf",
]


def load_cjk_font(size: int = 28):
    for path in CJK_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def make_text_image(lines: list[str], *, width: int = 640, line_gap: int = 26,
                    font_size: int = 28) -> Image.Image:
    """渲染多行中文文本图（白底黑字），供行切分 / RapidOCR 集成测试。"""
    font = load_cjk_font(font_size)
    height = line_gap * (len(lines) + 1) + font_size * len(lines)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    y = line_gap
    for text in lines:
        draw.text((20, y), text, fill=0, font=font)
        y += font_size + line_gap
    return img


def png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_image_data_url() -> str:
    """最小合法 PNG（契约测试不需要真实文本，OCR 由 Fake 承担）。"""
    return png_data_url(Image.new("L", (60, 30), 255))


def inked_image_data_url() -> str:
    """带一行真实墨迹的图：让管线能完成行切分，进入 OCR/LLM 阶段（OCR 仍由 Fake 承担）。"""
    return png_data_url(make_text_image(["我们要树立远大理想，努力学习。"]))


def sample_criteria() -> dict[str, Any]:
    return {
        "question_id": "Q18",
        "max_score": 6,
        "question_stem": "结合材料，谈谈青少年应如何将个人梦想与国家发展结合起来。",
        "reference_answer": "①树立远大理想；②结合材料说明具体行动。",
        "scoring_points": [
            {"id": "P1", "description": "指出树立远大理想的重要性", "score": 2},
            {"id": "P2", "description": "结合材料说明具体行动", "score": 4},
        ],
        "deduction_rules": ["未结合材料扣 1 分", "照抄题目不给分"],
    }


def sample_criteria5() -> dict[str, Any]:
    """五点各 2 分（和=10=满分），对应 ti2 类题型。"""
    base = sample_criteria()
    base.update({
        "max_score": 10,
        "scoring_points": [
            {"id": f"P{i}", "description": f"得分点{i}", "score": 2} for i in range(1, 6)
        ],
    })
    return base


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        llm_provider="openai",
        openai_api_key="",  # FakeLLM 注入，不需要真实密钥；杜绝测试夹具携带凭据形态字符串
        openai_model="fake-model",
        anthropic_api_key="",
        anthropic_model="fake-claude",
        ocr_provider="rapidocr",
        ocr_confidence_threshold=0.5,
        ocr_timeout_seconds=10.0,
        ocr_concurrency=2,
        max_image_mb=10.0,
        request_timeout_seconds=60.0,
        llm_timeout_seconds=20.0,
        debug_logging=False,
        ofa_model_path="",
    )
    base.update(overrides)
    return Settings(**base)


CLEANUP_OK = ('{"gradable": true, "text": "我们要树立远大理想，把个人梦想和国家发展结合起来。'
              '在学校认真学习，参加志愿服务。", "note": ""}')


def grading_json(
    awarded: dict[str, float] | None = None,
    deductions: list[dict] | None = None,
    *,
    evidence: str = "“我们要树立远大理想”",
) -> str:
    awarded = awarded or {"P1": 2, "P2": 4}
    deductions = deductions if deductions is not None else []
    points = []
    for pid, score in awarded.items():
        hit = score > 0
        points.append(
            {"id": pid, "hit": hit, "awarded_score": score, "evidence": evidence if hit else ""}
        )
    import json as _json

    return _json.dumps(
        {
            "scoring_points": points,
            "deductions": deductions,
            "reason_summary": "学生答出理想与行动，扣分项见 deductions。",
        },
        ensure_ascii=False,
    )


@pytest.fixture()
def app_client_factory():
    """工厂夹具：注入 Fake OCR/LLM 构建 TestClient。"""
    from fastapi.testclient import TestClient

    from grader.main import create_app

    def factory(ocr: FakeOCR | None = None, llm: FakeLLM | None = None,
                settings_overrides: dict | None = None) -> "TestClient":
        settings = make_settings(**(settings_overrides or {}))
        app = create_app(
            settings=settings,
            ocr_adapter=ocr or FakeOCR(),
            llm_adapter=llm or FakeLLM([CLEANUP_OK, grading_json()]),
        )
        return TestClient(app)

    return factory
