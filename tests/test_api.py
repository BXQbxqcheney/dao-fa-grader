"""端到端 API 测试（Fake OCR/LLM 注入）：AC1~AC16 中的契约类用例。

真实引擎（RapidOCR/OFA）用例见 test_engines_integration.py。
"""

import json as _json

from tests.conftest import (
    CLEANUP_OK,
    FakeLLM,
    FakeOCR,
    grading_json,
    inked_image_data_url,
    sample_criteria,
    sample_image_data_url,
)


def _post(client, image=None, criteria=None, **options):
    body = {
        # 默认带墨迹：保证管线能走完 切分→OCR→LLM 全链路（内容由 Fake 决定）
        "image_base64": image or inked_image_data_url(),
        "criteria": criteria or sample_criteria(),
    }
    if options:
        body["options"] = options
    return client.post("/api/v1/grade", json=body)


def test_healthz(app_client_factory):
    client = app_client_factory()
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_workbench_served(app_client_factory):
    client = app_client_factory()
    r = client.get("/workbench/")
    assert r.status_code == 200
    assert "试判工作台" in r.text


def test_ac1_full_score(app_client_factory):
    """清晰作答完全命中：score = max_score，得分点引用原文。"""
    llm = FakeLLM([CLEANUP_OK, grading_json({"P1": 2, "P2": 4}, [])])
    client = app_client_factory(llm=llm)
    r = _post(client)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "graded"
    assert data["score"] == 6 == data["max_score"]
    assert all(p["evidence"] for p in data["scoring_points"] if p["hit"])
    assert data["request_id"].startswith("req_")
    assert data["model"]["provider"] == "fake-llm"
    # ocr 文本回传（默认 return_ocr_text=true）
    assert data["ocr"]["text"]


def test_ac2_partial_hit_with_deduction(app_client_factory):
    """部分命中：score = Σawarded − Σdeductions（用户裁决 2026-11-07：得分=踩到的得分点之和减规则扣分）。"""
    llm = FakeLLM([CLEANUP_OK], point_awarded={"P1": 2, "P2": 3},
                  deductions=[{"reason": "未结合材料中的具体人物事例", "points": 1}])
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["score"] == 4  # Σawarded 5 − Σded 1
    assert data["deductions"] == [{"reason": "未结合材料中的具体人物事例", "points": 1}]


def test_score_is_sum_of_awarded_minus_rule_deductions(app_client_factory):
    """计分语义（用户裁决）：得分=踩到的得分点得分之和，规则扣分另减；每点有 evidence 依据。"""
    from tests.conftest import sample_criteria5

    llm = FakeLLM([CLEANUP_OK], point_awarded={"P1": 0, "P2": 2, "P3": 2, "P4": 1, "P5": 0},
                  deductions=[{"reason": "未结合自身做法", "points": 1}])
    client = app_client_factory(llm=llm)
    data = _post(client, criteria=sample_criteria5()).json()
    assert data["status"] == "graded"
    assert data["score"] == 4  # Σawarded 5 − Σded 1
    for p in data["scoring_points"]:
        if p["hit"]:
            assert p["evidence"]  # 每分必有原文依据


def test_ac4_low_confidence_needs_human(app_client_factory):
    """OCR 置信度低于阈值：needs_human + 原图，且不调用 LLM。"""
    ocr = FakeOCR([("模糊的文字", 0.38)])
    llm = FakeLLM([])
    client = app_client_factory(ocr=ocr, llm=llm)
    image = inked_image_data_url()
    r = _post(client, image=image)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "needs_human"
    assert data["reason_code"] == "low_ocr_confidence"
    assert data["image_url"] == image
    assert "model" not in data
    assert llm.calls == []  # 未调用大模型


def test_ac5_garbled_needs_human(app_client_factory):
    ocr = FakeOCR([("@@@###@@@###@@@", 0.9)])
    client = app_client_factory(ocr=ocr, llm=FakeLLM([]))
    data = _post(client).json()
    assert data["status"] == "needs_human" and data["reason_code"] == "garbled_text"


def test_empty_ocr_text(app_client_factory):
    ocr = FakeOCR([("", 1.0)])
    client = app_client_factory(ocr=ocr, llm=FakeLLM([]))
    data = _post(client).json()
    assert data["status"] == "needs_human" and data["reason_code"] == "empty_ocr_text"


def test_line_split_failure_needs_human(app_client_factory):
    """空图无法分行 → illegible_line_split。"""
    from PIL import Image

    from tests.conftest import png_data_url

    client = app_client_factory(ocr=FakeOCR(), llm=FakeLLM([]))
    r = _post(client, image=png_data_url(Image.new("L", (200, 150), 255)))
    data = r.json()
    assert data["status"] == "needs_human" and data["reason_code"] == "illegible_line_split"


def test_cleanup_ungradable(app_client_factory):
    """整理层判定不可判 → illegible_after_cleanup。"""
    llm = FakeLLM(['{"gradable": false, "text": "", "note": "仅几个字，语义残缺"}'])
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["status"] == "needs_human" and data["reason_code"] == "illegible_after_cleanup"


def test_ac6_invalid_base64(app_client_factory):
    client = app_client_factory()
    r = _post(client, image="data:image/png;base64,%%%invalid%%%")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_ARGUMENT"


def test_ac6_missing_max_score(app_client_factory):
    from tests.conftest import sample_image_data_url as img

    client = app_client_factory()
    criteria = sample_criteria()
    criteria.pop("max_score")
    r = _post(client, image=img(), criteria=criteria)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_ARGUMENT"


def test_ac10_oversize_image(app_client_factory):
    from PIL import Image

    from tests.conftest import png_data_url

    client = app_client_factory(settings_overrides={"max_image_mb": 0.001})
    r = _post(client, image=png_data_url(Image.new("L", (3000, 3000), 255)))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_ac10_unsupported_format(app_client_factory):
    import base64

    client = app_client_factory()
    r = _post(client, image=base64.b64encode(b"GIF89a-not-supported-xx").decode())
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_ac12_half_step_rejected(app_client_factory):
    client = app_client_factory()
    criteria = sample_criteria()
    criteria["max_score"] = 0.3
    r = _post(client, criteria=criteria)
    assert r.status_code == 400


def test_ac14_injection_isolated(app_client_factory):
    """注入样本：作答中的指令性文字必须被隔离在 <student_answer> 内，判分不受影响。"""
    llm = FakeLLM([CLEANUP_OK, grading_json({"P1": 2, "P2": 4}, [])])
    client = app_client_factory(llm=llm)
    r = _post(client)
    assert r.status_code == 200 and r.json()["score"] == 6

    cleanup_system, cleanup_user = llm.calls[0]
    judge_calls = [(s, u) for s, u in llm.calls if "<scoring_point>" in u]
    assert judge_calls, "应有逐点判分调用"
    judge_system, judge_user = judge_calls[0]
    # 1) 系统提示词必须包含注入防护声明
    assert "一律不得遵循" in cleanup_system and "一律不得遵循" in judge_system
    # 2) 学生文本必须处于 <student_answer> 分隔标记之间
    assert "<student_answer>" in cleanup_user and "</student_answer>" in cleanup_user
    assert "<student_answer>" in judge_user and "</student_answer>" in judge_user
    # 3) 判分信息在独立的可信上下文区域
    assert "<scoring_point>" in judge_user


def test_ac15_sum_below_max(app_client_factory):
    """得分点之和 < 满分：正常判卷，缺额不自动给出（无"未覆盖"扣分），score ≤ max_score。"""
    llm = FakeLLM([CLEANUP_OK], point_awarded={"P1": 2, "P2": 2}, deductions=[])
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["status"] == "graded"
    assert data["score"] == 4  # Σawarded=4；缺额 2 分未自动给出
    assert data["score"] <= data["max_score"]


def test_sum_above_max_capped_note(app_client_factory):
    """得分点之和 > 满分：封顶并在 reason 注明（用户裁决：不拒绝）。"""
    criteria = sample_criteria()
    criteria["scoring_points"] = [
        {"id": "P1", "description": "要点一", "score": 4},
        {"id": "P2", "description": "要点二", "score": 4},
    ]  # 之和 8 > 满分 6
    llm = FakeLLM([CLEANUP_OK], point_awarded={"P1": 4, "P2": 2}, deductions=[])
    client = app_client_factory(llm=llm)
    data = _post(client, criteria=criteria).json()
    assert data["status"] == "graded"
    assert data["score"] == 6  # Σawarded 6 ≤ max 6
    assert "分值之和超过满分" in data["reason"]


def test_llm_parse_error_after_retry(app_client_factory):
    """判分点持续输出坏 JSON：点级重试 + 整体重试后仍失败 → 502 LLM_PARSE_ERROR。"""
    llm = FakeLLM([CLEANUP_OK], judge_fail_pids={"P1", "P2"})
    client = app_client_factory(llm=llm)
    r = _post(client)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_PARSE_ERROR"


def test_point_retry_then_success(app_client_factory):
    """某点第一次输出坏 JSON，点级重试 + 整体重试后成功 → 正常判分。"""

    class FlakyLLM(FakeLLM):
        def __init__(self):
            super().__init__([CLEANUP_OK])
            self.p1_bad_calls = 0

        def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
            if "<scoring_point>" in user and '"id": "P1"' in user and self.p1_bad_calls < 1:
                self.p1_bad_calls += 1
                return "不是 JSON"
            return super().chat(system, user, json_mode=json_mode)

    llm = FlakyLLM()
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["status"] == "graded" and data["score"] == 6
    assert llm.p1_bad_calls == 1  # P1 恰好坏过一次，重试成功


def test_mmqf_cleanup_marks_unrecognizable(app_client_factory):
    """AI 识别返回 MMQF（面目全非）→ needs_human: illegible_after_cleanup。"""
    llm = FakeLLM(["MMQF"])
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["status"] == "needs_human"
    assert data["reason_code"] == "illegible_after_cleanup"
    assert "面目全非" in data["message"]


def test_vlm_aiocr_flow():
    """VLM 直读：AIOCR 转写（不确定的字不说）→ AI 识别 → 并发逐点判分；置信度为 None。"""
    from fastapi.testclient import TestClient

    from grader.main import create_app
    from tests.conftest import FakeLLM, make_settings, sample_criteria5

    class FakeVLM:
        name = "fake-vlm"
        max_chat_attempts = 1

        def __init__(self) -> None:
            self.vlm_calls = 0
            self._judge = FakeLLM([CLEANUP_OK], full_score=True)  # 脚本首项给整理，判分走路由

        def chat_vlm(self, system: str, user: str, image_url: str) -> str:
            self.vlm_calls += 1
            assert "不确定的字" in system  # AIOCR 提示词要求宁缺勿猜
            assert "答题区" in system
            assert "uncertain" in system  # 必须自报认不出的字数
            return '{"text": "小宇因为别人的一句话陷入烦恼，说明他没有正确认识自己。", "uncertain": 0}'

        def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
            return self._judge.chat(system, user, json_mode=json_mode)

    settings = make_settings(agnes_vlm_mode=True)
    app = create_app(settings=settings, ocr_adapter=FakeOCR(), llm_adapter=FakeVLM())
    client = TestClient(app)
    data = _post(client, criteria=sample_criteria5()).json()
    assert data["status"] == "graded"
    assert data["score"] == 10  # 5 点各 2 分，FakeLLM 路由全给满分
    assert data["ocr"]["confidence"] is None  # VLM 模式无数值置信度
    assert "树立远大理想" in data["ocr"]["text"]  # ocr.text = 整理后文本（Fake 链路固定文本）


def test_aiocr_uncertain_over_threshold_needs_human():
    """AIOCR 自报认不出 >6 字 → needs_human: low_ocr_confidence（不猜测原则）。"""
    from fastapi.testclient import TestClient

    from grader.main import create_app
    from tests.conftest import make_settings, sample_criteria5

    class FakeVLM:
        name = "fake-vlm"
        max_chat_attempts = 1

        def chat_vlm(self, system: str, user: str, image_url: str) -> str:
            return '{"text": "我们", "uncertain": 7}'

        def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:  # pragma: no cover
            raise AssertionError("不应调用 LLM")

    settings = make_settings(agnes_vlm_mode=True)
    app = create_app(settings=settings, ocr_adapter=FakeOCR(), llm_adapter=FakeVLM())
    client = TestClient(app)
    data = _post(client, criteria=sample_criteria5()).json()
    assert data["status"] == "needs_human"
    assert data["reason_code"] == "low_ocr_confidence"
    assert "7 个字无法辨认" in data["message"]


def test_cleanup_garbled_defensive_needs_human(app_client_factory):
    """整理结果仍含乱码（防御性校验）→ needs_human，不把乱码交给判分。"""
    llm = FakeLLM(['{"gradable": true, "text": "@@@@######@@@@ 大概", "note": ""}'])
    client = app_client_factory(llm=llm)
    data = _post(client).json()
    assert data["status"] == "needs_human"
    assert data["reason_code"] == "illegible_after_cleanup"
    assert "乱码" in data["message"]


def test_stage_events_carry_input_output(app_client_factory):
    """每个阶段事件必须携带发给 AI 的完整输入与 AI 原始返回（不裁剪）。"""
    llm = FakeLLM([CLEANUP_OK, grading_json()])
    client = app_client_factory(llm=llm)
    with client.stream("POST", "/api/v1/grade/stream", json={
        "image_base64": inked_image_data_url(),
        "criteria": sample_criteria(),
    }) as resp:
        events = [_json.loads(l[6:]) for l in resp.iter_lines() if l.startswith("data: ")]
    stages = [e for e in events if e["type"] == "stage"]
    # running/done 是同名阶段的两个事件，按阶段名合并（UI 同逻辑）
    merged: dict[str, dict] = {}
    for s in stages:
        merged.setdefault(s["stage"], {}).update({k: v for k, v in s.items() if v is not None})
    cleanup = merged["AI 识别整理"]
    assert "student_answer" in cleanup.get("input", "")   # 隔离注入的完整输入
    assert "gradable" in cleanup.get("output", "")        # AI 原始返回
    judge = next(v for k, v in merged.items() if k.startswith("判卷·"))
    assert "<scoring_point>" in judge.get("input", "")
    assert "awarded_score" in judge.get("output", "")


def test_llm_timeout_maps_504(app_client_factory):
    llm = FakeLLM([], delay=0.5)
    client = app_client_factory(llm=llm, settings_overrides={"llm_timeout_seconds": 0.1})
    r = _post(client)
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "LLM_TIMEOUT"


def test_ocr_timeout_maps_500(app_client_factory):
    ocr = FakeOCR(delay=0.5)
    client = app_client_factory(ocr=ocr, llm=FakeLLM([]), settings_overrides={"ocr_timeout_seconds": 0.1})
    # 使用带墨迹的图（避免先触发 illegible_line_split）
    from tests.conftest import make_text_image, png_data_url

    r = _post(client, image=png_data_url(make_text_image(["一行文字"])))
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "OCR_ERROR"


def test_return_ocr_text_false(app_client_factory):
    llm = FakeLLM([CLEANUP_OK])
    client = app_client_factory(llm=llm)
    data = _post(client, return_ocr_text=False).json()
    assert data["ocr"]["text"] == ""


def test_grading_rules_injected_into_prompt(app_client_factory):
    """判分准则（新参数）：必须逐条注入判分提示词，作为尺度指令。"""
    llm = FakeLLM([CLEANUP_OK])
    client = app_client_factory(llm=llm)
    criteria = sample_criteria()
    criteria["grading_rules"] = ["与得分点接近得一半分", "与得分点完全一致得满分"]
    r = _post(client, criteria=criteria)
    assert r.status_code == 200
    judge_calls = [(s, u) for s, u in llm.calls if "<scoring_point>" in u]
    assert judge_calls
    judge_system, judge_user = judge_calls[0]
    assert "完全一致" in judge_system and "一半分" in judge_system  # 默认尺度已内置
    assert "与得分点接近得一半分" in judge_user and "<grading_rules>" in judge_user
    # 未传准则的请求不带该块
    llm2 = FakeLLM([CLEANUP_OK])
    client2 = app_client_factory(llm=llm2)
    _post(client2)
    judge_user2 = next(u for s, u in llm2.calls if "<scoring_point>" in u)
    assert "<grading_rules>" not in judge_user2


def test_grade_stream_sse(app_client_factory):
    """SSE 流式判卷：start → stage 进度事件 → result 终态，content-type 为 event-stream。"""
    llm = FakeLLM([CLEANUP_OK, grading_json()])
    client = app_client_factory(llm=llm)
    with client.stream("POST", "/api/v1/grade/stream", json={
        "image_base64": inked_image_data_url(),
        "criteria": sample_criteria(),
    }) as resp:
        assert resp.status_code == 200
        assert "event-stream" in resp.headers["content-type"]
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(_json.loads(line[6:]))
    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert "result" in types
    stages = [e for e in events if e["type"] == "stage"]
    stage_names = [s["stage"] for s in stages]
    assert any("OCR" in n for n in stage_names)
    assert any("判卷" in n for n in stage_names)
    for name in set(stage_names):
        assert "done" in [s["status"] for s in stages if s["stage"] == name]
    # 至少两个阶段经历过 running（转写/OCR 与判卷）
    assert sum(1 for s in stages if s["status"] == "running") >= 2
    # result 载荷与 /api/v1/grade 同构
    result = next(e for e in events if e["type"] == "result")
    assert result["http_status"] == 200
    assert result["data"]["status"] == "graded"
    assert result["data"]["score"] == 6


def test_grade_stream_needs_human(app_client_factory):
    llm = FakeLLM([])
    ocr = FakeOCR([("模糊的文字", 0.38)])
    client = app_client_factory(ocr=ocr, llm=llm)
    with client.stream("POST", "/api/v1/grade/stream", json={
        "image_base64": inked_image_data_url(),
        "criteria": sample_criteria(),
    }) as resp:
        events = [_json.loads(l[6:]) for l in resp.iter_lines() if l.startswith("data: ")]
    result = next(e for e in events if e["type"] == "result")
    assert result["http_status"] == 200
    assert result["data"]["status"] == "needs_human"
    assert result["data"]["reason_code"] == "low_ocr_confidence"


def test_noise_lines_dropped(app_client_factory):
    """页眉/题号类噪声行（极低置信度且无正文）不拖垮整题置信度，也不进正文。"""
    from tests.conftest import grading_json, inked_image_data_url

    ocr = FakeOCR([
        ("民之大天D仁", 0.209),       # 噪声：conf<0.3 且汉字≤6（页眉残迹形态）
        ("3", 0.131),                 # 噪声：孤立题号
        ("我们要树立远大理想，努力学习。", 0.92),
        ("积极参加志愿服务活动。", 0.88),
    ])
    llm = FakeLLM([CLEANUP_OK, grading_json()])
    client = app_client_factory(ocr=ocr, llm=llm)
    data = _post(client, image=inked_image_data_url()).json()
    assert data["status"] == "graded"                       # 未被拖到 needs_human
    assert data["ocr"]["confidence"] >= 0.5
    assert data["ocr"]["removed_marks"] == 2                # 噪声行计入诊断计数
    assert "民之大天" not in data["ocr"]["text"]            # 噪声不进正文


def test_real_text_line_not_dropped(app_client_factory):
    """低置信度但有实质正文的行不能被当噪声剔除（宁严勿松，交给置信度分流）。"""
    from tests.conftest import grading_json, inked_image_data_url

    ocr = FakeOCR([
        ("这是一行潦草但内容完整的作答文字，包含很多汉字", 0.35),  # conf<0.3? 否——用 0.29
    ])
    llm = FakeLLM([])
    client = app_client_factory(ocr=ocr, llm=llm, settings_overrides={"ocr_confidence_threshold": 0.5})
    data = _post(client, image=inked_image_data_url()).json()
    # conf 0.35 > 0.3 不算噪声；0.35 < 0.5 → low_ocr_confidence（而非噪声剔除）
    assert data["status"] == "needs_human"
    assert data["reason_code"] == "low_ocr_confidence"
