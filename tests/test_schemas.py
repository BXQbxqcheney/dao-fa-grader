"""API 契约与校验测试：0.5 步长、必填项（含用户裁决的三个必填字段）。"""

import pytest
from pydantic import ValidationError

from tests.conftest import sample_criteria, sample_image_data_url


def _request(**criteria_overrides):
    payload = {
        "image_base64": sample_image_data_url(),
        "criteria": sample_criteria(),
    }
    payload["criteria"].update(criteria_overrides)
    return payload


def test_valid_request_passes():
    from grader.schemas.models import GradeRequest

    req = GradeRequest.model_validate(_request())
    assert req.criteria.max_score == 6
    assert req.criteria.question_stem  # 必填（用户裁决）
    assert req.criteria.reference_answer  # 必填（用户裁决）
    assert req.criteria.deduction_rules  # 必填（用户裁决）


def test_missing_max_score_rejected():
    from grader.schemas.models import GradeRequest

    payload = _request()
    payload["criteria"].pop("max_score")
    with pytest.raises(ValidationError):
        GradeRequest.model_validate(payload)


def test_missing_user_required_fields_rejected():
    """用户裁决：question_stem / reference_answer / deduction_rules 为必填。"""
    from grader.schemas.models import GradeRequest

    for field in ("question_stem", "reference_answer", "deduction_rules"):
        payload = _request()
        payload["criteria"].pop(field)
        with pytest.raises(ValidationError):
            GradeRequest.model_validate(payload)


@pytest.mark.parametrize("bad", [0.3, 1.3, 2.25, -1])
def test_half_step_violation_rejected(bad):
    """AC12：任一分值非 0.5 步长 → 400（Schema 层拒绝）。"""
    from grader.schemas.models import GradeRequest

    payload = _request()
    payload["criteria"]["max_score"] = bad
    with pytest.raises(ValidationError):
        GradeRequest.model_validate(payload)

    payload = _request()
    payload["criteria"]["scoring_points"][0]["score"] = 0.3
    with pytest.raises(ValidationError):
        GradeRequest.model_validate(payload)


def test_scoring_points_empty_rejected():
    from grader.schemas.models import GradeRequest

    payload = _request()
    payload["criteria"]["scoring_points"] = []
    with pytest.raises(ValidationError):
        GradeRequest.model_validate(payload)


def test_deduction_rules_must_be_strings():
    from grader.schemas.models import GradeRequest

    payload = _request()
    payload["criteria"]["deduction_rules"] = [123]
    with pytest.raises(ValidationError):
        GradeRequest.model_validate(payload)
