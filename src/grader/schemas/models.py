"""请求/响应 Pydantic 模型（API 契约唯一来源）。

依据 SPEC v0.2：
- question_stem / reference_answer / deduction_rules 为必填（用户裁决）；
- 所有分值必须为 0.5 的整数倍，违反返回 400 INVALID_ARGUMENT；
- 扣分语义：score = max_score − Σdeductions（服务端计算）。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# 浮点容差：0.5 步长与一致性校验共用
EPS = 1e-6


def is_half_step(value: float) -> bool:
    """判断是否为 0.5 的整数倍（0.5 / 1 / 1.5 / 2 …）。"""
    twice = value * 2
    return abs(twice - round(twice)) < EPS


class ScoringPointInput(BaseModel):
    id: str = Field(min_length=1, description="得分点编号，如 P1")
    description: str = Field(min_length=1, description="得分点描述")
    score: float = Field(description="该点分值，必须 > 0 且为 0.5 的整数倍")

    @field_validator("score")
    @classmethod
    def _check_score(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("得分点分值必须大于 0")
        if not is_half_step(v):
            raise ValueError(f"得分点分值必须为 0.5 的整数倍，收到 {v}")
        return v


class CriteriaInput(BaseModel):
    question_id: Optional[str] = None
    max_score: float = Field(description="本题满分，必须 > 0 且为 0.5 的整数倍")
    question_stem: str = Field(description="题干（必填，供判分参考）")
    reference_answer: str = Field(description="参考答案（必填）")
    scoring_points: list[ScoringPointInput] = Field(min_length=1, description="得分点列表，非空")
    deduction_rules: list[str] = Field(description="扣分规则（用户裁决必填；可为空列表，但字段必须出现）")
    grading_rules: list[str] = Field(
        default_factory=list,
        description="判分准则（可选，每条一个尺度指令，如『与得分点接近得一半分』），优先级高于默认尺度",
    )

    @field_validator("max_score")
    @classmethod
    def _check_max_score(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("criteria.max_score 必须大于 0")
        if not is_half_step(v):
            raise ValueError(f"criteria.max_score 必须为 0.5 的整数倍，收到 {v}")
        return v

    # question_stem / reference_answer / deduction_rules 的必填语义由字段定义
    # （无默认值即必填）直接保证；内容允许为空串，有效性由判分层判定。

    @field_validator("deduction_rules")
    @classmethod
    def _deduction_rules_list(cls, v: list[str]) -> list[str]:
        if any(not isinstance(item, str) for item in v):
            raise ValueError("criteria.deduction_rules 必须为字符串数组")
        return v

    @model_validator(mode="after")
    def _unique_point_ids(self) -> "CriteriaInput":
        """得分点 id 必须唯一：重复 id 会导致同一模型输出被复制到多个点，形成重复计分。"""
        ids = [p.id for p in self.scoring_points]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"scoring_points 的 id 必须唯一，重复: {dupes}")
        return self


class GradeOptions(BaseModel):
    return_ocr_text: bool = True


class GradeRequest(BaseModel):
    image_base64: str = Field(min_length=1, description="原始 Base64 或 data URL")
    criteria: CriteriaInput
    options: GradeOptions = Field(default_factory=GradeOptions)


# ---------- 响应模型 ----------


class GradedPoint(BaseModel):
    id: str
    criterion: str
    evidence: str = ""
    hit: bool
    awarded_score: float
    max_score: float


class Deduction(BaseModel):
    reason: str
    points: float


class OcrInfo(BaseModel):
    text: str = ""
    confidence: Optional[float] = None
    removed_marks: int = 0
    resized: bool = False


class ModelInfo(BaseModel):
    provider: str
    name: str


class GradedResponse(BaseModel):
    request_id: str
    status: Literal["graded"] = "graded"
    question_id: Optional[str] = None
    max_score: float
    score: float
    scoring_points: list[GradedPoint]
    deductions: list[Deduction]
    reason: str
    ocr: OcrInfo
    model: ModelInfo


class NeedsHumanResponse(BaseModel):
    request_id: str
    status: Literal["needs_human"] = "needs_human"
    reason_code: str
    message: str
    image_url: str
    ocr: OcrInfo


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
