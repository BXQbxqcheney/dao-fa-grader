"""模型输出解析与校验：JSON 多级容错链 + Schema 校验 + 0.5 步长/一致性校验。

容错链（代码生成铁则 7）：
  1. 直接 json.loads
  2. 剥 Markdown 代码围栏
  3. 大括号配平截取
  4. 正则逐字段提取（兜底）
  5. 全部失败 → JSONParseFailure
Schema/一致性失败 → SchemaFailure。两者由管线统一重试一次，仍失败 → 502 LLM_PARSE_ERROR。
"""

from __future__ import annotations

import json
import re
from typing import Any

from grader.schemas.models import EPS, is_half_step

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class JSONParseFailure(Exception):
    """模型输出无法解析为 JSON。"""


class SchemaFailure(Exception):
    """解析成功但未通过 Schema / 业务校验。"""


# ---------- JSON 多级容错链 ----------


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出中尽力提取 JSON 对象；全部策略失败抛 JSONParseFailure。"""
    text = (text or "").strip()
    if not text:
        raise JSONParseFailure("模型输出为空")

    # 0. 剥思考块（思考型模型即使被要求关闭，也可能残留 <think>…</think>）
    if "<think>" in text:
        # 闭合块整块剥除；未闭合只剥开标签保留内容（答案可能在块内）
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").strip()

    # 1. 直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. 剥代码围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. 大括号配平截取（取最外层 { … }）
    candidate = _balanced_braces(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 4. 正则逐字段提取（最后一个兜底）
    fields = _regex_extract(text)
    if fields:
        return fields

    raise JSONParseFailure(f"无法从模型输出解析 JSON: {text[:200]!r}")


def _balanced_braces(text: str) -> str | None:
    """取文本中**最后一个**顶层配平 JSON 对象（答案通常在思考/说明文字之后）。"""
    result = None
    depth = 0
    start = -1
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                result = text[start : idx + 1]  # 持续覆盖 → 保留最后一个
    return result


def _regex_extract(text: str) -> dict[str, Any] | None:
    """字段级兜底：能抓到多少抓多少（仅简单字段，嵌套列表靠 3 已覆盖大多数场景）。

    同时容忍字符串值未闭合（被截断的输出）。
    """
    fields: dict[str, Any] = {}

    def _quoted(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m is None:
            # 容忍未闭合的字符串（截断输出）
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)$', text)
        if m:
            return json.loads(f'"{m.group(1)}"')
        return None

    def _bool(key: str) -> bool | None:
        m = re.search(rf'"{key}"\s*:\s*(true|false)', text)
        return m.group(1) == "true" if m else None

    gradable = _bool("gradable")
    if gradable is not None:
        fields["gradable"] = gradable
    cleaned = _quoted("text")
    if cleaned is not None:
        fields["text"] = cleaned
    summary = _quoted("reason_summary")
    if summary is not None:
        fields["reason_summary"] = summary
    return fields or None


# ---------- 阶段结果校验 ----------


def validate_cleanup(obj: dict[str, Any]) -> tuple[bool, str, str]:
    """校验整理阶段输出 → (gradable, cleaned_text, note)。"""
    gradable = obj.get("gradable")
    text = obj.get("text")
    if not isinstance(gradable, bool):
        raise SchemaFailure("整理输出缺少布尔字段 gradable")
    if not isinstance(text, str):
        raise SchemaFailure("整理输出缺少字符串字段 text")
    note = obj.get("note")
    return gradable, text, (note if isinstance(note, str) else "")


def validate_garbled(text: str) -> bool:
    """OCR 层乱码判定（SPEC reason_code: garbled_text）：
    有内容但几乎无汉字，或出现超长重复串，视为严重混乱。"""
    stripped = text.strip()
    if not stripped:
        return False  # 空文本走 empty_ocr_text
    cjk = len(_CJK_RE.findall(stripped))
    if cjk / max(1, len(stripped)) < 0.15:
        return True
    # 连续重复同一字符 ≥ 10 次视为乱码
    if re.search(r"(.)\1{9,}", stripped):
        return True
    return False


# ---------- 判分结果校验与归一化 ----------


def validate_grading(
    obj: dict[str, Any],
    criteria: Any,
) -> dict[str, Any]:
    """校验判分阶段输出并归一化。

    返回结构：
    {
      "rubric_insufficient": False,                                          # 标准无法解析时为 True
      "points": [{id, criterion, evidence, hit, awarded_score, max_score}],  # 按传入顺序
      "deductions": [{reason, points}],
      "reason_summary": str,
    }
    分值非法 / 越界 / 证据缺失一律 SchemaFailure（管线据此重试）。
    """
    # 唯一例外分支：模型判定判卷标准语义不可用（SPEC reason_code: rubric_insufficient）
    if obj.get("rubric_insufficient") is True:
        reason = obj.get("reason_summary") or obj.get("reason") or "判卷标准无法解析"
        if not isinstance(reason, str) or not reason.strip():
            raise SchemaFailure("rubric_insufficient 缺少 reason_summary")
        return {"rubric_insufficient": True, "points": [], "deductions": [], "reason_summary": reason}
    raw_points = obj.get("scoring_points")
    raw_deductions = obj.get("deductions")
    if not isinstance(raw_points, list) or not isinstance(raw_deductions, list):
        raise SchemaFailure("判分输出缺少 scoring_points / deductions 数组")

    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_points:
        if not isinstance(item, dict):
            raise SchemaFailure("scoring_points 元素必须为对象")
        pid = item.get("id")
        if not isinstance(pid, str):
            raise SchemaFailure("得分点 id 缺失或非字符串")
        by_id[pid] = item

    points_out: list[dict[str, Any]] = []
    for sp in criteria.scoring_points:
        item = by_id.get(sp.id)
        if item is None:
            # 模型漏答的得分点：未命中、0 分，而不是让整次输出失败
            points_out.append(
                {"id": sp.id, "criterion": sp.description, "evidence": "", "hit": False,
                 "awarded_score": 0.0, "max_score": sp.score}
            )
            continue
        hit = item.get("hit")
        awarded = item.get("awarded_score")
        evidence = item.get("evidence", "")
        if not isinstance(hit, bool):
            raise SchemaFailure(f"得分点 {sp.id} 缺少布尔 hit")
        if not isinstance(awarded, (int, float)) or isinstance(awarded, bool):
            raise SchemaFailure(f"得分点 {sp.id} awarded_score 非数值")
        awarded = float(awarded)
        if not is_half_step(awarded):
            raise SchemaFailure(f"得分点 {sp.id} awarded_score 非 0.5 步长: {awarded}")
        if awarded < -EPS or awarded > sp.score + EPS:
            raise SchemaFailure(f"得分点 {sp.id} awarded_score 越界: {awarded} (0~{sp.score})")
        if hit and not isinstance(evidence, str):
            raise SchemaFailure(f"得分点 {sp.id} evidence 非字符串")
        if hit and not evidence.strip():
            # SPEC §3.1.4：命中得分点必须引用学生作答原文
            raise SchemaFailure(f"得分点 {sp.id} 命中但缺少原文依据 evidence")
        points_out.append(
            {"id": sp.id, "criterion": sp.description, "evidence": evidence,
             "hit": hit, "awarded_score": awarded, "max_score": sp.score}
        )

    deductions_out: list[dict[str, Any]] = []
    ded_total = 0.0
    for item in raw_deductions:
        if not isinstance(item, dict):
            raise SchemaFailure("deductions 元素必须为对象")
        reason = item.get("reason")
        value = item.get("points")
        if not isinstance(reason, str) or not reason.strip():
            raise SchemaFailure("deductions 缺少 reason")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaFailure("deductions points 非数值")
        value = float(value)
        if not is_half_step(value):
            raise SchemaFailure(f"扣分非 0.5 步长: {value}")
        if value <= 0:
            raise SchemaFailure(f"扣分必须大于 0: {value}")
        deductions_out.append({"reason": reason, "points": value})
        ded_total += value
    if ded_total > criteria.max_score + EPS:
        raise SchemaFailure(f"扣分总和 {ded_total} 超过满分 {criteria.max_score}")

    summary = obj.get("reason_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise SchemaFailure("判分输出缺少 reason_summary")

    return {"rubric_insufficient": False, "points": points_out, "deductions": deductions_out, "reason_summary": summary}
