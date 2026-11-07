"""主流程编排（SPEC §2 v0.3）：

请求校验 → 图像预处理 → 【OCR 路径】行切分/涂改/OCR/噪声过滤/置信度分流
  或【VLM 直读路径】AIOCR 转写（不确定的字不说，uncertain>6 转人工）
→ AI 识别整理（错别字语病；面目全非/严重不通/乱码 → MMQF 转人工）
→ AI 判卷：并发逐点独立判分（每点单次调用，走 Key 池）+ 扣分规则独立判定
→ 服务端计算 score = clamp(Σ awarded_score − Σ deductions.points, 0, max_score)。

needs_human 通过返回 NeedsHumanResponse 表达（HTTP 200）；
系统错误抛 GraderError 子类（4xx/5xx），两者严格区分。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

from grader.core.config import Settings
from grader.core.errors import (
    InternalError,
    LLMParseError,
    LLMTimeoutError,
    OCRError,
    OCRTimeoutError,
)
from grader.core.image import PreparedImage, prepare_image
from grader.grading import parser as P
from grader.grading.prompts import (
    AIOCR_SYSTEM,
    AIOCR_USER_TEMPLATE,
    CLEANUP_SYSTEM,
    JUDGE_DEDUCTIONS_SYSTEM,
    JUDGE_DEDUCTIONS_USER_TEMPLATE,
    JUDGE_POINT_SYSTEM,
    JUDGE_POINT_USER_TEMPLATE,
    MMQF_MARKER,
    RETRY_SUFFIX,
    build_cleanup_user,
    build_judge_deductions_user,
    build_judge_point_user,
)
from grader.llm.base import LLMAdapter
from grader.ocr.base import OCRAdapter, OCRResult
from grader.ocr.linesplit import LineSplitFailure, split_lines
from grader.ocr.marks import detect_and_mask
from grader.schemas.models import EPS, GradeRequest, GradedResponse, NeedsHumanResponse

logger = logging.getLogger("grader.pipeline")

T = TypeVar("T")

CONSISTENCY_EPS = 0.01  # §3.1.3 一致性容差
AIOCR_UNCERTAIN_MAX = 6  # AIOCR 认不出而省略的字符数上限，超过即转人工（用户裁决 2026-11-07）


class GradingPipeline:
    def __init__(
        self,
        settings: Settings,
        ocr_adapter: OCRAdapter,
        llm_adapter: LLMAdapter,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._ocr = ocr_adapter
        self._llm = llm_adapter
        self._executor = executor or ThreadPoolExecutor(max_workers=max(1, settings.ocr_concurrency))

    # ---------- 对外入口 ----------

    async def grade(
        self, request: GradeRequest, request_id: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> GradedResponse | NeedsHumanResponse:
        """on_progress（可选）：阶段进度回调，事件为 {"type":"stage", stage, status, detail}；
        在事件循环线程内调用（SSE 流式端点使用）。"""
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._grade_inner(request, request_id, on_progress),
                timeout=self._settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InternalError(
                f"请求总超时（>{self._settings.request_timeout_seconds:g}s），请检查 OCR/大模型预算"
            ) from exc
        self._log_result(request_id, result, time.monotonic() - started)
        return result

    @staticmethod
    def _progress(
        cb: Callable[[dict], None] | None, stage: str, status: str, detail: str = "",
        input_: str | None = None, output: str | None = None, input_kind: str | None = None,
    ) -> None:
        """阶段进度事件：detail 为摘要行；input/output 为发给 AI 的完整输入与 AI 原始返回（不裁剪）。"""
        if not cb:
            return
        evt: dict = {"type": "stage", "stage": stage, "status": status, "detail": detail}
        if input_ is not None:
            evt["input"] = input_
        if output is not None:
            evt["output"] = output
        if input_kind is not None:
            evt["input_kind"] = input_kind
        cb(evt)

    # ---------- 内部阶段 ----------

    async def _grade_inner(
        self, request: GradeRequest, request_id: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> GradedResponse | NeedsHumanResponse:
        prepared = await self._run(prepare_image, request.image_base64, self._settings)
        self._progress(on_progress, "校验图像", "done",
                       f"{prepared.format} · {prepared.image.width}×{prepared.image.height}px")

        # VLM 直读模式：AIOCR 转写（不确定的字不说）→ AI 识别 → 判分；跳过行切分/OCR/置信度闸
        if self._settings.agnes_vlm_mode and hasattr(self._llm, "chat_vlm"):
            return await self._vlm_flow(request, request_id, prepared, on_progress)

        self._progress(on_progress, "OCR 识别", "running")
        # 行切分（失败 → needs_human: illegible_line_split）
        try:
            line_images = await self._run(split_lines, prepared.image)
        except LineSplitFailure as exc:
            return self._needs_human(
                request_id, request, prepared, "illegible_line_split", f"行切分失败：{exc}"
            )

        # 涂改/圈划剔除（仅明确标记）
        masked_lines = []
        removed_total = 0
        for line_img in line_images:
            masked = await self._run(detect_and_mask, line_img)
            masked_lines.append(masked.image)
            removed_total += masked.removed_est_chars

        # OCR（线程池 + 独立超时；非 GraderError 异常统一归一为 OCR_ERROR，纵深防御）
        try:
            ocr_lines = await asyncio.wait_for(
                self._run(self._ocr.recognize_lines, masked_lines),
                timeout=self._settings.ocr_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise OCRTimeoutError(f"OCR 超时（>{self._settings.ocr_timeout_seconds:g}s）") from exc
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"OCR 引擎异常: {type(exc).__name__}: {exc}") from exc
        keep_lines, noise_count = self._drop_noise_lines(ocr_lines)
        ocr_result = OCRResult(
            lines=keep_lines, removed_marks=removed_total + noise_count, resized=prepared.resized
        )
        self._progress(
            on_progress, "OCR 识别", "done",
            f"识别 {len([l for l in keep_lines if l.text.strip()])} 行"
            f"，置信度 {ocr_result.confidence if ocr_result.confidence is not None else '—'}"
            f"，剔除标记 {ocr_result.removed_marks}\n{ocr_result.text}",
        )

        # 置信度分流（空文本 / 乱码 / 低置信度）
        guard = self._confidence_guard(request_id, request, prepared, ocr_result)
        if guard is not None:
            return guard

        # LLM 文本整理（阶段事件由 _cleanup 自持）
        cleaned, note = await self._cleanup(ocr_result.text, on_progress)
        if not cleaned:
            return self._needs_human(
                request_id, request, prepared, "illegible_after_cleanup",
                f"文本整理后仍不可判：{note or '语义不通或信息残缺'}", ocr_result,
            )
        if P.validate_garbled(cleaned):  # 防御性校验：整理结果仍含乱码 → 转人工
            return self._needs_human(
                request_id, request, prepared, "illegible_after_cleanup",
                "整理结果仍存在乱码（防御性校验），为确保不误判转人工", ocr_result,
            )

        # LLM 判分（阶段事件由 _grading 自持）
        return await self._grading(request_id, request, prepared, ocr_result, cleaned, on_progress)

    # ---------- VLM 直读流程（AIOCR 转写 → AI 识别整理 → 判分） ----------

    async def _vlm_flow(
        self, request: GradeRequest, request_id: str, prepared: PreparedImage,
        on_progress: Callable[[dict], None] | None = None,
    ) -> GradedResponse | NeedsHumanResponse:
        """AIOCR：图片交给多模态 LLM 转写答题区（不确定的字不说）→ AI 识别整理 → 判分。"""
        from grader.core.image import strip_data_url

        payload, _ = strip_data_url(request.image_base64)
        image_url = f"data:image/{prepared.format};base64,{payload}"
        ocr_result = OCRResult(lines=[], removed_marks=0, resized=prepared.resized)

        # 阶段 1：AIOCR 转写
        self._progress(on_progress, "AIOCR 转写", "running",
                       input_kind="image", input_="（作答图片，见左侧/上方预览）")
        raw = await self._chat_vlm_raw(AIOCR_SYSTEM, AIOCR_USER_TEMPLATE, image_url)
        transcribed, uncertain = self._parse_aiocr(raw)
        if uncertain > AIOCR_UNCERTAIN_MAX:
            return self._needs_human(
                request_id, request, prepared, "low_ocr_confidence",
                f"AIOCR 有 {uncertain} 个字无法辨认（超过 {AIOCR_UNCERTAIN_MAX} 字阈值），"
                "为确保不误判转人工", ocr_result,
            )
        if not transcribed.strip():
            return self._needs_human(
                request_id, request, prepared, "empty_ocr_text",
                "AIOCR 未识别出可辨认的学生作答", ocr_result,
            )
        self._progress(on_progress, "AIOCR 转写", "done", transcribed, output=raw)

        # 阶段 2：AI 识别（文本整理，面目全非/严重不通/错别字连篇/乱码 → MMQF）
        cleaned, note = await self._cleanup(transcribed, on_progress)
        if not cleaned:
            return self._needs_human(
                request_id, request, prepared, "illegible_after_cleanup",
                f"文本整理后仍不可判：{note or '面目全非'}", ocr_result,
            )
        if P.validate_garbled(cleaned):  # 防御性校验：整理结果仍含乱码 → 转人工
            return self._needs_human(
                request_id, request, prepared, "illegible_after_cleanup",
                "整理结果仍存在乱码（防御性校验），为确保不误判转人工", ocr_result,
            )
        return await self._grading(request_id, request, prepared, ocr_result, cleaned, on_progress)

    def _parse_aiocr(self, raw: str) -> tuple[str, int]:
        """AIOCR 输出解析 → (转写文本, 认不出而省略的字符数)。

        优先 {"text": …, "uncertain": N}；解析失败时把非空原文当转写结果（转写是散文体）。
        """
        try:
            obj = P.extract_json(raw)
            text = str(obj.get("text", "")).strip()
            try:
                uncertain = int(obj.get("uncertain", 0))
            except (TypeError, ValueError):
                uncertain = 0
            return text, max(0, uncertain)
        except P.JSONParseFailure:
            return raw.strip(), 0

    async def _chat_vlm_raw(self, system: str, user: str, image_url: str) -> str:
        """VLM 调用（线程池 + 外层预算内由适配器轮换 Key），返回原始输出。"""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self._llm.chat_vlm(system, user, image_url),
                ),
                timeout=self._llm_budget(),
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                f"AIOCR 转写超时（>{self._llm_budget():.0f}s，含全部 Key 尝试）"
            ) from exc

    # ---------- OCR 结果整备 ----------

    # 噪声行判定阈值：识别置信度极低且汉字极少的行，是页眉残迹/题号/污渍 band，
    # 不是作答内容——若计入「整题取各行最小值」会拖垮整题置信度造成误转人工。
    NOISE_CONF_MAX = 0.30
    NOISE_HANZI_MAX = 8

    def _drop_noise_lines(self, ocr_lines: list) -> tuple[list, int]:
        """分离噪声行；返回 (有效行, 噪声行数)。噪声行并入 removed_marks 计数（诊断用）。"""
        hanzi_re = re.compile(r"[\u4e00-\u9fff]")
        noise: list = []
        keep: list = []
        for line in ocr_lines:
            if line.confidence < self.NOISE_CONF_MAX and len(hanzi_re.findall(line.text)) <= self.NOISE_HANZI_MAX:
                noise.append(line)
            else:
                keep.append(line)
        return keep, len(noise)

    # ---------- 置信度分流 ----------

    def _confidence_guard(
        self, request_id: str, request: GradeRequest, prepared: PreparedImage, ocr_result: OCRResult
    ) -> NeedsHumanResponse | None:
        threshold = self._settings.ocr_confidence_threshold
        if not ocr_result.text.strip():
            return self._needs_human(
                request_id, request, prepared, "empty_ocr_text", "OCR 未识别出有效文本", ocr_result
            )
        if P.validate_garbled(ocr_result.text):
            return self._needs_human(
                request_id, request, prepared, "garbled_text", "OCR 文本严重混乱，无法整理", ocr_result
            )
        confidence = ocr_result.confidence
        if confidence is not None and confidence < threshold:
            return self._needs_human(
                request_id, request, prepared, "low_ocr_confidence",
                f"OCR 置信度 {confidence:.2f} 低于阈值 {threshold:g}，需人工判卷", ocr_result,
            )
        return None

    # ---------- LLM：整理 ----------

    async def _cleanup(self, ocr_text: str, on_progress: Callable[[dict], None] | None = None) -> tuple[str, str]:
        """AI 识别：返回 (整理后文本, 不可判说明)。

        模型判定面目全非/严重不通/错别字连篇/乱码时返回 MMQF 标记 → 视为无法识别（转人工）；
        解析失败重试一次，仍失败 → LLMParseError。
        """
        user = build_cleanup_user(ocr_text)
        self._progress(on_progress, "AI 识别整理", "running", input_=user)
        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(2):
            text_out = await self._chat(CLEANUP_SYSTEM, user if attempt == 0 else user + RETRY_SUFFIX)
            last_raw = text_out
            stripped = text_out.strip()
            if MMQF_MARKER in stripped and "{" not in stripped:
                self._progress(on_progress, "AI 识别整理", "done", "（MMQF）", input_=user, output=last_raw)
                return "", "模型判定文本面目全非/严重不通/错别字连篇/乱码（MMQF）"
            try:
                gradable, cleaned, note = P.validate_cleanup(P.extract_json(text_out))
            except (P.JSONParseFailure, P.SchemaFailure) as exc:
                last_error = exc
                continue
            if not gradable:
                self._progress(on_progress, "AI 识别整理", "done", f"（不可判：{note}）",
                               input_=user, output=last_raw)
                return "", note.strip() or "模型判定不可判"
            self._progress(on_progress, "AI 识别整理", "done", cleaned, input_=user, output=last_raw)
            return cleaned, ""
        raise LLMParseError(f"文本整理输出经重试仍无法解析: {last_error}")

    # ---------- LLM：判分 ----------

    async def _grading(
        self, request_id: str, request: GradeRequest, prepared: PreparedImage,
        ocr_result: OCRResult, cleaned: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> GradedResponse | NeedsHumanResponse:
        """并发逐点判卷：每个得分点独立一次判分 + 扣分规则独立一路，全部并发执行后合成。"""
        criteria = request.criteria
        self._progress(on_progress, "AI 判卷", "running",
                       detail=f"{len(criteria.scoring_points)} 个得分点 + 扣分规则，并发独立判")

        tasks = [
            self._judge_point(criteria, sp, cleaned, on_progress)
            for sp in criteria.scoring_points
        ]
        tasks.append(self._judge_deductions(criteria, cleaned, on_progress))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 失败的点整体重试一轮（点级已内含一次重试 + Key 轮换）
        retry_idx = [i for i, r in enumerate(results) if isinstance(r, BaseException)]
        if retry_idx:
            self._progress(on_progress, "AI 判卷", "running",
                           detail=f"{len(retry_idx)} 路判分异常，重试")
            retry_tasks = [
                (self._judge_point(criteria, criteria.scoring_points[i], cleaned, on_progress)
                 if i < len(criteria.scoring_points)
                 else self._judge_deductions(criteria, cleaned, on_progress))
                for i in retry_idx
            ]
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            for i, r in zip(retry_idx, retry_results):
                if isinstance(r, BaseException):
                    if isinstance(r, LLMTimeoutError):
                        raise r  # 超时保留 504 语义
                    raise LLMParseError(f"判分点经重试仍失败: {r}") from r
                results[i] = r

        points = [results[i] for i in range(len(criteria.scoring_points))]
        deductions = results[-1]
        reason_summary = "；".join(
            f"{p['id']} {p['awarded_score']:g}/{p['max_score']:g}" for p in points
        )
        if deductions:
            reason_summary += "；扣分：" + "；".join(
                f"{d['reason']}（{d['points']:g} 分）" for d in deductions
            )
        normalized = {
            "rubric_insufficient": False,
            "points": points,
            "deductions": deductions,
            "reason_summary": reason_summary,
        }
        response = self._build_graded(request_id, request, prepared, ocr_result, cleaned, normalized)
        self._progress(on_progress, "AI 判卷", "done",
                       f"得分 {response.score:g} / {criteria.max_score:g}")
        return response

    async def _judge_point(
        self, criteria: Any, sp: Any, student_text: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        """单点并发判分：判定该得分点 hit/awarded_score/evidence；失败自动重试一次。"""
        stage = f"判卷·{sp.id}"
        user = build_judge_point_user(criteria, sp, student_text)
        self._progress(on_progress, stage, "running", input_=user)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = await self._chat(
                    JUDGE_POINT_SYSTEM,
                    user if attempt == 0 else user + RETRY_SUFFIX,
                )
            except (LLMTimeoutError, LLMError) as exc:
                last_error = exc
                continue
            try:
                obj = P.extract_json(raw)
                awarded = float(obj.get("awarded_score", 0))
                hit = bool(obj.get("hit", awarded > 0))
                if not P.is_half_step(awarded) or awarded < -EPS or awarded > sp.score + EPS:
                    raise P.SchemaFailure(f"awarded_score 非法: {awarded} (应为 0~{sp.score} 的 0.5 步长)")
                evidence = obj.get("evidence", "")
                if hit and (not isinstance(evidence, str) or not evidence.strip()):
                    raise P.SchemaFailure("命中但缺少 evidence")
                self._progress(on_progress, stage, "done",
                               f"{sp.score:g} 分点得 {awarded:g}", output=raw)
                return {"id": sp.id, "criterion": sp.description, "evidence": evidence or "",
                        "hit": hit, "awarded_score": awarded, "max_score": sp.score}
            except (P.JSONParseFailure, P.SchemaFailure) as exc:
                last_error = exc
                continue
        raise LLMParseError(f"得分点 {sp.id} 经重试仍无法解析: {last_error}")

    async def _judge_deductions(
        self, criteria: Any, student_text: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> list[dict[str, Any]]:
        """扣分规则独立判定：返回 deductions 列表；失败自动重试一次。"""
        stage = "判卷·扣分"
        user = build_judge_deductions_user(criteria, student_text)
        self._progress(on_progress, stage, "running", input_=user)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = await self._chat(
                    JUDGE_DEDUCTIONS_SYSTEM,
                    user if attempt == 0 else user + RETRY_SUFFIX,
                )
            except (LLMTimeoutError, LLMError) as exc:
                last_error = exc
                continue
            try:
                obj = P.extract_json(raw)
                ded_list = obj.get("deductions", [])
                if not isinstance(ded_list, list):
                    raise P.SchemaFailure("deductions 非数组")
                out: list[dict[str, Any]] = []
                total = 0.0
                for item in ded_list:
                    if not isinstance(item, dict):
                        raise P.SchemaFailure("deductions 元素必须为对象")
                    reason = item.get("reason")
                    value = float(item.get("points", 0))
                    if not isinstance(reason, str) or not reason.strip():
                        raise P.SchemaFailure("deductions 缺少 reason")
                    if not P.is_half_step(value) or value <= 0:
                        raise P.SchemaFailure(f"扣分非法: {value}")
                    out.append({"reason": reason, "points": value})
                    total += value
                if total > criteria.max_score + EPS:
                    raise P.SchemaFailure(f"扣分总和 {total} 超过满分")
                self._progress(on_progress, stage, "done",
                               f"扣 {total:g} 分" if out else "无规则扣分", output=raw)
                return out
            except (P.JSONParseFailure, P.SchemaFailure) as exc:
                last_error = exc
                continue
        raise LLMParseError(f"扣分判定经重试仍无法解析: {last_error}")

    @staticmethod
    def _compute_score(normalized: dict[str, Any], criteria: Any) -> float:
        """计分语义（用户裁决 2026-11-07）：得分 = 踩到的得分点得分之和。

        score = clamp(Σawarded_score − Σdeductions.points, 0, max_score)；
        deductions 只承载规则性扣分（如不结合材料倒扣），得分点内部失分体现在 awarded_score。
        """
        awarded_sum = sum(p["awarded_score"] for p in normalized["points"])
        ded_sum = sum(d["points"] for d in normalized["deductions"])
        return max(0.0, min(criteria.max_score, awarded_sum - ded_sum))

    def _build_graded(
        self, request_id: str, request: GradeRequest, prepared: PreparedImage,
        ocr_result: OCRResult, cleaned: str, normalized: dict[str, Any],
    ) -> GradedResponse:
        criteria = request.criteria
        score = self._compute_score(normalized, criteria)
        notes: list[str] = []
        if sum(sp.score for sp in criteria.scoring_points) > criteria.max_score + EPS:
            notes.append("判卷标准分值之和超过满分，总分已按满分封顶")
        reason = normalized["reason_summary"].strip()
        if notes:
            reason = reason + "（" + "；".join(notes) + "）"

        model_name = self._model_name()
        return GradedResponse(
            request_id=request_id,
            question_id=criteria.question_id,
            max_score=criteria.max_score,
            score=score,
            scoring_points=[{**p} for p in normalized["points"]],  # type: ignore[arg-type]
            deductions=normalized["deductions"],  # type: ignore[arg-type]
            reason=reason,
            ocr={
                "text": cleaned if request.options.return_ocr_text else "",
                "confidence": ocr_result.confidence,
                "removed_marks": ocr_result.removed_marks,
                "resized": prepared.resized,
            },  # type: ignore[arg-type]
            model={"provider": self._llm.name, "name": model_name},  # type: ignore[arg-type]
        )

    def _model_name(self) -> str:
        s = self._settings
        return s.anthropic_model if s.llm_provider == "anthropic" else s.openai_model

    # ---------- 通用工具 ----------

    def _needs_human(
        self, request_id: str, request: GradeRequest, prepared: PreparedImage,
        reason_code: str, message: str, ocr_result: OCRResult | None = None,
    ) -> NeedsHumanResponse:
        ocr: dict[str, Any] = {"text": "", "confidence": None, "removed_marks": 0, "resized": prepared.resized}
        if ocr_result is not None:
            ocr = {
                "text": ocr_result.text if request.options.return_ocr_text else "",
                "confidence": ocr_result.confidence,
                "removed_marks": ocr_result.removed_marks,
                "resized": prepared.resized,
            }
        return NeedsHumanResponse(
            request_id=request_id,
            reason_code=reason_code,
            message=message,
            image_url=request.image_base64,  # 原样回传提交时的 data URL 或 Base64
            ocr=ocr,  # type: ignore[arg-type]
        )

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        """同步阻塞函数投递到线程池（OCR / 图像处理）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def _llm_budget(self) -> float:
        """LLM 外层超时预算：单次调用超时 × 适配器最大尝试次数（Key 池轮换）+ 少量余量。

        Agnes 适配器在单 Key 超时/失败时内部自动换 Key 重试，因此外层预算必须
        覆盖全部尝试，否则池还没转起来就被总闸杀掉。
        """
        attempts = max(1, int(getattr(self._llm, "max_chat_attempts", 1)))
        return self._settings.llm_timeout_seconds * attempts + min(2.0, self._settings.llm_timeout_seconds)

    async def _chat(self, system: str, user: str) -> str:
        """LLM 调用（SDK 为同步实现；外层预算覆盖 Key 池全部轮换尝试）。"""
        if self._settings.llm_disable_thinking:
            # 思考型模型（Qwen3 系）的官方软开关：关闭思考模式，直接输出答案
            system = system.rstrip() + "\n/no_think"
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, lambda: self._llm.chat(system, user, json_mode=True)),
                timeout=self._llm_budget(),
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                f"大模型调用超时（>{self._llm_budget():.0f}s，含全部 {getattr(self._llm, 'max_chat_attempts', 1)} 次 Key 尝试）"
            ) from exc

    def _log_result(self, request_id: str, result: Any, elapsed: float) -> None:  # noqa: C901
        """日志白名单：仅 request_id / 状态 / 耗时 / 指标，禁止图片与全文。"""
        if isinstance(result, GradedResponse):
            logger.info(
                "grade done request_id=%s status=graded score=%s confidence=%s removed_marks=%s "
                "provider=%s model=%s elapsed=%.2fs",
                request_id, result.score, result.ocr.confidence, result.ocr.removed_marks,
                result.model.provider, result.model.name, elapsed,
            )
        else:
            logger.info(
                "grade done request_id=%s status=needs_human reason_code=%s confidence=%s "
                "removed_marks=%s elapsed=%.2fs",
                request_id, result.reason_code, result.ocr.confidence, result.ocr.removed_marks, elapsed,
            )
