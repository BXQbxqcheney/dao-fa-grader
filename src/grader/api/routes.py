"""API 路由：判卷接口 + SSE 流式判卷 + 健康检查 + 工作台静态托管挂载点。"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from grader.core.errors import GraderError, InternalError
from grader.schemas.models import GradeRequest

router = APIRouter()
logger = logging.getLogger("grader.api")


def new_request_id() -> str:
    """格式：req_<UTC 日期>_<8 位随机串>（SPEC §3.1.4）。"""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"req_{date}_{secrets.token_hex(4)}"


def _log_request_start(request_id: str, auth_present: bool) -> None:
    # 隐私白名单：只记 request_id 与鉴权头是否存在（绝不记录其值）
    logger.info("grade start request_id=%s auth_header=%s", request_id, "present" if auth_present else "absent")


@router.get("/healthz")
async def healthz() -> dict:
    """探活，不检查外部模型可用性（SPEC §3.2）。"""
    return {"status": "ok"}


@router.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse(url="/workbench/")


@router.post("/api/v1/grade")
async def grade(payload: GradeRequest, request: Request) -> JSONResponse:
    request_id = new_request_id()
    _log_request_start(request_id, bool(request.headers.get("authorization")))
    started = time.monotonic()
    pipeline = request.app.state.pipeline
    try:
        result = await pipeline.grade(payload, request_id)
    except GraderError as exc:
        elapsed = time.monotonic() - started
        logger.warning(
            "grade failed request_id=%s code=%s elapsed=%.2fs", request_id, exc.code, elapsed
        )
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload(request_id))
    except Exception as exc:  # 兜底：未预期异常也必须以统一错误结构返回，不吞错
        logger.exception("grade unexpected error request_id=%s: %s", request_id, exc)
        unexpected = InternalError("服务内部错误，请稍后重试")
        return JSONResponse(status_code=unexpected.http_status, content=unexpected.to_payload(request_id))
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.post("/api/v1/grade/stream")
async def grade_stream(payload: GradeRequest, request: Request) -> StreamingResponse:
    """SSE 流式判卷：与 /api/v1/grade 同契约同载荷，另发阶段进度事件（工作台实时进度用）。

    事件均为 `data: {json}\\n\\n`：
      {"type":"start","request_id":…}
      {"type":"stage","stage":"OCR 识别","status":"running|done","detail":…}
      {"type":"result","data":{graded|needs_human|error 载荷},"http_status":…,"elapsed":…}
    """
    request_id = new_request_id()
    _log_request_start(request_id, bool(request.headers.get("authorization")))
    pipeline = request.app.state.pipeline
    started = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(evt: dict) -> None:
        queue.put_nowait(evt)  # 进度回调在事件循环线程内触发

    async def run() -> tuple[dict, int]:
        try:
            result = await pipeline.grade(payload, request_id, on_progress=emit)
            return result.model_dump(mode="json"), 200
        except GraderError as exc:
            logger.warning("grade stream failed request_id=%s code=%s", request_id, exc.code)
            return exc.to_payload(request_id), exc.http_status
        except Exception as exc:  # 兜底不吞错
            logger.exception("grade stream unexpected error request_id=%s: %s", request_id, exc)
            unexpected = InternalError("服务内部错误，请稍后重试")
            return unexpected.to_payload(request_id), unexpected.http_status

    async def gen():
        yield f"data: {json.dumps({'type': 'start', 'request_id': request_id}, ensure_ascii=False)}\n\n"
        runner = asyncio.create_task(run())
        final: tuple[dict, int] | None = None
        while final is None:
            if runner.done():
                while not queue.empty():  # 排空余下的阶段事件
                    yield f"data: {json.dumps(queue.get_nowait(), ensure_ascii=False)}\n\n"
                final = runner.result()
                break
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=0.25)
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue
        result_payload, http_status = final
        yield (
            "data: "
            + json.dumps(
                {"type": "result", "data": result_payload, "http_status": http_status,
                 "elapsed": round(time.monotonic() - started, 2)},
                ensure_ascii=False,
            )
            + "\n\n"
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
