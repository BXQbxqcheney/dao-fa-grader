"""FastAPI 应用入口。

启动：uvicorn grader.main:app --host 0.0.0.0 --port 8000
测试注入：create_app(settings=..., ocr_adapter=FakeOCR(), llm_adapter=FakeLLM(...))
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from grader import __version__
from grader.api.routes import new_request_id, router
from grader.core.config import Settings, load_settings
from grader.core.errors import GraderError, InvalidArgumentError, LLMError
from grader.core.pipeline import GradingPipeline
from grader.llm.base import create_llm_adapter
from grader.ocr.base import OCRAdapter


def build_ocr_adapter(settings: Settings) -> OCRAdapter:
    """按 OCR_PROVIDER 构建适配器；权重懒加载，首次识别时才读盘。"""
    if settings.ocr_provider == "rapidocr":
        from grader.ocr.rapidocr import RapidOCRAdapter

        return RapidOCRAdapter()
    from grader.ocr.ofa import OFAAdapter

    return OFAAdapter(settings.ofa_model_path)


def _configure_logging(settings: Settings) -> None:
    level = logging.DEBUG if settings.debug_logging else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("grader").setLevel(level)
    # 密钥缺失的处理：create_llm_adapter / build_ocr_adapter 在 create_app 期即 fail-fast
    # （LLMError / OCRError），保证「起不来」优于「起得来但每个请求都失败」。


def create_app(
    settings: Settings | None = None,
    ocr_adapter: OCRAdapter | None = None,
    llm_adapter: Any | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    _configure_logging(settings)

    app = FastAPI(title="道法主观题智能判卷服务", version=__version__)
    app.state.settings = settings

    executor = ThreadPoolExecutor(max_workers=max(2, settings.ocr_concurrency + 1))
    ocr = ocr_adapter or build_ocr_adapter(settings)
    try:
        llm = llm_adapter or create_llm_adapter(settings)
    except GraderError as exc:
        # 软启动：未配置 LLM（常见于首次体验）不阻塞服务——工作台/健康检查可用，
        # 判卷请求返回带配置指引的 502。
        logging.getLogger("grader").warning(
            "LLM 未配置，服务将以受限模式启动，判卷请求将返回 502：%s", exc
        )
        from grader.llm.base import UnconfiguredLLMAdapter

        llm = UnconfiguredLLMAdapter(str(exc))
    app.state.pipeline = GradingPipeline(settings, ocr, llm, executor)

    app.include_router(router)

    # 请求体 Pydantic 校验失败 → 400 INVALID_ARGUMENT（含必填缺失与非 0.5 步长）
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        messages = "; ".join(
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        error = InvalidArgumentError(f"请求参数不合法: {messages}")
        return JSONResponse(status_code=400, content=error.to_payload())

    # GraderError 全局兜底（路由内已捕获一层，此处防其他入口）
    @app.exception_handler(GraderError)
    async def _grader_error_handler(_: Request, exc: GraderError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    # Web 试判工作台（静态托管，不改 API 契约）
    workbench_dir = Path(__file__).resolve().parent / "workbench"
    if (workbench_dir / "index.html").is_file():
        app.mount("/workbench", StaticFiles(directory=str(workbench_dir), html=True), name="workbench")

    return app


def __getattr__(name: str) -> Any:
    """模块级惰性创建 uvicorn 入口 `grader.main:app`。

    延迟到真正启动服务时才读取环境并构建真实适配器；
    测试导入本模块（注入 Fake）不会触发真实 LLM/OCR 初始化。
    """
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
