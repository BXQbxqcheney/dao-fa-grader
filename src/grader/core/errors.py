"""统一错误体系：系统错误（4xx/5xx）与 needs_human 严格区分，绝不混用。"""

from __future__ import annotations


class GraderError(Exception):
    """所有可映射到 HTTP 错误响应的异常基类。"""

    http_status: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_payload(self, request_id: str | None = None) -> dict:
        error: dict = {"code": self.code, "message": self.message}
        if request_id:
            error["request_id"] = request_id
        return {"error": error}


class InvalidArgumentError(GraderError):
    http_status = 400
    code = "INVALID_ARGUMENT"


class ImageTooLargeError(GraderError):
    http_status = 413
    code = "IMAGE_TOO_LARGE"


class UnsupportedImageTypeError(GraderError):
    http_status = 415
    code = "UNSUPPORTED_IMAGE_TYPE"


class LLMError(GraderError):
    """大模型调用失败（网络/接口/欠费等）。"""

    http_status = 502
    code = "LLM_ERROR"


class LLMParseError(GraderError):
    """模型输出经重试仍无法通过 JSON Schema / 一致性校验。"""

    http_status = 502
    code = "LLM_PARSE_ERROR"


class LLMTimeoutError(GraderError):
    http_status = 504
    code = "LLM_TIMEOUT"


class OCRTimeoutError(GraderError):
    http_status = 500
    code = "OCR_ERROR"


class OCRError(GraderError):
    http_status = 500
    code = "OCR_ERROR"


class InternalError(GraderError):
    http_status = 500
    code = "INTERNAL_ERROR"

# 「需人工处理」不是错误：由管线直接返回 NeedsHumanResponse（HTTP 200 + image_url），
# 与本模块的 4xx/5xx 错误结构严格互斥（SPEC §3.1.5/§3.1.6）。
