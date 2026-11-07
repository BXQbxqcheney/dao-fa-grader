"""Anthropic 适配器。"""

from __future__ import annotations

from typing import Any

from grader.core.config import Settings
from grader.core.errors import LLMError, LLMTimeoutError


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY 未配置（环境变量注入，禁止写入代码）")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(f"anthropic SDK 未安装: {exc}") from exc
        self._sdk = anthropic
        self._client: Any = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.anthropic_model
        self._timeout = settings.llm_timeout_seconds

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=self._timeout,
            )
        except self._sdk.APITimeoutError as exc:
            raise LLMTimeoutError(f"大模型调用超时: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"大模型调用失败: {type(exc).__name__}: {exc}") from exc
        content = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        if not content.strip():
            raise LLMError("大模型返回空内容")
        return content
