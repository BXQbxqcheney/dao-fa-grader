"""OpenAI 兼容适配器（同时服务任何 OpenAI 兼容网关）。"""

from __future__ import annotations

from typing import Any

from grader.core.config import Settings
from grader.core.errors import LLMError, LLMTimeoutError


class OpenAICompatibleAdapter:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY 未配置（环境变量注入，禁止写入代码）")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(f"openai SDK 未安装: {exc}") from exc
        # max_retries=0：重试策略由管线统一控制（判分输出失败才重试一次），避免叠加
        self._client: Any = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.openai_model
        self._timeout = settings.llm_timeout_seconds

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._create(messages, json_mode=json_mode)

    def _create(self, messages: list[dict], *, json_mode: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "timeout": self._timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # 部分兼容网关不支持 response_format：去掉后重试一次
            if json_mode and "response_format" in str(exc):
                kwargs.pop("response_format", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    raise self._classify(exc2) from exc2
            else:
                raise self._classify(exc) from exc
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            raise LLMError("大模型返回空内容")
        return content

    @staticmethod
    def _classify(exc: Exception) -> LLMError:
        """SDK 异常归类：超时 → LLM_TIMEOUT，其余 → LLM_ERROR。"""
        name = type(exc).__name__.lower()
        message = str(exc)
        if "timeout" in name or "timeout" in message.lower() or "timed out" in message.lower():
            return LLMTimeoutError(f"大模型调用超时: {message}")
        return LLMError(f"大模型调用失败: {type(exc).__name__}: {message}")
