"""LLM 适配器协议与工厂（SPEC §6.1）。"""

from __future__ import annotations

from typing import Protocol

from grader.core.config import Settings
from grader.core.errors import LLMError


class LLMAdapter(Protocol):
    """所有模型供应商实现统一接口。"""

    name: str

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """同步调用；实现必须把 SDK 异常归类为 LLMError / LLMTimeoutError 抛出。"""
        ...


class UnconfiguredLLMAdapter:
    """未配置任何 LLM 时的占位适配器：服务可启动（工作台/健康检查可用），
    判卷请求返回带配置指引的 502，而不是启动即失败。"""

    name = "unconfigured"

    def __init__(self, message: str) -> None:
        self._message = message

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        raise LLMError(f"{self._message}（配置 .env 后重启服务即可）")


def create_llm_adapter(settings: Settings) -> LLMAdapter:
    """按配置创建供应商适配器；密钥缺失在创建期即报错，避免运行到一半才失败。"""
    if settings.llm_provider == "anthropic":
        from grader.llm.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(settings)
    if settings.llm_provider == "agnes":
        from grader.llm.agnes_adapter import AgnesAdapter

        return AgnesAdapter(settings)
    if settings.llm_provider == "openai":
        from grader.llm.openai_adapter import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter(settings)
    raise LLMError(f"未知 LLM_PROVIDER: {settings.llm_provider}")
