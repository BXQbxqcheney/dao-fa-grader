"""Agnes 适配器（OpenAI 兼容接口 + 多 Key 轮询池）。

接口规范：D:/zm/文本模型.txt（Agnes 2.5 Flash）
- Endpoint: POST https://api.agnes-ai.cn/v1/chat/completions
- 认证: Bearer <key>；模型: agnes-2.5-flash（兼容 agnes-2.0-flash）
- Thinking 默认关闭（chat_template_kwargs.enable_thinking 显式开启才启用）

Key 来源（三选一，优先级从高到低）：
1. AGNES_API_KEYS：逗号分隔的多 Key；
2. AGNES_KEY_FILE：Key 文件路径（每行一个 Key，sk- 开头行）；
3. AGNES_API_KEY：单 Key。
Key 严禁写入代码/示例/日志（密钥铁则）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from grader.core.config import Settings
from grader.core.errors import LLMError, LLMTimeoutError
from grader.llm.keypool import KeyPool

logger = logging.getLogger("grader.llm.agnes")

# 单次判卷请求内最多尝试的 Key 数（防止系统性故障把整个池打一遍）
MAX_KEY_ATTEMPTS = 4


def load_keys(settings: Settings) -> list[str]:
    """按优先级收集 Key；来源之间合并去重（env 逗号列表 + 文件 + 单 Key）。"""
    keys: list[str] = []
    if settings.agnes_api_keys:
        keys.extend(k.strip() for k in settings.agnes_api_keys.split(",") if k.strip())
    if settings.agnes_key_file:
        path = Path(settings.agnes_key_file)
        if not path.is_file():
            raise LLMError(f"AGNES_KEY_FILE 指向的文件不存在: {path}")
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("sk-"):
                    keys.append(line)
        except OSError as exc:
            raise LLMError(f"读取 Agnes Key 文件失败: {exc}") from exc
    if settings.agnes_api_key:
        keys.append(settings.agnes_api_key.strip())
    # 去重保序
    seen: set[str] = set()
    unique = [k for k in keys if not (k in seen or seen.add(k))]
    if not unique:
        raise LLMError(
            "Agnes 未配置任何 Key（AGNES_API_KEYS / AGNES_KEY_FILE / AGNES_API_KEY 三选一）"
        )
    return unique


class AgnesAdapter:
    name = "agnes"

    def __init__(self, settings: Settings) -> None:
        self._pool = KeyPool(load_keys(settings))
        self._base_url = settings.agnes_base_url or "https://api.agnes-ai.cn/v1"
        self._model = settings.agnes_model or "agnes-2.5-flash"
        self._timeout = settings.llm_timeout_seconds
        self._thinking = settings.agnes_thinking
        self._clients: dict[str, Any] = {}
        logger.info("Agnes 适配器就绪：Key 池 %d 个，model=%s，thinking=%s",
                    len(self._pool), self._model, self._thinking)

    @property
    def max_chat_attempts(self) -> int:
        """单次判卷内最多尝试的 Key 数（管线按此放大外层超时预算，让池转起来）。"""
        return min(MAX_KEY_ATTEMPTS, len(self._pool))

    def _client_for(self, key: str) -> Any:
        """按 Key 惰性创建 OpenAI 兼容客户端（SDK 复用连接池）。"""
        if key not in self._clients:
            from openai import OpenAI

            self._clients[key] = OpenAI(
                api_key=key, base_url=self._base_url, timeout=self._timeout, max_retries=0
            )
        return self._clients[key]

    def _thinking_kwargs(self) -> dict[str, Any]:
        """Agnes Thinking 模式参数（OpenAI 兼容格式；经 extra_body 传递非标准字段）。"""
        if self._thinking:
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        return {}

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """多 Key 轮询调用：单 Key 失败按错误类型标记池状态并切换下一个。"""
        pool_size = len(self._pool)
        attempts = min(MAX_KEY_ATTEMPTS, pool_size)
        last_error: LLMError | None = None
        for _ in range(attempts):
            key = self._pool.acquire()  # 全冷却时此处抛 LLMError
            try:
                return self._create(key, system, user, json_mode=json_mode)
            except LLMTimeoutError:
                self._pool.report_transient_error(key)
                last_error = LLMTimeoutError("Agnes 调用超时（该 Key 已短冷却，可换 Key 重试）")
                continue
            except LLMError as exc:
                # _create 已按状态码向池上报；此处继续换 Key
                last_error = exc
                continue
        raise last_error or LLMError("Agnes 调用失败（无可用 Key 尝试）")

    def _create(self, key: str, system: str, user: str, *, json_mode: bool) -> str:
        client = self._client_for(key)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "timeout": self._timeout,
            **self._thinking_kwargs(),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if json_mode and self._is_response_format_reject(exc):
                kwargs.pop("response_format", None)
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    raise self._classify_and_report(key, exc2) from exc2
            else:
                raise self._classify_and_report(key, exc) from exc
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            self._pool.report_transient_error(key)
            raise LLMError("Agnes 返回空内容")
        self._pool.report_success(key)
        return content

    def chat_vlm(self, system: str, user: str, image_data_url: str) -> str:
        """VLM 直读：图片以 base64 data URL 随消息发送（Agnes 2.5 Flash 已实测支持）。"""
        pool_size = len(self._pool)
        attempts = min(MAX_KEY_ATTEMPTS, pool_size)
        last_error: LLMError | None = None
        for _ in range(attempts):
            key = self._pool.acquire()
            client = self._client_for(key)
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "text", "text": user},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]},
                    ],
                    temperature=0,
                    timeout=self._timeout,
                )
            except Exception as exc:
                last_error = self._classify_and_report(key, exc)
                continue
            content = resp.choices[0].message.content if resp.choices else None
            if not content:
                self._pool.report_transient_error(key)
                last_error = LLMError("Agnes VLM 返回空内容")
                continue
            self._pool.report_success(key)
            return content
        raise last_error or LLMError("Agnes VLM 调用失败（无可用 Key 尝试）")

    @staticmethod
    def _is_response_format_reject(exc: Exception) -> bool:
        return "response_format" in str(exc)

    def _classify_and_report(self, key: str, exc: Exception) -> LLMError:
        """按状态码/文案分类：向池上报并返回类型化错误。"""
        name = type(exc).__name__.lower()
        message = str(exc)
        status = getattr(exc, "status_code", None)
        lowered = message.lower()
        if status == 429 or "rate limit" in lowered or "too many requests" in lowered:
            self._pool.report_rate_limited(key)
            return LLMError(f"Agnes 限流(429)（已换 Key 重试）: {message[:120]}")
        if status in (401, 403):
            self._pool.report_auth_failed(key)
            return LLMError(f"Agnes 认证失败({status})（该 Key 已禁用）: {message[:120]}")
        if status == 402 or any(w in lowered for w in ("quota", "insufficient", "余额", "欠费", "额度")):
            self._pool.report_quota(key)
            return LLMError(f"Agnes 额度不足({status or 402})（该 Key 已长冷却）: {message[:120]}")
        if "timeout" in name or "timed out" in lowered:
            return LLMTimeoutError(f"Agnes 调用超时: {message[:120]}")
        self._pool.report_transient_error(key)
        return LLMError(f"Agnes 调用失败: {type(exc).__name__}: {message[:120]}")
