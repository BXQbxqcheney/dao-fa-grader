"""Agnes 多 Key 池：轮询取用 + 冷却 + 失败标记（密钥铁则 4）。

策略（参考库内 QianLoop/core/adapter.py 已验证模式，同步版）：
- 轮询：游标顺序取下一个可用 Key（Least-Recently-Used 语义由 last_used 辅助）；
- 429 限流：指数冷却（30s × 2^连续失败次数，上限 3600s），切下一个 Key；
- 402/欠费：长冷却 1 小时；
- 401/403：Key 无效，永久禁用；
- 其他错误：短冷却 10s（网络抖动）；
- 成功：失败计数清零；
- 全部 Key 冷却/禁用：抛 LLMError（附最短剩余冷却时间），不无限等待。

线程安全：判卷服务的 LLM 调用跑在线程池里，所有状态变更持锁。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from grader.core.errors import LLMError

logger = logging.getLogger("grader.llm.keypool")

COOLDOWN_RATE_LIMIT_BASE = 30.0    # 429 基础冷却（秒），指数翻倍
COOLDOWN_RATE_LIMIT_MAX = 3600.0
COOLDOWN_QUOTA = 3600.0            # 402/欠费冷却
COOLDOWN_TRANSIENT = 10.0          # 网络/服务端抖动短冷却


@dataclass
class _KeyState:
    key: str
    cooldown_until: float = 0.0
    disabled: bool = False
    fail_count: int = 0
    last_used: float = 0.0

    @property
    def tail(self) -> str:
        """日志脱敏：只展示尾 4 位。"""
        return "…" + self.key[-4:] if len(self.key) >= 4 else "…"


class KeyPool:
    def __init__(self, keys: list[str]) -> None:
        cleaned = [k.strip() for k in keys if k and k.strip()]
        if not cleaned:
            raise LLMError("Agnes Key 池为空（未提供任何有效 Key）")
        self._states = [_KeyState(k) for k in cleaned]
        self._lock = threading.Lock()
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._states)

    def acquire(self) -> str:
        """轮询取下一个可用 Key；全部不可用时抛 LLMError（附最短剩余冷却）。"""
        now = time.monotonic()
        with self._lock:
            n = len(self._states)
            for offset in range(n):
                state = self._states[(self._cursor + offset) % n]
                if state.disabled or state.cooldown_until > now:
                    continue
                self._cursor = (self._cursor + offset + 1) % n
                state.last_used = now
                return state.key
            # 全部不可用：报最短剩余冷却，给调用方明确预期
            active = [s for s in self._states if not s.disabled]
            if active:
                wait = min(s.cooldown_until for s in active) - now
                raise LLMError(
                    f"全部 {len(self._states)} 个 Agnes Key 均在冷却中（最快 {max(0.0, wait):.0f}s 后可用，"
                    f"禁用 {n - len(active)} 个），请稍后重试"
                )
            raise LLMError(f"全部 {len(self._states)} 个 Agnes Key 均已禁用（401/403），请检查 Key 有效性")

    def report_success(self, key: str) -> None:
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.fail_count = 0
                    s.cooldown_until = 0.0

    def report_rate_limited(self, key: str) -> None:
        """429：指数冷却并标记失败。"""
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.fail_count += 1
                    wait = min(COOLDOWN_RATE_LIMIT_BASE * (2 ** (s.fail_count - 1)), COOLDOWN_RATE_LIMIT_MAX)
                    s.cooldown_until = time.monotonic() + wait
                    logger.warning("Agnes Key %s 触发 429，冷却 %.0fs（连续第 %d 次）", s.tail, wait, s.fail_count)

    def report_quota(self, key: str) -> None:
        """402/欠费：长冷却。"""
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.fail_count += 1
                    s.cooldown_until = time.monotonic() + COOLDOWN_QUOTA
                    logger.warning("Agnes Key %s 欠费/额度不足，冷却 %.0fs", s.tail, COOLDOWN_QUOTA)

    def report_auth_failed(self, key: str) -> None:
        """401/403：Key 无效，永久禁用。"""
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.disabled = True
                    logger.warning("Agnes Key %s 认证失败（401/403），已永久禁用", s.tail)

    def report_transient_error(self, key: str) -> None:
        """网络/5xx 抖动：短冷却后换下一个 Key。"""
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.fail_count += 1
                    s.cooldown_until = time.monotonic() + COOLDOWN_TRANSIENT

    def snapshot(self) -> dict:
        """诊断快照（脱敏）：可用/冷却/禁用计数。"""
        now = time.monotonic()
        with self._lock:
            disabled = sum(1 for s in self._states if s.disabled)
            cooling = sum(1 for s in self._states if not s.disabled and s.cooldown_until > now)
            return {
                "total": len(self._states),
                "available": len(self._states) - disabled - cooling,
                "cooling": cooling,
                "disabled": disabled,
            }
