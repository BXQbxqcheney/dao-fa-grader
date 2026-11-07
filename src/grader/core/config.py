"""配置单源：所有环境变量与超时预算在此读取，其他模块一律 import 本模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """轻量 .env 加载：KEY=VALUE，不覆盖已存在的环境变量（真实环境优先）。

    查找顺序：显式 path → 当前工作目录 → 仓库根（src/grader/core 向上 4 层）。
    """
    candidates = [path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]
    for candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError as exc:
            raise ValueError(f"读取 .env 失败: {candidate} ({exc})") from exc
        return


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 不是合法数字: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 不是合法整数: {raw!r}") from exc


def _default_ofa_model_path() -> str:
    """OFA 权重目录：优先环境变量；随后探测仓库内 model_assets/ 与包外相邻的 ocr/。"""
    env = os.getenv("OFA_MODEL_PATH", "").strip()
    if env:
        return env
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "model_assets",             # 仓库根 model_assets/（configs 随库分发 + 下载的 bin）
        here.parents[4] / "ocr",                      # 仓库同级 ocr/（部署方整目录提供）
    ]
    for path in candidates:
        if (path / "config.json").is_file() and (path / "pytorch_model.bin").is_file():
            return str(path)
    return ""


def _default_agnes_key_file() -> str:
    """Agnes Key 文件自动探测：环境变量优先，其次仓库根/仓库同级 agnes-key.txt。"""
    env = os.getenv("AGNES_KEY_FILE", "").strip()
    if env:
        return env
    here = Path(__file__).resolve()
    for candidate in (here.parents[3] / "agnes-key.txt", here.parents[4] / "agnes-key.txt"):
        if candidate.is_file():
            return str(candidate)
    return ""


@dataclass(frozen=True)
class Settings:
    """运行时配置（不可变）。创建一次后全局共享。"""

    # 大模型（默认 Agnes：放好 agnes-key.txt 即零配置；可经 LLM_PROVIDER 切换）
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "agnes").strip().lower())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    # 思考型模型（Qwen3 系等）：true 时在 system 末尾追加 /no_think 软开关关闭思考模式
    llm_disable_thinking: bool = field(default_factory=lambda: _env_bool("LLM_DISABLE_THINKING", False))

    # Agnes（OpenAI 兼容 + 多 Key 轮询池 + VLM 直读）
    agnes_api_key: str = field(default_factory=lambda: os.getenv("AGNES_API_KEY", ""))
    agnes_api_keys: str = field(default_factory=lambda: os.getenv("AGNES_API_KEYS", ""))
    agnes_key_file: str = field(default_factory=_default_agnes_key_file)
    agnes_base_url: str = field(default_factory=lambda: os.getenv("AGNES_BASE_URL", "https://api.agnes-ai.cn/v1"))
    agnes_model: str = field(default_factory=lambda: os.getenv("AGNES_MODEL", "agnes-2.5-flash"))
    # VLM 直读模式：AIOCR 阶段直接把作答图片交给 Agnes 读图转写（跳过 OCR 与置信度闸）
    agnes_vlm_mode: bool = field(default_factory=lambda: _env_bool("AGNES_VLM_MODE", True))
    # Agnes Thinking 模式（chat_template_kwargs.enable_thinking），判分推理质量更好
    agnes_thinking: bool = field(default_factory=lambda: _env_bool("AGNES_THINKING", True))

    # OCR
    # 默认 rapidocr（M2a 实测结论）：OFA 依赖 fairseq，Windows/py≥3.11 无预编译 wheel 且
    # 源码构建失败；具备 fairseq 的环境（如 Linux）仍可用 OCR_PROVIDER=ofa 启用。
    ocr_provider: str = field(default_factory=lambda: os.getenv("OCR_PROVIDER", "rapidocr").strip().lower())
    ocr_confidence_threshold: float = field(default_factory=lambda: _env_float("OCR_CONFIDENCE_THRESHOLD", 0.5))
    ocr_timeout_seconds: float = field(default_factory=lambda: _env_float("OCR_TIMEOUT_SECONDS", 10.0))
    ocr_concurrency: int = field(default_factory=lambda: _env_int("OCR_CONCURRENCY", 2))
    ofa_model_path: str = field(default_factory=_default_ofa_model_path)

    # 请求限制与超时预算（VLM 读图/本地大模型较慢，默认放宽：LLM 120s / 总闸 300s）
    max_image_mb: float = field(default_factory=lambda: _env_float("MAX_IMAGE_MB", 10.0))
    request_timeout_seconds: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT_SECONDS", 300.0))
    llm_timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 120.0))

    # 调试
    debug_logging: bool = field(default_factory=lambda: _env_bool("DEBUG_LOGGING", False))

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_mb * 1024 * 1024)


def load_settings() -> Settings:
    """读取当前环境变量生成配置；非法值在此统一报错，避免散落各处。"""
    load_dotenv()
    settings = Settings()
    if settings.llm_provider not in {"openai", "anthropic", "agnes"}:
        raise ValueError(f"LLM_PROVIDER 不支持: {settings.llm_provider}（可选 openai / anthropic / agnes）")
    if settings.ocr_provider not in {"ofa", "rapidocr"}:
        raise ValueError(f"OCR_PROVIDER 不支持: {settings.ocr_provider}（可选 ofa / rapidocr）")
    if not 0 < settings.ocr_confidence_threshold <= 1:
        raise ValueError("OCR_CONFIDENCE_THRESHOLD 必须在 (0, 1] 区间")
    return settings
