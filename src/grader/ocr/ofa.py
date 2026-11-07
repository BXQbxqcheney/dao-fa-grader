"""OFA 中文手写 OCR 适配器（ModelScope ofa-ocr-recognition）。

单行输入模型：由管线完成行切分后逐行调用本适配器。
置信度口径（SPEC §4.0）：该行生成序列 token 概率的几何平均 exp(平均对数概率)，
通过 generate(return_dict_in_generate=True, output_scores=True) 取逐步 logits 计算。

加载策略链（对 ModelScope 版本差异做防御，加载失败抛 OCRError，链路记录日志）：
  A. 直接 from_pretrained 加载 OFAModel + OFAPreprocessor；
  B. 经 modelscope pipeline 拿到底层 model / preprocessor / tokenizer。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch
from PIL import Image

from grader.core.errors import OCRError
from grader.ocr.base import OCRLine

logger = logging.getLogger("grader.ocr.ofa")

PROMPT = "图片中的文字是什么？"
MAX_NEW_TOKENS = 64


def _shim_huggingface_hub() -> None:
    """版本三角兼容垫片（实测）。

    modelscope 1.40.1 从 huggingface_hub.errors 导入 BucketNotFoundError（hub≥1.x 才有），
    而 transformers 4.46.3（OFA 模块钉板 tokenizers<0.21 所需）要求 hub<1.0。
    此垫片仅在缺失时补一个等价异常别名，行为等价、进程内生效，不改动环境。
    """
    try:
        import huggingface_hub.errors as hub_errors

        if not hasattr(hub_errors, "BucketNotFoundError"):
            hub_errors.BucketNotFoundError = type(
                "BucketNotFoundError", (FileNotFoundError,), {"__doc__": "compat shim for modelscope<->hub<1.0"}
            )
    except ImportError:  # pragma: no cover
        pass


class OFAAdapter:
    """懒加载适配器：构造不加载权重，首次 recognize_lines 时加载。"""

    name = "ofa"

    def __init__(self, model_path: str) -> None:
        if not model_path:
            raise OCRError("OFA_MODEL_PATH 未配置且未在默认位置找到 OFA 模型权重（config.json/pytorch_model.bin）")
        self._model_path = model_path
        self._model: Any | None = None
        self._preprocess: Callable[..., dict] | None = None
        self._tokenizer: Any | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    # ---------- 加载 ----------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        last_error: Exception | None = None
        for strategy in (self._load_direct, self._load_via_pipeline):
            try:
                strategy()
            except Exception as exc:  # 策略链：逐个尝试，全部失败才报错
                logger.warning("OFA 加载策略失败: %s: %s", type(exc).__name__, exc)
                last_error = exc
            else:
                self._loaded = True
                logger.info("OFA 模型加载成功: %s (device=%s)", self._model_path, self._device)
                return
        raise OCRError(f"OFA 模型加载失败（已尝试全部策略）: {last_error}")

    def _load_direct(self) -> None:
        from modelscope.models.multi_modal.ofa import OFAModel
        from modelscope.preprocessors import OFAPreprocessor

        model = OFAModel.from_pretrained(self._model_path)
        preprocessor = OFAPreprocessor(model_path)
        self._model = model.to(self._device).eval()
        self._preprocess = lambda image: preprocessor((PROMPT, image))  # noqa: E731
        self._tokenizer = getattr(preprocessor, "tokenizer", None)
        if self._tokenizer is None:
            raise RuntimeError("OFAPreprocessor 未携带 tokenizer，无法解码输出")

    def _load_via_pipeline(self) -> None:
        _shim_huggingface_hub()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        # trust_remote_code=True：OFA 的 configuration.json（allow_remote=true）要求显式信任；
        # 仅对部署方自行提供并信任的本地权重目录开启（README 有安全说明）。
        #
        # 依赖隔离（实测）：allow_remote 会触发 modelscope 自动 pip 安装模型目录的
        # requirements.txt（钉 transformers==4.48.3），与 modelscope OFA 模块自身
        # 钉板（tokenizers>=0.20,<0.21，即 transformers 4.46.x）冲突。此处进程内
        # 临时隔离该自动安装（恢复原函数），不修改用户的权重目录，不污染环境。
        import modelscope.utils.plugins as ms_plugins

        original_install = ms_plugins.install_requirements_by_files
        ms_plugins.install_requirements_by_files = lambda *_a, **_k: logger.warning(
            "已隔离 modelscope 对模型目录 requirements.txt 的自动安装（与 OFA 模块钉板冲突）"
        )
        try:
            pipe = pipeline(
                Tasks.ocr_recognition,
                model=self._model_path,
                device=str(self._device),
                trust_remote_code=True,
            )
        finally:
            ms_plugins.install_requirements_by_files = original_install
        model = getattr(pipe, "model", None)
        preprocessor = getattr(pipe, "preprocessor", None)
        if model is None or preprocessor is None:
            raise RuntimeError("modelscope pipeline 未暴露 model/preprocessor")
        self._model = model.to(self._device).eval()
        self._preprocess = lambda image: preprocessor((PROMPT, image))  # noqa: E731
        tokenizer = getattr(preprocessor, "tokenizer", None) or getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("pipeline/preprocessor 均未携带 tokenizer")
        self._tokenizer = tokenizer

    # ---------- 推理 ----------

    def recognize_lines(self, line_images: list[Image.Image]) -> list[OCRLine]:
        self._ensure_loaded()
        return [self._recognize_one(img) for img in line_images]

    def _recognize_one(self, image: Image.Image) -> OCRLine:
        inputs = self._preprocess(image)
        inputs = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                num_beams=1,
                max_length=MAX_NEW_TOKENS + 8,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = outputs.sequences  # [batch, start+T]
        scores = outputs.scores  # tuple[T]，每个 [batch, vocab]
        if sequences.shape[0] == 0 or sequences.shape[1] <= 1:
            return OCRLine(text="", confidence=1.0)
        tokens = sequences[0, 1:]  # 逐步生成 token（去掉 decoder 起始符）

        # 序列置信度：各生成 token 的几何平均概率
        log_probs: list[float] = []
        for step, token_id in enumerate(tokens.tolist()):
            if step >= len(scores):
                break
            logit = scores[step][0]
            log_prob = torch.log_softmax(logit.float(), dim=-1)[token_id].item()
            log_probs.append(log_prob)

        text = self._decode(sequences)
        if not log_probs:
            return OCRLine(text=text.strip(), confidence=1.0)
        # 截断到实际生成长度（平均对数概率 → 几何平均概率）
        confidence = math.exp(sum(log_probs) / len(log_probs))
        return OCRLine(text=text.strip(), confidence=max(0.0, min(1.0, confidence)))

    def _decode(self, sequences: torch.Tensor) -> str:
        text = self._tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return text[0] if text else ""
