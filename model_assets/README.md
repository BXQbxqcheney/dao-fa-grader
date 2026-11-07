# model_assets/ — OFA 模型配置目录

本目录存放 OFA 手写识别模型的**配置与词表文件**（非权重，随仓库分发）：

| 文件 | 用途 |
|---|---|
| `config.json` | 模型结构配置（transformers 加载用） |
| `configuration.json` | ModelScope 模型元信息（含 allow_remote 声明） |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.txt` | 分词器 |

权重 `pytorch_model.bin`（约 707MB）**不入库**，由 `python scripts/download_model.py`
一键下载到本目录（默认目标），或用 `OFA_MODEL_PATH` 指向其他位置。

注意：`.gitignore` 中 `*.bin` 规则确保权重永不被提交；`OCR_PROVIDER=ofa` 需要 fairseq
可用的环境（Windows/py≥3.11 实测不可装，详见 README「OFA 实测结论」）。
