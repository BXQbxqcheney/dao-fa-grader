# dao-fa-grader · 道法主观题智能判卷服务

> 版本 **0.2.0**（单点定义于 `pyproject.toml`）｜许可证 **AGPL-3.0-or-later**
> 作者：**白崇义（白小谦）** ｜ 联系：**bcybxq@163.com**
> 手写 OCR / VLM 直读 + 置信度分流 + 大模型依标判分。核心边界：**低置信度不判、不自行修复、每分必有原文依据、0.5 分步长**。
> 计分语义：**得分 = 踩到的得分点得分之和**（`score = clamp(Σawarded − Σdeductions, 0, max)`，deductions 只承载规则性扣分）。

本服务**不是大模型**：通过标准 API 接入你自有的大模型（OpenAI 兼容 / Anthropic），
本地负责图像预处理、行切分、涂改剔除、手写 OCR 与置信度分流；OCR 不可靠时明确返回
`needs_human` 转人工，绝不做猜测性判分。完整契约见 [docs/SPEC.md](docs/SPEC.md) 与 [docs/API.md](docs/API.md)。

## 快速开始

### 0. 一键启动（推荐）

```bat
start.bat
```

自动：安装预检 → 起服务（窗口保留日志）→ 等健康检查 → 打开工作台。`start.bat 9000` 可换端口；`stop.bat [port]` 停止。

前置条件只有两个：① `python -m pip install -e ".[dev]"`；② 把 Agnes Key 文件 `agnes-key.txt`（每行一个 `sk-` Key）放在仓库根或上级目录——**默认即用 Agnes + VLM 直读，无需其他配置**。

### 1. 环境准备（手动方式）

```bash
# Python ≥ 3.10（推荐 3.12）；GPU 推荐但非必需
cd dao-fa-grader
python -m pip install -e ".[dev]"        # 基础依赖 + 测试依赖
python -m pip install -e ".[ofa]"        # 可选：使用 OFA 中文手写 OCR 时安装（torch/modelscope/transformers）
cp .env.example .env                     # 默认已面向 Agnes；按需改
```

OFA 模型权重（约 700MB）**不随仓库分发**，两种获取方式任选：

```bash
# 方式一：一键下载脚本（断点续传，含 SSRF 防护，默认下载到仓库内 model_assets/）
python scripts/download_model.py

# 方式二：浏览器直接下载 pytorch_model.bin（约 707MB）放入 model_assets/
#        （config.json 等配置文件已随仓库 model_assets/ 分发，无需另行下载）：
# https://modelscope.cn/models/iic/ofa_ocr-recognition_handwriting_base_zh/resolve/master/pytorch_model.bin
```

或用环境变量 `OFA_MODEL_PATH` 指向任意本地权重目录。

### 2. 启动

```bash
uvicorn grader.main:app --host 0.0.0.0 --port 8000
```

- 工作台：浏览器打开 <http://127.0.0.1:8000/workbench/>（上传图片 → 填判卷标准 → 看结果）。
- 健康检查：`curl http://127.0.0.1:8000/healthz` → `{"status":"ok"}`

### 3. 一次完整的 curl 调用

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/grade \
  -H 'Content-Type: application/json' \
  -d @- <<'EOF'
{
  "image_base64": "data:image/png;base64,<你的作答图Base64>",
  "criteria": {
    "question_id": "Q18",
    "max_score": 6,
    "question_stem": "结合材料，谈谈青少年应如何将个人梦想与国家发展结合起来。",
    "reference_answer": "①树立远大理想；②结合材料说明具体行动。",
    "scoring_points": [
      { "id": "P1", "description": "指出树立远大理想的重要性", "score": 2 },
      { "id": "P2", "description": "结合材料说明具体行动", "score": 4 }
    ],
    "deduction_rules": ["未结合材料扣 1 分", "照抄题目不给分"]
  },
  "options": { "return_ocr_text": true }
}
EOF
```

### 4. 返回示例

可判（`status=graded`）：

```json
{
  "request_id": "req_20261107_9f3a2b1c",
  "status": "graded",
  "question_id": "Q18",
  "max_score": 6,
  "score": 5,
  "scoring_points": [
    { "id": "P1", "criterion": "指出树立远大理想的重要性",
      "evidence": "“我们要树立远大理想，把个人梦想和国家发展结合起来”", "hit": true,
      "awarded_score": 2, "max_score": 2 },
    { "id": "P2", "criterion": "结合材料说明具体行动",
      "evidence": "“在学校认真学习，参加志愿服务”", "hit": true, "awarded_score": 3, "max_score": 4 }
  ],
  "deductions": [ { "reason": "未结合材料中的具体人物事例", "points": 1 } ],
  "reason": "学生答出了理想与行动，但未结合材料中的具体人物事例，扣 1 分，故得 5 分。",
  "ocr": { "text": "整理后的作答文本", "confidence": 0.91, "removed_marks": 2, "resized": false },
  "model": { "provider": "openai", "name": "gpt-4o-mini" }
}
```

不可判（`status=needs_human`，HTTP 200，原图回传）：

```json
{
  "request_id": "req_20261107_7c1d0e22",
  "status": "needs_human",
  "reason_code": "low_ocr_confidence",
  "message": "OCR 置信度 0.38 低于阈值 0.5，需人工判卷",
  "image_url": "data:image/png;base64,iVBORw0KGgo...",
  "ocr": { "text": "识别片段", "confidence": 0.38, "removed_marks": 0, "resized": false }
}
```

系统错误（4xx/5xx，与 needs_human 严格互斥）：`{"error": {"code": "…", "message": "…", "request_id": "…"}}`。
全部错误码见 [docs/API.md](docs/API.md)。

## 模型配置说明

| 变量组                                                                     | 说明                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `LLM_PROVIDER=agnes \| openai \| anthropic`                             | 默认 **agnes**（放好 agnes-key.txt 即零配置）                                |
| `OPENAI_API_KEY` `OPENAI_BASE_URL` `OPENAI_MODEL`                       | OpenAI 兼容接入（兼容网关亦可）                                                |
| `ANTHROPIC_API_KEY` `ANTHROPIC_MODEL`                                   | Anthropic 接入                                                       |
| `AGNES_KEY_FILE`（每行一个 sk- Key）或 `AGNES_API_KEYS`（逗号分隔）/ `AGNES_API_KEY` | Agnes 多 Key 轮询池；Key 文件默认自动探测仓库根/上级 `agnes-key.txt`                 |
| `AGNES_BASE_URL` `AGNES_MODEL`                                          | 默认 `https://api.agnes-ai.cn/v1` / `agnes-2.5-flash`                |
| `AGNES_VLM_MODE=true`（默认开）                                              | **VLM 直读**：AIOCR 读图转写（不确定的字不说）→ AI 识别（面目全非返回 MMQF）→ 判分；跳过 OCR 置信度闸 |
| `OCR_PROVIDER=ofa \| rapidocr`                                          | OCR 引擎切换（均完整实现；VLM 模式下不使用）                                         |
| `OCR_CONFIDENCE_THRESHOLD=0.5`                                          | 两引擎同语义使用，低于即转人工                                                    |
| `OFA_MODEL_PATH`                                                        | OFA 权重目录（默认探测仓库内 `model_assets/`，其次仓库同级 `ocr/`）                    |
| `LLM_DISABLE_THINKING=true`                                             | 思考型模型（Qwen3 系）追加 /no_think 关闭思考模式                                  |

密钥只经环境变量注入，禁止写入代码、示例与日志。切换供应商只需改环境变量（AC7）。

## OCR 引擎与 OFA 实测结论（M2a）

| 引擎         | 默认   | 说明                                                        |
| ---------- | ---- | --------------------------------------------------------- |
| `rapidocr` | ✅ 默认 | onnxruntime 轻量引擎，零额外依赖，随装随用                               |
| `ofa`      | 可选   | ModelScope ofa-ocr-recognition 手写模型（Apache-2.0），需 fairseq |

**OFA 兼容性实测结论（2026-11-07，Windows / Python 3.12 / modelscope 1.40.1）**：

1. modelscope 的 OFA 实现硬依赖 `fairseq`（`multihead_attention.py` 等多处 `from fairseq import …`）；
2. fairseq 在 PyPI 无 Windows/py≥3.11 预编译 wheel，源码构建在本机失败（缺 MSVC 工具链，且其
   omegaconf/hydra 依赖链与 py3.12 不兼容）；
3. 同时存在版本三角：modelscope 1.40.x 需要 hub≥1.x 的符号（`BucketNotFoundError` 等），
   而 OFA 钉板 tokenizers<0.21 ⇒ transformers 4.46.x ⇒ hub<1.0，包级无解。

**结论**：Windows/py3.12 部署请使用默认 `rapidocr`；具备 fairseq 的环境（如 Linux + py3.10）
可用 `OCR_PROVIDER=ofa` 启用手写识别（权重约 707MB，一键下载见上文，置信度走 generate
token 概率几何平均）。适配器代码保留且测试覆盖。

## 硬件与性能预期

- **GPU 推荐**：OFA 为 700MB 视觉-语言模型，单行推理在 RTX 4060 级 GPU 上约百毫秒级；
  **CPU 可运行**但逐行推理显著变慢（秒级/行）。
- 超时预算：OCR ≤ 10s（`OCR_TIMEOUT_SECONDS`）、大模型单次 ≤ 20s（`LLM_TIMEOUT_SECONDS`）、
  请求总闸 60s（`REQUEST_TIMEOUT_SECONDS`）；OCR 超时 → 500 `OCR_ERROR`，大模型超时 → 504 `LLM_TIMEOUT`。
- OCR 线程池并发默认 2（`OCR_CONCURRENCY`）；大模型调用异步。
- 60s 总预算内最多发生：1 次整理 + 2 次判分（含一次重试）；若整体超时返回 500 `INTERNAL_ERROR`。

## 判分语义（重要）

- `score = max_score − Σdeductions`，服务端计算，恒在 `[0, max_score]`；
  `awarded_score` 为逐点诊断值，一致性约束 `Σawarded = 满分 − Σdeductions`（不符自动重试一次）。
- 得分点分值之和 ≠ 满分**不会**被拒绝：之和小于满分属正常（缺额以扣分体现，不自动给出）；
  之和超过满分时封顶并在 `reason` 注明。
- 所有分值为 0.5 的整数倍，其余小数直接 400。
- 学生作答按不可信数据处理（`<student_answer>` 隔离注入 + 固定声明），提示词注入不生效。

## 测试

```bash
python -m pip install -e ".[dev]"
python -m pytest            # 单元 + API 契约（Fake OCR/LLM，无外部依赖）
RUN_OFA_TESTS=1 python -m pytest tests/test_engines_integration.py -k ofa   # OFA 实测（需 fairseq 可用环境）
```

## 目录结构

见 [docs/SPEC.md](docs/SPEC.md) §7。工程要点：配置单源（`core/config.py`）、错误体系统一
（`core/errors.py`）、JSON 多级容错解析（`grading/parser.py`）、双 OCR 统一 0~1 置信度口径
（`ocr/base.py`）、依赖注入可测试（`main.create_app`）。

## 复用优先走查记录

【复用优先走查】需求：道法主观题 LLM 判卷服务（OCR + 置信度分流 + 依标判分）。
第 1 级 本地库：`find.py` 检索 判卷/判分/OCR/手写识别/图像识别/FastAPI/Python服务 —— 仅命中
无关的「隐私打码 OCR 工具」（新小工具集合），无 ≥70% 契合资产。未命中。
第 2 级 开源：EvaluAI（整平台）、Moodle 手写评分流水线（LMS 集成）、AWESOME-OCR-LLM（清单）、
olmOCR（文档 OCR）—— 均为整平台/整 LMS 形态，与「单题 HTTP + needs_human 分流」契合度不足。未采用。
结论：第 3 级自研本仓库；工程模式借鉴（JSON 多级容错链、密钥仅环境变量）为通用实践移植。

## 许可证与作者

**AGPL-3.0-or-later**，全文见 [LICENSE](LICENSE)。

- 作者：白崇义（白小谦）<bcybxq@163.com>
- 模型权重不随仓库分发；示例与测试不包含任何真实密钥、学校信息或学生个人信息。
- 本程序按 AGPL-3.0-or-later 授权，不提供任何担保；再分发时须遵守 AGPL 条款
  （包括网络交互时提供完整源代码）。
