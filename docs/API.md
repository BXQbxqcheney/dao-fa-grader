# API 文档 — 道法主观题智能判卷服务

> 契约唯一来源：`docs/SPEC.md`（SPEC v0.2，已获用户批准）。本文档与其一致。

## 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/grade` | 单题判卷 |
| GET | `/healthz` | 探活（不检查外部模型） |
| GET | `/` | 重定向到工作台 |
| GET | `/workbench/` | Web 试判工作台（静态页） |

## 判卷 `POST /api/v1/grade`

Content-Type：`application/json`。单次请求只允许一道题。
服务仅识别并记录 `Authorization` 头是否存在（审计用），**不透传给任何上游模型服务**，无业务语义。

### 请求体

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgo...",
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
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image_base64` | string | 是 | 原始 Base64 或 data URL；支持 JPEG/PNG/WebP；解码后 ≤ `MAX_IMAGE_MB`（默认 10 MB）；最长边 > 8192 px 自动等比缩放并标记 `ocr.resized`；格式按**魔数嗅探**校验，不仅信前缀 |
| `criteria.max_score` | number | 是 | 满分；> 0 且为 0.5 的整数倍 |
| `criteria.question_stem` | string | 是 | 题干（用户裁决必填） |
| `criteria.reference_answer` | string | 是 | 参考答案（用户裁决必填） |
| `criteria.scoring_points` | array | 是 | 非空；`id` 非空、`description` 非空、`score` > 0 且 0.5 步长 |
| `criteria.question_id` | string | 否 | 题号，原样回传 |
| `criteria.deduction_rules` | array | 是 | 扣分规则（用户裁决必填；可为空列表） |
| `options.return_ocr_text` | boolean | 否 | 是否回传 OCR 文本，默认 `true` |

### 校验规则（400 `INVALID_ARGUMENT`）

- 必填缺失：`image_base64`、`criteria.max_score`、`criteria.scoring_points`、`criteria.question_stem`、`criteria.reference_answer`、`criteria.deduction_rules`。
- Base64 非法 / 解码失败；魔数不符或声明的类型与魔数不一致（415 单列）。
- `max_score` ≤ 0、`scoring_points` 为空、任一得分点 `score` ≤ 0。
- 任一分值不是 0.5 的整数倍（AC12）。

### 分值口径与扣分语义

- **不因得分点分值之和 ≠ 满分而拒绝**：之和 < 满分属正常（缺额以扣分体现，不自动给出）；之和 > 满分时总分封顶并
  在 `reason` 注明「判卷标准分值之和超过满分」。
- **扣分语义（钉死）**：`score = max_score − Σ deductions.points`（服务端计算，下限 0，上限 max_score）；
  `awarded_score` 为逐点诊断值；一致性约束 `|score − Σawarded| ≤ 0.01`，不符重试一次，仍不符 → 502
  `LLM_PARSE_ERROR`，不得返回伪结果。
- 所有分值恒为 0.5 的整数倍。

### 可判分响应 HTTP 200（status=graded）

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
      "evidence": "“在学校认真学习，参加志愿服务”", "hit": true,
      "awarded_score": 3, "max_score": 4 }
  ],
  "deductions": [ { "reason": "未结合材料中的具体人物事例", "points": 1 } ],
  "reason": "学生答出了理想与行动，但未结合材料中的具体人物事例，扣 1 分，故得 5 分。",
  "ocr": { "text": "整理后的作答文本", "confidence": 0.91, "removed_marks": 2, "resized": false },
  "model": { "provider": "openai", "name": "gpt-4o-mini" }
}
```

### 不可判响应 HTTP 200（status=needs_human）

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

`reason_code` 取值：

| 值 | 判定层 | 含义 |
|---|---|---|
| `low_ocr_confidence` | OCR 层 | 整题置信度（各行最小值）低于阈值 |
| `empty_ocr_text` | OCR 层 | 未识别出有效文本 |
| `garbled_text` | OCR 层 | 行内无有效汉字序列、超长重复串等严重混乱 |
| `illegible_line_split` | 切分层 | 行切分失败 |
| `illegible_after_cleanup` | 整理层 | 整理后仍语义不通/信息残缺 |
| `rubric_insufficient` | 判分层 | 标准结构合法但语义无法解析（如得分点互斥） |

### 系统错误 HTTP 4xx/5xx

```json
{ "error": { "code": "INVALID_ARGUMENT", "message": "…", "request_id": "req_…" } }
```

| HTTP | code | 场景 |
|---|---|---|
| 400 | `INVALID_ARGUMENT` | 参数缺失/非法（含非 0.5 步长） |
| 413 | `IMAGE_TOO_LARGE` | 图片解码后超限 |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | 格式不支持或魔数不符 |
| 502 | `LLM_ERROR` | 大模型调用失败 |
| 502 | `LLM_PARSE_ERROR` | 模型输出经重试仍无法通过 Schema/一致性校验 |
| 504 | `LLM_TIMEOUT` | 大模型超时 |
| 500 | `OCR_ERROR` | OCR 加载/推理失败（含 OCR 超时） |
| 500 | `INTERNAL_ERROR` | 其他内部错误（含请求总超时） |

`needs_human` 与错误结构严格互斥（AC16）。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `anthropic` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | — | OpenAI 兼容接入 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — | Anthropic 接入 |
| `OCR_PROVIDER` | `ofa` | `ofa` / `rapidocr` |
| `OCR_CONFIDENCE_THRESHOLD` | `0.5` | 两引擎同语义使用 |
| `OCR_TIMEOUT_SECONDS` | `10` | OCR 单次预算 |
| `OCR_CONCURRENCY` | `2` | OCR 线程池并发 |
| `OFA_MODEL_PATH` | 自动探测 | OFA 权重目录 |
| `MAX_IMAGE_MB` | `10` | 图片解码后上限 |
| `REQUEST_TIMEOUT_SECONDS` | `60` | 请求总闸 |
| `LLM_TIMEOUT_SECONDS` | `20` | 大模型单次调用超时（重试计入总预算） |
| `DEBUG_LOGGING` | `false` | 调试日志（仍禁止记录图片与全文） |

## 隐私与日志

- 无状态：默认不持久化图片与判分结果。
- 日志白名单：request_id、状态、耗时、confidence、removed_marks、score、provider/model。
- 禁止记录：图片内容、OCR 全文、学生原文、判分理由全文。
