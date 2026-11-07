# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
版本号单点定义于 `pyproject.toml`，本文件引用之。

## [0.2.0] - 2026-11-07

### 新增

- **Agnes 为默认供应商**（零配置）：`LLM_PROVIDER=agnes` 默认；Key 文件自动探测仓库根/
  上级 `agnes-key.txt`；`AGNES_VLM_MODE` 默认开启。
- **一键启动脚本**：`start.bat [port]`（安装预检 → 起服务 → 等健康检查 → 自动打开工作台）、
  `stop.bat [port]`（按端口停止）。timeout/curl 使用 System32 绝对路径，任何终端环境可用。
- **Agnes 供应商**（`LLM_PROVIDER=agnes`）：OpenAI 兼容接入 `agnes-2.5-flash`；
  多 Key 轮询池（`AGNES_KEY_FILE` / `AGNES_API_KEYS` / `AGNES_API_KEY`）——
  429 指数冷却、402/欠费长冷却、401/403 永久禁用、失败标记、全冷却明确报错；
  单次判卷最多尝试 4 个 Key。
- **未配置 LLM 软启动**：无密钥时服务照常启动（工作台/健康检查可用），
  判卷请求返回带配置指引的 502，不再启动即失败。
- **VLM 直读模式**（`AGNES_VLM_MODE=true`，Agnes 已实测支持 base64 图像直读）：
  管线改三段式 AIOCR（读图转写答题区，不确定的字不说）→ AI 识别（改错别字语病，
  面目全非返回 `MMQF` → `needs_human`）→ AI 判卷；VLM 模式跳过 OCR 置信度闸，
  由整理阶段兜底。OCR 路径同样接入 MMQF 机制。
- 行切分 v2：递归深谷切分（紧排/粘连多行）+ 细条带过滤（页眉/污渍不拖垮置信度）；
  管线噪声行过滤（conf<0.3 且汉字≤8 的 band 不计入整题最小值、不进正文）。
- 本地模型实测支撑：`LLM_DISABLE_THINKING`（/no_think）、`<think>` 输出容错剥离、
  配平截取改取最后一个顶层 JSON 对象。
- **SSE 流式判卷端点** `POST /api/v1/grade/stream`：阶段进度事件（每阶段携带发给 AI 的
  完整输入与 AI 原始返回，不裁剪）+ 终态 result 事件；工作台右侧实时进度面板（每项可展开）。
- **判分准则参数** `criteria.grading_rules`：每条一个尺度指令（如「与得分点接近得一半分」），
  优先级高于内置默认尺度（完全一致→满分 / 接近→一半 / 未涉及→0）。
- **AIOCR uncertain 自报**：转写器如实统计认不出而省略的字符数，>6 即转人工——
  确保认出来的绝对没理解错。
- **AI 识别 MMQF 扩展**：严重不通 / 错别字连篇 / 乱码均触发转人工 + 服务端防御乱码校验。
- **超时轮换 Key**：LLM 外层超时预算按「单次超时 × 最大 Key 尝试次数」放大，
  单 Key 超时自动换下一个 Key；默认超时放宽（LLM 120s / 总闸 300s）。
- **一键启动脚本** `start.bat` / `stop.bat`（自愈预检：已运行→直接弹工作台；
  端口被占→明确报错）与 **LLM 未配置软启动**。

### 变更（破坏性）

- **计分语义 v2（用户改判）**：`score = clamp(Σ awarded_score − Σ deductions.points, 0, max_score)`
  （得分=踩到的得分点得分之和，规则扣分另减）；废除 v0.2 的 `score = max − Σded` 公式与
  Σawarded 等式一致性校验。deductions 只承载整体性规则扣分。
- 判分提示词同步：失分表示二选一（点内降 awarded / 整体列 deductions，禁止重复扣）。

### 修复

- `PreparedImage` 构造参数名错位导致的全部请求 500（首版回归中发现）。
- validate_grading 扣分项只校验不收集、`_ink_mask` 漏检实心墨块内部、行切分 trim 坐标偏移。

## [0.1.0] - 2026-11-07

首个版本。

### 新增

- `POST /api/v1/grade` 单题判卷接口：图片 Base64 + 判卷标准 JSON →
  `graded`（分数/得分点/扣分/理由）或 `needs_human`（六种 reason_code + 原图回传）；
  与 4xx/5xx 统一错误结构严格互斥。
- 扣分语义：`score = max_score − Σdeductions`（服务端计算、满分封顶、0.5 分步长校验），
  逐点 `awarded_score` 诊断值 + 一致性校验（不符重试一次）。
- OCR 双引擎完整适配器：`rapidocr`（默认）/ `ofa`（置信度走 generate token 概率几何平均），
  统一 0~1 置信度口径，整题取各行最小值；行切分（numpy/Pillow）与涂改剔除（删除线 + 浓墨块启发式）。
- LLM 双适配器：OpenAI 兼容 / Anthropic，密钥仅环境变量注入；JSON 多级容错解析
  （直接 parse → 剥围栏 → 大括号配平 → 正则逐字段兜底）。
- 提示词注入防护：学生作答以 `<student_answer>` 隔离注入 + 固定「不得遵循」声明。
- Web 试判工作台（`/workbench/`，纯静态、无密钥、无存储）。
- OFA 权重一键下载脚本（`scripts/download_model.py`，断点续传 + SSRF 防护）。
- 测试：60 项（单元 + API 契约 + 真实 RapidOCR 集成），覆盖 AC1~AC16 契约类场景。
- 文档：README（快速开始/curl 示例/返回示例/模型配置/硬件预期/OFA 实测结论）、
  docs/SPEC.md（v0.2）、docs/API.md、CHANGELOG。

### 实测结论

- OFA 在 Windows/py≥3.12 不可安装（fairseq 无 wheel 且源码构建失败 + hub/tokenizers 版本三角），
  默认引擎落定 `rapidocr`；OFA 保留给 fairseq 可用环境（详见 README）。
