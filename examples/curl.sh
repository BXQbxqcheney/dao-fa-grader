#!/usr/bin/env bash
# 道法主观题智能判卷服务 —— curl 调用示例
# 用法：把作答图片转为 Base64 或 data URL 后替换 <IMAGE_BASE64_OR_DATA_URL>
# 启动服务：uvicorn grader.main:app --host 0.0.0.0 --port 8000

set -euo pipefail

IMAGE='PASTE_IMAGE_DATA_URL_OR_BASE64_HERE'

curl -sS -X POST http://127.0.0.1:8000/api/v1/grade \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
{
  "image_base64": "${IMAGE}",
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

echo
echo "# 健康检查：curl http://127.0.0.1:8000/healthz"
