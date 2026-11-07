"""道法主观题智能判卷服务 —— Python 客户端示例。

本脚本会：
1. 用 PIL 生成一张三行中文文本的示例作答图（无需任何真实学生数据）；
2. 组装判卷标准 JSON；
3. 调用 POST /api/v1/grade 并打印可判 / 不可判两种返回。

安全约定：脚本仅允许调用本机服务（127.0.0.1 / localhost / ::1），
拒绝其他目标与非 HTTP(S) 协议，且不跟随重定向 —— 防止被当作 SSRF 跳板。
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
ALLOWED_SCHEMES = {"http", "https"}

CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def validate_base_url(raw: str) -> str:
    """仅允许本机地址与 HTTP(S)；解析结果必须为环回地址，防 SSRF。"""
    parts = urlsplit(raw)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"仅允许 {sorted(ALLOWED_SCHEMES)} 协议，收到 {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise ValueError("URL 缺少主机名")
    if host.lower() not in ALLOWED_HOSTS:
        raise ValueError(f"示例脚本仅允许本机服务（{sorted(ALLOWED_HOSTS)}），收到 {host!r}")
    ip = ipaddress.ip_address(socket.gethostbyname(host))
    if not ip.is_loopback:
        raise ValueError(f"主机 {host} 解析结果不是环回地址")
    return raw.rstrip("/")


def render_sample_image() -> str:
    """渲染示例作答图并编码为 data URL。"""
    lines = [
        "我们要树立远大理想，把个人梦想和国家发展结合起来。",
        "在学校认真学习，积极参加志愿服务活动。",
        "努力学习科学文化知识，锻炼强健体魄。",
    ]
    font = None
    for path in CJK_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, 30)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    line_h = 30 + 26
    img = Image.new("L", (760, line_h * len(lines) + 26), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((20, 26 + i * line_h), text, fill=0, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64


def build_payload() -> dict:
    return {
        "image_base64": render_sample_image(),
        "criteria": {
            "question_id": "Q18",
            "max_score": 6,
            "question_stem": "结合材料，谈谈青少年应如何将个人梦想与国家发展结合起来。",
            "reference_answer": "①树立远大理想；②结合材料说明具体行动。",
            "scoring_points": [
                {"id": "P1", "description": "指出树立远大理想的重要性", "score": 2},
                {"id": "P2", "description": "结合材料说明具体行动", "score": 4},
            ],
            "deduction_rules": ["未结合材料扣 1 分", "照抄题目不给分"],
        },
        "options": {"return_ocr_text": True},
    }


def post_grade(base_url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(
        url=base_url + "/api/v1/grade",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(req, timeout=70) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main() -> int:
    base_url = validate_base_url(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
    status, body = post_grade(base_url, build_payload())

    if body.get("status") == "graded":
        print(f"[可判] HTTP {status}  score={body['score']}/{body['max_score']}")
        for p in body["scoring_points"]:
            mark = "命中" if p["hit"] else "未命中"
            print(f"  {p['id']} {mark} {p['awarded_score']}/{p['max_score']}  依据：{p['evidence'] or '—'}")
        for d in body["deductions"]:
            print(f"  扣 {d['points']} 分：{d['reason']}")
        print(f"  理由：{body['reason']}")
        print(f"  OCR 置信度：{body['ocr']['confidence']}")
    elif body.get("status") == "needs_human":
        print(f"[需人工] HTTP {status}  reason_code={body['reason_code']}")
        print(f"  说明：{body['message']}")
    else:
        print(f"[错误] HTTP {status}  {json.dumps(body, ensure_ascii=False)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
