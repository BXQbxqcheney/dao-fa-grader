"""OFA 手写识别模型权重一键下载脚本。

模型来源（ModelScope，Apache-2.0）：
https://modelscope.cn/models/iic/ofa_ocr-recognition_handwriting_base_zh

用法：
    python scripts/download_model.py                # 下载到仓库内 model_assets/（配置文件已在，权重不入仓库）
    python scripts/download_model.py --dest <目录>  # 自定义目录（配合 OFA_MODEL_PATH 使用）

安全约定（强制）：
- 仅允许 https；目标主机仅允许 modelscope.cn 及其重定向目标（逐一校验）；
- 每次请求前 DNS 解析并拒绝环回 / 私网 / 链路本地 / 保留地址（防 SSRF）；
- 重定向目标必须再次通过全部校验，禁止跨到未验证主机。
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

BASE = "https://modelscope.cn/models/iic/ofa_ocr-recognition_handwriting_base_zh/resolve/master"
ALLOWED_SCHEMES = {"https"}
# 允许的最终下载主机（ModelScope 页面域；重定向到 CDN 时按地址校验兜底）
ALLOWED_HOSTS = {"modelscope.cn", "www.modelscope.cn"}

# 该 ModelScope 仓库仅托管权重文件（用户提供信息，2026-11-07 实测确认）；
# config/tokenizer 等配置文件随本仓库 model_assets/ 分发（非权重，可入库）。
FILES = [
    "pytorch_model.bin",
]


def _check_ip(host: str) -> None:
    """DNS 解析后拒绝一切非公网地址（环回/私网/链路本地/保留）。"""
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"目标 {host} 解析到非公网地址 {ip}，已拒绝（防 SSRF）")


def validate_url(url: str) -> None:
    """仅允许 https + 已知主机 + 公网解析结果；失败即抛异常。"""
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"仅允许 {sorted(ALLOWED_SCHEMES)}，收到 {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise ValueError("URL 缺少主机名")
    if host.lower() not in ALLOWED_HOSTS:
        raise ValueError(f"主机 {host!r} 不在允许清单 {sorted(ALLOWED_HOSTS)} 中")
    _check_ip(host)


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """重定向目标必须再次通过完整校验，否则中止。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dest: Path) -> None:
    validate_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：已有部分文件则从其大小继续
    offset = dest.stat().st_size if dest.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    opener = urllib.request.build_opener(_GuardedRedirect)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(req, timeout=60) as resp:
            total_hdr = resp.headers.get("Content-Length")
            total = int(total_hdr) + offset if total_hdr else None
            mode = "ab" if offset and resp.status == 206 else "wb"
            done = offset if mode == "ab" else 0
            with dest.open(mode) as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r  {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                              end="", flush=True)
        print(f"\r  {dest.name}: 完成（{dest.stat().st_size / 1e6:.1f} MB）")
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:  # 已完整下载
            print(f"  {dest.name}: 已存在且完整，跳过")
            return
        raise RuntimeError(f"下载失败 HTTP {exc.code}: {url}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="OFA 手写识别模型一键下载（仅权重；配置已随仓库 model_assets/ 分发）")
    parser.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "model_assets"),
                        help="权重目标目录（默认仓库内 model_assets/，与配置文件同处）")
    args = parser.parse_args()
    dest_dir = Path(args.dest)

    print(f"目标目录: {dest_dir}")
    for name in FILES:
        url = f"{BASE}/{name}"
        target = dest_dir / name
        if target.is_file() and target.stat().st_size > 100 * 1024 * 1024:
            print(f"  {name}: 已存在且完整，跳过")
            continue
        print(f"下载 {url}")
        try:
            download(url, target)
        except Exception as exc:  # 兜底不吞错：失败即退出并保留已下载部分（支持续传重跑）
            print(f"\n[失败] {name}: {exc}", file=sys.stderr)
            print("重跑本脚本可断点续传。", file=sys.stderr)
            return 1
    print("全部就绪。OFA_MODEL_PATH 默认自动探测仓库内 model_assets/（config.json + pytorch_model.bin 同目录）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
