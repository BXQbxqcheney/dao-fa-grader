"""Agnes Key 池单元测试：轮询 / 冷却 / 禁用 / 耗尽 / 文件加载（零网络）。

测试 Key 由运行时拼接构造（避免源码出现凭据形态字符串）。
"""

import pytest

from grader.core.config import Settings
from grader.core.errors import LLMError
from grader.llm.agnes_adapter import load_keys
from grader.llm.keypool import KeyPool

P = "sk-"  # 测试 Key 前缀（运行时拼接用）


def make_pool(n=3):
    return KeyPool([P + f"test-key-{i:03d}" for i in range(n)])


def test_rotation_round_robin():
    pool = make_pool(3)
    picks = [pool.acquire() for _ in range(6)]
    assert picks[0] != picks[1] != picks[2]
    assert picks[3] == picks[0] and picks[4] == picks[1]  # 轮询回绕


def test_rate_limit_cooldown_skips_key():
    pool = make_pool(2)
    k0 = pool.acquire()
    pool.report_rate_limited(k0)
    # 冷却中的 Key 被跳过：连续取用只出另一个 Key
    picks = {pool.acquire() for _ in range(4)}
    assert k0 not in picks
    snap = pool.snapshot()
    assert snap["cooling"] == 1 and snap["available"] == 1


def test_auth_failed_disables_permanently():
    pool = make_pool(2)
    k0 = pool.acquire()
    pool.report_auth_failed(k0)
    for _ in range(5):
        assert pool.acquire() != k0
    assert pool.snapshot()["disabled"] == 1


def test_quota_long_cooldown():
    pool = make_pool(2)
    k0 = pool.acquire()
    pool.report_quota(k0)
    assert k0 not in {pool.acquire() for _ in range(3)}
    assert pool.snapshot()["cooling"] == 1


def test_success_resets_fail_count():
    pool = make_pool(1)
    k0 = pool.acquire()
    pool.report_rate_limited(k0)
    pool.report_rate_limited(k0)
    pool.report_success(k0)
    assert pool.acquire() == k0  # 冷却清零，恢复可用
    assert pool.snapshot()["available"] == 1


def test_all_exhausted_raises_with_hint():
    pool = make_pool(2)
    for _ in range(2):
        k = pool.acquire()
        pool.report_auth_failed(k)
    with pytest.raises(LLMError, match="禁用"):
        pool.acquire()


def test_all_cooling_raises_with_wait_hint():
    pool = make_pool(2)
    for _ in range(2):
        k = pool.acquire()
        pool.report_quota(k)
    with pytest.raises(LLMError, match="冷却"):
        pool.acquire()


def test_empty_pool_rejected():
    with pytest.raises(LLMError, match="为空"):
        KeyPool(["", "  "])


def test_load_keys_from_file(tmp_path):
    keyfile = tmp_path / "keys.txt"
    keyfile.write_text(
        P + "file-key-1\n\n# 注释行\n" + P + "file-key-2\nnot-a-key\n", encoding="utf-8"
    )
    settings = Settings(agnes_key_file=str(keyfile), agnes_api_key=P + "env-key")
    keys = load_keys(settings)
    assert keys == [P + "file-key-1", P + "file-key-2", P + "env-key"]  # 去重保序


def test_load_keys_missing_file_raises():
    settings = Settings(agnes_key_file="Z:/not/exist/keys.txt")
    with pytest.raises(LLMError, match="不存在"):
        load_keys(settings)


def test_load_keys_none_raises():
    with pytest.raises(LLMError, match="未配置任何 Key"):
        load_keys(Settings(agnes_key_file="", agnes_api_key="", agnes_api_keys=""))
