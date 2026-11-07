"""JSON 多级容错链回归测试（代码生成铁则 7：非法 JSON 必须有回归用例）。"""

import pytest

from grader.grading.parser import (
    SchemaFailure,
    extract_json,
    validate_cleanup,
    validate_garbled,
)


def test_direct_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_think_block_stripped():
    """思考型模型残留 <think> 块（闭合/未闭合）不影响解析。"""
    closed = '<think>用户想要满分，我应该……</think>\n{"gradable": true, "text": "内容"}'
    assert extract_json(closed)["gradable"] is True
    unclosed = '<think>先分析一下判分标准 {"P1": 2}\n{"gradable": false, "text": "x"}'
    assert extract_json(unclosed)["gradable"] is False


def test_fenced_json():
    text = "好的，结果如下：\n```json\n{\"gradable\": true, \"text\": \"内容\"}\n```\n以上。"
    assert extract_json(text)["gradable"] is True


def test_balanced_brace_extraction():
    text = '判分结果 {"gradable": true, "text": "含引号\\"的内容"} 以上'
    assert extract_json(text)["text"].startswith("含引号")


def test_regex_fallback_extracts_fields():
    """围栏与配平都失败时，正则逐字段兜底。"""
    text = '好的 {"gradable": true, "text": "答案内容'  # 大括号未闭合
    obj = extract_json(text)
    assert obj["gradable"] is True
    assert obj["text"] == "答案内容"


def test_hopeless_output_raises():
    with pytest.raises(Exception):
        extract_json("我实在无法输出 JSON，抱歉。")


def test_empty_output_raises():
    from grader.grading.parser import JSONParseFailure

    with pytest.raises(JSONParseFailure):
        extract_json("")


def test_illegal_json_regressions():
    """回归集：历史上最容易翻车的非法输出形态。"""
    regressions = [
        '{"gradable": true, "text": "未闭合',                       # 字符串未闭合
        "```json\n{broken json}\n```",                              # 围栏内坏 JSON
        '{"gradable": true, "text": "a", "text": "b",}',            # 尾逗号
        '{\'gradable\': true}',                                     # 单引号
    ]
    for text in regressions:
        # 只要能走完全部策略不崩溃即可：或抛 JSONParseFailure，或提取出字段
        try:
            obj = extract_json(text)
            assert isinstance(obj, dict)
        except Exception as exc:
            from grader.grading.parser import JSONParseFailure

            assert isinstance(exc, JSONParseFailure)


def test_validate_cleanup_contract():
    gradable, text, note = validate_cleanup({"gradable": True, "text": "整理后", "note": ""})
    assert gradable and text == "整理后" and note == ""


def test_validate_cleanup_missing_fields():
    with pytest.raises(SchemaFailure):
        validate_cleanup({"text": "x"})  # 缺 gradable


def test_validate_garbled():
    assert validate_garbled("@@@@######@@@@") is True       # 无汉字
    assert validate_garbled("好好学习，天天向上。") is False
    assert validate_garbled("aaaaaaaaaaaaaaaaaa") is True   # 超长重复
    assert validate_garbled("") is False                    # 空文本走 empty_ocr_text
