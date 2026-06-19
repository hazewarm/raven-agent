from __future__ import annotations

from raven_agent.llm_json import load_json_object_loose


def test_load_json_object_loose_reads_plain_object() -> None:
    """测试可以解析纯 JSON object。

    返回:
        None。
    """

    payload = load_json_object_loose('{"ok": true}')

    assert payload == {"ok": True}


def test_load_json_object_loose_reads_fenced_json() -> None:
    """测试可以解析 Markdown fenced JSON。

    返回:
        None。
    """

    payload = load_json_object_loose('```json\n{"ok": true}\n```')

    assert payload == {"ok": True}


def test_load_json_object_loose_extracts_object_from_text() -> None:
    """测试可以从解释文字中提取 JSON object。

    返回:
        None。
    """

    payload = load_json_object_loose('结果如下：\n{"ok": true}\n请查收。')

    assert payload == {"ok": True}


def test_load_json_object_loose_repairs_common_llm_json() -> None:
    """测试可以修复常见 LLM 不严格 JSON。

    返回:
        None。
    """

    payload = load_json_object_loose("{ok: true, name: 'Raven',}")

    assert payload == {"ok": True, "name": "Raven"}


def test_load_json_object_loose_rejects_non_object() -> None:
    """测试非 object JSON 会返回 None。

    返回:
        None。
    """

    assert load_json_object_loose('[1, 2, 3]') is None
    assert load_json_object_loose('not json') is None