from __future__ import annotations

from raven_agent.memory2.rule_schema import (
    build_procedure_rule_schema,
    procedure_rules_conflict,
    resolve_procedure_rule_schema,
)


def test_build_procedure_rule_schema_infers_required_and_forbidden_tools() -> None:
    """测试从中文 procedure 摘要中推断 required/forbidden tools。"""

    schema = build_procedure_rule_schema("查 Steam 信息时必须先使用 steam_mcp，不能直接使用 web_search。")

    assert "steam_mcp" in schema["required_tools"]
    assert "web_search" in schema["forbidden_tools"]
    assert "steam" in schema["mentioned_tools"]


def test_build_procedure_rule_schema_prefers_tool_requirement() -> None:
    """测试 tool_requirement 会进入 required_tools。"""

    schema = build_procedure_rule_schema(
        "用户发送网页链接时应抓取网页。",
        tool_requirement="web_fetch",
    )

    assert schema["required_tools"] == ["web_fetch"]
    assert "web_fetch" in schema["mentioned_tools"]


def test_resolve_procedure_rule_schema_uses_extra_json() -> None:
    """测试从 extra_json 中解析 rule_schema。"""

    schema = resolve_procedure_rule_schema(
        "运行命令前必须说明风险。",
        {"tool_requirement": "shell", "steps": ["先检查命令"]},
    )

    assert "shell" in schema["required_tools"]


def test_procedure_rules_conflict_detects_opposite_tool_direction() -> None:
    """测试 procedure_rules_conflict 能识别工具方向冲突。"""

    new_schema = {
        "required_tools": ["steam_mcp"],
        "forbidden_tools": ["web_search"],
        "mentioned_tools": ["steam_mcp", "web_search"],
    }
    old_schema = {
        "required_tools": ["web_search"],
        "forbidden_tools": ["steam_mcp"],
        "mentioned_tools": ["steam_mcp", "web_search"],
    }

    assert procedure_rules_conflict(new_schema, old_schema) is True