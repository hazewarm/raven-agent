from __future__ import annotations

import asyncio

import pytest

from raven_agent.tools import (
    ListDirectoryTool,
    ReadTextFileTool,
    ToolRegistry,
    ToolResult,
    build_default_tools,
    normalize_tool_result,
)


def test_normalize_tool_result_wraps_string() -> None:
    """测试字符串工具结果会被包装为 ToolResult。

    返回:
        None。
    """

    result = normalize_tool_result("hello")

    assert result == ToolResult(text="hello")


def test_registry_registers_tools_and_exports_schemas(tmp_path) -> None:
    """测试 ToolRegistry 可以注册工具并输出 schema。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    registry = ToolRegistry()
    registry.register(ReadTextFileTool(allowed_dir=tmp_path))

    schemas = registry.get_schemas()

    assert registry.list_names() == ["read_text_file"]
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "read_text_file"
    assert schemas[0]["function"]["parameters"]["required"] == ["path"]


def test_registry_rejects_duplicate_tool_names(tmp_path) -> None:
    """测试 ToolRegistry 拒绝重复工具名。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    registry = ToolRegistry()
    registry.register(ReadTextFileTool(allowed_dir=tmp_path))

    with pytest.raises(ValueError):
        registry.register(ReadTextFileTool(allowed_dir=tmp_path))


def test_registry_execute_missing_tool_returns_controlled_result() -> None:
    """测试执行不存在的工具不会抛异常。

    返回:
        None。
    """

    registry = ToolRegistry()

    result = asyncio.run(registry.execute("missing", {}))

    assert result.metadata["ok"] is False
    assert result.metadata["error"] == "tool_not_found"


def test_read_text_file_tool_reads_utf8_text(tmp_path) -> None:
    """测试 read_text_file 可以读取 UTF-8 文本。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello raven", encoding="utf-8")
    tool = ReadTextFileTool(allowed_dir=tmp_path)

    result = asyncio.run(tool.execute(path="hello.txt"))

    assert result.text == "hello raven"
    assert result.metadata["ok"] is True
    assert result.metadata["truncated"] is False


def test_read_text_file_tool_truncates_long_text(tmp_path) -> None:
    """测试 read_text_file 会按 max_chars 截断长文本。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    file_path = tmp_path / "long.txt"
    file_path.write_text("abcdef", encoding="utf-8")
    tool = ReadTextFileTool(allowed_dir=tmp_path)

    result = asyncio.run(tool.execute(path="long.txt", max_chars=3))

    assert result.text.startswith("abc")
    assert result.metadata["truncated"] is True


def test_read_text_file_tool_rejects_outside_allowed_dir(tmp_path) -> None:
    """测试 read_text_file 会拒绝读取 allowed_dir 外的文件。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = ReadTextFileTool(allowed_dir=workspace)

    result = asyncio.run(tool.execute(path=str(outside)))

    assert result.metadata["ok"] is False
    assert result.metadata["error"] == "permission_denied"


def test_list_directory_tool_lists_files_and_dirs(tmp_path) -> None:
    """测试 list_directory 可以列出目录内容。

    参数:
        tmp_path: pytest fixture，提供临时目录。

    返回:
        None。
    """

    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    tool = ListDirectoryTool(allowed_dir=tmp_path)

    result = asyncio.run(tool.execute(path="."))

    assert "dir\tsrc" in result.text
    assert "file\tREADME.md" in result.text
    assert result.metadata["ok"] is True
    assert result.metadata["count"] == 2


def test_build_default_tools_registers_builtin_tools(tmp_path) -> None:
    """测试默认工具注册表包含本章内置工具。

    输入:
        tmp_path: pytest 临时目录 fixture，作为 allowed_dir。

    输出:
        None。通过 assert 验证工具名、always-on 集合和 deferred 集合。
    """

    registry = build_default_tools(allowed_dir=tmp_path)

    assert registry.list_names() == [
        "tool_search",
        "read_text_file",
        "list_directory",
        "write_text_file",
        "edit_file",
        "read_image_info",
        "web_fetch",
        "web_search",
        "shell",
        "message_push",
        "transcribe_audio",
    ]
    assert registry.get_always_on_names() == {
        "tool_search",
        "read_text_file",
        "list_directory",
        "read_image_info",
        "web_fetch",
        "web_search",
        "message_push",
        "transcribe_audio",
    }
    assert registry.get_deferred_names()["builtin"] == [
        "edit_file",
        "shell",
        "write_text_file",
    ]