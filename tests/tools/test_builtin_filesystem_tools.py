from __future__ import annotations

import asyncio
import json

from PIL import Image

from raven_agent.tools import EditFileTool, ReadImageInfoTool, WriteTextFileTool


def _run(coro):
    """同步运行异步测试调用。

    输入:
        coro: 要运行的 coroutine。

    输出:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def test_write_text_file_creates_file(tmp_path) -> None:
    """测试 write_text_file 可以创建新文件。

    输入:
        tmp_path: pytest 临时目录 fixture。

    输出:
        None。通过 assert 验证文件内容和 ToolResult metadata。
    """

    tool = WriteTextFileTool(allowed_dir=tmp_path)

    result = _run(tool.execute(path="notes/hello.txt", content="hello raven"))

    assert result.metadata["ok"] is True
    assert (tmp_path / "notes" / "hello.txt").read_text(encoding="utf-8") == "hello raven"


def test_write_text_file_rejects_outside_allowed_dir(tmp_path) -> None:
    """测试 write_text_file 拒绝越界写入。

    输入:
        tmp_path: pytest 临时目录 fixture。

    输出:
        None。通过 assert 验证 permission_denied。
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    tool = WriteTextFileTool(allowed_dir=workspace)

    result = _run(tool.execute(path=str(outside), content="secret"))

    assert result.metadata["ok"] is False
    assert result.metadata["error"] == "permission_denied"


def test_edit_file_replaces_one_match_and_returns_diff(tmp_path) -> None:
    """测试 edit_file 替换单个匹配并返回 diff。

    输入:
        tmp_path: pytest 临时目录 fixture。

    输出:
        None。通过 assert 验证文件已修改且结果包含 diff。
    """

    target = tmp_path / "hello.py"
    target.write_text("name = 'old'\n", encoding="utf-8")
    tool = EditFileTool(allowed_dir=tmp_path)

    result = _run(tool.execute(path="hello.py", old_text="old", new_text="new"))

    assert result.metadata["ok"] is True
    assert "new" in target.read_text(encoding="utf-8")
    assert "```diff" in result.text


def test_edit_file_rejects_multiple_matches_without_replace_all(tmp_path) -> None:
    """测试 edit_file 默认拒绝多处匹配。

    输入:
        tmp_path: pytest 临时目录 fixture。

    输出:
        None。通过 assert 验证 multiple_matches 错误。
    """

    target = tmp_path / "hello.txt"
    target.write_text("x\nx\n", encoding="utf-8")
    tool = EditFileTool(allowed_dir=tmp_path)

    result = _run(tool.execute(path="hello.txt", old_text="x", new_text="y"))

    assert result.metadata["ok"] is False
    assert result.metadata["error"] == "multiple_matches"


def test_read_image_info_returns_metadata(tmp_path) -> None:
    """测试 read_image_info 返回图片元信息。

    输入:
        tmp_path: pytest 临时目录 fixture。

    输出:
        None。通过 assert 验证宽、高和格式。
    """

    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 8), color="red").save(image_path)
    tool = ReadImageInfoTool(allowed_dir=tmp_path)

    result = _run(tool.execute(path="image.png"))
    payload = json.loads(result.text)

    assert result.metadata["ok"] is True
    assert payload["width"] == 16
    assert payload["height"] == 8
    assert payload["format"] == "PNG"