from __future__ import annotations

import asyncio
import json

from raven_agent.tools import ShellTool


def _run(coro):
    """同步运行异步测试调用。

    输入:
        coro: 要运行的 coroutine。

    输出:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def test_shell_tool_executes_command(tmp_path) -> None:
    """测试 shell 工具可以执行普通命令。

    输入:
        tmp_path: pytest 临时目录 fixture，作为 shell 工作目录。

    输出:
        None。通过 assert 验证 exit_code 和 output。
    """

    tool = ShellTool(working_dir=tmp_path)

    result = _run(tool.execute(command="python -c \"print('hello')\"", description="打印 hello", timeout=5))
    payload = json.loads(result.text)

    assert result.metadata["ok"] is True
    assert payload["exit_code"] == 0
    assert "hello" in payload["output"]


def test_shell_tool_times_out(tmp_path) -> None:
    """测试 shell 工具超时后会中断命令。

    输入:
        tmp_path: pytest 临时目录 fixture，作为 shell 工作目录。

    输出:
        None。通过 assert 验证 interrupted=True 和 exit_code=-1。
    """

    tool = ShellTool(working_dir=tmp_path)

    result = _run(tool.execute(command="python -c \"import time; time.sleep(2)\"", description="测试超时", timeout=1))
    payload = json.loads(result.text)

    assert result.metadata["ok"] is True
    assert payload["interrupted"] is True
    assert payload["exit_code"] == -1