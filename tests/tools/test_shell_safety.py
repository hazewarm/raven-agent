from __future__ import annotations

import asyncio

from raven_agent.tools import ShellSafetyHook
from raven_agent.tools.hooks import ToolExecutionRequest, ToolHookContext


def _run(coro):
    """同步运行异步测试调用。

    输入:
        coro: 要运行的 coroutine。

    输出:
        coroutine 的返回值。
    """

    return asyncio.run(coro)


def _context(command: str) -> ToolHookContext:
    """创建 shell safety 测试上下文。

    输入:
        command: 要模拟的 shell 命令。

    输出:
        ToolHookContext，包含 shell 工具调用请求和当前参数。
    """

    return ToolHookContext(
        event="pre_tool_use",
        request=ToolExecutionRequest(call_id="c1", tool_name="shell", arguments={"command": command, "description": "测试命令"}),
        current_arguments={"command": command, "description": "测试命令"},
    )


def test_shell_safety_blocks_interactive_editor() -> None:
    """测试 shell safety 拦截交互式编辑器。

    输入:
        无。

    输出:
        None。通过 assert 验证 deny 和原因文本。
    """

    outcome = _run(ShellSafetyHook().run(_context("vim README.md")))

    assert outcome.decision == "deny"
    assert "交互式界面" in outcome.reason


def test_shell_safety_blocks_sudo_without_non_interactive_flag() -> None:
    """测试 shell safety 拦截未带 -n 的 sudo。

    输入:
        无。

    输出:
        None。通过 assert 验证 deny 和原因文本。
    """

    outcome = _run(ShellSafetyHook().run(_context("sudo apt update")))

    assert outcome.decision == "deny"
    assert "sudo -n" in outcome.reason


def test_shell_safety_blocks_package_write_without_confirm() -> None:
    """测试 shell safety 拦截缺少非交互确认的包管理器写操作。

    输入:
        无。

    输出:
        None。通过 assert 验证 deny 和原因文本。
    """

    outcome = _run(ShellSafetyHook().run(_context("apt install nginx")))

    assert outcome.decision == "deny"
    assert "非交互确认" in outcome.reason


def test_shell_safety_blocks_dangerous_recursive_rm() -> None:
    """测试 shell safety 拦截危险递归删除。

    输入:
        无。

    输出:
        None。通过 assert 验证 deny 和原因文本。
    """

    outcome = _run(ShellSafetyHook().run(_context("rm -rf /")))

    assert outcome.decision == "deny"
    assert "递归删除" in outcome.reason


def test_shell_safety_allows_harmless_command() -> None:
    """测试 shell safety 放行普通命令。

    输入:
        无。

    输出:
        None。通过 assert 验证 decision=pass。
    """

    outcome = _run(ShellSafetyHook().run(_context("python -V")))

    assert outcome.decision == "pass"