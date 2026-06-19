from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from raven_agent.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 30_000


class ShellTool(Tool):
    """执行前台 shell 命令的工具。

    输入:
        working_dir: 构造函数参数，命令执行时使用的工作目录。

    输出:
        一个 Tool 实例。执行 execute() 后返回命令执行 JSON 结果。
    """

    name = "shell"
    description = (
        "在 shell 中执行一条前台命令并返回 JSON 结果。"
        "适合运行测试、查看版本、执行诊断命令。长时间后台任务会在后续 Background Runtime 章节实现。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令。"},
            "description": {"type": "string", "description": "用一句话说明命令目的。"},
            "timeout": {
                "type": "integer",
                "description": "超时秒数，默认 30，最大 120。",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT,
            },
        },
        "required": ["command", "description"],
    }

    def __init__(self, working_dir: Path | None = None) -> None:
        """初始化 shell 工具。

        输入:
            working_dir: 命令执行工作目录；为 None 时使用当前进程工作目录。

        输出:
            None。初始化后的状态保存在 self._working_dir。
        """

        self._working_dir = working_dir

    async def execute(
        self,
        command: str,
        description: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行 shell 命令。

        输入:
            command: 要执行的 shell 命令。
            description: 对命令用途的一句话说明。
            timeout: 超时秒数；为 None 时使用默认值。
            **kwargs: 预留扩展参数，当前不使用。

        输出:
            ToolResult。text 是 JSON 字符串，包含 exit_code、interrupted、duration_ms、output 等字段。
        """

        clean_command = command.strip()
        if not clean_command:
            return ToolResult(text=json.dumps({"error": "命令不能为空"}, ensure_ascii=False), metadata={"ok": False, "error": "empty_command"})

        timeout_s = min(max(1, int(timeout or _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)
        started = time.perf_counter()
        proc: Any = None

        try:
            proc = await asyncio.create_subprocess_shell(clean_command, **_subprocess_options(self._working_dir))
            interrupted = False
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                interrupted = True
                _kill_process(proc)
                stdout, _ = await proc.communicate()

            duration_ms = int((time.perf_counter() - started) * 1000)
            output = stdout.decode(errors="replace") if stdout else "（无输出）"
            visible_output, truncated = _truncate_output(output)
            exit_code = -1 if interrupted else int(proc.returncode or 0)
            payload = {
                "command": clean_command,
                "description": description,
                "exit_code": exit_code,
                "interrupted": interrupted,
                "duration_ms": duration_ms,
                "truncated": truncated,
                "output": visible_output,
            }
            return ToolResult(text=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"ok": True, "exit_code": exit_code, "interrupted": interrupted})
        except Exception as exc:
            if proc is not None:
                _kill_process(proc)
            return ToolResult(text=json.dumps({"command": clean_command, "error": str(exc)}, ensure_ascii=False), metadata={"ok": False, "error": "shell_failed"})


def _subprocess_options(working_dir: Path | None) -> dict[str, Any]:
    """构造 asyncio subprocess 参数。

    输入:
        working_dir: 命令执行工作目录；为 None 时使用当前进程工作目录。

    输出:
        可传给 asyncio.create_subprocess_shell 的参数字典。
    """

    options: dict[str, Any] = {
        "cwd": str(working_dir) if working_dir is not None else None,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return options


def _kill_process(proc: Any) -> None:
    """终止 shell 子进程。

    输入:
        proc: asyncio subprocess process 对象。

    输出:
        None。进程不存在时静默返回。
    """

    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _truncate_output(output: str) -> tuple[str, bool]:
    """截断过长命令输出。

    输入:
        output: 原始命令输出文本。

    输出:
        二元组 `(visible_output, truncated)`；visible_output 为返回给模型的文本，truncated 表示是否截断。
    """

    if len(output) <= _MAX_OUTPUT_CHARS:
        return output, False
    half = _MAX_OUTPUT_CHARS // 2
    omitted = len(output) - _MAX_OUTPUT_CHARS
    return output[:half] + f"\n\n...[已截断 {omitted} 个字符]...\n\n" + output[-half:], True