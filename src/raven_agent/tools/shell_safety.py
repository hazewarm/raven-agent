from __future__ import annotations

import shlex
from pathlib import Path

from raven_agent.tools.hooks import ToolHook, ToolHookContext, ToolHookOutcome

_INTERACTIVE_COMMANDS = {"vi", "vim", "nvim", "nano", "less", "more", "top", "htop", "man", "ssh"}
_DESTRUCTIVE_COMMANDS = {"mkfs", "shutdown", "reboot", "poweroff", "halt"}
_APT_WRITE_ACTIONS = {"install", "remove", "upgrade", "dist-upgrade", "autoremove"}
_PACMAN_WRITE_PREFIXES = ("-S", "-R", "-U")


class ShellSafetyHook(ToolHook):
    """拦截高风险 shell 调用的 pre_tool_use Hook。

    输入:
        无构造参数。运行时通过 ToolHookContext 接收 shell 调用信息。

    输出:
        一个 ToolHook 实例。run() 返回 ToolHookOutcome，可能放行或 deny。
    """

    name = "shell_safety"
    event = "pre_tool_use"

    def matches(self, context: ToolHookContext) -> bool:
        """判断当前 Hook 是否匹配工具调用。

        输入:
            context: 当前 ToolHookContext。

        输出:
            bool。True 表示当前调用是 shell 工具，需要执行安全检查。
        """

        return context.request.tool_name == "shell"

    async def run(self, context: ToolHookContext) -> ToolHookOutcome:
        """执行 shell 安全检查。

        输入:
            context: 当前 ToolHookContext，包含 command 参数。

        输出:
            ToolHookOutcome。安全时 decision=pass；危险时 decision=deny 且 reason 说明原因。
        """

        command = str(context.current_arguments.get("command") or "").strip()
        reason = _deny_reason(command)
        if not reason:
            return ToolHookOutcome()
        return ToolHookOutcome(decision="deny", reason=reason)


def _deny_reason(command: str) -> str:
    """返回 shell 命令被拒绝的原因。

    输入:
        command: 待检查 shell 命令。

    输出:
        字符串。空字符串表示允许执行；非空字符串表示拒绝原因。
    """

    if not command:
        return "shell_safety 拦截：命令不能为空。"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "shell_safety 拦截：命令解析失败，请检查引号是否匹配。"
    if not tokens:
        return "shell_safety 拦截：命令不能为空。"

    interactive = _find_interactive_command(tokens)
    if interactive:
        return f"shell_safety 拦截：{interactive} 会打开交互式界面，请改用非交互命令。"
    destructive = _find_destructive_command(tokens)
    if destructive:
        return f"shell_safety 拦截：{destructive} 属于高风险系统命令。"
    if _sudo_needs_password(tokens):
        return "shell_safety 拦截：sudo 可能等待密码，请改用 sudo -n 让它在没有缓存时立即失败。"
    package_manager = _package_write_needs_confirm(tokens)
    if package_manager:
        return f"shell_safety 拦截：{package_manager} 写操作需要显式非交互确认参数。"
    if _dangerous_recursive_rm(tokens):
        return "shell_safety 拦截：递归删除根目录、当前目录、父目录或通配路径过于危险。"
    return ""


def _find_interactive_command(tokens: list[str]) -> str:
    """查找命令 token 中的交互式程序。

    输入:
        tokens: shlex.split 后的命令 token 列表。

    输出:
        命中的交互式程序名；没有命中时返回空字符串。
    """

    for token in tokens:
        name = Path(token).name
        if name in _INTERACTIVE_COMMANDS:
            return name
    return ""


def _find_destructive_command(tokens: list[str]) -> str:
    """查找命令 token 中的系统级破坏性命令。

    输入:
        tokens: shlex.split 后的命令 token 列表。

    输出:
        命中的高风险命令名；没有命中时返回空字符串。
    """

    for token in tokens:
        name = Path(token).name
        if name in _DESTRUCTIVE_COMMANDS:
            return name
    return ""


def _sudo_needs_password(tokens: list[str]) -> bool:
    """判断 sudo 是否可能等待密码。

    输入:
        tokens: shlex.split 后的命令 token 列表。

    输出:
        bool。True 表示存在未带 -n 的 sudo 调用。
    """

    for index, token in enumerate(tokens):
        if Path(token).name == "sudo" and not _sudo_has_non_interactive_option(tokens[index + 1 :]):
            return True
    return False


def _sudo_has_non_interactive_option(tokens: list[str]) -> bool:
    """判断 sudo 参数中是否包含非交互选项 -n。

    输入:
        tokens: sudo 后面的参数 token 列表。

    输出:
        bool。True 表示 sudo 会在需要密码时立即失败而不是等待输入。
    """

    for token in tokens:
        if token == "--":
            return False
        if not token.startswith("-") or token == "-":
            return False
        if token == "-n" or (token.startswith("-") and not token.startswith("--") and "n" in token[1:]):
            return True
    return False


def _package_write_needs_confirm(tokens: list[str]) -> str:
    """判断包管理器写操作是否缺少非交互确认参数。

    输入:
        tokens: shlex.split 后的命令 token 列表。

    输出:
        命中的包管理器名称；如果没有风险则返回空字符串。
    """

    for index, token in enumerate(tokens):
        name = Path(token).name
        args = tokens[index + 1 :]
        if name in {"apt", "apt-get"} and any(arg in _APT_WRITE_ACTIONS for arg in args) and "-y" not in args and "--yes" not in args:
            return name
        if name == "pacman" and any(arg.startswith(_PACMAN_WRITE_PREFIXES) for arg in args) and "--noconfirm" not in args:
            return name
    return ""


def _dangerous_recursive_rm(tokens: list[str]) -> bool:
    """判断 rm 递归删除目标是否明显危险。

    输入:
        tokens: shlex.split 后的命令 token 列表。

    输出:
        bool。True 表示命令尝试递归删除根目录、当前目录、父目录或通配路径。
    """

    if Path(tokens[0]).name != "rm":
        return False
    recursive = any(token in {"-r", "-R", "--recursive"} or (token.startswith("-") and "r" in token.lower()) for token in tokens[1:])
    if not recursive:
        return False
    targets = [token for token in tokens[1:] if not token.startswith("-")]
    dangerous_targets = {"/", ".", "..", "~", "$HOME"}
    return any(target in dangerous_targets or target.endswith("/*") or target in {"./*", "../*"} for target in targets)