from __future__ import annotations

from raven_agent.lifecycle import BeforeTurnCtx
from raven_agent.plugins import Plugin, on_before_turn


class StatusCommandsPlugin(Plugin):
    """状态命令内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        StatusCommandsPlugin 实例。
    """

    name = "status_commands"
    version = "0.1.0"
    desc = "提供 /status /tools /memory_status 等只读状态命令"

    @on_before_turn()
    async def handle_command(self, ctx: BeforeTurnCtx) -> BeforeTurnCtx:
        """在命中状态命令时提前结束本轮。

        输入:
            ctx: 当前 BeforeTurnCtx。

        输出:
            命中命令时 abort 的 BeforeTurnCtx；否则原样返回。
        """

        command = _normalize_command(ctx.content)
        if command in {"/status", "/runtime"}:
            ctx.abort = True
            ctx.abort_reply = self._render_runtime_status()
            ctx.outbound_metadata["status_command"] = command
        elif command in {"/tools", "/tool_status"}:
            ctx.abort = True
            ctx.abort_reply = self._render_tool_status()
            ctx.outbound_metadata["status_command"] = command
        elif command in {"/memory_status", "/memorystatus"}:
            ctx.abort = True
            ctx.abort_reply = self._render_memory_status(ctx.session_key)
            ctx.outbound_metadata["status_command"] = command
        return ctx

    def _render_runtime_status(self) -> str:
        """渲染运行时状态文本。

        输入:
            无。

        输出:
            运行时状态字符串。
        """

        engine = getattr(self.context.memory_engine, "describe", None)
        engine_name = engine().name if callable(engine) else "unknown"
        tool_count = len(self.context.tool_registry.list_names()) if self.context.tool_registry else 0
        lines = [
            "🟢 raven-agent 运行状态",
            "",
            f"记忆引擎：{engine_name}",
            f"已注册工具数：{tool_count}",
            f"workspace：{self.context.workspace or '（未设置）'}",
        ]
        return "\n".join(lines)

    def _render_tool_status(self) -> str:
        """渲染工具状态文本。

        输入:
            无。

        输出:
            工具状态字符串。
        """

        registry = self.context.tool_registry
        if registry is None:
            return "暂无工具注册表。"
        names = sorted(registry.list_names())
        always_on = registry.get_always_on_names()
        lines = [f"🛠 工具总数：{len(names)}", ""]
        for name in names:
            mark = "●" if name in always_on else "○"
            lines.append(f"{mark} {name}")
        lines.append("")
        lines.append("● = always-on，○ = 需 tool_search 解锁")
        return "\n".join(lines)

    def _render_memory_status(self, session_key: str) -> str:
        """渲染记忆整理状态文本。

        输入:
            session_key: 当前会话 key。

        输出:
            记忆整理状态字符串。
        """

        sessions = self.context.session_manager
        if sessions is None:
            return "暂无会话管理器，无法查看记忆整理状态。"
        session = sessions.get_or_create(session_key)
        total = len(session.messages)
        consolidated = max(0, min(int(getattr(session, "last_consolidated", 0)), total))
        pending = total - consolidated
        lines = [
            "🧠 记忆整理状态",
            "",
            f"当前会话消息数：{total}",
            f"已整理到下标：{consolidated}",
            f"尚未整理消息数：{pending}",
        ]
        return "\n".join(lines)


def _normalize_command(content: str) -> str:
    """把用户输入归一化为命令 token。

    输入:
        content: 用户输入文本。

    输出:
        小写命令首词；非命令返回空字符串。
    """

    parts = (content or "").strip().split(maxsplit=1)
    if not parts:
        return ""
    head = parts[0].lower()
    if "@" in head:
        head = head.split("@", 1)[0]
    return head if head.startswith("/") else ""