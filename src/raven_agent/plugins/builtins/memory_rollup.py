from __future__ import annotations

from raven_agent.lifecycle import BeforeTurnCtx
from raven_agent.plugins import Plugin, on_before_turn


class MemoryRollupPlugin(Plugin):
    """手动触发记忆整理与归档的内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        MemoryRollupPlugin 实例。
    """

    name = "memory_rollup"
    version = "0.1.0"
    desc = "手动 review/归档 markdown 记忆与 PENDING 候选"

    @on_before_turn()
    async def handle_command(self, ctx: BeforeTurnCtx) -> BeforeTurnCtx:
        """拦截记忆整理命令并执行整理。

        输入:
            ctx: 当前 BeforeTurnCtx。

        输出:
            命中命令时 abort 的 BeforeTurnCtx；否则原样返回。
        """

        command = (ctx.content or "").strip().lower()
        if command == "/memory_rollup":
            ctx.abort = True
            ctx.abort_reply = await self._run_optimizer()
            ctx.outbound_metadata["status_command"] = command
        elif command == "/consolidate":
            ctx.abort = True
            ctx.abort_reply = await self._run_consolidation(ctx.session_key)
            ctx.outbound_metadata["status_command"] = command
        return ctx

    async def _run_optimizer(self) -> str:
        """执行一次 PENDING -> MEMORY 归档。

        输入:
            无。

        输出:
            执行结果说明文本。
        """

        optimizer = self.context.memory_optimizer
        if optimizer is None:
            return "memory_rollup：当前未装配 MemoryOptimizer。"
        try:
            await optimizer.optimize()
        except Exception as exc:
            return f"memory_rollup：归档失败：{exc}"
        return "memory_rollup：已把 PENDING.md 归档进 MEMORY.md / SELF.md。"

    async def _run_consolidation(self, session_key: str) -> str:
        """对当前 session 强制执行一次 markdown consolidation。

        输入:
            session_key: 当前会话 key。

        输出:
            执行结果说明文本。
        """

        maintenance = self.context.memory_maintenance
        sessions = self.context.session_manager
        if maintenance is None or sessions is None:
            return "memory_rollup：当前未装配记忆维护器。"
        session = sessions.get_or_create(session_key)
        try:
            result = await maintenance.consolidate_session(session, force=True)
        except Exception as exc:
            return f"memory_rollup：consolidation 失败：{exc}"
        sessions.save(session)
        if result.skipped:
            return "memory_rollup：当前没有需要整理的新消息。"
        return f"memory_rollup：已整理 {result.consolidated_count} 条消息。"