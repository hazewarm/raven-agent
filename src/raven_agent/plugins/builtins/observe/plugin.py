from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from raven_agent.lifecycle import AfterTurnCtx
from raven_agent.plugins import Plugin, on_after_turn, on_tool_error, on_tool_post
from raven_agent.plugins.context import PluginToolHookEvent
from raven_agent.plugins.builtins.observe.events import ToolCallTrace, TurnTrace
from raven_agent.plugins.builtins.observe.retention import run_retention_if_needed
from raven_agent.plugins.builtins.observe.writer import TraceWriter
from raven_agent.plugins.builtins.observe.bridge import ObserveBridge

logger = logging.getLogger("plugin.observe")


class ObservePlugin(Plugin):
    """把对话与工具调用写入 observe.db 的内置插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        ObservePlugin 实例。
    """

    name = "observe"
    version = "0.1.0"
    desc = "记录每轮对话与工具调用到 observe.db"

    async def initialize(self) -> None:
        """启动 observe 写库后台任务与淘汰任务。

        输入:
            无。

        输出:
            None。缺少 workspace 时跳过启动。
        """

        workspace = self.context.workspace
        if workspace is None:
            logger.warning("observe 插件缺少 workspace，跳过加载")
            self._writer = None
            return
        db_path = workspace / "observe" / "observe.db"
        self._writer = TraceWriter(db_path)
        self._writer_task = asyncio.create_task(self._writer.run(), name="observe_writer")
        self._retention_task = asyncio.create_task(
            run_retention_if_needed(db_path), name="observe_retention"
        )
        # 创建并启动 ObserveBridge——监听 EventBus 的检索/记忆写入事件。
        # 桥接器是可选的旁路组件，初始化失败不应影响 turn/tool_call 记录。
        try:
            event_bus = self.context.event_bus
            self._bridge = ObserveBridge(getattr(self, "_writer", None), event_bus)
            self._bridge.start()
        except Exception:
            logger.warning("ObserveBridge 启动失败，检索/记忆写入不记录", exc_info=True)
            self._bridge = None

    async def terminate(self) -> None:
        """停止 observe 后台任务。

        输入:
            无。

        输出:
            None。
        """
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            bridge.stop()
        
        for task in (getattr(self, "_retention_task", None), getattr(self, "_writer_task", None)):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @on_after_turn()
    async def record_turn(self, ctx: AfterTurnCtx) -> None:
        """记录一轮对话。

        输入:
            ctx: 当前 AfterTurnCtx。

        输出:
            None。
        """

        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        cited = ctx.outbound_metadata.get("cited_memory_ids")
        writer.emit(
            TurnTrace(
                session_key=ctx.session_key,
                channel=ctx.channel,
                chat_id=ctx.chat_id,
                user_msg="",
                reply=ctx.reply,
                tools_used=list(ctx.tools_used),
                cited_memory_ids=list(cited) if isinstance(cited, list) else [],
            )
        )

    @on_tool_post()
    async def record_tool_success(self, event: PluginToolHookEvent) -> None:
        """记录一次成功工具调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            None。
        """

        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        writer.emit(
            ToolCallTrace(
                session_key=event.session_key,
                tool_name=event.tool_name,
                arguments=dict(event.arguments),
                status="success",
                plugin_source=self.context.plugin_id,
            )
        )

    @on_tool_error()
    async def record_tool_error(self, event: PluginToolHookEvent) -> None:
        """记录一次失败工具调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            None。
        """

        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        writer.emit(
            ToolCallTrace(
                session_key=event.session_key,
                tool_name=event.tool_name,
                arguments=dict(event.arguments),
                status="error",
                plugin_source=self.context.plugin_id,
                error=event.error or "",
            )
        )