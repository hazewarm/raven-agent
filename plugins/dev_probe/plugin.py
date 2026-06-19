from raven_agent.plugins import (
    Plugin,
    on_after_reasoning,
    on_after_turn,
    on_before_turn,
    on_tool_post,
    tool,
)


class DevProbe(Plugin):
    """开发验证插件。

    输入:
        无。PluginManager 实例化后注入 context。

    输出:
        DevProbe 实例。
    """

    name = "dev_probe"

    @on_before_turn()
    async def command(self, ctx):
        """拦截 /probe 命令。

        输入:
            ctx: 当前 BeforeTurnCtx。

        输出:
            修改后的 BeforeTurnCtx。
        """

        if ctx.content.strip() == "/probe":
            ctx.abort = True
            ctx.abort_reply = "dev_probe ok"
            ctx.outbound_metadata["probe"] = True
        return ctx

    @on_after_reasoning()
    async def clean(self, ctx):
        """清理内部标签。

        输入:
            ctx: 当前 AfterReasoningCtx。

        输出:
            修改后的 AfterReasoningCtx。
        """

        ctx.reply = ctx.reply.replace("<debug>", "").strip()
        return ctx

    @on_after_turn()
    async def count(self, ctx):
        """统计完成轮数。

        输入:
            ctx: 当前 AfterTurnCtx。

        输出:
            None。
        """

        self.context.kv_store.increment("turn_count")

    @tool("dev_echo", risk="read-only", always_on=True)
    async def dev_echo(self, event, text: str) -> str:
        """Echo text.

        Args:
            text: Text to echo.
        """

        return f"dev:{text}"

    @on_tool_post(tool_name="dev_echo")
    async def observe_echo(self, event):
        """记录 dev_echo 成功调用。

        输入:
            event: PluginToolHookEvent。

        输出:
            None。
        """

        self.context.kv_store.set("last_tool", event.tool_name)