from raven_agent.plugins import Plugin, on_turn_completed, tool


class HelloPlugin(Plugin):
    """本地 hello 插件。

    输入:
        无。

    输出:
        HelloPlugin 实例。
    """

    name = "hello"

    @on_turn_completed()
    async def count_turn(self, event):
        """统计完成轮数。

        输入:
            event: TurnCompleted 事件。

        输出:
            None。
        """

        self.context.kv_store.increment("turn_count")

    @tool("hello_echo", risk="read-only", always_on=True)
    async def hello_echo(self, event, text: str) -> str:
        """Echo text.

        Args:
            text: Text to echo.
        """

        return f"hello: {text}"