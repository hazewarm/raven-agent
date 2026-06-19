from __future__ import annotations

from raven_agent.channels.base import ChannelAdapter


class ChannelManager:
    """统一管理所有 ChannelAdapter 的生命周期。

    输入:
        无。通过 register() 逐个注册 Channel。

    输出:
        ChannelManager 实例。
    """

    def __init__(self) -> None:
        self._channels: list[ChannelAdapter] = []
        self._started = False

    def register(self, channel: ChannelAdapter) -> None:
        """注册一个 Channel。

        输入:
            channel: ChannelAdapter 子类实例。

        输出:
            None。
        """
        if self._started:
            raise RuntimeError("ChannelManager 已启动，不能再注册新 Channel")
        self._channels.append(channel)

    def get(self, channel_name: str) -> ChannelAdapter | None:
        """按名称查找 Channel。

        输入:
            channel_name: Channel 名称。

        输出:
            匹配的 ChannelAdapter；不存在时返回 None。
        """
        for channel in self._channels:
            if channel.channel_name == channel_name:
                return channel
        return None

    def list_channels(self) -> list[ChannelAdapter]:
        """返回已注册 Channel 列表。

        输入:
            无。

        输出:
            ChannelAdapter 列表副本。
        """
        return list(self._channels)

    async def start_all(self) -> None:
        """按注册顺序启动所有 Channel。

        输入:
            无。

        输出:
            None。
        """
        if self._started:
            return
        for channel in self._channels:
            await channel.start()
        self._started = True

    async def stop_all(self) -> None:
        """按注册逆序停止所有 Channel。

        输入:
            无。

        输出:
            None。
        """
        if not self._started:
            return
        for channel in reversed(self._channels):
            await channel.stop()
        self._started = False