from __future__ import annotations

import asyncio

from raven_agent.channels.base import ChannelAdapter
from raven_agent.channels.manager import ChannelManager


class FakeChannel(ChannelAdapter):
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    @property
    def channel_name(self) -> str:
        return self._name

    async def start(self) -> None:
        self._calls.append(f"start:{self._name}")

    async def stop(self) -> None:
        self._calls.append(f"stop:{self._name}")


def test_channel_manager_starts_and_stops_in_order() -> None:
    async def run() -> None:
        calls: list[str] = []
        manager = ChannelManager()
        manager.register(FakeChannel("a", calls))
        manager.register(FakeChannel("b", calls))

        await manager.start_all()
        await manager.stop_all()

        assert calls == ["start:a", "start:b", "stop:b", "stop:a"]

    asyncio.run(run())


def test_channel_manager_get_returns_registered_channel() -> None:
    calls: list[str] = []
    manager = ChannelManager()
    channel = FakeChannel("cli", calls)
    manager.register(channel)

    assert manager.get("cli") is channel
    assert manager.get("missing") is None
