from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from raven_agent.channels.base import ChannelAdapter
from raven_agent.events import InboundMessage, OutboundMessage, StreamToken
from raven_agent.event_bus import EventBus

if TYPE_CHECKING:
    from raven_agent.message_bus import MessageBus
    from raven_agent.session import SessionManager

logger = logging.getLogger(__name__)
CHANNEL = "cli"


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int] | None:
    """解析 TCP endpoint。

    输入:
        endpoint: 形如 "127.0.0.1:8765" 的字符串。

    输出:
        (host, port)；如果不是合法 TCP endpoint，则返回 None。
    """
    if endpoint.count(":") != 1:
        return None
    host, port = endpoint.rsplit(":", 1)
    if not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


class IPCServerChannel(ChannelAdapter):
    """IPC Server Channel，支持 CLI session 选择协议。

    输入:
        bus: MessageBus。
        socket_path: Unix socket 路径或 TCP endpoint。
        sessions: SessionManager，用于 session.continue_latest / session.list。

    输出:
        IPCServerChannel 实例。
    """

    def __init__(
        self,
        bus: "MessageBus",
        socket_path: str,
        sessions: "SessionManager | None" = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._bus = bus
        self._socket_path = socket_path
        self._sessions = sessions
        self._event_bus = event_bus
        # chat_id -> writer：仅保存已经完成 session 绑定的连接
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.AbstractServer | None = None

    @property
    def channel_name(self) -> str:
        """返回 Channel 名称。

        输入:
            无。

        输出:
            固定字符串 "cli"。IPC 客户端本质上仍是 CLI Channel。
        """
        return CHANNEL

    async def start(self) -> None:
        """启动 IPC Server。

        输入:
            无。

        输出:
            None。启动后开始监听 socket / TCP。
        """
        self._bus.subscribe_outbound(self.channel_name, self._on_outbound)
        if self._event_bus is not None:
            self._event_bus.on(StreamToken, self._on_stream_token)
        tcp_endpoint = parse_tcp_endpoint(self._socket_path)
        if tcp_endpoint is not None:
            host, port = tcp_endpoint
            self._server = await asyncio.start_server(
                self._handle_connection,
                host=host,
                port=port,
            )
            logger.info("IPC server listening on tcp://%s:%s", host, port)
            return

        if not hasattr(asyncio, "start_unix_server"):
            raise RuntimeError(
                "Unix sockets are unavailable on this platform; use host:port instead."
            )
        Path(self._socket_path).unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self._socket_path,
        )
        os.chmod(self._socket_path, 0o600)
        logger.info("IPC server listening on %s", self._socket_path)

    async def stop(self) -> None:
        """停止 IPC Server 并关闭所有客户端连接。

        输入:
            无。

        输出:
            None。
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for writer in list(self._writers.values()):
            writer.close()
            await writer.wait_closed()
        self._writers.clear()

        if parse_tcp_endpoint(self._socket_path) is None:
            Path(self._socket_path).unlink(missing_ok=True)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理单个 IPC 客户端连接。

        输入:
            reader: 客户端读取流。
            writer: 客户端写入流。

        输出:
            None。连接断开后返回。
        """
        peer = writer.get_extra_info("peername") or "local"
        connection_id = f"conn-{uuid4().hex[:8]}"
        # bound_chat_id 在客户端完成 session.* 绑定后才会写入
        bound_chat_id: str | None = None
        logger.info("[cli] client connected conn=%s peer=%s", connection_id, peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[cli] received non-JSON payload")
                    continue

                if data.get("type") == "command":
                    bound_chat_id = await self._handle_command(
                        data, writer, bound_chat_id
                    )
                    continue

                content = str(data.get("content", "")).strip()
                if not content:
                    continue
                if bound_chat_id is None:
                    # 还没绑定 session 就发了普通消息：兜底新建一个
                    bound_chat_id = self._new_chat_id()
                    self._writers[bound_chat_id] = writer
                    await self._write_json(
                        writer,
                        {
                            "type": "command_result",
                            "command": "session.bound",
                            "ok": True,
                            "chat_id": bound_chat_id,
                            "created": True,
                        },
                    )
                await self._bus.publish_inbound(
                    InboundMessage(
                        channel=self.channel_name,
                        sender="cli-user",
                        chat_id=bound_chat_id,
                        content=content,
                    )
                )
        finally:
            if bound_chat_id is not None:
                self._writers.pop(bound_chat_id, None)
            writer.close()
            await writer.wait_closed()
            logger.info("[cli] client disconnected conn=%s", connection_id)

    async def _handle_command(
        self,
        data: dict[str, Any],
        writer: asyncio.StreamWriter,
        bound_chat_id: str | None,
    ) -> str | None:
        """处理 IPC 控制命令，并返回最新的 bound_chat_id。

        输入:
            data: JSON 解析后的命令字典。
            writer: 当前客户端 writer。
            bound_chat_id: 当前连接已经绑定的 chat_id；尚未绑定时为 None。

        输出:
            处理后的 bound_chat_id。session.list 不改变绑定，返回原值。
        """
        cmd = str(data.get("command", ""))

        if cmd == "session.new":
            return self._bind(writer, self._new_chat_id(), created=True, prev=bound_chat_id)

        if cmd == "session.continue_latest":
            chat_id = self._latest_chat_id()
            created = chat_id is None
            if chat_id is None:
                chat_id = self._new_chat_id()
            return self._bind(writer, chat_id, created=created, prev=bound_chat_id)

        if cmd == "session.attach":
            chat_id = str(data.get("chat_id", "")).strip()
            if not chat_id:
                await self._write_json(
                    writer,
                    {
                        "type": "command_result",
                        "command": "session.attach",
                        "ok": False,
                        "message": "缺少 chat_id",
                    },
                )
                return bound_chat_id
            return self._bind(writer, chat_id, created=False, prev=bound_chat_id)

        if cmd == "session.list":
            await self._write_session_list(writer, limit=int(data.get("limit", 20)))
            return bound_chat_id

        await self._write_json(
            writer,
            {
                "type": "command_result",
                "command": cmd,
                "ok": False,
                "message": f"unknown command: {cmd!r}",
            },
        )
        return bound_chat_id

    def _bind(
        self,
        writer: asyncio.StreamWriter,
        chat_id: str,
        *,
        created: bool,
        prev: str | None,
    ) -> str:
        """把当前连接绑定到指定 chat_id，并回执 session.bound。

        输入:
            writer: 当前客户端 writer。
            chat_id: 要绑定的 chat_id。
            created: 是否是新建 session。
            prev: 之前绑定的 chat_id；存在时先解绑。

        输出:
            绑定后的 chat_id。
        """
        if prev is not None:
            self._writers.pop(prev, None)
        # 同一个 chat_id 若已被其它连接占用，后绑定者覆盖（本地 CLI 简化策略）
        self._writers[chat_id] = writer
        asyncio.ensure_future(
            self._write_json(
                writer,
                {
                    "type": "command_result",
                    "command": "session.bound",
                    "ok": True,
                    "chat_id": chat_id,
                    "created": created,
                },
            )
        )
        return chat_id

    async def _write_session_list(
        self,
        writer: asyncio.StreamWriter,
        *,
        limit: int,
    ) -> None:
        """把可恢复的 CLI session 摘要发回客户端。

        输入:
            writer: 当前客户端 writer。
            limit: 最多返回多少个 session。

        输出:
            None。
        """
        sessions: list[dict[str, Any]] = []
        if self._sessions is not None:
            summaries = self._sessions.list_session_summaries(
                channel=self.channel_name, limit=limit
            )
            for index, summary in enumerate(summaries, start=1):
                sessions.append(
                    {
                        "index": index,
                        "chat_id": summary.chat_id,
                        "session_key": summary.session_key,
                        "updated_at": summary.updated_at,
                        "message_count": summary.message_count,
                        "preview": summary.preview,
                    }
                )
        await self._write_json(
            writer,
            {
                "type": "command_result",
                "command": "session.list",
                "ok": True,
                "sessions": sessions,
            },
        )

    def _new_chat_id(self) -> str:
        """生成一个新的 CLI chat_id。

        输入:
            无。

        输出:
            形如 1a2b3c4d5e6f 的 chat_id。
            注意 chat_id 不带 channel 前缀：session_key = f"{channel}:{chat_id}"
            会由 InboundMessage 拼接出 "cli:1a2b3c4d5e6f"，无需在此重复 "cli-"。
        """
        return uuid4().hex[:12]

    def _latest_chat_id(self) -> str | None:
        """返回最近一次 CLI session 的 chat_id。

        输入:
            无。

        输出:
            最近 session 的 chat_id；没有历史或没有 SessionManager 时返回 None。
        """
        if self._sessions is None:
            return None
        summaries = self._sessions.list_session_summaries(
            channel=self.channel_name, limit=1
        )
        if not summaries:
            return None
        return summaries[0].chat_id

    async def _on_outbound(self, message: OutboundMessage) -> None:
        """把 OutboundMessage 写回对应 IPC 客户端。

        输入:
            message: Agent 出站消息。

        输出:
            None。
        """
        writer = self._writers.get(message.chat_id)
        if writer is None or writer.is_closing():
            return
        await self._write_json(
            writer,
            {
                "type": "assistant",
                "content": message.content,
                "metadata": message.metadata or {},
            },
        )
    
    async def _on_stream_token(self, event: StreamToken) -> StreamToken | None:
        """把流式 token 作为一行 JSON 写回对应 IPC 客户端。

        输入:
            event: StreamToken 事件。

        输出:
            None——不改写事件，让其他 handler 继续收到原事件。
        """
        writer = self._writers.get(event.chat_id)
        if writer is None or writer.is_closing():
            return None
        await self._write_json(
            writer,
            {
                "type": "stream_token",
                "token": event.token,
            },
        )
        return None

    @staticmethod
    async def _write_json(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        """写入一行 JSON。

        输入:
            writer: 目标 StreamWriter。
            payload: 要编码的 JSON 字典。

        输出:
            None。
        """
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()