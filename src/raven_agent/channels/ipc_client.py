from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Literal

from raven_agent.channels.ipc_server import parse_tcp_endpoint

_EXIT_CMDS = {"exit", "quit", "q"}

SessionMode = Literal["new", "continue", "resume"]


class IPCClient:
    """连接 IPC Server 的命令行客户端，支持 CLI session 选择。

    输入:
        socket_path: Unix socket 路径或 TCP endpoint。
        mode: 启动模式，new / continue / resume。

    输出:
        IPCClient 实例。
    """

    def __init__(self, socket_path: str, *, mode: SessionMode = "new") -> None:
        self._socket_path = socket_path
        self._mode = mode

    async def run(self) -> None:
        """运行客户端：先握手选择 session，再进入交互循环。

        输入:
            无。

        输出:
            None。用户输入 exit/quit/q 后返回。
        """
        try:
            reader, writer = await self._connect()
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            print(f"无法连接到 agent（{self._socket_path}），请先启动：python main.py serve")
            return

        # 1. 握手阶段：由 handshake 独占读取，确定并绑定 session
        chat_id = await self._handshake(reader, writer)
        if chat_id is None:
            writer.close()
            await writer.wait_closed()
            return

        print(f"Raven Agent CLI  |  session={chat_id}  |  输入 exit 退出")

        # 2. 交互阶段：严格回合制——发一条 → 等到 assistant 回复 → 打印 → 再读下一条。
        # 不用后台接收任务，避免输入提示与回复交错、提示重复。
        try:
            while True:
                text = await self._read_line()  # 写出 "\nYou> " 并读取一行
                stripped = text.strip()
                if stripped.lower() in _EXIT_CMDS:
                    break
                if not stripped:
                    continue
                await self._send_content(writer, stripped)
                reply = await self._await_reply(reader)
                if reply is None:
                    print("\n连接已断开")
                    break
                if reply != "":
                    # 如果流式输出了，我们默认传回空字符串，避免重复打印。
                    print(f"Raven> {reply}")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print("\n再见")

    async def _handshake(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> str | None:
        """根据 mode 完成 session 选择，返回绑定的 chat_id。

        输入:
            reader: 服务端读取流。
            writer: 服务端写入流。

        输出:
            绑定成功的 chat_id；用户取消或失败时返回 None。
        """
        if self._mode == "new":
            await self._send_command(writer, {"command": "session.new"})
            return await self._await_bound(reader)

        if self._mode == "continue":
            await self._send_command(writer, {"command": "session.continue_latest"})
            return await self._await_bound(reader)

        # resume：先列出，再让用户选择
        await self._send_command(writer, {"command": "session.list", "limit": 20})
        listing = await self._await_command_result(reader, "session.list")
        if listing is None:
            print("获取 session 列表失败")
            return None
        sessions = listing.get("sessions", [])
        chosen_chat_id = self._prompt_select(sessions)
        if chosen_chat_id is None:
            # 用户选择新建
            await self._send_command(writer, {"command": "session.new"})
        else:
            await self._send_command(
                writer, {"command": "session.attach", "chat_id": chosen_chat_id}
            )
        return await self._await_bound(reader)

    def _prompt_select(self, sessions: list[dict[str, Any]]) -> str | None:
        """打印 session 列表并读取用户选择。

        输入:
            sessions: 服务端返回的 session 摘要列表。

        输出:
            用户选择的 chat_id；用户输入 n（新建）或列表为空时返回 None。
        """
        if not sessions:
            print("没有可恢复的历史 CLI session，将新建一个。\n")
            return None

        print("请选择要恢复的 CLI session：\n")
        for item in sessions:
            updated = item.get("updated_at", "")
            count = item.get("message_count", 0)
            preview = item.get("preview", "")
            print(f"[{item.get('index')}] {updated}  {count} messages")
            if preview:
                print(f"    {preview}")
        print("\n输入编号恢复，或输入 n 新建 session：")

        sys.stdout.write("> ")
        sys.stdout.flush()
        choice = sys.stdin.readline().strip()
        if choice.lower() in {"n", "new", ""}:
            return None
        for item in sessions:
            if str(item.get("index")) == choice:
                return str(item.get("chat_id"))
        print("输入无效，将新建 session。\n")
        return None

    async def _await_bound(self, reader: asyncio.StreamReader) -> str | None:
        """等待 session.bound 回执。

        输入:
            reader: 服务端读取流。

        输出:
            绑定的 chat_id；失败时返回 None。
        """
        result = await self._await_command_result(reader, "session.bound")
        if result is None or not result.get("ok"):
            return None
        return str(result.get("chat_id"))

    async def _await_command_result(
        self,
        reader: asyncio.StreamReader,
        command: str,
    ) -> dict[str, Any] | None:
        """读取直到匹配指定 command 的 command_result。

        输入:
            reader: 服务端读取流。
            command: 期望的命令名，例如 "session.bound"。

        输出:
            匹配的结果字典；连接断开时返回 None。
        """
        while True:
            line = await reader.readline()
            if not line:
                return None
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "command_result" and data.get("command") == command:
                return data

    @staticmethod
    async def _send_command(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        """发送一条控制命令。

        输入:
            writer: 服务端写入流。
            payload: 命令字典，会自动补上 type=command。

        输出:
            None。
        """
        message = {"type": "command", **payload}
        writer.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """连接 IPC Server。

        输入:
            无。使用 self._socket_path。

        输出:
            (reader, writer) 元组。
        """
        endpoint = parse_tcp_endpoint(self._socket_path)
        if endpoint is not None:
            return await asyncio.open_connection(*endpoint)
        if not hasattr(asyncio, "open_unix_connection"):
            raise OSError("Unix sockets are unavailable on this platform.")
        return await asyncio.open_unix_connection(self._socket_path)

    @staticmethod
    async def _send_content(writer: asyncio.StreamWriter, content: str) -> None:
        """发送一条普通对话消息。

        输入:
            writer: 服务端写入流。
            content: 用户输入内容。

        输出:
            None。
        """
        payload = json.dumps({"content": content}, ensure_ascii=False) + "\n"
        writer.write(payload.encode("utf-8"))
        await writer.drain()

    @staticmethod
    async def _await_reply(reader: asyncio.StreamReader) -> str | None:
        """等待服务端的 assistant 回复。

        输入:
            reader: 服务端读取流。

        输出:
            assistant 消息内容；连接断开时返回 None。
            其它类型（如 command_result）会被跳过。
        """
        _streamed = False
        while True:
            line = await reader.readline()
            if not line:
                return None
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "stream_token":
                if not _streamed:
                    print("Raven> ", end="", flush=True)
                    _streamed = True
                print(data.get("token", ""), end="", flush=True)
                continue
            elif data.get("type") == "assistant":
                if _streamed:
                    print()  # 如果之前有流式输出，先换行
                    return ""  # 流式输出时，传回空字符串，避免重复打印
                return str(data.get("content", ""))

    @staticmethod
    async def _read_line() -> str:
        """异步读取用户输入。

        输入:
            无。

        输出:
            用户输入的一行文本。
        """
        loop = asyncio.get_event_loop()
        sys.stdout.write("\nYou> ")
        sys.stdout.flush()
        return await loop.run_in_executor(None, sys.stdin.readline)


def run_client(socket_path: str, *, mode: SessionMode = "new") -> None:
    """同步方式运行 IPCClient。

    输入:
        socket_path: Unix socket 路径或 TCP endpoint。
        mode: 启动模式，new / continue / resume。

    输出:
        None。
    """
    asyncio.run(IPCClient(socket_path, mode=mode).run())