"""
Telegram Channel

将 Telegram Bot 接入 MessageBus，支持 allowFrom 白名单。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.error import Conflict, NetworkError, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from raven_agent.channels.base import (
    AttachmentStore,
    ChannelAdapter,
    MessageDeduper,
    SessionIdentityIndex,
)
from raven_agent.channels.telegram.utils import (
    TelegramLiveEditQueue,
    TelegramLiveTextMessage,
    TelegramOutboundLimiter,
    _format_tool_intent,
    _format_tool_target,
    _format_turn_live,
    send_markdown,

)
from raven_agent.events import (
    InboundMessage,
    OutboundMessage,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from raven_agent.event_bus import EventBus

if TYPE_CHECKING:
    from raven_agent.message_bus import MessageBus
    from raven_agent.session import SessionManager
    from raven_agent.background.interrupt import InterruptManager

logger = logging.getLogger(__name__)
_CHANNEL = "telegram"
_SEEN_MSG_MAXSIZE = 500


class TelegramChannel(ChannelAdapter):
    """Telegram Bot Channel 适配器。

    通过 python-telegram-bot 接收 Telegram 消息，
    转换为 InboundMessage 发布到 MessageBus，
    并订阅出站消息回送到 Telegram。

    输入:
        token: Telegram Bot Token。
        bus: MessageBus，用于发布入站消息和订阅出站消息。
        session_manager: SessionManager，用于 username → chat_id 索引。
        allow_from: 用户白名单列表（user id 或 @username）；空列表表示允许所有人。
        workspace: 工作区根目录，用于附件存储。

    输出:
        TelegramChannel 实例。
    """

    def __init__(
        self,
        token: str,
        bus: "MessageBus",
        session_manager: "SessionManager",
        allow_from: list[str] | None = None,
        workspace: str | Path | None = None,
        interrupt_manager: "InterruptManager | None" = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._bus = bus
        self._session_manager = session_manager
        self._interrupt_manager = interrupt_manager
        self._event_bus = event_bus
        self._allow_from: set[str] = set(allow_from) if allow_from else set()

        # 消息去重：防止 Telegram 重投
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)

        # 附件存储
        ws_path = Path(workspace) if workspace else None
        self._attachments = AttachmentStore(
            ws_path / "uploads" if ws_path else None,
            channel="telegram",
        )

        # username → chat_id 索引
        self._identity_index = SessionIdentityIndex(
            session_manager,
            channel=_CHANNEL,
            metadata_key="username",
            normalizer=lambda value: value.lower(),
        )

        # Telegram Bot Application
        self._app = Application.builder().token(token).build()
        self._register_handlers()

        # 出站限流器
        self._telegram_outbound_limiter = TelegramOutboundLimiter()

        # typing 持续刷新：chat_id → 后台刷新 task
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}

        # Conflict 处理
        self._polling_conflict_task: asyncio.Task[None] | None = None

        # 连接韧性
        self._watchdog_task: asyncio.Task[None] | None = None
        self._last_polling_ok: float = 0.0
        self._known_chat_ids: set[str] = set()
        self._outage_warning_sent: bool = False
        self._RECONNECT_NOTIFY_THRESHOLD = 60.0

        # ── Live 消息状态 ──
        self._live_edit_queue = TelegramLiveEditQueue(
            limiter=self._telegram_outbound_limiter,
        )
        self._live_messages: dict[str, TelegramLiveTextMessage] = {}
        self._tool_lines: dict[str, list[dict[str, str]]] = {}
        self._reply_buffers: dict[str, str] = {}

    # ── ChannelAdapter 接口 ──────────────────────────────────────

    @property
    def channel_name(self) -> str:
        """返回 Channel 名称。

        输入:
            无。

        输出:
            固定字符串 "telegram"。
        """
        return _CHANNEL

    async def start(self) -> None:
        """启动 Telegram Channel。

        输入:
            无。

        输出:
            None。启动后开始接收 Telegram 消息并订阅出站。
        """
        self._identity_index.rebuild()
        self._bus.subscribe_outbound(_CHANNEL, self._on_outbound)
        if self._event_bus is not None:
            self._event_bus.on(TurnStarted, self._on_turn_started)
            self._event_bus.on(ToolCallStarted, self._on_tool_call_started)
            self._event_bus.on(ToolCallCompleted, self._on_tool_call_completed)

        await self._app.initialize()
        await self._app.start()
        await self._register_bot_commands()
        updater = self._app.updater
        if updater is None:
            raise RuntimeError("Telegram updater 未初始化")
        await updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            error_callback=self._on_polling_error,
        )
        # 启动 polling 存活看门狗
        self._watchdog_task = asyncio.create_task(self._polling_watchdog())

        logger.info(
            "[telegram] TelegramChannel 已启动  已知用户: %d",
            len(self._identity_index.mapping),
        )

    async def stop(self) -> None:
        """停止 Telegram Channel。

        输入:
            无。

        输出:
            None。停止后释放 polling、bot 和所有连接。
        """
        # 停止看门狗
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self._polling_conflict_task and not self._polling_conflict_task.done():
            await self._polling_conflict_task
        updater = self._app.updater
        if updater and updater.running:
            await updater.stop()
        await self._app.stop()
        for task in list(self._typing_tasks.values()):
            if not task.done():
                task.cancel()
        self._typing_tasks.clear()
        await self._app.shutdown()
        logger.info("[telegram] TelegramChannel 已停止")

    # ── 公开发送方法（供 MessagePushTool 注册） ────────────────

    async def send(self, chat_id: str, message: str) -> None:
        """发送 Markdown 文本消息。

        输入:
            chat_id: 目标 chat_id（数字字符串或 @username）。
            message: Markdown 格式文本。

        输出:
            None。
        """
        await send_markdown(
            self._app.bot,
            self._resolve_chat_id(chat_id),
            message,
            self._telegram_outbound_limiter,
        )

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
    ) -> None:
        """发送文件。

        输入:
            chat_id: 目标 chat_id。
            file_path: 本地文件路径。
            name: 可选文件名（用于 Telegram 显示）。

        输出:
            None。
        """
        cid = int(self._resolve_chat_id(chat_id))
        await self._telegram_outbound_limiter.run(
            cid,
            kind="send",
            label="send_document",
            action=lambda: self._send_document_file(cid, file_path, name),
        )

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片。

        输入:
            chat_id: 目标 chat_id。
            image: 本地文件路径或 HTTP(S) URL。

        输出:
            None。
        """
        cid = int(self._resolve_chat_id(chat_id))
        if image.startswith(("http://", "https://")):
            await self._telegram_outbound_limiter.run(
                cid,
                kind="send",
                label="send_photo",
                action=lambda: self._app.bot.send_photo(chat_id=cid, photo=image),
            )
        else:
            await self._telegram_outbound_limiter.run(
                cid,
                kind="send",
                label="send_photo",
                action=lambda: self._send_photo_file(cid, image),
            )

    # ── 私有方法：身份解析 ──────────────────────────────────────

    def _resolve_chat_id(self, chat_id: str) -> str:
        """将 username 或 chat_id 解析为数字 chat_id。

        输入:
            chat_id: 可能是数字字符串（如 "123456789"）或 @username。

        输出:
            数字 chat_id 字符串。

        异常:
            ValueError: 当 username 找不到对应的 chat_id 时。
        """
        resolved = chat_id.lstrip("@").lower()
        if not resolved.lstrip("-").isdigit():
            resolved = self._identity_index.resolve(resolved)
            if not resolved:
                known = list(self._identity_index.mapping.keys()) or ["（无）"]
                raise ValueError(
                    f"找不到用户 {chat_id!r} 的 chat_id，该用户需先给 bot 发一条消息。"
                    f"已知用户：{known}"
                )
        return resolved

    def _is_allowed(self, user) -> bool:
        """检查用户是否在白名单中。

        输入:
            user: Telegram User 对象。

        输出:
            True 表示允许该用户使用 Bot。
        """
        if not self._allow_from:
            return True
        return str(user.id) in self._allow_from or (
            user.username
            and user.username.lower() in {u.lower() for u in self._allow_from}
        )

    # ── 私有方法：消息处理 handler ──────────────────────────────

    def _register_handlers(self) -> None:
        """注册 Telegram 消息处理器。

        输入:
            无。

        输出:
            None。
        """
        # /stop 走控制面：绕过 MessageBus，直接调用 interrupt_controller
        self._app.add_handler(
            CommandHandler("stop", self._on_stop_command)
        )
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            MessageHandler(filters.COMMAND, self._on_command)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO & ~filters.COMMAND, self._on_photo)
        )
        self._app.add_handler(
            MessageHandler(filters.Document.ALL & ~filters.COMMAND, self._on_document)
        )
        self._app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self._on_voice)
        )

    async def _on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理普通文本消息。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。消息被转换为 InboundMessage 并发布到 MessageBus。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.text or not chat or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权用户  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        # 去重
        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            logger.warning(
                "[telegram] 重复消息已忽略  chat_id=%s  message_id=%s",
                chat.id, msg.message_id,
            )
            return

        chat_id_str = str(chat.id)
        self._known_chat_ids.add(chat_id_str)
        await self._remember_username(chat_id_str, user.username)
        self._launch_typing(chat.id, context)

        inbound_text, reply_meta = _build_inbound_text_with_reply(
            msg.text, msg.reply_to_message
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=chat_id_str,
                content=inbound_text,
                metadata={
                    "username": user.username or "",
                    **reply_meta,
                },
            )
        )
        preview = msg.text[:60] + "..." if len(msg.text) > 60 else msg.text
        logger.info(
            "[telegram] 收到消息  chat_id=%s  user=@%s  内容: %r",
            chat.id, user.username or user.id, preview,
        )

    async def _on_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理 Telegram 命令（如 /start）。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权命令  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        chat_id_str = str(chat.id)
        self._known_chat_ids.add(chat_id_str)
        await self._remember_username(chat_id_str, user.username)
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=chat_id_str,
                content=str(getattr(msg, "text", "") or ""),
                metadata={"username": user.username or ""},
            )
        )

    async def _on_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理 /stop 命令 —— 绕过 MessageBus，直接调用中断控制面。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。中断结果直接回复到 Telegram，不入队。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权 /stop  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        chat_id_str = str(chat.id)
        session_key = f"{_CHANNEL}:{chat_id_str}"
        self._known_chat_ids.add(chat_id_str)

        if self._interrupt_manager is None:
            await send_markdown(
                self._app.bot,
                chat_id_str,
                "中断系统未启用。",
                self._telegram_outbound_limiter,
            )
            return

        result = self._interrupt_manager.request_interrupt(
            session_key=session_key,
            sender=str(user.id),
            command="/stop",
        )
        await send_markdown(
            self._app.bot,
            chat_id_str,
            result.message,
            self._telegram_outbound_limiter,
        )
        logger.info(
            "[telegram] /stop 已处理  chat_id=%s  status=%s",
            chat_id_str, result.status,
        )

    async def _on_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理图片消息。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。图片被下载到本地后，路径放入 InboundMessage.media。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.photo or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权用户  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            logger.warning(
                "[telegram] 重复图片消息已忽略  chat_id=%s  message_id=%s",
                chat.id, msg.message_id,
            )
            return

        chat_id_str = str(chat.id)
        self._known_chat_ids.add(chat_id_str)
        await self._remember_username(chat_id_str, user.username)
        self._launch_typing(chat.id, context)

        # 下载最高分辨率图片
        try:
            tg_file = await context.bot.get_file(msg.photo[-1].file_id)
            tmp = self._attachments.create_path("photo_", ".jpg")
            await tg_file.download_to_drive(tmp)
        except Exception as e:
            logger.error(
                "[telegram] 图片下载失败  chat_id=%s  user=@%s  err=%s",
                chat.id, user.username or user.id, e,
            )
            return
        logger.info(
            "[telegram] 收到图片  chat_id=%s  user=@%s  path=%s",
            chat.id, user.username or user.id, tmp,
        )

        caption_text = msg.caption or ""
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            caption_text, msg.reply_to_message
        )
        media = [str(tmp)]
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=chat_id_str,
                content=inbound_text,
                media=media,
                metadata={
                    "username": user.username or "",
                    **reply_meta,
                },
            )
        )

    async def _on_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理文件消息。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。文件被下载到本地后，路径放入 InboundMessage.media。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.document or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权用户  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        chat_id_str = str(chat.id)
        self._known_chat_ids.add(chat_id_str)
        await self._remember_username(chat_id_str, user.username)
        self._launch_typing(chat.id, context)

        doc = msg.document
        suffix = ""
        if doc.file_name and "." in doc.file_name:
            suffix = "." + doc.file_name.rsplit(".", 1)[-1]
        try:
            tg_file = await context.bot.get_file(doc.file_id)
            tmp = self._attachments.create_path("doc_", suffix)
            await tg_file.download_to_drive(tmp)
        except Exception as e:
            logger.error(
                "[telegram] 文件下载失败  chat_id=%s  user=@%s  filename=%r  err=%s",
                chat.id, user.username or user.id, doc.file_name, e,
            )
            return
        logger.info(
            "[telegram] 收到文件  chat_id=%s  user=@%s  filename=%r  tmp=%s",
            chat.id, user.username or user.id, doc.file_name, tmp,
        )

        caption_text = msg.caption or ""
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            caption_text, msg.reply_to_message
        )
        if doc.file_name:
            inbound_text = f"[文件: {doc.file_name}]\n{inbound_text}".strip()
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=chat_id_str,
                content=inbound_text,
                media=[str(tmp)],
                metadata={
                    "username": user.username or "",
                    "document_filename": doc.file_name or "",
                    "document_mime_type": doc.mime_type or "",
                    **reply_meta,
                },
            )
        )
    
    async def _on_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """处理语音消息。

        输入:
            update: Telegram Update 对象。
            context: Telegram 上下文。

        输出:
            None。语音文件被下载到本地后，路径放入 InboundMessage.media。
        """
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.voice or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                "[telegram] 拒绝未授权用户  id=%s  username=@%s",
                user.id, user.username,
            )
            return

        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            logger.warning(
                "[telegram] 重复语音消息已忽略  chat_id=%s  message_id=%s",
                chat.id, msg.message_id,
            )
            return

        chat_id_str = str(chat.id)
        self._known_chat_ids.add(chat_id_str)
        await self._remember_username(chat_id_str, user.username)
        self._launch_typing(chat.id, context)

        # 下载语音文件
        try:
            tg_file = await context.bot.get_file(msg.voice.file_id)
            tmp = self._attachments.create_path("voice_", ".ogg")
            await tg_file.download_to_drive(tmp)
        except Exception as e:
            logger.error(
                "[telegram] 语音下载失败  chat_id=%s  user=@%s  err=%s",
                chat.id, user.username or user.id, e,
            )
            return
        logger.info(
            "[telegram] 收到语音  chat_id=%s  user=@%s  duration=%ss  path=%s",
            chat.id, user.username or user.id, msg.voice.duration, tmp,
        )

        # 构造入站消息
        duration = msg.voice.duration or 0
        voice_text = f"[语音消息，{duration}秒]"
        if msg.caption:
            voice_text = f"{msg.caption}\n{voice_text}"

        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=chat_id_str,
                content=voice_text,
                media=[str(tmp)],
                metadata={
                    "username": user.username or "",
                    "voice_duration": duration,
                },
            )
        )

    # ── 私有方法：出站回复 ──────────────────────────────────────

    async def _on_outbound(self, msg: OutboundMessage) -> None:
        """处理出站消息，发送到 Telegram。

        输入:
            msg: Agent 产生的 OutboundMessage。

        输出:
            None。
        """
        cid = int(self._resolve_chat_id(msg.chat_id))
        # —— 删除 Live 进度消息 ——
        session_key = f"{_CHANNEL}:{msg.chat_id}"
        if session_key in self._live_messages:
            await self._cleanup_live_message(session_key)
        # 取消可能存在的旧 typing...
        self._cancel_typing(cid)
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info("[telegram] 发送回复  chat_id=%s  内容: %r", msg.chat_id, preview)

        # 发送文本
        if msg.content.strip():
            await send_markdown(
                self._app.bot,
                msg.chat_id,
                msg.content,
                self._telegram_outbound_limiter,
            )

        # 发送附件（按文件类型分发）
        _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        for file_path in (msg.media or []):
            try:
                ext = Path(file_path).suffix.lower()
                if ext in _IMAGE_EXTS:
                    await self.send_image(str(msg.chat_id), file_path)
                else:
                    # HTML / PDF / 其他文件 → 以 Document 形式发送
                    await self.send_file(str(msg.chat_id), file_path)
            except Exception as e:
                logger.warning(
                    "[telegram] 附件发送失败  chat_id=%s  path=%s  err=%s",
                    msg.chat_id, file_path, e,
                )

    # ── 私有方法：辅助 ──────────────────────────────────────────

    async def _register_bot_commands(self) -> None:
        """向 Telegram 注册 Bot 命令列表。

        输入:
            无。

        输出:
            None。
        """
        commands = [
            BotCommand("start", "开始使用"),
        ]
        await self._app.bot.set_my_commands(commands)

    async def _remember_username(self, chat_id: str, username: str | None) -> None:
        """记录 username → chat_id 映射。

        输入:
            chat_id: Telegram chat_id。
            username: Telegram @username（可能为 None）。

        输出:
            None。
        """
        if username:
            await self._identity_index.remember(username, chat_id)

    async def _safe_send_typing(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int
    ) -> None:
        """安全发送 typing 指示器。

        输入:
            context: Telegram 上下文。
            chat_id: 目标聊天 ID。

        输出:
            None。失败时仅记录日志，不影响主流程。
        """
        try:
            await self._telegram_outbound_limiter.run(
                chat_id,
                kind="typing",
                label="send_chat_action",
                action=lambda: context.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING
                ),
            )
        except Exception as e:
            logger.warning(
                "[telegram] send_chat_action 失败，已跳过 typing chat_id=%s err=%s",
                chat_id, e,
            )

    async def _send_document_file(
        self,
        chat_id: int,
        file_path: str,
        name: str | None,
    ) -> object:
        """发送本地文件。

        输入:
            chat_id: 目标聊天 ID。
            file_path: 本地文件路径。
            name: 可选显示名称。

        输出:
            Telegram Message 对象。
        """
        with open(file_path, "rb") as f:
            return await self._app.bot.send_document(
                chat_id=chat_id, document=f, filename=name,
            )

    async def _send_photo_file(self, chat_id: int, image: str) -> object:
        """发送本地图片。

        输入:
            chat_id: 目标聊天 ID。
            image: 本地图片路径。

        输出:
            Telegram Message 对象。
        """
        with open(image, "rb") as f:
            return await self._app.bot.send_photo(chat_id=chat_id, photo=f)

    # ── 私有方法：polling 错误处理 ──────────────────────────────

    def _on_polling_error(self, exc: TelegramError) -> None:
        """处理 Telegram polling 异常。

        输入:
            exc: Telegram 异常。

        输出:
            None。Conflict 时自动停止 polling 并保留发送能力。
        """
        if isinstance(exc, Conflict):
            if self._polling_conflict_task is None:
                logger.error(
                    "[telegram] 检测到 getUpdates 冲突，已暂停 Telegram 接收。"
                    "请确保同一 bot token 仅运行一个轮询实例。"
                )
                self._polling_conflict_task = asyncio.create_task(
                    self._disable_polling_on_conflict()
                )
            return
        logger.warning("[telegram] polling 异常，框架将自动重试: %s", exc)

    async def _disable_polling_on_conflict(self) -> None:
        """Conflict 时关闭 updater 轮询，保留 bot 发送能力。

        输入:
            无。

        输出:
            None。
        """
        updater = self._app.updater
        if updater is None or not updater.running:
            return
        try:
            await updater.stop()
            logger.warning(
                "[telegram] polling 已停止；当前进程不再接收 Telegram 消息。"
            )
        except Exception as e:
            logger.warning("[telegram] 停止 polling 失败: %s", e)


    # ── 私有方法：polling 存活看门狗 ─────────────────────────


    async def _polling_watchdog(self) -> None:
        """每 30s 检查 polling 存活状态。

        - polling 正常：更新 _last_polling_ok，重置断联状态
        - polling 死亡：尝试自动重启
        - 断联超过阈值未恢复：向已知用户发送断联警告
        - 重启成功 + 之前断联过：发送恢复通知

        输入:
            无。

        输出:
            None。作为后台 Task 持续运行直到被 cancel。
        """
        self._last_polling_ok = time.monotonic()
        outage_start: float | None = None

        while True:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return

            updater = self._app.updater
            if updater is None:
                continue

            polling_alive = updater.running
            now = time.monotonic()

            if polling_alive:
                # ── polling 存活 ──
                self._last_polling_ok = now
                if outage_start is not None:
                    outage_duration = now - outage_start
                    outage_start = None
                    self._outage_warning_sent = False
                    if outage_duration > self._RECONNECT_NOTIFY_THRESHOLD:
                        logger.warning(
                            "[telegram] polling 已恢复（中断 %.0fs），通知已知用户",
                            outage_duration,
                        )
                        await self._notify_reconnect(outage_duration)
                    else:
                        logger.info(
                            "[telegram] polling 已恢复（短暂中断 %.0fs，不通知用户）",
                            outage_duration,
                        )
            else:
                # ── polling 已停止 ──
                if outage_start is None:
                    outage_start = now
                    uptime_before_death = now - self._last_polling_ok
                    logger.warning(
                        "[telegram] polling 已停止！上次存活 %.0fs 前，尝试看门狗重启...",
                        uptime_before_death,
                    )

                # 断联超过阈值且尚未发过警告 → 发断联警告
                outage_duration = now - outage_start
                if (
                    outage_duration > self._RECONNECT_NOTIFY_THRESHOLD
                    and not self._outage_warning_sent
                ):
                    logger.warning(
                        "[telegram] 断联超过 %.0fs 未恢复，发送断联警告",
                        outage_duration,
                    )
                    await self._notify_disconnect(outage_duration)
                    self._outage_warning_sent = True

                # 尝试重启 polling
                try:
                    await updater.start_polling(
                        allowed_updates=Update.ALL_TYPES,
                        error_callback=self._on_polling_error,
                    )
                    logger.info("[telegram] polling 已通过看门狗重启成功")
                except Exception as e:
                    logger.error(
                        "[telegram] polling 看门狗重启失败（30s 后重试）: %s", e
                    )

    async def _notify_disconnect(self, outage_seconds: float) -> None:
        """向所有已知 Telegram 用户发送断联警告。

        bot.send_message() 走独立 HTTP POST，不依赖 polling，
        即使 polling 死了也能发出。

        输入:
            outage_seconds: 断联已持续的秒数。

        输出:
            None。单个用户通知失败不影响其他用户。
        """
        chat_ids: set[str] = set(self._known_chat_ids)
        chat_ids.update(self._identity_index.mapping.values())

        if not chat_ids:
            logger.info("[telegram] 无已知用户，跳过断联警告")
            return

        minutes = int(outage_seconds / 60)
        if minutes >= 1:
            msg = (
                f"⚠️ Raven 暂时断联（已持续约 {minutes} 分钟）。"
                f"正在尝试重连，恢复后会自动通知你。"
            )
        else:
            msg = (
                f"⚠️ Raven 暂时断联（已持续约 {int(outage_seconds)} 秒）。"
                f"正在尝试重连，恢复后会自动通知你。"
            )

        count = 0
        for chat_id in chat_ids:
            try:
                await send_markdown(
                    self._app.bot,
                    chat_id,
                    msg,
                    self._telegram_outbound_limiter,
                )
                count += 1
            except Exception as e:
                logger.warning(
                    "[telegram] 断联警告发送失败 chat_id=%s: %s", chat_id, e,
                )

        logger.info(
            "[telegram] 已向 %d/%d 位用户发送断联警告",
            count, len(chat_ids),
        )

    async def _notify_reconnect(self, outage_seconds: float) -> None:
        """向所有已知 Telegram 用户发送恢复通知。

        输入:
            outage_seconds: 中断时长（秒）。

        输出:
            None。单个用户通知失败不影响其他用户。
        """
        chat_ids: set[str] = set(self._known_chat_ids)
        chat_ids.update(self._identity_index.mapping.values())

        if not chat_ids:
            logger.info("[telegram] 无已知用户，跳过恢复通知")
            return

        minutes = int(outage_seconds / 60)
        if minutes >= 1:
            msg = (
                f"⚡ Raven 已恢复在线（中断了约 {minutes} 分钟）。"
                f"继续为你服务。"
            )
        else:
            msg = (
                f"⚡ Raven 已恢复在线（中断了约 {int(outage_seconds)} 秒）。"
                f"继续为你服务。"
            )

        count = 0
        for chat_id in chat_ids:
            try:
                await send_markdown(
                    self._app.bot,
                    chat_id,
                    msg,
                    self._telegram_outbound_limiter,
                )
                count += 1
            except Exception as e:
                logger.warning(
                    "[telegram] 恢复通知发送失败 chat_id=%s: %s", chat_id, e,
                )

        logger.info(
            "[telegram] 已向 %d/%d 位用户发送恢复通知",
            count, len(chat_ids),
        )


    # ── 私有方法：typing 刷新循环 ──────────────────────────────
    async def _typing_loop(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """后台循环：每 4 秒重发一次 typing，直到被 cancel。

        输入:
            chat_id: 目标聊天 ID。
            context: Telegram 上下文。

        输出:
            None。协程结束时 typing 自然停止。
        """
        try:
            while True:
                await self._safe_send_typing(context, chat_id)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    def _launch_typing(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """启动 typing 刷新循环。自动取消同一 chat 已有的循环。

        输入:
            chat_id: 目标聊天 ID。
            context: Telegram 上下文。

        输出:
            None。
        """
        self._cancel_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(
            self._typing_loop(chat_id, context)
        )

    def _cancel_typing(self, chat_id: int) -> None:
        """取消指定 chat 的 typing 刷新循环。

        输入:
            chat_id: 目标聊天 ID。

        输出:
            None。
        """
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
    
    # ── 事件处理：Live 消息 ──────────────────────────────────────

    async def _on_turn_started(self, event: TurnStarted) -> TurnStarted | None:
        """新轮次开始时清理上一轮的 Live 状态。

        输入:
            event: TurnStarted 事件。

        输出:
            None——不改写事件。
        """
        if event.inbound.channel != _CHANNEL:
            return None
        session_key = event.session_key
        self._tool_lines.pop(session_key, None)
        self._reply_buffers.pop(session_key, None)
        live_msg = self._live_messages.pop(session_key, None)
        if live_msg is not None:
            asyncio.create_task(live_msg.delete())
        return None

    async def _on_tool_call_started(
        self, event: ToolCallStarted
    ) -> ToolCallStarted | None:
        """工具调用开始时追加工具行并刷新 Live 消息。

        输入:
            event: ToolCallStarted 事件。

        输出:
            None——不改写事件。
        """
        if event.channel != _CHANNEL:
            return None
        cid = int(self._resolve_chat_id(event.chat_id))
        if cid <= 0:
            return None
        lines = self._tool_lines.setdefault(event.session_key, [])
        lines.append({
            "tool_name": event.tool_name,
            "intent": _format_tool_intent(event.arguments),
            "target": _format_tool_target(event.arguments),
            "status": "running",
        })
        asyncio.create_task(
            self._sync_live_message(event.session_key, cid)
        )
        return None

    async def _on_tool_call_completed(
        self, event: ToolCallCompleted
    ) -> ToolCallCompleted | None:
        """工具调用完成时更新工具状态并刷新 Live 消息。

        输入:
            event: ToolCallCompleted 事件。

        输出:
            None——不改写事件。
        """
        if event.channel != _CHANNEL:
            return None
        cid = int(self._resolve_chat_id(event.chat_id))
        if cid <= 0:
            return None
        lines = self._tool_lines.setdefault(event.session_key, [])
        matched = False
        for line in reversed(lines):
            if (
                line["status"] == "running"
                and line["tool_name"] == event.tool_name
            ):
                line["status"] = "done" if event.status == "success" else "error"
                matched = True
                break
        if not matched:
            lines.append({
                "tool_name": event.tool_name,
                "intent": _format_tool_intent(
                    event.final_arguments or event.arguments
                ),
                "target": _format_tool_target(
                    event.final_arguments or event.arguments
                ),
                "status": "done" if event.status == "success" else "error",
            })
        asyncio.create_task(
            self._sync_live_message(event.session_key, cid)
        )
        return None

    async def _sync_live_message(
        self,
        session_key: str,
        chat_id: int,
        *,
        terminal: bool = False,
    ) -> None:
        """把当前工具行和回复缓冲渲染为 Live 消息。

        输入:
            session_key: 会话 key。
            chat_id: 目标聊天 ID。
            terminal: 是否为最终状态。

        输出:
            None。
        """
        text = _format_turn_live(
            self._tool_lines.get(session_key, []),
            self._reply_buffers.get(session_key, ""),
            terminal=terminal,
        )
        if not text:
            return
        message = self._live_messages.get(session_key)
        if message is None:
            message = TelegramLiveTextMessage(
                self._app.bot,
                self._live_edit_queue,
                chat_id,
            )
            self._live_messages[session_key] = message
        await message.update(text, force=terminal)

    async def _cleanup_live_message(self, session_key: str) -> None:
        """删除 Live 消息并清理所有相关状态。

        输入:
            session_key: 会话 key。

        输出:
            None。
        """
        message = self._live_messages.pop(session_key, None)
        if message is not None:
            await message.delete()
        self._tool_lines.pop(session_key, None)
        self._reply_buffers.pop(session_key, None)
    
    
    # ── MessagePushTool sender 注册 ────────────────────────────────

    def register_push_senders(self, push_tool: Any) -> None:
        """向 MessagePushTool 注册本 Channel 的 sender。

        输入:
            push_tool: MessagePushTool 实例。

        输出:
            None。
        """
        push_tool.register_channel(
            _CHANNEL,
            text=self.send,
            file=self.send_file,
            image=self.send_image,
        )


# ── 模块级工具函数 ────────────────────────────────────────────────

def _build_inbound_text_with_reply(
    user_text: str,
    reply_msg,
) -> tuple[str, dict[str, str | int]]:
    """将 Telegram reply 上下文合并进入站文本。

    输入:
        user_text: 用户当前消息文本。
        reply_msg: Telegram 被回复的 Message 对象。

    输出:
        (合并后的文本, 附加元数据字典)。
    """
    text = (user_text or "").strip()
    if not reply_msg:
        return text, {}

    reply_text = (reply_msg.text or reply_msg.caption or "").strip()
    if not reply_text:
        if getattr(reply_msg, "photo", None):
            reply_text = "[图片]"
        else:
            return text, {"reply_to_message_id": int(reply_msg.message_id)}

    reply_sender = ""
    from_user = getattr(reply_msg, "from_user", None)
    if from_user:
        reply_sender = from_user.username or str(from_user.id)
    sender_label = f"@{reply_sender}" if reply_sender else "未知发送者"

    merged = (
        "【你正在回复一条历史消息】\n"
        f"被回复消息（来自 {sender_label}）：\n"
        f"{reply_text}\n\n"
        "【你当前新消息】\n"
        f"{text}"
    ).strip()
    return merged, {
        "reply_to_message_id": int(reply_msg.message_id),
        "reply_to_sender": sender_label,
    }