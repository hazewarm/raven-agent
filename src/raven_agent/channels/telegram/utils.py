"""
Telegram Markdown 发送工具

将 Markdown 文本转换成 Telegram text+entities 后发送：
- 自动分段（超出 4096 字符时）
- 转换失败时降级为纯文本
- 提供 TelegramOutboundLimiter 遵守 API 频率限制
"""

from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from telegram import Bot
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import MessageEntity, split_entities

logger = logging.getLogger(__name__)
_TELEGRAM_MSG_LIMIT = 4096
_T = TypeVar("_T")
_LIVE_EDIT_MIN_INTERVAL_S = 1.0  # 两次编辑最小间隔
_LIVE_MAX_FLOOD_STRIKES = 3      # 连续限流后停止更新
_LIVE_MESSAGE_LIMIT = 3900       # Live 消息最大字符数（UTF-16 code units）


class TelegramOutboundLimiter:
    """Telegram API 出站频率限制器。

    遵守 Telegram 的 send/edit/typing 各操作的最小间隔，避免触发 flood control。

    输入:
        send_interval_s: 两次 send 之间的最小间隔，默认 2.0s。
        edit_interval_s: 两次 edit 之间的最小间隔，默认 5.0s。
        typing_interval_s: 两次 typing 之间的最小间隔，默认 8.0s。
        global_interval_s: 全局最小间隔，默认 0.25s。
        retry_padding_s: RetryAfter 惩罚的额外缓冲，默认 1.0s。
        max_attempts: 最大重试次数，默认 5。

    输出:
        TelegramOutboundLimiter 实例。
    """

    def __init__(
        self,
        *,
        send_interval_s: float = 2.0,
        edit_interval_s: float = 5.0,
        typing_interval_s: float = 8.0,
        global_interval_s: float = 0.25,
        retry_padding_s: float = 1.0,
        max_attempts: int = 5,
    ) -> None:
        self._send_interval_s = send_interval_s
        self._edit_interval_s = edit_interval_s
        self._typing_interval_s = typing_interval_s
        self._global_interval_s = global_interval_s
        self._retry_padding_s = retry_padding_s
        self._max_attempts = max_attempts
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._typing_locks: dict[int, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._next_chat_at: dict[int, float] = {}
        self._next_typing_at: dict[int, float] = {}
        self._next_global_at = 0.0

    async def run(
        self,
        chat_id: int | str,
        *,
        kind: str,
        label: str,
        action: Callable[[], Awaitable[_T]],
        max_attempts: int | None = None,
    ) -> _T:
        """在频率限制保护下执行一个 Telegram API 调用。

        输入:
            chat_id: 目标聊天 ID。
            kind: 操作类型，send / edit / typing。
            label: 日志标签。
            action: 要执行的异步操作。
            max_attempts: 最大重试次数；不传则使用实例默认值。

        输出:
            action 的返回值。
        """
        cid = int(chat_id)
        if kind == "typing":
            return await self._run_typing(cid, label=label, action=action)
        attempts = max_attempts or self._max_attempts
        lock = self._chat_locks.setdefault(cid, asyncio.Lock())
        async with lock:
            last_err: Exception | None = None
            for attempt in range(1, attempts + 1):
                await self._wait_for_chat_slot(cid)
                try:
                    result = await self._run_with_global_slot(action)
                    self._mark_used(cid, kind)
                    return result
                except RetryAfter as e:
                    last_err = e
                    delay = max(
                        float(getattr(e, "retry_after", 1.0) or 1.0) + self._retry_padding_s,
                        self._interval(kind),
                    )
                    self._cooldown(cid, delay)
                    logger.warning(
                        "[telegram] %s 命中限流，按 retry_after 冷却 chat_id=%s attempt=%d/%d delay=%.1fs",
                        label, cid, attempt, attempts, delay,
                    )
                except (TimedOut, NetworkError) as e:
                    last_err = e
                    delay = min(0.8 * (2 ** (attempt - 1)), 8.0)
                    self._cooldown(cid, delay)
                    logger.warning(
                        "[telegram] %s 网络失败，准备重试 chat_id=%s attempt=%d/%d delay=%.1fs err=%s",
                        label, cid, attempt, attempts, delay, e,
                    )
                if attempt >= attempts:
                    break
                await self._sleep_until_ready(cid)
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"{label} failed without exception")

    async def _run_typing(
        self,
        chat_id: int,
        *,
        label: str,
        action: Callable[[], Awaitable[_T]],
    ) -> _T:
        """执行 typing 状态发送（有独立的锁和间隔）。"""
        lock = self._typing_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            now = asyncio.get_running_loop().time()
            wait_s = self._next_typing_at.get(chat_id, 0.0) - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            try:
                result = await action()
                self._next_typing_at[chat_id] = (
                    asyncio.get_running_loop().time() + self._typing_interval_s
                )
                return result
            except RetryAfter as e:
                delay = (
                    float(getattr(e, "retry_after", 1.0) or 1.0)
                    + self._retry_padding_s
                )
                self._next_typing_at[chat_id] = asyncio.get_running_loop().time() + delay
                raise

    async def _wait_for_chat_slot(self, chat_id: int) -> None:
        """等待当前 chat 的时间槽可用。"""
        now = asyncio.get_running_loop().time()
        wait_s = self._next_chat_at.get(chat_id, 0.0) - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    async def _run_with_global_slot(
        self,
        action: Callable[[], Awaitable[_T]],
    ) -> _T:
        """在全局锁保护下执行操作。"""
        async with self._global_lock:
            now = asyncio.get_running_loop().time()
            wait_s = self._next_global_at - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            try:
                return await action()
            finally:
                self._next_global_at = (
                    asyncio.get_running_loop().time() + self._global_interval_s
                )

    async def _sleep_until_ready(self, chat_id: int) -> None:
        """等待当前 chat 的下一个可用时刻。"""
        now = asyncio.get_running_loop().time()
        wait_s = self._next_chat_at.get(chat_id, 0.0) - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    def _mark_used(self, chat_id: int, kind: str) -> None:
        """标记一次操作完成，更新下次可用时间。"""
        now = asyncio.get_running_loop().time()
        self._next_chat_at[chat_id] = now + self._interval(kind)

    def _cooldown(self, chat_id: int, delay: float) -> None:
        """施加额外的冷却时间。"""
        now = asyncio.get_running_loop().time()
        self._next_chat_at[chat_id] = max(
            self._next_chat_at.get(chat_id, 0.0),
            now + delay,
        )
        self._next_global_at = max(self._next_global_at, now + self._global_interval_s)

    def _interval(self, kind: str) -> float:
        """返回指定操作类型的最小间隔。"""
        if kind == "edit":
            return self._edit_interval_s
        if kind == "typing":
            return self._typing_interval_s
        return self._send_interval_s


async def send_markdown(
    bot: Bot,
    chat_id: int | str,
    text: str,
    limiter: TelegramOutboundLimiter | None = None,
) -> None:
    """将 Markdown 文本转换为 Telegram text+entities 并发送。

    输入:
        bot: Telegram Bot 实例。
        chat_id: 目标聊天 ID。
        text: Markdown 文本。
        limiter: 可选限流器；不传则无频率限制。

    输出:
        None。发送成功或降级为纯文本发送。
    """
    cid = int(chat_id)
    try:
        rendered_text, entities, _segments = convert_with_segments(text)
        chunks = split_entities(rendered_text, entities, 4090)
    except Exception as e:
        logger.warning("[telegram] Markdown 转换失败，降级纯文本: %s", e)
        for chunk in _split_text(text, 4090):
            await _run_outbound(
                limiter, cid, kind="send",
                action=lambda c=chunk: bot.send_message(chat_id=cid, text=c),
                label="send_message(plain)",
            )
        return
    for chunk_text, chunk_entities in chunks:
        chunk_text, chunk_entities = _strip_chunk(chunk_text, chunk_entities)
        if not chunk_text:
            continue
        serialized = [entity.to_dict() for entity in chunk_entities] if chunk_entities else None
        await _run_outbound(
            limiter, cid, kind="send",
            action=lambda ct=chunk_text, se=serialized: bot.send_message(
                chat_id=cid, text=ct, entities=se,
            ),
            label="send_message(markdown)",
        )


async def _run_outbound(
    limiter: TelegramOutboundLimiter | None,
    chat_id: int,
    *,
    kind: str,
    label: str,
    action: Callable[[], Awaitable[_T]],
) -> _T:
    """在限流器保护下执行一次出站操作。

    输入:
        limiter: 限流器或 None。
        chat_id: 目标聊天 ID。
        kind: 操作类型。
        label: 日志标签。
        action: 要执行的异步操作。

    输出:
        action 的返回值。
    """
    if limiter is not None:
        return await limiter.run(chat_id, kind=kind, label=label, action=action)
    return await _send_with_retry_result(action, label=label)


async def _send_with_retry_result(
    send_coro_factory: Callable[[], Awaitable[_T]],
    *,
    label: str,
    max_attempts: int = 3,
    base_delay: float = 0.8,
) -> _T:
    """带重试的发送，返回结果。

    输入:
        send_coro_factory: 无参异步可调用对象。
        label: 日志标签。
        max_attempts: 最大尝试次数。
        base_delay: 基础退避延迟秒数。

    输出:
        成功时返回 send_coro_factory 的结果。

    异常:
        所有重试耗尽后抛出最后一个异常。
    """
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await send_coro_factory()
        except RetryAfter as e:
            last_err = e
            if attempt >= max_attempts:
                break
            delay = max(float(getattr(e, "retry_after", 1.0) or 1.0), base_delay)
            logger.warning(
                "[telegram] %s 命中限流，准备重试 attempt=%d/%d delay=%.1fs err=%s",
                label, attempt, max_attempts, delay, e,
            )
            await asyncio.sleep(delay)
        except (TimedOut, NetworkError) as e:
            last_err = e
            if attempt >= max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "[telegram] %s 发送失败，准备重试 attempt=%d/%d delay=%.1fs err=%s",
                label, attempt, max_attempts, delay, e,
            )
            await asyncio.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"{label} failed without exception")


def _split_text(text: str, limit: int) -> list[str]:
    """按行切分文本，每段不超过 limit 字符。

    输入:
        text: 原始文本。
        limit: 每段最大字符数。

    输出:
        切分后的文本段列表。
    """
    chunks, current = [], []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _strip_chunk(
    text: str,
    entities: list[MessageEntity],
) -> tuple[str, list[MessageEntity]]:
    """去除文本首尾换行，并调整 entity 偏移量。

    输入:
        text: 原始文本。
        entities: telegramify-markdown 返回的 entity 列表。

    输出:
        (去除首尾换行后的文本, 调整偏移后的 entity 列表)。
    """
    leading = len(text) - len(text.lstrip("\n"))
    trailing = len(text) - len(text.rstrip("\n"))
    if leading == 0 and trailing == 0:
        return text, entities

    end = len(text) - trailing if trailing else len(text)
    stripped = text[leading:end]
    if not stripped:
        return "", []

    stripped_utf16_len = len(stripped.encode("utf-16-le")) // 2
    adjusted: list[MessageEntity] = []
    for entity in entities:
        new_offset = entity.offset - leading
        new_end = new_offset + entity.length
        if new_end <= 0 or new_offset >= stripped_utf16_len:
            continue
        new_offset = max(0, new_offset)
        new_end = min(new_end, stripped_utf16_len)
        new_length = new_end - new_offset
        if new_length <= 0:
            continue
        adjusted.append(
            MessageEntity(
                type=entity.type,
                offset=new_offset,
                length=new_length,
                url=entity.url,
                language=entity.language,
                custom_emoji_id=entity.custom_emoji_id,
            )
        )
    return stripped, adjusted


# ── Live 消息编辑队列 ──────────────────────────────────────────────


class TelegramLiveEditQueue:
    """Telegram Live 消息编辑队列。

    管理 edit_message_text 的频率以防止 Telegram flood control。
    连续限流 3 次后放弃后续编辑（保留最后可见状态）。

    输入:
        min_interval_s: 两次编辑的最小间隔秒数。
        limiter: 可选的 TelegramOutboundLimiter，用于频率控制。

    输出:
        TelegramLiveEditQueue 实例。
    """

    def __init__(
        self,
        min_interval_s: float = _LIVE_EDIT_MIN_INTERVAL_S,
        limiter: TelegramOutboundLimiter | None = None,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._limiter = limiter
        self._locks: dict[int, asyncio.Lock] = {}
        self._next_allowed_at: dict[int, float] = {}
        self._flood_strikes: dict[int, int] = {}

    async def run(
        self,
        chat_id: int,
        *,
        label: str,
        force: bool = False,
        action: Callable[[], Awaitable[object | None]],
    ) -> object | None:
        """在频率控制下执行一次编辑操作。

        输入:
            chat_id: 目标聊天 ID。
            label: 日志标签。
            force: 为 True 时跳过 flood strike 检查。
            action: 要执行的异步操作；返回 Telegram Message 或 None。

        输出:
            action 的返回值；限流跳过时返回 None。
        """
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            strikes = self._flood_strikes.get(chat_id, 0)
            if strikes >= _LIVE_MAX_FLOOD_STRIKES and not force:
                return None

            now = asyncio.get_running_loop().time()
            wait_s = self._next_allowed_at.get(chat_id, 0.0) - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)

            try:
                if self._limiter is not None:
                    result = await self._limiter.run(
                        chat_id,
                        kind="edit" if "edit" in label else "send",
                        label=label,
                        action=action,
                        max_attempts=1,
                    )
                else:
                    result = await action()
                self._flood_strikes[chat_id] = 0
                self._next_allowed_at[chat_id] = (
                    asyncio.get_running_loop().time() + self._min_interval_s
                )
                return result
            except RetryAfter as e:
                self._flood_strikes[chat_id] = strikes + 1
                delay = float(getattr(e, "retry_after", 1.0) or 1.0)
                self._next_allowed_at[chat_id] = (
                    asyncio.get_running_loop().time() + delay
                )
                logger.warning(
                    "[telegram] %s 命中限流 strikes=%d delay=%.1fs",
                    label,
                    self._flood_strikes[chat_id],
                    delay,
                )
                return None
            except (TimedOut, NetworkError) as e:
                logger.warning("[telegram] %s 网络失败: %s", label, e)
                return None


# ── Live 文本消息 ──────────────────────────────────────────────────


class TelegramLiveTextMessage:
    """一条可编辑的 Telegram Live 消息。

    首次调用 update() 时发送新消息，后续调用 edit_message_text 更新。
    内容未变化时跳过编辑。

    输入:
        bot: Telegram Bot 实例。
        queue: TelegramLiveEditQueue。
        chat_id: 目标聊天 ID。

    输出:
        TelegramLiveTextMessage 实例。
    """

    def __init__(
        self,
        bot: Bot,
        queue: TelegramLiveEditQueue,
        chat_id: int,
    ) -> None:
        self._bot = bot
        self._queue = queue
        self._chat_id = chat_id
        self._message_id: int | None = None
        self._last_plain = ""
        self._is_deleted = False
        self._lock = asyncio.Lock()

    async def update(self, text: str, *, force: bool = False) -> None:
        """更新 Live 消息内容。

        输入:
            text: 纯文本内容。
            force: 强制更新。

        输出:
            None。
        """
        async with self._lock:
            await self._update_locked(text, force=force)

    async def _update_locked(self, text: str, *, force: bool) -> None:
        """在锁保护下执行实际更新。

        输入:
            text: 纯文本内容。
            force: 是否强制更新。

        输出:
            None。
        """
        if self._is_deleted:
            return

        plain = _clip_live_text(text.strip())
        if not plain:
            return
        if not force and plain == self._last_plain:
            return

        if self._message_id is None:
            sent = await self._queue.run(
                self._chat_id,
                label="send_message(live)",
                action=lambda: self._bot.send_message(
                    chat_id=self._chat_id,
                    text=f"<pre>{html.escape(plain)}</pre>",
                    parse_mode="HTML",
                ),
            )
            if sent is None:
                return
            # ── 幽灵消息防护 ──
            # send_message 等待网络响应期间，delete() 可能已被调用。
            # 此时 _is_deleted 为 True，需要立即清理刚创建的这条消息。
            if self._is_deleted:
                new_id = (
                    int(getattr(sent, "message_id", 0) or 0) or None
                )
                if new_id is not None:
                    await self._queue.run(
                        self._chat_id,
                        label="delete_message(live-ghost)",
                        force=True,
                        action=lambda: self._bot.delete_message(
                            chat_id=self._chat_id,
                            message_id=new_id,
                        ),
                    )
                return
            self._message_id = (
                int(getattr(sent, "message_id", 0) or 0) or None
            )
        else:
            ok = await self._queue.run(
                self._chat_id,
                label="edit_message(live)",
                force=force,
                action=lambda: _safe_edit_live(
                    self._bot, self._chat_id, self._message_id, plain
                ),
            )
            if not ok:
                return

        self._last_plain = plain

    async def delete(self) -> None:
        """删除 Live 消息。

        先标记 _is_deleted = True（防止并发 update 的 send_message
        返回后写入 message_id），再尝试删除已知消息。

        输入:
            无。

        输出:
            None。
        """
        async with self._lock:
            self._is_deleted = True
            if self._message_id is not None:
                msg_id = self._message_id
                self._message_id = None
                await self._queue.run(
                    self._chat_id,
                    label="delete_message(live)",
                    force=True,
                    action=lambda: self._bot.delete_message(
                        chat_id=self._chat_id,
                        message_id=msg_id,
                    ),
                )
            self._last_plain = ""


# ── Live 消息辅助函数 ──────────────────────────────────────────────


async def _safe_edit_live(
    bot: Bot,
    chat_id: int,
    message_id: int | None,
    plain_text: str,
) -> bool:
    """安全编辑 Live 消息，HTML 解析失败时降级纯文本。

    输入:
        bot: Telegram Bot 实例。
        chat_id: 目标聊天 ID。
        message_id: 要编辑的消息 ID。
        plain_text: 纯文本内容。

    输出:
        True 表示编辑成功；False 表示失败。
    """
    import re
    _PARSE_ERR_RE = re.compile(
        r"can't parse entities|parse entities|find end of the entity", re.I
    )
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=f"<pre>{html.escape(plain_text)}</pre>",
            parse_mode="HTML",
        )
        return True
    except BadRequest as e:
        if _PARSE_ERR_RE.search(str(e)):
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=plain_text,
            )
            return True
        raise


def _clip_live_text(text: str) -> str:
    """截断 Live 消息文本到 Telegram 允许的长度。

    输入:
        text: 原始文本。

    输出:
        截断后的文本。
    """
    utf16_len = len(text.encode("utf-16-le")) // 2
    if utf16_len <= _LIVE_MESSAGE_LIMIT:
        return text
    suffix = "\n..."
    cut = _utf16_cut(text, _LIVE_MESSAGE_LIMIT - len(suffix))
    return text[:cut] + suffix


def _utf16_cut(text: str, max_utf16: int) -> int:
    """返回前 max_utf16 个 UTF-16 code units 对应的字符切点。

    输入:
        text: 原始文本。
        max_utf16: 最大 UTF-16 code units 数。

    输出:
        Python str 切片索引。
    """
    utf16_count = 0
    for i, ch in enumerate(text):
        utf16_count += 2 if ord(ch) > 0xFFFF else 1
        if utf16_count > max_utf16:
            return i
    return len(text)


def _format_turn_live(
    lines: list[dict[str, str]],
    reply: str,
    *,
    terminal: bool = False,
) -> str:
    """格式化工具进度 + 回复预览的 Live 文本。

    输入:
        lines: 工具行列表，每项为
            {"tool_name": ..., "intent": ..., "target": ..., "status": ...}。
        reply: 当前累积的回复文本。
        terminal: 是否为最终版本。

    输出:
        格式化后的纯文本。
    """
    blocks: list[str] = []

    if lines:
        shown = lines[-12:]
        hidden = len(lines) - len(shown)
        rows = ["工具调用"]
        if hidden > 0:
            rows.append(f"... {hidden} more")
        for line in shown:
            status_icon = {
                "running": "...",
                "done": "✅",
                "error": "✗",
            }.get(line.get("status", ""), "...")
            tool_name = line.get("tool_name", "?")
            intent = line.get("intent", "")
            target = line.get("target", "")
            target_str = f" {target}" if target else ""
            rows.append(
                f"{_tool_emoji(tool_name)} {tool_name}: "
                f"{intent}{target_str} {status_icon}"
            )
        if all(line.get("status") != "running" for line in lines):
            rows.append(f"Done · {len(lines)} tools")
        blocks.append("\n".join(rows))

    reply_body = reply.strip()
    if reply_body:
        if terminal:
            blocks.append(reply_body)
        else:
            blocks.append(f"回复预览\n{reply_body[:600]}")

    if terminal and not blocks:
        return "本轮预览完成"
    return "\n\n".join(blocks)


def _tool_emoji(tool_name: str) -> str:
    """根据工具名返回 emoji。

    输入:
        tool_name: 工具名称。

    输出:
        对应 emoji 字符串。
    """
    name = tool_name.lower()
    if "search" in name:
        return "🔍"
    if "web" in name or "url" in name:
        return "🌐"
    if "file" in name or "read" in name or "write" in name:
        return "📄"
    if "shell" in name or "exec" in name:
        return "⚙"
    if "memory" in name:
        return "🧠"
    if "schedule" in name:
        return "⏰"
    if "spawn" in name:
        return "🤖"
    if "mcp" in name:
        return "📡"
    return "🔧"


def _format_tool_intent(arguments: dict[str, Any], limit: int = 60) -> str:
    """从工具参数中提取意图摘要。

    输入:
        arguments: 工具参数字典。
        limit: 最大字符数。

    输出:
        截断后的意图文本。
    """
    value = arguments.get("description")
    if not value:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_tool_target(arguments: dict[str, Any], limit: int = 50) -> str:
    """从工具参数中提取目标摘要。

    输入:
        arguments: 工具参数字典。
        limit: 最大字符数。

    输出:
        截断后的目标文本；无目标时返回 ""。
    """
    for key in ("cmd", "command", "query", "url", "path", "file", "name"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            text = " ".join(value.split())
            if len(text) <= limit:
                return f'"{text}"'
            return f'"{text[:limit - 3]}..."'
    return ""