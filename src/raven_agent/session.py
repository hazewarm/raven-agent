from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from raven_agent.session_store import SessionStore, SessionSummary

from raven_agent.messages import (
    ChatMessage,
    MessageRole,
    ToolCall,
    assistant_message,
    tool_message,
    user_message,
)


def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    """把 ToolCall 转换为可写入 JSON 的字典。

    参数:
        tool_call: 要序列化的 ToolCall。

    返回:
        包含 id、name、arguments 的字典。
    """

    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _tool_call_from_dict(payload: dict[str, Any]) -> ToolCall:
    """从 JSON 字典恢复 ToolCall。

    参数:
        payload: 包含工具调用字段的字典。

    返回:
        ToolCall 实例。
    """

    arguments = payload.get("arguments", {})
    return ToolCall(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        arguments=arguments if isinstance(arguments, dict) else {},
    )


# 下面两个函数当前服务 JSON SessionStore。
# 改成 SQLite 后主链路不再依赖它们，但可以保留用于旧 JSON 迁移和调试。
def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
    """把 ChatMessage 转换为可写入 JSON 的字典。

    参数:
        message: 要序列化的 ChatMessage。

    返回:
        包含 id、seq、role、content、tool_calls、tool_call_id、reasoning_content 的字典。
    """

    return {
        "id": message.id,
        "seq": message.seq,
        "role": message.role,
        "content": message.content,
        "tool_calls": [_tool_call_to_dict(call) for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "reasoning_content": message.reasoning_content,
    }


def _message_from_dict(payload: dict[str, Any]) -> ChatMessage:
    """从 JSON 字典恢复 ChatMessage。

    参数:
        payload: 包含消息字段的字典。

    返回:
        ChatMessage 实例；非法 role 会降级为 user。
    """

    role = str(payload.get("role", "user"))
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    raw_tool_calls = payload.get("tool_calls", [])
    tool_calls = [
        _tool_call_from_dict(cast(dict[str, Any], item))
        for item in raw_tool_calls
        if isinstance(item, dict)
    ]
    return ChatMessage(
        role=cast(MessageRole, role),
        content=str(payload.get("content", "")),
        tool_calls=tool_calls,
        tool_call_id=str(payload.get("tool_call_id", "")),
        reasoning_content=str(payload.get("reasoning_content", "")),
        id=str(payload.get("id", "")),
        seq=int(payload.get("seq", -1)),
    )

# 新增
def _parse_datetime(value: object) -> datetime:
    """把外部时间字段解析为 datetime。

    参数:
        value: ISO 字符串、datetime 或其他对象。

    返回:
        datetime；解析失败时返回当前时间。
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()
    return datetime.now()


@dataclass
class Session:
    """单个会话的运行时历史记录。

    参数:
        key: 全局唯一 session key，例如 cli:default。
        messages: 当前运行时可见的消息列表。
        created_at: session 创建时间。
        updated_at: session 最近更新时间。
        metadata: 会话扩展元数据，后续 Dashboard / Channel metadata 会使用。
        last_consolidated: 已被长期记忆整理过的列表游标；它是 session.messages 的下标，不是 seq。
        consolidation_requested: 是否被外部请求强制整理。

    返回:
        Session 实例。
    """

    key: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0
    consolidation_requested: bool = False

    def add_user_message(self, content: str, *, media: list[str] | None = None) -> None:
        """向会话追加一条尚未持久化的用户消息。

        参数:
            content: 用户输入内容。
            media: 附件文件路径列表（如 Telegram 下载到本地的图片/语音）。

        返回:
            None。
        """

        text = content or ""
        if media:
            media_lines = "\n".join(f"  - {p}" for p in media)
            text = f"{text}\n\n[附件]\n{media_lines}"
        msg = user_message(text)
        if media:
            if not hasattr(self, "_media_map"):
                self._media_map: dict[str, list[str]] = {}
            self._media_map[text] = media
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def add_assistant_message(self, content: str) -> None:
        """向会话追加一条尚未持久化的助手消息。

        参数:
            content: 助手回复内容。

        返回:
            None。
        """

        self.messages.append(assistant_message(content))
        self.updated_at = datetime.now()

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """向会话追加一条尚未持久化的工具结果消息。

        参数:
            tool_call_id: 对应 assistant tool call 的 id。
            content: 工具执行结果文本。

        返回:
            None。
        """

        self.messages.append(tool_message(tool_call_id=tool_call_id, content=content))
        self.updated_at = datetime.now()

    def history_for_prompt(self, max_messages: int = 20) -> list[ChatMessage]:
        """返回最近一段会话历史，供 PromptBuilder 使用。

        参数:
            max_messages: 最多返回多少条历史消息；小于等于 0 时返回空列表。

        返回:
            ChatMessage 列表，按时间从旧到新排列。
        """

        if max_messages <= 0:
            return []
        return self.messages[-max_messages:]

    def clear(self) -> None:
        """清空当前运行时可见历史。

        参数:
            无。

        返回:
            None。
        """

        self.messages = []
        self.updated_at = datetime.now()
        self.last_consolidated = 0
        self.consolidation_requested = False

    # 下面两个方法当前服务 JSON SessionStore。
    # 改成 SQLite 后主链路不再依赖它们，但可以保留用于旧 JSON 迁移和调试。
    def to_dict(self) -> dict[str, Any]:
        """把 Session 转换为可写入 JSON 的字典。

        参数:
            无。

        返回:
            包含 key、created_at、updated_at、metadata、last_consolidated、messages 的字典。
        """

        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "last_consolidated": self.last_consolidated,
            "consolidation_requested": self.consolidation_requested,
            "messages": [_message_to_dict(message) for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Session:
        """从 JSON 字典恢复 Session。

        参数:
            payload: 包含 key、messages、last_consolidated 等字段的字典。

        返回:
            Session 实例。
        """

        raw_messages = payload.get("messages", [])
        messages = [
            _message_from_dict(cast(dict[str, Any], item))
            for item in raw_messages
            if isinstance(item, dict)
        ]
        metadata = payload.get("metadata", {})
        return cls(
            key=str(payload.get("key", "")),
            messages=messages,
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
            metadata=metadata if isinstance(metadata, dict) else {},
            last_consolidated=int(payload.get("last_consolidated", 0) or 0),
            consolidation_requested=bool(payload.get("consolidation_requested", False)),
        )
    
def _guess_media_type(path: str) -> str:
    """从文件路径推断媒体类型。

    参数:
        path: 本地文件路径。

    返回:
        "image" / "audio" / "video" / "file"。
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
        return "image"
    if ext in ("ogg", "mp3", "wav", "m4a", "flac", "opus"):
        return "audio"
    if ext in ("mp4", "mov", "webm", "avi"):
        return "video"
    return "file"


def _message_extra(message: ChatMessage, *, media: list[str] | None = None) -> dict[str, Any]:
    """提取 ChatMessage 需要写入 messages.extra 的附加字段。

    参数:
        message: 要持久化的 ChatMessage。
        media: 附件文件路径列表（仅 user 消息传入）。

    返回:
        包含 tool_call_id / reasoning_content / media 的字典；没有附加字段时返回空字典。
    """

    extra: dict[str, Any] = {}
    if message.tool_call_id:
        extra["tool_call_id"] = message.tool_call_id
    if message.reasoning_content:
        extra["reasoning_content"] = message.reasoning_content
    if media:
        extra["media"] = [
            {"path": p, "type": _guess_media_type(p)}
            for p in media
        ]
    return extra

class SessionManager:
    """按 session key 管理运行时 Session，并负责与 SessionStore 适配。

    参数:
        store: SQLite SessionStore。

    返回:
        SessionManager 实例。
    """

    def __init__(self, store: SessionStore) -> None:
        """初始化 SessionManager。

        参数:
            store: SQLite SessionStore。

        返回:
            None。
        """

        self._store = store
        self._cache: dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        """获取已有 Session；不存在则创建新 Session。

        参数:
            key: session key。

        返回:
            对应 key 的 Session。
        """

        if key in self._cache:
            return self._cache[key]
        session = self._load(key)
        if session is None:
            session = Session(key=key)
            self._ensure_session_meta(session)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """从 SQLite 组装 Session。

        参数:
            key: session key。

        返回:
            Session；如果 sessions 和 messages 都不存在，返回 None。
        """

        meta = self._store.get_session_meta(key)
        messages = self._store.fetch_session_messages(key)
        if meta is None and not messages:
            return None
        metadata = meta.get("metadata", {}) if meta else {}
        return Session(
            key=key,
            messages=messages,
            created_at=_parse_datetime(meta.get("created_at") if meta else None),
            updated_at=_parse_datetime(meta.get("updated_at") if meta else None),
            metadata=metadata if isinstance(metadata, dict) else {},
            last_consolidated=int(meta.get("last_consolidated", 0)) if meta else 0,
        )

    def _ensure_session_meta(self, session: Session) -> None:
        """确保 sessions 元数据行存在。

        参数:
            session: 要确保存在的 Session。

        返回:
            None。
        """

        self._store.upsert_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            last_consolidated=session.last_consolidated,
            metadata=session.metadata,
        )

    def _persist_messages(self, session: Session) -> int:
        """把 Session 中尚未持久化的消息追加到 SQLite。

        参数:
            session: 要持久化新增消息的 Session。

        返回:
            本次新插入的消息数量。
        """

        next_seq = self._store.next_seq(session.key)
        inserted = 0
        for index, message in enumerate(session.messages):
            if message.id:
                continue
            media = getattr(session, "_media_map", {}).get(message.content)
            row = self._store.insert_message(
                session.key,
                role=message.role,
                content=message.content,
                ts=datetime.now().astimezone().isoformat(),
                seq=next_seq,
                tool_chain=[_tool_call_to_dict(call) for call in message.tool_calls] or None,
                extra=_message_extra(message, media=media),
            )
            session.messages[index] = replace(
                message,
                id=str(row["id"]),
                seq=int(row["seq"]),
            )
            next_seq += 1
            inserted += 1
        return inserted

    def save(self, session: Session) -> None:
        """保存 Session 元数据，并追加写入尚未持久化的消息。

        参数:
            session: 要保存的 Session。

        返回:
            None。
        """

        session.updated_at = datetime.now()
        self._ensure_session_meta(session)
        self._persist_messages(session)
        self._store.upsert_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            last_consolidated=session.last_consolidated,
            metadata=session.metadata,
        )
        self._cache[session.key] = session

    def clear(self, key: str) -> Session:
        """清空某个 Session 的运行时可见历史。

        参数:
            key: session key。

        返回:
            清空后的 Session。
        """

        session = self.get_or_create(key)
        session.clear()
        self._cache[key] = session
        return session

    def peek_next_message_id(self, session_key: str) -> str:
        """预览某个 session 下一条消息会获得的 message id。

        参数:
            session_key: session key。

        返回:
            形如 cli:default:14 的 message id。
        """

        next_seq = self._store.next_seq(session_key)
        return f"{session_key}:{next_seq}"

    def close(self) -> None:
        """关闭底层 SessionStore。

        参数:
            无。

        返回:
            None。
        """

        closer = getattr(self._store, "close", None)
        if callable(closer):
            closer()
    
    
    def get_channel_metadata(self, channel: str) -> list[dict[str, object]]:
        """返回某个 Channel 下所有 Session 的 metadata。

        输入:
            channel: Channel 名称，例如 "telegram"。

        输出:
            字典列表。每项包含 chat_id、session_key、metadata。
        """
        prefix = f"{channel}:"
        results: list[dict[str, object]] = []
        for key in self._store.list_session_keys(prefix=prefix):
            session = self.get_or_create(key)
            chat_id = key.removeprefix(prefix)
            results.append(
                {
                    "session_key": key,
                    "chat_id": chat_id,
                    "metadata": dict(session.metadata),
                }
            )
        return results

    async def save_async(self, session: Session) -> None:
        """异步兼容保存接口。

        输入:
            session: 要保存的 Session。

        输出:
            None。当前实现内部调用同步 save()。
        """
        self.save(session)

    
    def list_session_summaries(
        self,
        *,
        channel: str,
        limit: int = 20,
    ) -> list[SessionSummary]:
        """转发到 SessionStore.list_session_summaries。

        输入:
            channel: Channel 名称。
            limit: 最多返回多少个 session。

        输出:
            SessionSummary 列表。
        """
        return self._store.list_session_summaries(channel=channel, limit=limit)