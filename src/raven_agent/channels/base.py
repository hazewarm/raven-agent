from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from raven_agent.events import OutboundMessage
from raven_agent.session import SessionManager

_DEFAULT_UPLOAD_DIR = Path.home() / ".raven-agent" / "workspace" / "uploads"


class ChannelAdapter(ABC):
    """所有外部消息入口的统一抽象。

    输入:
        无。子类通常在 __init__ 中接收 MessageBus、配置、SessionManager 等依赖。

    输出:
        ChannelAdapter 子类实例。实例通过 start()/stop() 接入 Runtime 生命周期。
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """返回 Channel 名称。

        输入:
            无。

        输出:
            Channel 名称，例如 "cli"、"telegram"、"qq"。
        """
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        """启动 Channel。

        输入:
            无。

        输出:
            None。启动后 Channel 可以接收或发送消息。
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """停止 Channel。

        输入:
            无。

        输出:
            None。停止后应释放 socket、后台任务、客户端连接等资源。
        """
        raise NotImplementedError


class AttachmentStore:
    """Channel 媒体附件落盘工具。

    输入:
        root: 附件存储根目录；不传则使用 ~/.raven-agent/workspace/uploads。

    输出:
        AttachmentStore 实例。
    """

    def __init__(self, root: Path | None = None, *, channel: str = "") -> None:
        self.root = root or _DEFAULT_UPLOAD_DIR
        self._channel = channel or "unknown"

    def _resolve_root(self) -> Path:
        """解析可写入的附件根目录。

        输入:
            无。

        输出:
            可写入的 Path。首选 self.root，失败时回退到系统临时目录下的 raven_uploads。
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if os.access(self.root, os.W_OK):
                return self.root
        except Exception:
            pass
        fallback = Path(tempfile.gettempdir()) / "raven_uploads"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def create_path(self, prefix: str, suffix: str) -> Path:
        """生成唯一附件路径，格式为 root/channel/YYYY-MM-DD/uuid.ext。

        输入:
            prefix: 文件名前缀，例如 "photo_"、"doc_"。
            suffix: 文件后缀，例如 ".jpg"、".pdf"。

        输出:
            尚未写入的唯一 Path。中间目录自动创建（mkdir -p）。
        """
        from datetime import date

        root = self._resolve_root()
        today = date.today().isoformat()  # "2026-06-06"
        target_dir = root / self._channel / today
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{prefix}{uuid4().hex}{suffix}"

    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path:
        """写入字节附件。

        输入:
            data: 要写入的字节内容。
            prefix: 文件名前缀。
            suffix: 文件后缀。

        输出:
            实际写入的 Path。
        """
        path = self.create_path(prefix, suffix)
        path.write_bytes(data)
        return path


class MessageDeduper:
    """滑动窗口消息去重器。

    输入:
        max_size: 最多保留多少个已见消息 key。

    输出:
        MessageDeduper 实例。
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._seen: set[str] = set()
        self._order: deque[str] = deque()

    def seen(self, key: str) -> bool:
        """判断 key 是否已见，并把新 key 写入窗口。

        输入:
            key: 消息唯一标识，例如平台 message_id。

        输出:
            True 表示重复消息；False 表示第一次看到。
        """
        if key in self._seen:
            return True
        self._seen.add(key)
        self._order.append(key)
        while len(self._order) > self._max_size:
            self._seen.discard(self._order.popleft())
        return False


class SessionIdentityIndex:
    """维护 identity 到 chat_id 的索引，并同步写入 Session metadata。

    输入:
        session_manager: SessionManager，用于读取和保存 session metadata。
        channel: Channel 名称，例如 "telegram"。
        metadata_key: metadata 中保存 identity 的字段名，例如 "username"。
        normalizer: 可选归一化函数，例如 lower/去 @ 前缀。

    输出:
        SessionIdentityIndex 实例。
    """

    def __init__(
        self,
        session_manager: SessionManager,
        *,
        channel: str,
        metadata_key: str,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._channel = channel
        self._metadata_key = metadata_key
        self._normalizer = normalizer or (lambda value: value)
        self.mapping: dict[str, str] = {}

    def rebuild(self) -> dict[str, str]:
        """从 Session metadata 重建 identity -> chat_id 索引。

        输入:
            无。

        输出:
            重建后的索引副本。
        """
        self.mapping.clear()
        for entry in self._session_manager.get_channel_metadata(self._channel):
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            raw_value = metadata.get(self._metadata_key)
            if not isinstance(raw_value, str):
                continue
            normalized = self._normalize(raw_value)
            chat_id = str(entry.get("chat_id", ""))
            if normalized and chat_id:
                self.mapping[normalized] = chat_id
        return dict(self.mapping)

    def resolve(self, identity: str) -> str | None:
        """查询 identity 对应的 chat_id。

        输入:
            identity: 外部身份，例如 "@alice"。

        输出:
            chat_id；找不到时返回 None。
        """
        normalized = self._normalize(identity)
        if not normalized:
            return None
        return self.mapping.get(normalized)

    async def remember(self, identity: str, chat_id: str) -> None:
        """记录 identity 与 chat_id 的映射，并写入 Session metadata。

        输入:
            identity: 外部身份。
            chat_id: Channel 内会话 ID。

        输出:
            None。
        """
        normalized = self._normalize(identity)
        if not normalized:
            return
        self.mapping[normalized] = chat_id
        session = self._session_manager.get_or_create(f"{self._channel}:{chat_id}")
        if session.metadata.get(self._metadata_key) == normalized:
            return
        session.metadata[self._metadata_key] = normalized
        await self._session_manager.save_async(session)

    def _normalize(self, value: str) -> str:
        """归一化 identity。

        输入:
            value: 原始 identity 字符串。

        输出:
            去除首尾空白并经过 normalizer 处理后的字符串。
        """
        return self._normalizer((value or "").strip())