from __future__ import annotations

from raven_agent.channels.telegram.channel import _build_inbound_text_with_reply


class _FakeReplyMsg:
    """模拟 Telegram reply 消息对象。"""
    def __init__(self, text: str, message_id: int, from_username: str = "alice"):
        self.text = text
        self.message_id = message_id
        self.caption = ""
        self.photo = None
        self.from_user = _FakeUser(from_username)


class _FakeUser:
    def __init__(self, username: str):
        self.username = username
        self.id = 123456


def test_build_inbound_text_no_reply() -> None:
    """验证无 reply 时原文不变。"""
    text, meta = _build_inbound_text_with_reply("hello", None)
    assert text == "hello"
    assert meta == {}


def test_build_inbound_text_with_reply() -> None:
    """验证 reply 上下文被合并进入站文本。"""
    reply_msg = _FakeReplyMsg("原始消息", 42, "alice")
    text, meta = _build_inbound_text_with_reply("我的回复", reply_msg)
    assert "正在回复一条历史消息" in text
    assert "原始消息" in text
    assert "我的回复" in text
    assert "@alice" in text
    assert meta["reply_to_message_id"] == 42
    assert meta["reply_to_sender"] == "@alice"


def test_build_inbound_text_reply_without_username() -> None:
    """验证 reply 发送者无 username 时使用 id。"""
    reply_msg = _FakeReplyMsg("msg", 1, "")  # type: ignore[arg-type]
    reply_msg.from_user = _FakeUser("")  # type: ignore[attr-defined]
    reply_msg.from_user.id = 99999  # type: ignore[attr-defined]
    text, meta = _build_inbound_text_with_reply("hi", reply_msg)
    assert "99999" in text