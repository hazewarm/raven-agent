from __future__ import annotations

import pytest

from raven_agent.messages import (
    ToolCall,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)


def test_chat_message_to_openai_dict() -> None:
    """测试 ChatMessage 能转换为 OpenAI 消息格式。

    返回:
        None。
    """

    message = user_message("hello")

    assert message.to_openai_dict() == {"role": "user", "content": "hello"}


def test_message_helpers_create_expected_roles() -> None:
    """测试消息便捷函数会创建正确角色。

    返回:
        None。
    """

    assert system_message("s").role == "system"
    assert user_message("u").role == "user"
    assert assistant_message("a").role == "assistant"
    assert tool_message("call_1", "result").role == "tool"


def test_tool_call_converts_to_openai_dict() -> None:
    """测试 ToolCall 会转换为 OpenAI tool_call 格式。

    返回:
        None。
    """

    tool_call = ToolCall(
        id="call_1",
        name="read_text_file",
        arguments={"path": "pyproject.toml"},
    )

    assert tool_call.to_openai_dict() == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read_text_file",
            "arguments": '{"path": "pyproject.toml"}',
        },
    }


def test_assistant_message_can_include_tool_calls() -> None:
    """测试 assistant 消息可以携带 tool_calls。

    返回:
        None。
    """

    message = assistant_message(
        "",
        tool_calls=[ToolCall(id="call_1", name="list_directory", arguments={"path": "."})],
    )

    payload = message.to_openai_dict()

    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["function"]["name"] == "list_directory"


def test_assistant_message_preserves_reasoning_content() -> None:
    """测试 assistant 消息会把 reasoning_content 回传给 API。

    返回:
        None。
    """

    message = assistant_message(
        "",
        tool_calls=[ToolCall(id="call_1", name="list_directory", arguments={"path": "."})],
        reasoning_content="model reasoning",
    )

    payload = message.to_openai_dict()

    assert payload["reasoning_content"] == "model reasoning"


def test_tool_message_requires_tool_call_id() -> None:
    """测试 tool 消息缺少 tool_call_id 时会报错。

    返回:
        None。
    """

    message = tool_message("", "result")

    with pytest.raises(ValueError):
        message.to_openai_dict()