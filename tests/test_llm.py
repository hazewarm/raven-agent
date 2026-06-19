from __future__ import annotations

from raven_agent.config import LLMConfig
from raven_agent.llm import LLMProvider, LLMResponse


def test_llm_response_holds_content() -> None:
    """测试 LLMResponse 可以保存模型文本。

    返回:
        None。
    """

    response = LLMResponse(content="hello")

    assert response.content == "hello"
    assert response.tool_calls == []
    assert response.reasoning_content == ""


def test_llm_provider_can_be_created() -> None:
    """测试 LLMProvider 可以用最小配置初始化。

    返回:
        None。
    """

    config = LLMConfig(
        provider="test",
        model="test-model",
        api_key="test-key",
        base_url="https://example.com/v1",
        max_tokens=128,
    )

    provider = LLMProvider(config=config)

    assert provider is not None