from __future__ import annotations

from raven_agent.config import load_config, resolve_env_value


def test_resolve_env_value_reads_environment(monkeypatch) -> None:
    """测试 ${ENV_NAME} 可以从环境变量读取。

    参数:
        monkeypatch: pytest fixture，用于临时修改环境变量。

    返回:
        None。
    """

    monkeypatch.setenv("RAVEN_TEST_KEY", "secret-value")

    assert resolve_env_value("${RAVEN_TEST_KEY}") == "secret-value"


def test_resolve_env_value_keeps_plain_text() -> None:
    """测试普通字符串不会被环境变量解析逻辑修改。

    返回:
        None。
    """

    assert resolve_env_value("plain-value") == "plain-value"


def test_load_config_from_toml(tmp_path, monkeypatch) -> None:
    """测试可以从 TOML 文件加载完整配置。

    参数:
        tmp_path: pytest fixture，提供临时目录路径。
        monkeypatch: pytest fixture，用于临时设置环境变量。

    返回:
        None。
    """

    monkeypatch.setenv("RAVEN_TEST_API_KEY", "test-api-key")
    monkeypatch.setenv("RAVEN_TEST_EMBEDDING_KEY", "test-embedding-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "deepseek"
model = "deepseek-chat"
api_key = "${RAVEN_TEST_API_KEY}"
base_url = "https://api.deepseek.com/v1"
max_tokens = 1024

[agent]
system_prompt = "You are Raven."

[memory]
enabled = true

[memory.embedding]
enabled = true
provider = "openai-compatible"
model = "text-embedding-v3"
api_key = "${RAVEN_TEST_EMBEDDING_KEY}"
base_url = "https://example.test/v1"
dimensions = 1024
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.provider == "deepseek"
    assert config.llm.model == "deepseek-chat"
    assert config.llm.api_key == "test-api-key"
    assert config.llm.base_url == "https://api.deepseek.com/v1"
    assert config.llm.max_tokens == 1024
    assert config.agent.system_prompt == "You are Raven."
    assert config.memory.enabled is True
    assert config.memory.embedding.enabled is True
    assert config.memory.embedding.provider == "openai-compatible"
    assert config.memory.embedding.model == "text-embedding-v3"
    assert config.memory.embedding.api_key == "test-embedding-key"
    assert config.memory.embedding.base_url == "https://example.test/v1"
    assert config.memory.embedding.dimensions == 1024


def test_load_config_defaults_memory_to_disabled(tmp_path, monkeypatch) -> None:
    """测试未配置 memory 时默认关闭 Memory2。

    参数:
        tmp_path: pytest fixture，提供临时目录路径。
        monkeypatch: pytest fixture，用于临时设置环境变量。

    返回:
        None。
    """

    monkeypatch.setenv("RAVEN_TEST_API_KEY", "test-api-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "deepseek"
model = "deepseek-chat"
api_key = "${RAVEN_TEST_API_KEY}"
base_url = "https://api.deepseek.com/v1"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.memory.enabled is False
    assert config.memory.embedding.enabled is False