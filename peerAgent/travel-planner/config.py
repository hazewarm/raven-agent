"""
peerAgent/travel-planner/config.py —— 读取 Peer Agent 自身的 config.toml。

设计原则:
  - Peer Agent 拥有独立的 config.toml，完全自包含
  - API Key 通过 ${ENV_NAME} 引用环境变量，与 raven-agent 共用
  - MCP server 配置（command + env）由 config.toml 定义
  - 可自由选择模型：ReAct 调用量大，用便宜模型更经济

使用方式:
  >>> from config import load_config
  >>> cfg = load_config()
  >>> print(cfg.llm.model)
  'deepseek-chat'
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


# ── 环境变量占位符: ${ENV_NAME} ──
_ENV_PATTERN = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


def _resolve(value: str) -> str:
    """解析 ${ENV_NAME} 占位符为真实环境变量值。

    输入:
        value: 原始配置值，可能是 "${DEEPSEEK_API_KEY}" 或普通字符串。

    输出:
        环境变量的实际值，或原样返回非占位符字符串。

    异常:
        RuntimeError: 环境变量未设置时抛出。
    """
    m = _ENV_PATTERN.match(value.strip())
    if not m:
        return value
    env_val = os.getenv(m.group(1))
    if not env_val:
        raise RuntimeError(
            f"环境变量 {m.group(1)} 未设置，config.toml 引用了它"
        )
    return env_val


# ── 配置数据类 ──

@dataclass(frozen=True)
class LLMConfig:
    """LLM 连接配置。

    字段:
        provider: 供应商名 (如 "deepseek")。
        model: 模型名 (如 "deepseek-chat")。
        api_key: 已解析的 API Key。
        base_url: API Base URL。
        max_tokens: 单次回复最大 token 数。
    """
    provider: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 8192


@dataclass(frozen=True)
class AmapConfig:
    """高德 MCP 配置。

    字段:
        server_command: MCP server 启动命令 (如 ["uvx", "amap-mcp-server"])。
        api_key: 高德 Web API Key（注入为 AMAP_MAPS_API_KEY 环境变量）。
    """
    server_command: tuple[str, ...] = ("uvx", "amap-mcp-server")
    api_key: str = ""


@dataclass(frozen=True)
class XHSConfig:
    """小红书 MCP 配置。

    字段:
        server_command: MCP server 启动命令 (如 ["stride28-search-mcp"])。
        search_mcp_home: STRIDE28_SEARCH_MCP_HOME 环境变量值。
        headless: STRIDE28_XHS_HEADLESS 环境变量值。
    """
    server_command: tuple[str, ...] = ("stride28-search-mcp",)
    search_mcp_home: str = ".raven/stride28-search"
    headless: str = "true"


@dataclass(frozen=True)
class TripPlannerConfig:
    """行程规划参数。

    字段:
        max_react_rounds: ReAct 单步最大对话轮数。
        max_synthesis_retries: JSON 校验失败最大重试次数。
        output_dir: JSON 落盘目录。
    """
    max_react_rounds: int = 50
    max_synthesis_retries: int = 3
    output_dir: str = "./outputs"


@dataclass(frozen=True)
class Config:
    """Peer Agent 完整运行时配置。

    字段:
        llm: LLM 配置。
        amap: 高德 MCP 配置。
        xhs: 小红书 MCP 配置。
        trip_planner: 规划参数。
    """
    llm: LLMConfig
    amap: AmapConfig
    xhs: XHSConfig
    trip_planner: TripPlannerConfig = field(default_factory=TripPlannerConfig)


def load_config(config_path: str | Path | None = None) -> Config:
    """从 Peer Agent 自身的 config.toml 加载配置。

    输入:
        config_path: 配置文件路径，默认为同目录下的 config.toml。

    输出:
        Config 实例。

    异常:
        FileNotFoundError: 配置文件不存在。
        KeyError: 必需字段缺失。
        RuntimeError: 环境变量未设置。
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.toml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    # ── LLM ──
    llm_data = data["llm"]
    llm = LLMConfig(
        provider=str(llm_data["provider"]),
        model=str(llm_data["model"]),
        api_key=_resolve(str(llm_data["api_key"])),
        base_url=str(llm_data["base_url"]),
        max_tokens=int(llm_data.get("max_tokens", 8192)),
    )

    # ── 高德 MCP ──
    amap_data = data.get("amap", {})
    amap = AmapConfig(
        server_command=_parse_command(amap_data.get("server_command", ["uvx", "amap-mcp-server"])),
        api_key=_resolve(str(amap_data.get("api_key", ""))),
    )

    # ── 小红书 MCP ──
    xhs_data = data.get("xhs", {})
    xhs = XHSConfig(
        server_command=_parse_command(xhs_data.get("server_command", ["stride28-search-mcp"])),
        search_mcp_home=str(xhs_data.get("search_mcp_home", ".raven/stride28-search")),
        headless=str(xhs_data.get("headless", "true")),
    )

    # ── 规划参数 ──
    tp_data = data.get("trip_planner", {})
    trip_planner = TripPlannerConfig(
        max_react_rounds=int(tp_data.get("max_react_rounds", 50)),
        max_synthesis_retries=int(tp_data.get("max_synthesis_retries", 3)),
        output_dir=str(tp_data.get("output_dir", "./outputs")),
    )

    return Config(llm=llm, amap=amap, xhs=xhs, trip_planner=trip_planner)


def _parse_command(value: object) -> tuple[str, ...]:
    """将 TOML 中的 command 字段解析为字符串元组。

    输入:
        value: 字符串或字符串列表。

    输出:
        字符串元组。
    """
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()
