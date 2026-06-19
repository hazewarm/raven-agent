from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


_ENV_PATTERN = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
DEFAULT_CHANNEL_SOCKET = (
    "127.0.0.1:8765" if os.name == "nt" else "/tmp/raven-agent.sock"
)


@dataclass(frozen=True)
class LLMConfig:
    """大模型连接配置。

    参数:
        provider: 模型供应商名称，例如 deepseek、qwen、openai。
        model: 具体模型名称。
        api_key: 真实 API Key，支持从环境变量解析。
        base_url: OpenAI 兼容 API 的 base URL。
        max_tokens: 单次回复允许的最大 token 数。
    """

    provider: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 2048
    streaming: bool = True

@dataclass(frozen=True)
class VLConfig:
    """VL（Vision-Language）视觉模型配置。

    当主模型不支持多模态时（multimodal=False），raven-agent 会使用独立
    的 VL 模型来分析图片内容，将结果以文本形式返回给主模型。

    输入:
        multimodal: 主模型是否原生支持多模态。True 表示主模型能直接处理
            image_url；False 表示主模型是纯文本模型，需要外接 VL 模型。
        model: VL 模型名称，例如 "qwen-vl-max"、"gpt-4o-mini"。
        api_key: VL 模型 API Key，支持环境变量占位符 ${VAR_NAME}。
            若为空字符串则复用主模型 api_key。
        base_url: OpenAI 兼容 API 的 base URL。若为空则复用主模型 base_url。

    输出:
        不可变 VLConfig 实例。
    """

    multimodal: bool = True
    model: str = ""
    api_key: str = ""
    base_url: str = ""

@dataclass(frozen=True)
class AudioConfig:
    """音频转录（STT）配置。

    输入:
        enabled: 是否启用音频转录工具。默认 True。
        model: local 后端使用的 Whisper 模型规格。
            "tiny"（~75MB，最快）/ "small"（~450MB，推荐）/
            "medium"（~1.5GB，更准）/ "large-v3"（~3GB，最准）。
    输出:
        不可变 AudioConfig 实例。
    """

    enabled: bool = True
    model: str = "small"


@dataclass(frozen=True)
class AgentConfig:
    """Agent 行为配置。

    参数:
        system_prompt: 注入给模型的系统提示词。
        subagent_model: 本地 SubAgent 使用的模型名；为空时复用 llm.model。
    """

    system_prompt: str
    subagent_model: str = ""


@dataclass(frozen=True)
class WebSearchConfig:
    """Web Search 工具配置。

    输入:
        api_key: SerpAPI API Key，可以来自 config.toml 字面值，也可以来自环境变量占位符。
        gl: Google 搜索国家代码，默认 cn。
        hl: Google 搜索语言，默认 zh-cn。

    输出:
        不可变配置对象，供 AppRuntime 创建 WebSearchTool 时注入。
    """

    api_key: str = ""
    gl: str = "cn"
    hl: str = "zh-cn"


@dataclass(frozen=True)
class ToolsConfig:
    """工具相关配置。

    输入:
        web_search: Web Search 工具配置。

    输出:
        不可变配置对象，挂载在根 Config.tools 下。
    """

    web_search: WebSearchConfig

@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding API 配置。

    参数:
        enabled: 是否启用真实 embedding API。
        provider: embedding provider 类型描述，当前支持 openai-compatible。
        model: embedding 模型名。
        api_key: embedding API key，支持环境变量占位符。
        base_url: OpenAI-compatible API base URL。
        dimensions: 可选输出维度；0 表示不传 dimensions。
    """

    enabled: bool = False
    provider: str = "openai-compatible"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    dimensions: int = 0


@dataclass(frozen=True)
class MemoryConfig:
    """Memory2 配置。

    参数:
        enabled: 是否启用结构化语义记忆。
        embedding: Embedding API 配置。
    """

    enabled: bool = False
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

@dataclass(frozen=True)
class BuiltinPluginsConfig:
    """内置后端插件开关配置。

    输入:
        shell_safety: 是否启用 shell 安全插件，默认 True。
        tool_loop_guard: 是否启用工具循环保护插件，默认 True。
        context_pressure: 是否启用上下文压力插件，默认 True。
        status_commands: 是否启用状态命令插件，默认 True。
        citation: 是否启用记忆引用协议插件，默认 False。
        memory_rollup: 是否启用记忆整理命令插件，默认 False。
        observe: 是否启用 observe 写库插件，默认 False。

    输出:
        BuiltinPluginsConfig 实例。
    """

    shell_safety: bool = True
    tool_loop_guard: bool = True
    context_pressure: bool = True
    status_commands: bool = True
    citation: bool = False
    memory_rollup: bool = False
    observe: bool = False

    def is_enabled(self, name: str) -> bool:
        """判断某个内置插件是否启用。

        输入:
            name: 内置插件名，例如 "shell_safety"。

        输出:
            True 表示该内置插件启用。未知名字返回 False。
        """

        return bool(getattr(self, name, False))




@dataclass(frozen=True)
class PluginsConfig:
    """插件系统配置。

    输入:
        enabled: 是否启用外部插件目录加载。
        dirs: 外部插件根目录列表。
        builtins: 内置后端插件开关。

    输出:
        不可变配置对象，挂载在根 Config.plugins 下。
    """

    enabled: bool = False
    dirs: tuple[str, ...] = ("plugins",)
    builtins: BuiltinPluginsConfig = field(default_factory=BuiltinPluginsConfig)


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram Bot 配置。

    输入:
        enabled: 是否启用 Telegram Channel。
        token: Telegram Bot Token，支持环境变量占位符。
        allow_from: 用户白名单列表（user id 或 @username）；空列表表示允许所有人。

    输出:
        TelegramConfig 实例。
    """

    enabled: bool = False
    token: str = ""
    allow_from: tuple[str, ...] = ()



@dataclass(frozen=True)
class ChannelsConfig:
    """Channel 系统配置。

    输入:
        socket: IPC Server 监听地址。Linux/WSL 推荐 Unix socket；Windows 使用 TCP。
        cli_enabled: 默认入口是否启用嵌入式 CLI。
        ipc_enabled: 默认服务启动时是否启用 IPC Server。
        telegram: Telegram Bot 配置。

    输出:
        ChannelsConfig 实例。
    """

    socket: str = DEFAULT_CHANNEL_SOCKET
    cli_enabled: bool = True
    ipc_enabled: bool = False
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler 调度器配置。

    输入:
        enabled: 是否启用定时任务调度器。默认 True。

    输出:
        SchedulerConfig 实例。
    """

    enabled: bool = True
    timezone: str = "UTC"

@dataclass(frozen=True)
class McpConfig:
    """MCP（Model Context Protocol）配置。

    输入:
        enabled: 是否启用 MCP 支持。默认 True。
        auto_connect: 启动时自动连接的 MCP server 名称列表。
            对应的 server 配置保存在 .raven/mcp_servers.json 中。

    输出:
        McpConfig 实例。
    """

    enabled: bool = True
    auto_connect: tuple[str, ...] = ()

@dataclass(frozen=True)
class ProactiveConfig:
    """Proactive 主动推送系统配置。

    输入:
        enabled: 是否启用 Proactive 系统。默认 False（需要显式配置推送目标）。
        default_channel: 默认推送渠道，如 "telegram"。
        default_chat_id: 默认推送目标会话 ID。
        model: Proactive Agent 使用的模型名；为空则复用主模型。

        tick_interval_s0: base_score ≤ 0.20 时的 tick 间隔（秒），默认 4800（~80 min）。
        tick_interval_s1: base_score > 0.20 时的 tick 间隔（秒），默认 2400（~40 min）。
        tick_interval_s2: base_score > 0.40 时的 tick 间隔（秒），默认 1080（~18 min）。
        tick_interval_s3: base_score > 0.70 时的 tick 间隔（秒），默认 420（~7 min）。
        tick_jitter: 随机抖动比例，默认 0.30（±30%）。

        score_weight_energy: D_energy 的权重，默认 0.40。
        score_weight_content: D_content 的权重，默认 0.40。
        score_weight_recent: D_recent 的权重，默认 0.20。
        score_recent_scale: d_recent 的对数归一化尺度，默认 10.0。

        interval_seconds: 固定间隔秒数（无 presence 时的回退值），默认 1800。

    输出:
        ProactiveConfig 实例。
    """

    enabled: bool = False
    default_channel: str = "telegram"
    default_chat_id: str = ""
    model: str = ""

    tick_interval_s0: int = 4800
    tick_interval_s1: int = 2400
    tick_interval_s2: int = 1080
    tick_interval_s3: int = 420
    tick_jitter: float = 0.30

    score_weight_energy: float = 0.40
    score_weight_content: float = 0.40
    score_weight_recent: float = 0.20
    score_recent_scale: float = 10.0

    interval_seconds: int = 1800

    judge_balance_daily_max: int = 8

    # ── Agent Loop ──
    proactive_max_steps: int = 50
    proactive_web_fetch_max_chars: int = 8_000

    # ── 上下文采集 ──
    recent_chat_messages: int = 20

    # ── Drift 参数 ──
    drift_enabled: bool = False
    drift_max_steps: int = 20
    drift_min_interval_hours: float = 3.0
    drift_web_fetch_max_chars: int = 8_000

    # ── 静默时段 ──
    quiet_hours_start: int = 22
    quiet_hours_end: int = 8
    quiet_hours_drift: bool = True


@dataclass(frozen=True)
class PeerAgentConfig:
    """单个 Peer Agent 的配置。

    冷启动架构下，TOML 是这些值的唯一来源——工具注册在启动阶段完成，
    此时 peer agent 子进程尚未拉起，AgentCard 端点不存在。

    输入:
        name: Agent 标识（必填）。ProcessManager 字典键、工具名 delegate_<slug>。
        base_url: A2A 服务 HTTP 端点 URL（必填）。
        launcher: 冷启动命令列表（必填）。
        cwd: 子进程工作目录。
        description: 工具描述（必填）。这就是 LLM 看到的内容。
        health_path: 健康检查路径，默认 "/health"。
        startup_timeout_s: 启动超时（秒），默认 30。
        shutdown_timeout_s: 关闭超时（秒），默认 10。

    输出:
        PeerAgentConfig 实例。
    """

    name: str = ""
    base_url: str = ""
    launcher: tuple[str, ...] = ()
    cwd: str | None = None
    description: str = ""
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard HTTP API 配置。

    输入:
        enabled: 是否在 AppRuntime 启动时挂载 Dashboard API 服务。默认 False。
        host: HTTP 监听地址。默认 "127.0.0.1"。
        port: HTTP 监听端口。默认 2236。
        api_key: API 认证密钥。为空字符串时不启用认证，
                 /api/dashboard/runtime/status 健康检查端点始终免认证。

    输出:
        DashboardConfig 实例。
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 2236
    api_key: str = ""



@dataclass(frozen=True)
class Config:
    """raven-agent 的根配置对象。

    输入:
        llm: 大模型连接配置。
        agent: Agent 行为配置。
        tools: 工具配置。
        memory: Memory2 配置。
        plugins: 插件系统配置。
        channels: 通信系统配置。
        scheduler: 定时任务调度器配置。
        mcp: MCP 配置。
        peer: Peer Agent 配置。
        proactive: Proactive 主动推送系统配置。
        vl: VL 视觉模型配置。
        audio: 音频转录配置。
        dashboard: Dashboard 配置。

    输出:
        根配置对象。
    """

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    peer_agents: tuple[PeerAgentConfig, ...] = ()
    vl: VLConfig = field(default_factory=VLConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def resolve_env_value(value: str) -> str:
    """解析形如 ${ENV_NAME} 的环境变量占位符。

    参数:
        value: 原始配置值。可以是普通字符串，也可以是 ${ENV_NAME}。

    返回:
        如果 value 是环境变量占位符，返回对应环境变量值；否则返回原字符串。

    异常:
        RuntimeError: 当配置引用了环境变量但该变量不存在时抛出。
    """

    match = _ENV_PATTERN.match(value.strip())
    if match is None:
        return value
    env_name = match.group(1)
    env_value = os.getenv(env_name)
    if not env_value:
        raise RuntimeError(f"环境变量 {env_name} 未设置，但 config.toml 引用了它")
    return env_value


def load_config(path: str | Path = "config.toml") -> Config:
    """从 TOML 文件加载 raven-agent 配置。

    参数:
        path: 配置文件路径，默认读取当前目录下的 config.toml。

    返回:
        Config 对象，包含 llm 与 agent 两组配置。

    异常:
        FileNotFoundError: 当配置文件不存在时抛出。
        KeyError: 当必需字段缺失时抛出。
        RuntimeError: 当环境变量占位符无法解析时抛出。
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    llm_data = data["llm"]
    agent_data = data.get("agent", {})
    tools_data = data.get("tools", {})
    memory_data = data.get("memory", {})
    embedding_data = memory_data.get("embedding", {})
    web_search_data = tools_data.get("web_search", {})
    plugins_data = data.get("plugins", {})
    channels_data = data.get("channels", {})
    telegram_data = channels_data.get("telegram", {})
    scheduler_data = data.get("scheduler", {})
    mcp_data = data.get("mcp", {})
    proactive_data = data.get("proactive", {})
    drift_data = proactive_data.get("drift", {})
    vl_data = data.get("vl", {})
    audio_data = data.get("audio", {})
    dashboard_data = data.get("dashboard", {})

    # ── VL 视觉模型 ──
    vl = VLConfig(
        multimodal=bool(vl_data.get("multimodal", True)),
        model=str(vl_data.get("model", "")),
        api_key=resolve_env_value(str(vl_data.get("api_key", ""))),
        base_url=str(vl_data.get("base_url", "")),
    )
    # ── Audio 音频转录 ──
    audio = AudioConfig(
        enabled=bool(audio_data.get("enabled", True)),
        model=str(audio_data.get("model", "small")),
    )
    
    
    llm = LLMConfig(
        provider=str(llm_data["provider"]),
        model=str(llm_data["model"]),
        api_key=resolve_env_value(str(llm_data["api_key"])),
        base_url=str(llm_data["base_url"]),
        max_tokens=int(llm_data.get("max_tokens", 2048)),
        streaming=bool(llm_data.get("streaming", True)),
    )
    agent = AgentConfig(
        system_prompt=str(
            agent_data.get(
                "system_prompt",
                "You are Raven, a concise and helpful AI assistant.",
            )
        ),
        subagent_model=str(agent_data.get("subagent_model", "")).strip(),
    )
    tools = ToolsConfig(
        web_search=WebSearchConfig(
            api_key=resolve_env_value(str(web_search_data.get("api_key", ""))),
            gl=str(web_search_data.get("gl", "cn")),
            hl=str(web_search_data.get("hl", "zh-cn")),
        )
    )
    memory = MemoryConfig(
        enabled=bool(memory_data.get("enabled", False)),
        embedding=EmbeddingConfig(
            enabled=bool(embedding_data.get("enabled", False)),
            provider=str(embedding_data.get("provider", "openai-compatible")),
            model=str(embedding_data.get("model", "")),
            api_key=resolve_env_value(str(embedding_data.get("api_key", ""))),
            base_url=str(embedding_data.get("base_url", "")),
            dimensions=int(embedding_data.get("dimensions", 0)),
        ),
    )
    plugins = PluginsConfig(
        enabled=bool(plugins_data.get("enabled", False)),
        dirs=_load_plugin_dirs(plugins_data.get("dirs", ["plugins"])),
        builtins=_load_builtin_plugins(plugins_data.get("builtins", {})),
    )
    channels = ChannelsConfig(
        socket=str(channels_data.get("socket", DEFAULT_CHANNEL_SOCKET)),
        cli_enabled=bool(channels_data.get("cli_enabled", True)),
        ipc_enabled=bool(channels_data.get("ipc_enabled", False)),
        telegram=TelegramConfig(
            enabled=bool(telegram_data.get("enabled", False)),
            token=resolve_env_value(str(telegram_data.get("token", ""))),
            allow_from=_load_telegram_allow_from(telegram_data.get("allow_from", [])),
        ),
    )
    scheduler = SchedulerConfig(enabled=bool(scheduler_data.get("enabled", True)), timezone=str(scheduler_data.get("timezone", "UTC")))
    mcp = McpConfig(
        enabled=bool(mcp_data.get("enabled", True)),
        auto_connect=tuple(
            str(item).strip()
            for item in mcp_data.get("auto_connect", [])
            if str(item).strip()
        ),
    )
    proactive = ProactiveConfig(
        enabled=bool(proactive_data.get("enabled", False)),
        default_channel=str(proactive_data.get("default_channel", "telegram")),
        default_chat_id=str(proactive_data.get("default_chat_id", "")),
        model=str(proactive_data.get("model", "")),
        tick_interval_s0=int(proactive_data.get("tick_interval_s0", 4800)),
        tick_interval_s1=int(proactive_data.get("tick_interval_s1", 2400)),
        tick_interval_s2=int(proactive_data.get("tick_interval_s2", 1080)),
        tick_interval_s3=int(proactive_data.get("tick_interval_s3", 420)),
        tick_jitter=float(proactive_data.get("tick_jitter", 0.30)),
        score_weight_energy=float(proactive_data.get("score_weight_energy", 0.40)),
        score_weight_content=float(proactive_data.get("score_weight_content", 0.40)),
        score_weight_recent=float(proactive_data.get("score_weight_recent", 0.20)),
        score_recent_scale=float(proactive_data.get("score_recent_scale", 10.0)),
        interval_seconds=int(proactive_data.get("interval_seconds", 1800)),
        judge_balance_daily_max=int(proactive_data.get("judge_balance_daily_max", 8)),
        # --- Agent Loop ---
        proactive_max_steps=int(proactive_data.get("proactive_max_steps", 50)),
        proactive_web_fetch_max_chars=int(proactive_data.get("proactive_web_fetch_max_chars", 8_000)),
        # --- Context ---
        recent_chat_messages=int(proactive_data.get("recent_chat_messages", 20)),

        # --- Drift ---
        drift_enabled=bool(drift_data.get("enabled", False)),
        drift_max_steps=int(drift_data.get("max_steps", 20)),
        drift_min_interval_hours=float(drift_data.get("min_interval_hours", 3.0)),
        drift_web_fetch_max_chars=int(drift_data.get("web_fetch_max_chars", 8_000)),

        # --- Quiet Hours ---
        quiet_hours_start=int(proactive_data.get("quiet_hours_start", 22)),
        quiet_hours_end=int(proactive_data.get("quiet_hours_end", 8)),
        quiet_hours_drift=bool(proactive_data.get("quiet_hours_drift", True)),
    )

    # ── Peer Agent ──
    peer_enabled = bool(agent_data.get("peer_enabled", False))
    peer_toml_path = str(agent_data.get("peer_agents_toml", "peer_agents.toml"))
    peer_agents: list[PeerAgentConfig] = []
    if peer_enabled:
        try:
            peer_toml = tomllib.loads(
                Path(peer_toml_path).read_text(encoding="utf-8")
            )
            for pa in peer_toml.get("peer_agents", []):
                if not isinstance(pa, dict):
                    continue
                launcher_raw = pa.get("launcher", [])
                if isinstance(launcher_raw, list):
                    launcher = tuple(
                        str(item).strip()
                        for item in launcher_raw
                        if str(item).strip()
                    )
                else:
                    launcher = ()
                peer_agents.append(
                    PeerAgentConfig(
                        base_url=str(pa.get("base_url", "")).strip(),
                        launcher=launcher,
                        cwd=str(pa["cwd"]).strip() if pa.get("cwd") else None,
                        name=str(pa.get("name", "")).strip(),
                        description=str(pa.get("description", "")).strip(),
                        health_path=str(pa.get("health_path", "/health")).strip(),
                        startup_timeout_s=int(pa.get("startup_timeout_s", 30)),
                        shutdown_timeout_s=int(pa.get("shutdown_timeout_s", 10)),
                    )
                )
        except FileNotFoundError:
            pass  # peer_agents.toml 不存在时 agents 为空
    
    # ── Dashboard ──
    dashboard = DashboardConfig(
        enabled=bool(dashboard_data.get("enabled", False)),
        host=str(dashboard_data.get("host", "127.0.0.1")),
        port=int(dashboard_data.get("port", 2236)),
        api_key=str(dashboard_data.get("api_key", "")).strip(),
    )

    return Config(
        llm=llm,
        agent=agent,
        tools=tools,
        memory=memory,
        plugins=plugins,
        channels=channels,
        scheduler=scheduler,
        mcp=mcp,
        proactive=proactive,
        peer_agents=tuple(peer_agents),
        vl=vl,
        audio=audio,
        dashboard=dashboard,
    )

def _load_plugin_dirs(raw_value: object) -> tuple[str, ...]:
    """把 TOML 中的 plugins.dirs 解析为字符串元组。

    输入:
        raw_value: TOML 中的 dirs 字段，可以是字符串、字符串列表或缺失值。

    输出:
        插件目录字符串元组；为空时返回 ("plugins",)。
    """

    if isinstance(raw_value, str):
        value = raw_value.strip()
        return (value,) if value else ("plugins",)
    if isinstance(raw_value, list):
        values = tuple(str(item).strip() for item in raw_value if str(item).strip())
        return values or ("plugins",)
    return ("plugins",)

def _load_builtin_plugins(raw_value: object) -> BuiltinPluginsConfig:
    """把 TOML 中的 plugins.builtins 解析为 BuiltinPluginsConfig。

    输入:
        raw_value: TOML 中的 builtins 字段，期望是 dict；其他类型按默认值处理。

    输出:
        BuiltinPluginsConfig；缺省字段使用默认开关。
    """

    if not isinstance(raw_value, dict):
        return BuiltinPluginsConfig()
    data = cast(dict[str, object], raw_value)
    defaults = BuiltinPluginsConfig()
    return BuiltinPluginsConfig(
        shell_safety=bool(data.get("shell_safety", defaults.shell_safety)),
        tool_loop_guard=bool(data.get("tool_loop_guard", defaults.tool_loop_guard)),
        context_pressure=bool(data.get("context_pressure", defaults.context_pressure)),
        status_commands=bool(data.get("status_commands", defaults.status_commands)),
        citation=bool(data.get("citation", defaults.citation)),
        memory_rollup=bool(data.get("memory_rollup", defaults.memory_rollup)),
        observe=bool(data.get("observe", defaults.observe)),
    )


def _load_telegram_allow_from(raw_value: object) -> tuple[str, ...]:
    """把 TOML 中的 channels.telegram.allow_from 解析为字符串元组。

    输入:
        raw_value: TOML 中的 allow_from 字段，可以是字符串或字符串列表。

    输出:
        允许的用户标识字符串元组。
    """
    if isinstance(raw_value, str):
        return (raw_value.strip(),) if raw_value.strip() else ()
    if isinstance(raw_value, list):
        return tuple(str(item).strip() for item in raw_value if str(item).strip())
    return ()