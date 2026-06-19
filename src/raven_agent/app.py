from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging
import time

from raven_agent.agent import ReactAgent
from raven_agent.config import Config
from raven_agent.event_bus import EventBus
from raven_agent.events import InboundMessage, OutboundMessage, TurnCompleted
from raven_agent.llm import LLMProvider
from raven_agent.message_bus import MessageBus, OutboundHandler
from raven_agent.prompt import PromptBuilder
from raven_agent.runtime import handle_inbound_message
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore
from raven_agent.tools import (
    MemoryToolContextHook,
    ShellSafetyHook,
    ToolExecutor,
    ToolRegistry,
    build_default_tools,
    register_memory_tools,
)
from raven_agent.memory import (
    DisabledMemoryEngine,
    MarkdownMemoryMaintenance,
    MarkdownMemoryStore,
    MemoryEngine,
    MemoryOptimizer,
    MemoryOptimizerLoop,
)
from raven_agent.memory2 import (
    DisabledEmbeddingProvider,
    Memorizer,
    Memory2Engine,
    MemoryStore2,
    OpenAICompatibleEmbeddingProvider,
    ProcedureTagger,
    ProfileFactExtractor,
    Retriever,
)
from raven_agent.workspace import Workspace

from raven_agent.lifecycle import LifecycleModules
from raven_agent.plugins import PluginManager
from raven_agent.turn_pipeline import PassiveTurnPipeline, PassiveTurnPipelineDeps
from raven_agent.plugins import PluginManager, load_builtin_plugin_specs
from raven_agent.channels import ChannelManager, CLIChannel, IPCServerChannel, TelegramChannel
from raven_agent.scheduler import LatencyTracker, SchedulerService

from raven_agent.background.interrupt import TurnInterruptState
from raven_agent.background import (
    BackgroundJobRunner,
    BackgroundRuntime,
    InterruptManager,
)

from raven_agent.mcp.registry import McpServerRegistry
from raven_agent.mcp.manage_tools import (
    McpAddTool,
    McpListTool,
    McpRemoveTool,
)

from raven_agent.proactive import PresenceStore, ProactiveLoop
from raven_agent.proactive.drift_state import DriftStateStore

from raven_agent.proactive.contracts import (
    AlertContract,
    ContentContract,
    ContextContract,
)
from raven_agent.proactive.mcp_sources import McpSourceFetcher, SourceStore
from raven_agent.proactive.source_tools import (
    ProactiveSourceAddTool,
    ProactiveSourceListTool,
    ProactiveSourceRemoveTool,
)
from raven_agent.proactive.state import ProactiveStateStore

import httpx
from raven_agent.peer import PeerAgentPoller, PeerProcessManager

from raven_agent.background.subagent_profiles import SubagentRuntime
from raven_agent.background.subagents import (
    SpawnAwareBackgroundJobRunner,
    SubagentJobRunner,
    SubagentManager,
)
from raven_agent.tools.spawn import SpawnManageTool, SpawnTool, SpawnToolContextHook

from raven_agent.plugins.builtins.observe.backup import (
    backup_databases,
    schedule_backup,
)

from raven_agent.skills import SkillsLoader


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CoreRuntime:
    """当前后端核心对象集合。

    参数:
        config: 当前应用配置。
        provider: 调用大模型的 LLMProvider。
        tools: 工具注册表。
        tool_executor: 工具执行器。
        agent: 执行 ReAct 推理的 ReactAgent。
        message_bus: 入站/出站消息总线。
        event_bus: 事件总线。
        workspace: 本地运行工作区。
        sessions: 会话管理器。
        memory: Markdown 记忆存储。
        memory_maintenance: 对话完成后的 Markdown 记忆维护器。
        memory_optimizer: 把 PENDING.md 归档进长期记忆的优化器。
        optimizer_loop: 定时触发 memory_optimizer 的循环。
        memory_engine: 结构化语义记忆引擎。
        prompt_builder: PromptBuilder。
        procedure_tagger: Procedure 写入时使用的 trigger tagger。
        profile_extractor: Profile 事实抽取器。
        plugin_manager: 插件管理器。
        channel_manager: 通信管理器。
        scheduler: 定时任务调度器。
        interrupt_manager: 中断管理器。
        background_runtime: 后台运行时。
        mcp_registry: MCP 服务器注册表。
        proactive_loop: 主动推送循环。
        subagent_manager: 本地 SubAgent 任务管理器，供 spawn 工具使用。
    """

    config: Config
    provider: LLMProvider
    tools: ToolRegistry
    tool_executor: ToolExecutor
    agent: ReactAgent
    message_bus: MessageBus
    event_bus: EventBus
    workspace: Workspace
    sessions: SessionManager
    memory: MarkdownMemoryStore
    memory_maintenance: MarkdownMemoryMaintenance
    memory_optimizer: MemoryOptimizer
    optimizer_loop: MemoryOptimizerLoop
    memory_engine: MemoryEngine
    prompt_builder: PromptBuilder
    procedure_tagger: ProcedureTagger | None
    profile_extractor: ProfileFactExtractor | None
    plugin_manager: PluginManager | None
    channel_manager: ChannelManager
    scheduler: SchedulerService | None
    interrupt_manager: InterruptManager | None
    background_runtime: BackgroundRuntime | None
    mcp_registry: McpServerRegistry | None
    proactive_loop: ProactiveLoop | None
    proactive_state: ProactiveStateStore | None = None
    # ── Peer Agent ──
    peer_process_manager: PeerProcessManager | None = None
    peer_poller: PeerAgentPoller | None = None        
    peer_client: httpx.AsyncClient | None = None

    # ——subagent相关资源——
    subagent_manager: SubagentManager | None = None  
        

class AppRuntime:
    """raven-agent 应用运行时。

    参数:
        core: 当前核心运行时对象集合。
        workspace: 本地工作目录，用于保存运行数据。
    """

    def __init__(self, core: CoreRuntime, workspace: Path) -> None:
        self.core = core
        self.workspace = workspace
        self._started = False
        self._stopped = False
        self._pipeline = None
        self._interrupt_manager = core.interrupt_manager
        self._background_runtime = core.background_runtime
        self._proactive_loop = core.proactive_loop
        self._optimizer_loop = core.optimizer_loop
        self._proactive_state = core.proactive_state
        self._peer_pm = core.peer_process_manager              
        self._peer_poller = core.peer_poller  
        self._peer_client = core.peer_client
        self._subagent_manager = core.subagent_manager
        self._dashboard_server = None
        self._dashboard_task = None
        self._backup_task = None                   

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        workspace: str | Path = ".raven",
        allowed_dir: str | Path | None = None,
    ) -> AppRuntime:
        """根据配置创建 AppRuntime。

        参数:
            config: 应用配置。
            workspace: 本地运行数据目录。
            allowed_dir: 默认工具允许访问的目录；不传则使用当前工作目录。

        返回:
            AppRuntime 实例。
        """

        workspace_model = Workspace(workspace)
        workspace_path = workspace_model.root
        allowed_path = Path(allowed_dir) if allowed_dir is not None else Path.cwd()

        # ── Scheduler ──（必须在 build_default_tools 之前创建）
        scheduler = None
        if config.scheduler.enabled:
            scheduler = SchedulerService(
                store_path=workspace_model.schedules_file,
                push_tool=None,               # 在 start() 中绑定
                agent_loop_provider=None,      # 在 start() 中绑定
                tracker=LatencyTracker(),
            )

        # ── Interrupt & Background ──
        interrupt_manager = InterruptManager()
        background_runtime = BackgroundRuntime(max_concurrent=3)

        provider = LLMProvider(config=config.llm)

        # ── VL 视觉模型 Provider ──
        # 仅当主模型不支持多模态（multimodal=False）且配置了 vl_model 时创建。
        # 这样 ReadImageVisionTool 可以用独立的 VL 模型分析图片。
        vl_provider: LLMProvider | None = None
        vl_model: str = ""
        if not config.vl.multimodal and config.vl.model:
            vl_model = config.vl.model
            vl_api_key = config.vl.api_key or config.llm.api_key
            vl_base_url = config.vl.base_url or config.llm.base_url
            vl_config_dict = {
                "provider": config.vl.model,          # provider 字段用于日志标识
                "model": config.vl.model,
                "api_key": vl_api_key,
                "base_url": vl_base_url,
                "max_tokens": 2048,
                "streaming": False,                   # VL 调用不需要流式
            }
            from raven_agent.config import LLMConfig
            vl_provider = LLMProvider(config=LLMConfig(**vl_config_dict))

        # ——subagent——
        subagent_runtime = SubagentRuntime(
            provider=provider,
            model=config.agent.subagent_model or config.llm.model,
            web_search_api_key=config.tools.web_search.api_key,
            web_search_gl=config.tools.web_search.gl,
            web_search_hl=config.tools.web_search.hl,
        )

        tools = build_default_tools(
            allowed_dir=allowed_path,
            web_search_api_key=config.tools.web_search.api_key,
            web_search_gl=config.tools.web_search.gl,
            web_search_hl=config.tools.web_search.hl,
            scheduler=scheduler,
            scheduler_tz=config.scheduler.timezone,
            vl_provider=vl_provider,
            vl_model=vl_model,
            audio_model=config.audio.model,
            audio_enabled=config.audio.enabled,
        )

        # ── MCP ──（必须在 tools 创建之后——mcp_registry 需要注入 tool_registry；
        # MCP 管理工具也在这里注册，因为 registry 依赖已建好的 tools）
        mcp_registry = None
        if config.mcp.enabled:
            mcp_registry = McpServerRegistry(
                config_path=workspace_model.mcp_servers_file,
                tool_registry=tools,
                auto_connect=config.mcp.auto_connect,
            )
            tools.register(
                McpAddTool(mcp_registry),
                risk="external-side-effect",
                search_hint="mcp 添加服务器 连接外部工具 注册MCP server add connect",
            )
            tools.register(
                McpRemoveTool(mcp_registry),
                risk="external-side-effect",
                search_hint="mcp 移除服务器 断开连接 删除MCP server remove disconnect",
            )
            tools.register(
                McpListTool(mcp_registry),
                risk="read-only",
                search_hint="mcp 列表 查看外部工具 已连接服务器 list servers",
                always_on=True,
            )

        # tool_executor = ToolExecutor([ShellSafetyHook(), MemoryToolContextHook()])
        tool_executor = ToolExecutor([MemoryToolContextHook(), SpawnToolContextHook()])
        message_bus = MessageBus()
        event_bus = EventBus()
        agent = ReactAgent(
            provider=provider,
            tools=tools,
            tool_executor=tool_executor,
            interrupt_manager=interrupt_manager,
            event_bus=event_bus,
            streaming_enabled=config.llm.streaming,
        )

        subagent_manager = SubagentManager(
            runtime=subagent_runtime,
            workspace=workspace_model.root,
            task_root=workspace_model.subagents_dir,
            bus=message_bus,
            background_runtime=background_runtime,
        )
        
        # ── Peer Agent ──
        peer_process_manager = None
        peer_poller = None
        peer_client = None
        if config.peer_agents:
            from raven_agent.peer.builder import build_peer_agent_resources

            peer_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
            peer_process_manager, peer_poller = build_peer_agent_resources(
                config=config,
                bus=message_bus,
                client=peer_client,
                log_dir=workspace_model.peer_agents_dir,
            )
        
        
        session_store = SessionStore(workspace_model.sessions_db_file)
        sessions = SessionManager(session_store)
        memory = MarkdownMemoryStore(workspace_model.memory_dir)
        memory_maintenance = MarkdownMemoryMaintenance(
            store=memory,
            provider=provider,
            sessions=sessions,
        )
        memory_optimizer = MemoryOptimizer(store=memory, provider=provider)
        optimizer_loop = MemoryOptimizerLoop(
            optimizer=memory_optimizer,
            interval_seconds=64800,  # 每 18 小时
        )
        if config.memory.enabled:
            store = MemoryStore2(
                workspace_model.memory2_db_file,
                vec_dim=config.memory.embedding.dimensions,
            )
            if config.memory.embedding.enabled:
                embedder = OpenAICompatibleEmbeddingProvider(
                    api_key=config.memory.embedding.api_key,
                    base_url=config.memory.embedding.base_url,
                    model=config.memory.embedding.model,
                    dimensions=config.memory.embedding.dimensions,
                )
            else:
                embedder = DisabledEmbeddingProvider()
            retriever = Retriever(store=store, embedder=embedder)
            procedure_tagger = ProcedureTagger(
                provider=provider,
                tools_fn=tools.list_names,
            )
            profile_extractor = ProfileFactExtractor(provider=provider)
            memorizer = Memorizer(
                store=store,
                embedder=embedder,
                procedure_tagger=procedure_tagger,
                event_bus=event_bus
            )
            memory_engine = Memory2Engine(
                store=store,
                embedder=embedder,
                retriever=retriever,
                memorizer=memorizer,
            )
        else:
            procedure_tagger = None
            profile_extractor = None
            memory_engine = DisabledMemoryEngine()

        # 按 engine.tool_profile() 注册记忆工具（disabled engine 返回空 profile，不注册任何工具）
        register_memory_tools(tools, memory_engine, event_bus=event_bus)
        # ── Spawn / SubAgent 工具 ──
        tools.register(
            SpawnTool(subagent_manager),
            risk="external-side-effect",
            always_on=True,
            search_hint="spawn subagent 子任务 后台任务 委托 调研 多步任务",
        )
        tools.register(
            SpawnManageTool(subagent_manager),
            risk="external-side-effect",
            always_on=True,
            search_hint="spawn 管理 后台任务 列表 取消 job_id",
        )

        _repo_root = Path(__file__).parent.parent.parent
        skills_loader = SkillsLoader(skills_dir=_repo_root / "skills")

        prompt_builder = PromptBuilder(
            system_prompt=config.agent.system_prompt,
            history_window=20,
            memory_store=memory,
            skills_loader=skills_loader, 
        )

        builtin_specs = load_builtin_plugin_specs(config.plugins.builtins)
        plugin_dirs = [Path(item) for item in config.plugins.dirs] if config.plugins.enabled else []
        plugin_manager = None
        if builtin_specs or plugin_dirs:
            plugin_manager = PluginManager(
                plugin_dirs=plugin_dirs,
                event_bus=event_bus,
                tool_registry=tools,
                workspace=workspace_model.root,
                session_manager=sessions,
                memory_engine=memory_engine,
                memory_maintenance=memory_maintenance,
                memory_optimizer=memory_optimizer,
                builtin_specs=builtin_specs,
            )

        event_bus.on(TurnCompleted, memory_maintenance.on_turn_completed)

        channel_manager = ChannelManager()
        if config.channels.cli_enabled:
            channel_manager.register(
                CLIChannel(
                    bus=message_bus,
                    chat_id="default",
                    sender="local",
                    event_bus=event_bus,
                )
            )
        if config.channels.ipc_enabled:
            channel_manager.register(
                IPCServerChannel(
                    bus=message_bus,
                    socket_path=config.channels.socket,
                    sessions=sessions,  # 支持 session.continue_latest / session.list
                    event_bus=event_bus,
                )
            )
        if config.channels.telegram.enabled:
            telegram = TelegramChannel(
                token=config.channels.telegram.token,
                bus=message_bus,
                session_manager=sessions,
                allow_from=list(config.channels.telegram.allow_from),
                workspace=workspace_model.root,
                interrupt_manager=interrupt_manager,
                event_bus=event_bus,
            )
            channel_manager.register(telegram)

        
        drift_store = None
        if config.proactive.drift_enabled:
            _repo_root = Path(__file__).parent.parent.parent
            drift_source_dir = _repo_root / "drift" / "skills"
            drift_state_dir = workspace_model.drift_dir

            drift_store = DriftStateStore(
                source_dir=drift_source_dir,
                state_dir=drift_state_dir,
            )
        
        # ── Proactive source 管理工具 ──
        if config.proactive.enabled:
            source_store = SourceStore(workspace_model.proactive_sources_file)
            tools.register(
                ProactiveSourceAddTool(source_store),
                risk="write",
                search_hint="proactive 数据源 添加来源 add source alert content context",
            )
            tools.register(
                ProactiveSourceRemoveTool(source_store),
                risk="write",
                search_hint="proactive 移除来源 删除源 remove source",
            )
            tools.register(
                ProactiveSourceListTool(source_store),
                risk="read-only",
                search_hint="proactive 查看来源 列出数据源 list sources",
            )

        

        # ── Proactive ──
        proactive_loop = None
        if config.proactive.enabled:
            target_chat_id = config.proactive.default_chat_id
            target_key = ""
            if config.proactive.default_channel and target_chat_id:
                channel = config.proactive.default_channel
                # Telegram channel 且非数字 → username 解析
                if channel == "telegram" and not target_chat_id.lstrip("-").isdigit():
                    from raven_agent.channels.base import SessionIdentityIndex
                    identity_index = SessionIdentityIndex(
                        sessions,
                        channel=channel,
                        metadata_key="username",
                        normalizer=lambda v: v.lower(),
                    )
                    identity_index.rebuild()
                    normalized = target_chat_id.lstrip("@").lower()
                    resolved = identity_index.resolve(normalized)
                    if resolved:
                        logger.info(
                            "[proactive] %r 解析为 chat_id=%s", target_chat_id, resolved,
                        )
                        target_key = f"{channel}:{resolved}"
                    else:
                        logger.warning(
                            "[proactive] 目标用户 %r 尚未给 bot 发过消息，"
                            "请先发送一条消息后再启动 Proactive。",
                            target_chat_id,
                        )
                else:
                    target_key = f"{channel}:{target_chat_id}"
            if not target_key:
                logger.warning(
                    "Proactive 已启用但未配置 default_channel/default_chat_id——"
                    "ProactiveLoop 不会启动"
                )
            else:
                presence = PresenceStore(session_store)
                source_fetcher = None
                proactive_state = None
                if mcp_registry is not None:
                    source_store_for_fetcher = SourceStore(
                        workspace_model.proactive_sources_file,
                    )
                    source_fetcher = McpSourceFetcher(
                        mcp_registry, source_store_for_fetcher,
                    )
                    proactive_state = ProactiveStateStore(
                        workspace_model.proactive_state_db_file,
                    )
                proactive_loop = ProactiveLoop(
                    presence=presence,
                    target_session_key=target_key,
                    workspace_root=workspace_model.root,
                    tick_s0=config.proactive.tick_interval_s0,
                    tick_s1=config.proactive.tick_interval_s1,
                    tick_s2=config.proactive.tick_interval_s2,
                    tick_s3=config.proactive.tick_interval_s3,
                    tick_jitter=config.proactive.tick_jitter,
                    w_e=config.proactive.score_weight_energy,
                    w_c=config.proactive.score_weight_content,
                    w_r=config.proactive.score_weight_recent,
                    recent_scale=config.proactive.score_recent_scale,
                    interval_seconds=config.proactive.interval_seconds,
                    model=config.proactive.model or config.llm.model,
                    sessions=sessions,
                    memory=memory,
                    memory_engine=memory_engine,
                    cfg=config.proactive,
                    drift_store=drift_store,
                    tool_hooks=None,
                    tools=tools,
                    source_fetcher=source_fetcher,
                    state_store=proactive_state,
                )

        core = CoreRuntime(
            config=config,
            provider=provider,
            tools=tools,
            tool_executor=tool_executor,
            agent=agent,
            message_bus=message_bus,
            event_bus=event_bus,
            workspace=workspace_model,
            memory=memory,
            sessions=sessions,
            memory_maintenance=memory_maintenance,
            memory_optimizer=memory_optimizer,
            optimizer_loop=optimizer_loop,
            memory_engine=memory_engine,
            prompt_builder=prompt_builder,
            procedure_tagger=procedure_tagger,
            profile_extractor=profile_extractor,
            plugin_manager=plugin_manager,
            channel_manager=channel_manager,
            scheduler=scheduler,
            interrupt_manager=interrupt_manager,
            background_runtime=background_runtime,
            mcp_registry=mcp_registry,
            proactive_loop=proactive_loop,
            peer_process_manager=peer_process_manager,     
            peer_poller=peer_poller,    
            peer_client=peer_client,
            subagent_manager=subagent_manager,
        )
        return cls(core=core, workspace=workspace_path)
    
    async def start(self) -> None:
        """启动 AppRuntime。

        输入:
            无。

        输出:
            None。
        """

        if self._started:
            return
        self.core.workspace.ensure()
        lifecycle_modules = LifecycleModules()
        if self.core.plugin_manager is not None:
            await self.core.plugin_manager.load_all()
            self.core.tool_executor.add_hooks(self.core.plugin_manager.tool_hooks)
            lifecycle_modules = self.core.plugin_manager.lifecycle_modules()
        if self._proactive_loop is not None:
            self._proactive_loop._tool_hooks = (
                list(self.core.plugin_manager.tool_hooks)
                if self.core.plugin_manager is not None
                else []
            )
        self._pipeline = PassiveTurnPipeline(
            PassiveTurnPipelineDeps(
                sessions=self.core.sessions,
                prompt_builder=self.core.prompt_builder,
                agent=self.core.agent,
                event_bus=self.core.event_bus,
                lifecycle_modules=lifecycle_modules,
                interrupt_manager=self.core.interrupt_manager,
                multimodal=self.core.config.vl.multimodal,
            )
        )
        # 注册各 Channel 的 sender 到 MessagePushTool
        push_tool = self.core.tools.get("message_push")
        if push_tool is not None:
            from raven_agent.tools.message_push import MessagePushTool
            if isinstance(push_tool, MessagePushTool):
                for ch in self.core.channel_manager.list_channels():
                    register_fn = getattr(ch, "register_push_senders", None)
                    if callable(register_fn):
                        register_fn(push_tool)
        await self.core.channel_manager.start_all()
        # ── 绑定 Scheduler 运行时依赖并启动 ──
        if self.core.scheduler is not None:
            # push_tool: 此时 sender 已注册完毕
            self.core.scheduler.push_tool = push_tool
            # agent_loop_provider: 指向已创建的 pipeline
            self.core.scheduler._agent_loop_provider = lambda: self._pipeline
            # 启动调度循环（后台 asyncio Task）
            self.core.scheduler.start()
        # ── Dashboard API ──
        if self.core.config.dashboard.enabled:
            try:
                self._dashboard_server = self.create_dashboard_server()
                self._dashboard_task = asyncio.create_task(
                    self._dashboard_server.serve(),
                    name="dashboard_api",
                )
                # 让出事件循环，等 Dashboard 的启动日志先输出
                await asyncio.sleep(0)
                logger.info(
                    "Dashboard API 已启动: http://%s:%d",
                    self.core.config.dashboard.host,
                    self.core.config.dashboard.port,
                )
            except ImportError as exc:
                logger.warning(
                    "Dashboard API 启动失败（缺少依赖）: %s", exc
                )
        
        # ── 每日数据库备份 ──
        try:
            # 启动时立即执行一次备份
            await asyncio.to_thread(backup_databases, self.core.workspace.root)
            # 然后注册每日定时任务
            self._backup_task = asyncio.create_task(
                schedule_backup(self.core.workspace.root),
                name="daily_backup",
            )
            logger.info("每日数据库备份已启动")
        except Exception as exc:
            logger.warning("数据库备份启动失败: %s", exc)
        
        # ── 绑定 Pipeline 依赖到 BackgroundRuntime ──
        if self._background_runtime is not None:
            default_runner = BackgroundJobRunner(lambda: self._pipeline)
            if self._subagent_manager is not None:
                self._subagent_manager.set_tool_hooks(
                    list(self.core.plugin_manager.tool_hooks)
                    if self.core.plugin_manager is not None
                    else []
                )
                runner = SpawnAwareBackgroundJobRunner(
                    default_runner=default_runner,
                    subagent_runner=SubagentJobRunner(self._subagent_manager),
                )
            else:
                runner = default_runner
            self._background_runtime.set_runner(runner)

            self._background_runtime.on_complete(
                lambda job: logger.info(
                    "Background job done: job_id=%s status=%s exit_reason=%s",
                    job.job_id,
                    job.status,
                    job.exit_reason,
                )
            )
            if self._subagent_manager is not None:
                self._background_runtime.on_complete(
                    self._subagent_manager.announce_completion
                )
            await self._background_runtime.start()
        # ── MCP: 后台重连已配置的 server ──
        if self.core.mcp_registry is not None:
            self.core.mcp_registry.start_connect_all_background()
        # ── 启动 ProactiveLoop ──
        if self._proactive_loop is not None:
            if push_tool is not None:
                self._proactive_loop._push_tool = push_tool
            self._proactive_loop._provider = self.core.provider
            self._proactive_loop.start()
        # ── 启动 MemoryOptimizerLoop ──
        self._optimizer_loop.start()
        # ── Proactive: 预拉取 content feeds ──
        if self._proactive_loop is not None:
            fetcher = getattr(self._proactive_loop, "_source_fetcher", None)
            if fetcher is not None:
                # 延迟拉取，保证所有 MCP 连接都已准备就绪，能正确处理 fetcher 可能触发的工具调用。
                async def _delayed_poll():
                    await asyncio.sleep(15)
                    await fetcher.poll_feeds()
                asyncio.create_task(
                    _delayed_poll(), name="proactive_poll_feeds",
                )
        # ── Peer Agent: 发现并注册工具 + 启动 Poller ──   
        if (
            self._peer_poller is not None
            and self._peer_pm is not None
            and self._peer_client is not None
        ):
            from raven_agent.peer.registry import PeerAgentRegistry

            registry = PeerAgentRegistry(
                process_manager=self._peer_pm,
                poller=self._peer_poller,
                client=self._peer_client,
            )
            peer_tools = await registry.discover_all(self.core.config.peer_agents)
            for t in peer_tools:
                self.core.tools.register(
                    t,
                    risk="external-side-effect",
                    always_on=True,
                )
            self._peer_poller.start()
        
        self._started = True
        self._stopped = False

        print(f"Raven Agent 服务已启动  |  CLI 连接地址: {self.core.config.channels.socket}")
        print("按 Ctrl+C 停止服务")

    async def stop(self) -> None:
        """停止 AppRuntime。

        输入:
            无。

        输出:
            None。
        """

        if self._stopped:
            return
        # 停止 Scheduler
        if self.core.scheduler is not None:
            self.core.scheduler.stop()
        # 停止 BackgroundRuntime
        if self._background_runtime is not None:
            await self._background_runtime.stop()
        # 关闭 MCP 连接
        if self.core.mcp_registry is not None:
            await self.core.mcp_registry.shutdown()
        # 关闭 ProactiveStateStore
        if self._proactive_state is not None:
            self._proactive_state.close()
        # 停止 MemoryOptimizerLoop
        self._optimizer_loop.stop()
        # ── 停止 Dashboard API ──
        if self._dashboard_server is not None:
            self._dashboard_server.should_exit = True
        if self._dashboard_task is not None and not self._dashboard_task.done():
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass
            logger.info("Dashboard API 已停止")
        if self._backup_task is not None and not self._backup_task.done():
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        # 停止 ProactiveLoop
        if self._proactive_loop is not None:
            self._proactive_loop.stop()
        # ── Peer Agent: 停止 Poller + 销毁子进程 ── 
        if self._peer_poller is not None:
            await self._peer_poller.stop()
        if self._peer_pm is not None:
            await self._peer_pm.shutdown_all()
            
        await self.core.channel_manager.stop_all()
        if self.core.plugin_manager is not None:
            self.core.tool_executor.remove_hooks_by_prefix("plugin:")
            await self.core.plugin_manager.terminate_all()
        if self.core.plugin_manager is not None:
            await self.core.plugin_manager.terminate_all()
        session_closer = getattr(self.core.sessions, "close", None)
        if callable(session_closer):
            session_closer()
        closer = getattr(self.core.memory_engine, "close", None)
        if callable(closer):
            result = closer()
            if hasattr(result, "__await__"):
                await result
        self._stopped = True

    # 薄封装，让 main.py 不再直接访问太多内部对象
    # 后续如果 MessageBus 变复杂，main.py 不需要跟着改太多。
    def subscribe_outbound(self, channel: str, handler: OutboundHandler) -> None:
        """订阅某个 channel 的出站消息。

        参数:
            channel: 要订阅的渠道名称。
            handler: 收到出站消息时执行的函数。

        返回:
            None。
        """

        self.core.message_bus.subscribe_outbound(channel, handler)

    async def publish_inbound(self, message: InboundMessage) -> None:
        """发布入站消息。

        参数:
            message: 要发布的 InboundMessage。

        返回:
            None。
        """

        await self.core.message_bus.publish_inbound(message)

    async def consume_inbound(self) -> InboundMessage:
        """消费一条入站消息。

        返回:
            下一条 InboundMessage。
        """

        return await self.core.message_bus.consume_inbound()

    async def publish_outbound(self, message: OutboundMessage) -> None:
        """发布出站消息。

        参数:
            message: 要发布的 OutboundMessage。

        返回:
            None。
        """

        await self.core.message_bus.publish_outbound(message)

    async def dispatch_outbound_once(self) -> OutboundMessage:
        """分发一条出站消息。

        返回:
            被分发的 OutboundMessage。
        """

        return await self.core.message_bus.dispatch_outbound_once()
    

    async def process_inbound(self, inbound: InboundMessage) -> OutboundMessage:
        """处理一条入站消息。

        输入:
            inbound: 当前入站消息。

        输出:
            当前轮产生的 OutboundMessage。
        """

        if self._pipeline is None:
            raise RuntimeError("AppRuntime.process_inbound 需要先调用 start()")
        return await self._pipeline.run(inbound)
    
    def clear_session(self, key: str) -> None:
        """清空某个会话。

        参数:
            key: session key。

        返回:
            None。
        """

        self.core.sessions.clear(key)

    def create_dashboard_server(self) -> object:
        """创建 Dashboard API 的 uvicorn Server 对象。

        输入:
            无。使用 self.core 中已组装的核心对象。

        输出:
            uvicorn.Server 实例，尚未启动。
            调用方需 await server.serve() 来启动。

        异常:
            ImportError: 当 fastapi 或 uvicorn 未安装时。
        """
        try:
            import uvicorn
            from raven_agent.api.dashboard import create_dashboard_app
        except ImportError as exc:
            raise ImportError(
                "Dashboard API 需要 fastapi 和 uvicorn。"
                " 请执行: uv pip install fastapi uvicorn[standard]"
            ) from exc

        core = self.core
        ws = core.workspace

        # 确定 observe.db 路径
        observe_db = ws.root / "observe" / "observe.db"

        # 确定 proactive_state.db 路径
        proactive_db = ws.root / "proactive_state.db"

        # 构造 manual_consolidator：复用 MarkdownMemoryMaintenance
        manual_consolidator = core.memory_maintenance

        # 构造 manual_memory_optimizer：复用 MemoryOptimizer
        manual_memory_optimizer = core.memory_optimizer

        # 获取 SessionStore（从 SessionManager 的内部 store）
        store = core.sessions._store

        cfg = core.config.dashboard

        # 项目根目录与插件根目录（从当前文件路径推导）
        _here = Path(__file__).resolve()               # src/raven_agent/app.py
        pkg_root = _here.parent.parent.parent          # raven-agent/
        plugins_root = pkg_root / "plugins"
        static_dir = pkg_root / "static" / "dashboard"
        trips_dir = plugins_root / "travel" / "output"

        app = create_dashboard_app(
            workspace=ws.root,
            store=store,
            sessions=core.sessions,
            memory_admin=core.memory_engine,
            api_key=cfg.api_key,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
            observe_db_path=observe_db if observe_db.exists() else None,
            proactive_db_path=proactive_db if proactive_db.exists() else None,
            project_root=pkg_root,
            plugins_root=plugins_root,
            static_dir=static_dir,
            trips_dir=trips_dir,
        )

        # 注入 Runtime 状态信息到 app.state
        app.state.scheduler_info = {
            "enabled": core.scheduler is not None,
        }
        app.state.background_info = {
            "enabled": self._background_runtime is not None,
        }
        app.state.proactive_info = {
            "enabled": self._proactive_loop is not None,
        }
        app.state.mcp_info = {
            "enabled": core.mcp_registry is not None,
        }
        app.state.vision_info = {
            "enabled": core.config.vl is not None,
            "multimodal": getattr(core.config.vl, "multimodal", True),
            "vl_model": getattr(core.config.vl, "model", ""),
        }
        app.state.audio_info = {
            "enabled": core.config.audio.enabled,
            "model": core.config.audio.model,
        }

        
        config = uvicorn.Config(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level="info",
        )
        return uvicorn.Server(config)
    
    
    async def optimize_memory(self) -> None:
        """手动触发一次 PENDING.md 到 MEMORY.md 的归档。

        返回:
            None。
        """

        await self.core.memory_optimizer.optimize()
    

    async def process_bus_message_once(
        self, inbound: InboundMessage
    ) -> OutboundMessage:
        """处理一条 MessageBus 入站消息，并发布对应出站消息。

        输入:
            inbound: 从 MessageBus 取出的 InboundMessage。

        输出:
            当前轮最终发布的 OutboundMessage。
        """
        command_reply = self._handle_runtime_command(inbound)
        if command_reply is not None:
            outbound = command_reply
        else:
            try:
                # 检查并处理中断恢复
                inbound = self._check_interrupt_resume(inbound)  # ← 新增
                # 创建 asyncio Task 是为了让 InterruptManager 可以 cancel 它
                outbound = await self.process_inbound(inbound)
            except asyncio.CancelledError:
                # 当前 turn 被 /stop 中断——返回一个占位 outbound
                # 实际的下一条用户消息会触发恢复流程
                outbound = OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content="",
                    metadata={"interrupted": True},
                )
            except Exception as exc:
                outbound = OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=f"调用模型失败: {exc}",
                )
        # 更新用户在线心跳（供 Proactive 系统使用）
        self._record_presence(inbound)
        await self.publish_outbound(outbound)
        await self.dispatch_outbound_once()
        return outbound

    def _check_interrupt_resume(
        self, inbound: InboundMessage
    ) -> InboundMessage:
        """检查并处理中断恢复。

        如果 inbound.session_key 存在中断态，把中断上下文拼入用户消息，
        让 LLM 在继续的对话中看到之前的进度。

        输入:
            inbound: 当前入站消息。

        输出:
            可能被改写后的 InboundMessage（拼接了中断上下文的 content）。
        """
        if self._interrupt_manager is None:
            return inbound

        state = self._interrupt_manager.pop_interrupt_state(inbound.session_key)
        if state is None:
            return inbound

        resumed_content = self._interrupt_manager.build_resume_content(
            state, inbound.content
        )
        logger.info(
            "Resuming interrupted turn for %s  partial_reply_len=%d  tools_used=%d",
            inbound.session_key,
            len(state.partial_reply),
            len(state.tools_used),
        )
        return InboundMessage(
            channel=inbound.channel,
            sender=inbound.sender,
            chat_id=inbound.chat_id,
            content=resumed_content,
            timestamp=inbound.timestamp,
            metadata={
                **inbound.metadata,
                "resumed_from_interrupt": True,
            },
        )
    
    
    def _handle_runtime_command(self, inbound: InboundMessage) -> OutboundMessage | None:
        """处理 Runtime 层内置命令。

        输入:
            inbound: 当前入站消息。

        输出:
            如果命中了内置命令，返回对应 OutboundMessage；否则返回 None。
        """
        content = inbound.content.strip()
        if content == "/clear":
            self.clear_session(inbound.session_key)
            return OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content="当前会话历史已清空。",
            )
        return None
    
    # 活跃度更新函数
    def _record_presence(self, inbound: InboundMessage) -> None:
        """在收到用户消息后更新 Presence 心跳。

        输入:
            inbound: 当前入站消息。

        输出:
            None。
        """
        if self._proactive_loop is None:
            return
        presence = self._proactive_loop._presence
        if presence is not None:
            presence.record_user_message(inbound.session_key)
    

    async def run_cli_loop(self) -> None:
        """运行嵌入式 CLI 模式。

        输入:
            无。

        输出:
            None。CLI 输入循环结束后返回。
        """
        channel = self.core.channel_manager.get("cli")
        if not isinstance(channel, CLIChannel):
            raise RuntimeError("CLIChannel 未启用")

        # 严格回合制：读一行 → 处理（含打印回复）→ 再读下一行。
        # 不并发读取，避免“还没等回复就跳到下一个输入提示”。
        channel.print_banner()
        while not self._stopped:
            text = await channel.read_input()
            if text is None:  # EOF / Ctrl-C / Ctrl-D
                print("\nBye.")
                break
            if text.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break
            if not text:
                continue
            inbound = InboundMessage(
                channel=channel.channel_name,
                sender=channel.sender,
                chat_id=channel.chat_id,
                content=text,
            )
            # CLI 是严格回合制，无法在推理期间输入 /stop，无需 Task 追踪
            await self.process_bus_message_once(inbound)

    async def run_serve_loop(self) -> None:
        """运行 IPC serve 模式。"""
        last_heartbeat = time.monotonic()
        try:
            while not self._stopped:
                try:
                    inbound = await asyncio.wait_for(
                        self.consume_inbound(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    if time.monotonic() - last_heartbeat > 300:
                        logger.debug(
                            "[serve] 主循环存活  inbound_q=%d  outbound_q=%d",
                            self.core.message_bus.inbound_size,
                            self.core.message_bus.outbound_size,
                        )
                        last_heartbeat = time.monotonic()
                    continue

                # ── 使用 Task 追踪，支持 /stop 中断 ──
                session_key = inbound.session_key
                turn_state = TurnInterruptState(
                    session_key=session_key,
                    original_user_message=inbound.content,
                    original_metadata=dict(inbound.metadata),
                )
                task = asyncio.create_task(
                    self.process_bus_message_once(inbound),
                    name=f"turn:{session_key}",
                )
                if self._interrupt_manager is not None:
                    self._interrupt_manager.track_task(
                        session_key, task, turn_state
                    )
                try:
                    await task
                except asyncio.CancelledError:
                    # 被 /stop 中断——正常情况，不需要额外处理
                    pass
                finally:
                    if self._interrupt_manager is not None:
                        self._interrupt_manager.untrack_task(session_key)
        except KeyboardInterrupt:
            return