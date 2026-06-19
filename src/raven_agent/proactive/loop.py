"""
ProactiveLoop —— 主动触达核心循环。

独立于 PassiveTurnPipeline，定期执行自适应 tick：
  1. 根据 Presence 数据计算互动电量
  2. 三维加权合成 base_score
  3. base_score → 自适应等待秒数（含随机抖动）
  4. 执行 _tick()

架构位置：
  ProactiveLoop 与 PassiveTurnPipeline 平级——两者都是 AppRuntime 的
  子组件。ProactiveLoop 通过 PresenceStore 感知用户活跃度，
  通过 MessagePushTool 发送主动消息（第 31 章就位）。
"""

from __future__ import annotations

import asyncio
import logging
import random as _random_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import hashlib

from raven_agent.proactive.energy import (
    compute_energy,
    composite_score,
    d_energy,
    d_content,
    d_recent,
    next_tick_from_score,

)
from raven_agent.proactive.presence import PresenceStore

from raven_agent.proactive.sensor import Sensor

from raven_agent.proactive.drift_context import DriftAgentTickContext
from raven_agent.proactive.drift_turn import DriftTurnPipeline

from raven_agent.proactive.agent_context import AgentTickContext
from raven_agent.proactive.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

_PROACTIVE_CONTEXT_FILE = "PROACTIVE_CONTEXT.md"
_PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 主 agent 负责维护这份文件。
- proactive agent 每轮都会读取它，并把它视为需要遵守的规则，不是普通参考建议。
- 这里适合写白名单、黑名单、过滤条件、优先级、必须先验证的步骤。
- 这里不提供新闻事实，不提供候选内容，只定义规则。
- 写结论即可，不要写冗长过程。
"""

def _utcnow() -> datetime:
    """返回当前 UTC 时间（aware）。

    输出:
        datetime 对象。
    """
    return datetime.now(timezone.utc)


class ProactiveLoop:
    """Proactive 主动触达循环。

    独立于被动 Turn Pipeline，拥有自己的 asyncio 循环，
    根据用户 Presence 数据自适应调整 tick 间隔。

    参数:
        presence: PresenceStore 实例（用户在线心跳表）。
        target_session_key: 目标会话 key（"channel:chat_id"）。
        workspace_root: workspace 根目录（PROACTIVE_CONTEXT.md 存放于此）。
        tick_s0..s3: 四档 tick 间隔秒数。
        tick_jitter: 随机抖动比例。
        w_e/w_c/w_r: 三维打分权重。
        recent_scale: d_recent 的对数归一化尺度。
        interval_seconds: 无 presence 时的固定回退间隔。
        rng: 可选的 random.Random 实例（确定性测试用）。
        model: Proactive Agent 使用的模型名（预留，第 31 章就位）。
        provider: LLMProvider 实例（预留，第 31 章就位）。
        push_tool: MessagePushTool 实例（预留，第 31 章就位）。
    """

    def __init__(
        self,
        *,
        presence: PresenceStore,
        target_session_key: str,
        workspace_root: Path,
        tick_s0: int = 4800,
        tick_s1: int = 2400,
        tick_s2: int = 1080,
        tick_s3: int = 420,
        tick_jitter: float = 0.30,
        w_e: float = 0.40,
        w_c: float = 0.40,
        w_r: float = 0.20,
        recent_scale: float = 10.0,
        interval_seconds: int = 1800,
        rng: _random_module.Random | None = None,
        model: str = "",
        provider: Any = None,
        push_tool: Any = None,
        sessions: Any = None,
        memory: Any = None,
        memory_engine: Any = None,
        cfg: Any = None,
        drift_store: Any = None,
        tool_hooks: list | None = None,
        tools: Any = None,
        source_fetcher: Any = None,
        state_store: Any = None,
        passive_busy_fn: Any = None,
    ) -> None:
        self._presence = presence
        self._target_session_key = target_session_key
        self._workspace_root = workspace_root
        self._tick_s0 = tick_s0
        self._tick_s1 = tick_s1
        self._tick_s2 = tick_s2
        self._tick_s3 = tick_s3
        self._tick_jitter = tick_jitter
        self._w_e = w_e
        self._w_c = w_c
        self._w_r = w_r
        self._recent_scale = recent_scale
        self._interval_seconds = interval_seconds
        self._rng = rng

        self._model = model
        self._provider = provider
        self._push_tool = push_tool
        self._sessions = sessions
        self._memory = memory
        self._memory_engine = memory_engine
        self._cfg = cfg

        self._drift_store = drift_store
        self._tool_hooks = tool_hooks
        self._tools = tools

        self._running = False
        self._task: asyncio.Task[None] | None = None

        self._source_fetcher = source_fetcher
        self._state_store = state_store
        self._passive_busy_fn = passive_busy_fn

        # web_fetch 工具（外部传入或懒加载）
        self._web_fetch_tool = None
        self._web_fetch_tool_ready = False
        self._web_fetch_tool_failed = False

        # 确保规则面板文件存在
        self._ensure_proactive_context_file()

    # ── 生命周期 ───────────────────────────────────────────────────

    async def run(self) -> None:
        """启动 Proactive 主循环（阻塞当前协程）。

        输入:
            无。

        输出:
            None。循环持续到 stop() 被调用。
        """
        self._running = True
        logger.info(
            "[proactive] ProactiveLoop 已启动  target=%s  "
            "tick_s0=%ds  tick_s1=%ds  tick_s2=%ds  tick_s3=%ds  jitter=%.0f%%",
            self._target_session_key,
            self._tick_s0, self._tick_s1, self._tick_s2, self._tick_s3,
            self._tick_jitter * 100,
        )
        try:
            # 真正的引擎循环
            await self._run_loop()
        finally:
            logger.info("[proactive] ProactiveLoop 已退出")

    def start(self) -> asyncio.Task[None]:
        """在后台启动 Proactive 循环并返回 Task。

        输入:
            无。

        输出:
            运行 ProactiveLoop.run() 的 asyncio.Task。
        """
        # 供 AppRuntime 在服务启动时挂起该任务
        self._task = asyncio.create_task(self.run(), name="proactive_loop")
        return self._task

    def stop(self) -> None:
        """停止 Proactive 循环。

        输入:
            无。

        输出:
            None。幂等——重复调用无副作用。
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()

    
    @property
    def _sensor(self) -> "Sensor":
        """懒加载 Sensor 实例。

        延迟创建的原因：Sensor 依赖 sessions / presence / memory / cfg，
        这些依赖在 __init__ 时已注入，但 Sensor 对象不需要在 ProactiveLoop
        启动前就创建——只有第一个 tick 真正需要采集数据时才实例化。

        输出:
            Sensor 实例。
        """
        if getattr(self, "_sensor_obj", None) is None:
            from raven_agent.proactive.sensor import Sensor
            self._sensor_obj = Sensor(
                sessions=self._sessions,
                presence=self._presence,
                memory=self._memory,
                workspace_root=self._workspace_root,
                cfg=self._cfg,
                rng=self._rng,
                source_fetcher=self._source_fetcher,
                state_store=self._state_store,
            )
        return self._sensor_obj

    def _get_web_fetch_tool(self):
        """懒加载 WebFetchTool 实例。

        只尝试创建一次，失败后标记 _web_fetch_tool_failed，
        后续 tick 不再重试，避免反复报错。

        输出:
            WebFetchTool 实例或 None。
        """
        if self._web_fetch_tool_ready:
            return self._web_fetch_tool
        if self._web_fetch_tool_failed:
            return None
        try:
            from raven_agent.tools.web_fetch import WebFetchTool
            self._web_fetch_tool = WebFetchTool()
            self._web_fetch_tool_ready = True
            logger.info("[proactive] WebFetchTool 已就绪")
        except Exception as exc:
            self._web_fetch_tool_failed = True
            logger.warning("[proactive] WebFetchTool 初始化失败，已禁用: %s", exc)
        return self._web_fetch_tool
 
    
    # ── 内部函数 ───────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """自适应 tick 主循环。"""
        # 暂存上一个周期的综合得分，用于推演下一个周期的冷却时长
        last_base_score: float | None = None
        
        while self._running:
            # 1. 根据上一次的打分，动态计算出当前需要等待多久才进行下一次 tick
            interval = self._next_interval(last_base_score)
            logger.info("[proactive] 下次 tick 间隔=%ds  base_score=%.3f",
                        interval, last_base_score or 0.0)
            
            # 2. 可中断的分段防御性等待（每秒检查一次 _running 状态，防止stop后后台需要等待）
            elapsed = 0.0
            step = 1.0  # 每秒检查一次 _running
            while elapsed < interval and self._running:
                # 最多等待1s
                await asyncio.sleep(min(step, interval - elapsed))
                elapsed += step
            # 3. 检查是否在等待期间系统触发了停机流程
            if not self._running:
                break

            # 4. 触发扫描逻辑
            try:
                last_base_score = await self._tick()
            except Exception:
                logger.exception("[proactive] tick 异常")
                last_base_score = None

    async def _tick(self) -> float | None:
        """执行一次完整五段式 Proactive tick。

        Gate → Fetch → Agent Loop → Resolve → Deliver

        输出:
            本次计算的 base_score；无数据时返回 None。
        """
        session_key = self._target_session_key
        if not session_key:
            return None

        sensor = self._sensor

        # ═══════════════ Phase 1: Gate（准入检查）═══════════════════════
        if self._passive_busy_fn and self._passive_busy_fn(session_key):
            logger.info("[proactive] gate: passive_busy → skip")
            return None

        if self._presence is None:
            return None

        # 静默时段检查：在非推送时间窗口内跳过 Fetch/Agent Loop，直接跑 drift
        now = _utcnow()
        now_hour = int(now.astimezone().strftime("%H"))
        qs = int(getattr(self._cfg, "quiet_hours_start", 0))
        qe = int(getattr(self._cfg, "quiet_hours_end", 8))
        in_quiet = (
            now_hour >= qs or now_hour < qe
            if qs > qe
            else now_hour >= qs and now_hour < qe
        )
        if in_quiet and getattr(self._cfg, "quiet_hours_drift", True):
            logger.info(
                "[proactive] gate: 静默时段 (%02d:00-%02d:00) hour=%d → drift",
                qs, qe, now_hour,
            )
            if getattr(self._cfg, "drift_enabled", False):
                await self._maybe_run_drift(session_key, now)
            self._maybe_cleanup_state()
            return None

        # ═══════════════ Phase 2: Fetch（采集数据）═════════════════════
        from raven_agent.proactive.agent_context import AgentTickContext

        ctx = AgentTickContext(
            session_key=session_key,
            now_utc=_utcnow(),
            max_steps=self._cfg.proactive_max_steps,
        )

        # 2a. 近期对话
        ctx.recent_chat_raw = sensor.collect_recent_chat(session_key)
        ctx.recent_chat = self._format_recent_chat(ctx.recent_chat_raw)

        # 2b. 规则面板 + 长期记忆
        ctx.context_rules = sensor.read_proactive_context()
        ctx.memory_text = sensor.read_long_term_memory()

        # 2c. 近期主动消息
        recent_proactive_list = sensor.collect_recent_proactive(session_key)
        ctx.recent_proactive_text = "\n".join(
            f"[{r.get('sent_at', '?')[:19]}] {r.get('content', '')[:200]}"
            for r in recent_proactive_list
        )

        # 2d. 外部数据源预轮询（per-tick 异步触发）
        _now_ts = _utcnow().timestamp()
        _last_poll = getattr(self, "_last_poll_ts", 0.0)
        if _now_ts - _last_poll > 1800:
            self._last_poll_ts = _now_ts
            fetcher = getattr(sensor, "_source_fetcher", None)
            if fetcher is not None:
                asyncio.create_task(
                    fetcher.poll_feeds(), name="proactive_poll_feeds_tick",
                )

        # 2e. 外部数据采集
        fetcher = getattr(sensor, "_source_fetcher", None)
        if fetcher is not None:
            try:
                external_data = await sensor.collect_external(session_key)
                ctx.alerts = external_data.get("alert_items", [])
                ctx.contents = external_data.get("content_items", [])
                ctx.contexts = external_data.get("context_items", [])
            except Exception as exc:
                logger.warning("[proactive] 外部数据采集失败: %s", exc)

        # 2e-2. 正文预取（旁路：失败不阻断主流程）
        if ctx.contents:
            await self._prefetch_content_bodies(ctx)

        # 2f. base_score（仅影响 tick 间隔，不影响推送决策）
        last_user_at = self._presence.get_last_user_at(session_key)
        energy = compute_energy(last_user_at)
        de = d_energy(energy)
        dc = d_content(len(ctx.alerts) + len(ctx.contents), halfsat=3.0)
        dr = d_recent(len(ctx.recent_chat_raw), self._recent_scale)
        base_score = composite_score(de, dc, dr, self._w_e, self._w_c, self._w_r)

        # 2g. Alert 内容级去重：不同 event_id 但内容相同的 alert，
        #     若已在 deliveries 表中（曾经推送过），标记 seen 并移出 alerts，
        #     防止重复 alert 反复触发 Alert 快速路径堵住 Content。
        if ctx.alerts:
            state_store = getattr(sensor, "_state_store", None)
            if state_store is not None:
                window_hours = int(getattr(self._cfg, "judge_balance_daily_max", 8))
                surviving = []
                for a in ctx.alerts:
                    ref = self._build_ref_for_item(a.item_id, ctx)
                    dk = hashlib.sha1(ref.encode()).hexdigest()[:16]
                    if state_store.is_delivery_duplicate(session_key, dk, window_hours):
                        parts = a.item_id.split(":", 1)
                        sk = parts[0] if len(parts) >= 2 else ""
                        state_store.mark_items_seen([(sk, a.item_id)])
                        logger.info(
                            "[proactive] fetch: alert 内容重复 → seen item=%s ref=%s",
                            a.item_id, ref,
                        )
                    else:
                        surviving.append(a)
                if len(surviving) < len(ctx.alerts):
                    ctx.alerts = surviving
        
        
        # 2h. 无 alert 且无 content → 低概率让 LLM 看 context
        if not ctx.alerts and not ctx.contents:
            rng = self._rng or _random_module
            if ctx.contexts and rng.random() < 0.03:
                logger.info("[proactive] fetch: context fallback 触发（3%概率）")
                # 继续进 Agent Loop，LLM 看 context 决定有没有值得说的
            else:
                logger.info("[proactive] fetch: 无 alert/content, context fallback 未触发 → drift")
                if getattr(self._cfg, "drift_enabled", False):
                    await self._maybe_run_drift(session_key, ctx.now_utc)
                self._maybe_cleanup_state()
                return base_score

        # ═══════════════ Phase 3: Agent Loop =══════════════════════════
        if self._provider is None:
            self._maybe_cleanup_state()
            return base_score

        try:
            await self._run_agent_loop(ctx)
        except Exception as exc:
            logger.exception("[proactive] Agent Loop 异常: %s", exc)
            await self._ack_and_mark_all_seen(ctx, sensor)
            self._maybe_cleanup_state()
            return base_score

        # ═══════════════ Phase 4: Resolve（去重）═══════════════════════
        if ctx.terminal_action != "reply":
            logger.info(
                "[proactive] resolve: action=%s steps=%d interesting=%d discarded=%d "
                "skip_reason=%s", ctx.terminal_action, ctx.steps_taken,
                len(ctx.interesting_ids), len(ctx.discarded_ids), ctx.skip_reason,
            )
            # skip 路径也做 ACK + 本地标记，避免下次 tick 重复拉取同批条目
            await self._ack_and_mark_all_seen(ctx, sensor)
            self._maybe_cleanup_state()
            return base_score

        # 4a. 逐条去重：每条 cited item 独立生成 ref，逐一检查 deliveries 表
        state_store = getattr(sensor, "_state_store", None)
        window_hours = int(getattr(self._cfg, "judge_balance_daily_max", 8))
        duplicate_found = False
        delivery_keys: list[str] = []

        if state_store is not None and ctx.cited_ids:
            for item_id in ctx.cited_ids:
                ref = self._build_ref_for_item(item_id, ctx)
                dk = hashlib.sha1(ref.encode()).hexdigest()[:16]
                delivery_keys.append(dk)
                if state_store.is_delivery_duplicate(session_key, dk, window_hours):
                    logger.info(
                        "[proactive] resolve: delivery_dedupe hit — item=%s ref=%s",
                        item_id, ref,
                    )
                    duplicate_found = True

        if duplicate_found:
            # 只把命中 dedup 的 alert 写本地 seen_items，不动 contents
            if state_store is not None:
                entries: list[tuple[str, str]] = []
                for item_id in ctx.cited_ids:   # cited_ids 就是 dedup 命中的条目
                    parts = item_id.split(":", 1)
                    source_key = parts[0] if len(parts) >= 2 else ""
                    entries.append((source_key, item_id))
                if entries:
                    try:
                        state_store.mark_items_seen(entries)
                        logger.info("[proactive] dedup: mark_seen %d alerts", len(entries))
                    except Exception:
                        logger.warning("[proactive] dedup: mark_seen 失败", exc_info=True)
            self._maybe_cleanup_state()
            return base_score

        # ═══════════════ Phase 5: Deliver（发送 + ACK）════════════════
        if self._push_tool is not None:
            try:
                parts = session_key.split(":", 1)
                channel = parts[0]
                chat_id = parts[1] if len(parts) > 1 else ""
                await self._push_tool.execute(
                    channel=channel or self._cfg.default_channel,
                    chat_id=chat_id or str(getattr(self._cfg, "default_chat_id", "")),
                    message=ctx.final_message,
                    proactive=True,
                )
                self._presence.record_proactive_sent(session_key)
                sensor.record_delivery(session_key)
                logger.info(
                    "[proactive] 主动消息已推送 session=%s cited=%s",
                    session_key, ctx.cited_ids,
                )

                # 逐条写入 delivery 表，每条 cited item 一条记录
                if state_store is not None and delivery_keys:
                    for dk in delivery_keys:
                        state_store.mark_delivery(session_key, dk, content="")

                if ctx.cited_ids:
                    await self._ack_and_mark_all_seen(ctx, sensor)

            except Exception as exc:
                logger.exception("[proactive] 推送失败: %s", exc)

        self._maybe_cleanup_state()
        return base_score

    # ── Agent Loop ──────────────────────────────────────────────────────

    @staticmethod
    def _build_assistant_dict(
        response: Any,
        *,
        content: str = "",
        tool_calls: list[dict] | None = None,
    ) -> dict[str, Any]:
        """从 LLM 响应构建 assistant 消息 dict。

        thinking 模式（DeepSeek 等）要求将 reasoning_content 原样回传给 API，
        否则报 400 invalid_request_error。

        输入:
            response: chat_dicts 返回的响应对象，含 content / reasoning_content。
            content: 覆盖 content（如仅展示 tool_call 名称时）。
            tool_calls: 覆盖 tool_calls 列表。

        输出:
            符合 OpenAI API 格式的 assistant 消息 dict。
        """
        rc = getattr(response, "reasoning_content", "") or ""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or response.content or "",
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if rc:
            msg["reasoning_content"] = rc
        return msg

    async def _run_agent_loop(self, ctx: "AgentTickContext") -> None:
        """Agent Loop 主循环。

        三阶段：
          1. 主 loop — LLM 调工具，直到 finish_turn 或达到步数上限
          2. 完整性检查 — LLM finish_skip 了但还有未分类 Content → 纠正
          3. 反思收尾 — LLM 标了 interesting 但没 finish_turn → 催促

        输入:
            ctx: AgentTickContext，包含所有已采集数据。

        输出:
            None。结果写入 ctx.terminal_action / ctx.final_message / ctx.cited_ids。
        """
        from raven_agent.proactive.tools import TOOL_SCHEMAS

        # 构造 messages：system + runtime context + kickoff
        system_prompt = self._build_agent_system_prompt()
        user_prompt = self._build_agent_user_prompt(ctx)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": (
                "开始本轮 proactive 处理。"
                "请基于上面的候选内容和规则，必须通过工具逐步完成分类，"
                "最后通过 message_push + finish_turn(decision=reply)，"
                "或 finish_turn(decision=skip, reason=...) 收尾。"
            )},
        ]

        # ═══ 阶段 1：主 loop ══════════════════════════════════════════
        while ctx.steps_taken < ctx.max_steps and ctx.terminal_action is None:
            try:
                response = await self._provider.chat_dicts(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    model=self._model,
                    max_tokens=4096,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.error("[proactive] LLM 失败 (step %d): %s", ctx.steps_taken, exc)
                ctx.terminal_action = "skip"
                ctx.skip_reason = "llm_error"
                return

            ctx.steps_taken += 1

            if not response.tool_calls:
                messages.extend([
                    self._build_assistant_dict(response),
                    {"role": "user", "content": (
                        "请通过调用工具完成本轮决策。"
                        "如果没有任何内容可推送，请调用 "
                        "finish_turn(decision='skip', reason='no_content')。"
                    )},
                ])
                continue

            for tc in response.tool_calls:
                tool_result = await self._dispatch_tool(tc.name, tc.arguments, ctx)
                messages.extend([
                    self._build_assistant_dict(
                        response,
                        content=f"调用 {tc.name}",
                        tool_calls=[{
                            "id": tc.id, "type": "function",
                            "function": {"name": tc.name,
                                         "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                        }],
                    ),
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_result},
                ])
                if ctx.terminal_action is not None:
                    break

        # ═══ 阶段 2：完整性检查 ════════════════════════════════════════
        # LLM finish_skip 了但还有未分类 Content → 注入纠正消息，强制补全
        if ctx.terminal_action == "skip" and ctx.contents:
            unclassified = ctx.all_content_ids - ctx.interesting_ids - ctx.discarded_ids
            if unclassified:
                ctx.terminal_action = None
                ctx.skip_reason = ""
                titles_hint = "; ".join(
                    f"{cid}（{self._lookup_title(cid, ctx)}）"
                    for cid in sorted(unclassified)
                )
                completeness_msg = (
                    f"【系统提示】以下 {len(unclassified)} 个条目尚未完成分类：\n"
                    f"{titles_hint}\n"
                    "请对每条调用 mark_interesting 或 mark_not_interesting，"
                    "全部分类完毕后再调用 message_push + finish_turn(decision=reply)，"
                    "或 finish_turn(decision=skip, reason=...)。"
                )
                logger.info(
                    "[proactive] completeness: %d unclassified → %s",
                    len(unclassified), sorted(unclassified),
                )
                messages.append({"role": "user", "content": completeness_msg})
                for _ in range(20):
                    if ctx.steps_taken >= ctx.max_steps or ctx.terminal_action is not None:
                        break
                    try:
                        response = await self._provider.chat_dicts(
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            model=self._model,
                            max_tokens=4096,
                            tool_choice="auto",
                        )
                    except Exception:
                        break
                    ctx.steps_taken += 1
                    if not response.tool_calls:
                        continue
                    for tc in response.tool_calls:
                        tool_result = await self._dispatch_tool(tc.name, tc.arguments, ctx)
                        messages.extend([
                            self._build_assistant_dict(
                                response,
                                content=f"调用 {tc.name}",
                                tool_calls=[{"id": tc.id, "type": "function",
                                              "function": {"name": tc.name,
                                                           "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}],
                            ),
                            {"role": "tool", "tool_call_id": tc.id, "content": tool_result},
                        ])
                        if ctx.terminal_action is not None:
                            break

        # ═══ 阶段 3：反思收尾 ══════════════════════════════════════════
        # LLM 标了 interesting 但没调 finish_turn → 注入催促消息
        if (
            ctx.terminal_action is None
            and ctx.interesting_ids
            and ctx.steps_taken < ctx.max_steps
        ):
            ids_str = ", ".join(sorted(ctx.interesting_ids))
            reflection = (
                f"【系统提示】你已将以下条目标记为 interesting：{ids_str}。\n"
                "所有条目均已分类完毕。你必须现在调用 message_push 撰写推送，"
                "然后调用 finish_turn(decision=reply)；"
                "或直接调用 finish_turn(decision=skip, reason=...)。不允许直接结束。"
            )
            logger.info(
                "[proactive] reflection: interesting=%d → injecting",
                len(ctx.interesting_ids),
            )
            messages.append({"role": "user", "content": reflection})
            for _ in range(10):
                if ctx.steps_taken >= ctx.max_steps or ctx.terminal_action is not None:
                    break
                try:
                    response = await self._provider.chat_dicts(
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        model=self._model,
                        max_tokens=4096,
                        tool_choice="auto",
                    )
                except Exception:
                    break
                ctx.steps_taken += 1
                if not response.tool_calls:
                    continue
                for tc in response.tool_calls:
                    tool_result = await self._dispatch_tool(tc.name, tc.arguments, ctx)
                    messages.extend([
                        self._build_assistant_dict(
                            response,
                            content=f"调用 {tc.name}",
                            tool_calls=[{"id": tc.id, "type": "function",
                                          "function": {"name": tc.name,
                                                       "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}],
                        ),
                        {"role": "tool", "tool_call_id": tc.id, "content": tool_result},
                    ])
                    if ctx.terminal_action is not None:
                        break

        # 最终兜底
        if ctx.terminal_action is None:
            logger.warning(
                "[proactive] 达到 max_steps=%d 未 finish_turn — 强制 skip",
                ctx.max_steps,
            )
            ctx.terminal_action = "skip"
            ctx.skip_reason = "max_steps_reached"

    @staticmethod
    def _lookup_title(item_id: str, ctx: "AgentTickContext") -> str:
        """从 ctx 中查找 item_id 对应的标题（用于完整性检查的提示）。"""
        for c in ctx.contents:
            if c.item_id == item_id:
                return getattr(c, "title", "")[:40]
        return item_id[:40]

    @staticmethod
    def _build_ref_for_item(item_id: str, ctx: "AgentTickContext") -> str:
        """为单条 cited item 构建去重锚点 ref。

        优先级（对齐 akashic-agent proactive_turn.py:_build_delivery_refs）：
          1. URL（最稳定——同一篇文章不管从哪个源抓来 URL 都一样）
          2. source + title（同源内标题去重）
          3. title only（跨源标题去重——不同源报道同一事件时标题往往相同或高度相似）
          4. item_id（兜底）
        """
        for item in ctx.alerts + ctx.contents:
            if item.item_id != item_id:
                continue
            url = getattr(item, "url", "") or ""
            if url:
                from urllib.parse import urlsplit, urlunsplit
                parts = urlsplit(url)
                normalized = urlunsplit((
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    parts.path.rstrip("/") or parts.path,
                    parts.query,
                    "",
                ))
                return f"url:{normalized}"
            source = getattr(item, "source", "") or ""
            title = getattr(item, "title", "") or ""
            if title:
                title_norm = " ".join(title.strip().lower().split())
                if source.strip():
                    return f"title:{source.strip().lower()}|{title_norm}"
                return f"title:*|{title_norm}"
            break
        return f"id:{item_id}"

    async def _prefetch_content_bodies(self, ctx: "AgentTickContext") -> None:
        """对 content 条目预取正文，存入 ctx.content_store。

        并发上限 3，单条超时 15s，失败不阻断主流程。
        仅对有 URL 的条目预取；无 URL 的条目跳过。

        输入:
            ctx: AgentTickContext。
        """
        wf = self._get_web_fetch_tool()
        if wf is None:
            return

        max_chars = 4000
        concurrency = 3

        # 只取有 URL 的条目
        candidates = [
            c for c in ctx.contents
            if getattr(c, "url", "") and getattr(c, "has_valid_url", False)
        ]
        if not candidates:
            return

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _fetch_one(c) -> tuple[str, str]:
            """拉取单条正文，返回 (item_id, text)。异常时返回空文本。"""
            async with sem:
                try:
                    result = await asyncio.wait_for(
                        wf.execute(
                            url=getattr(c, "url", ""),
                            format="text",
                            max_chars=max_chars,
                        ),
                        timeout=15.0,
                    )
                    import json as _json
                    data = _json.loads(result.text)
                    return c.item_id, (data.get("text", "") or "")
                except asyncio.TimeoutError:
                    logger.debug("[proactive] prefetch 超时: %s", c.item_id)
                except Exception as exc:
                    logger.debug("[proactive] prefetch 失败: %s err=%s", c.item_id, exc)
                return c.item_id, ""

        tasks = [_fetch_one(c) for c in candidates]
        results: list[tuple[str, str]] = await asyncio.gather(*tasks)
        for item_id, text in results:
            ctx.content_store[item_id] = text
        filled = sum(1 for v in ctx.content_store.values() if v)
        logger.info(
            "[proactive] prefetch 完成: %d/%d 条正文已就绪",
            filled, len(candidates),
        )

    # ── Tool Dispatch ───────────────────────────────────────────────────

    async def _dispatch_tool(
        self, tool_name: str, arguments: dict, ctx: "AgentTickContext",
    ) -> str:
        """执行 Agent Loop 中的单个工具调用。

        输入:
            tool_name: 工具名称。
            arguments: LLM 传入的工具参数字典。
            ctx: 当前 tick 的 AgentTickContext。

        输出:
            工具执行结果的字符串。
        """
        if tool_name == "mark_interesting":
            item_ids = self._parse_item_ids(arguments)
            reason = str(arguments.get("reason", "") or "")
            if item_ids:
                for item_id in item_ids:
                    ctx.interesting_ids.add(item_id)
                    ctx.discarded_ids.discard(item_id)
                logger.info(
                    "[proactive] mark_interesting: %d items reason=%s",
                    len(item_ids), reason,
                )
            return f"ok: {len(item_ids)} items → interesting"

        elif tool_name == "mark_not_interesting":
            item_ids = self._parse_item_ids(arguments)
            reason = str(arguments.get("reason", "") or "")
            if item_ids:
                for item_id in item_ids:
                    ctx.discarded_ids.add(item_id)
                    ctx.interesting_ids.discard(item_id)
                logger.info(
                    "[proactive] mark_not_interesting: %d items reason=%s",
                    len(item_ids), reason,
                )
            return f"ok: {len(item_ids)} items → not_interesting"

        elif tool_name == "message_push":
            ctx.final_message = str(arguments.get("message", "") or "")
            evidence = arguments.get("evidence") or []
            ctx.cited_ids = [str(e) for e in evidence if e]
            logger.info("[proactive] message_push: %d chars, %d evidence",
                        len(ctx.final_message), len(ctx.cited_ids))
            return f"ok: draft ({len(ctx.final_message)} chars, {len(ctx.cited_ids)} cited)"

        elif tool_name == "finish_turn":
            decision = str(arguments.get("decision", "skip")).strip()
            reason = str(arguments.get("reason", "") or "").strip()
            if decision not in ("reply", "skip"):
                decision = "skip"; reason = "invalid_decision"
            if decision == "reply" and not ctx.final_message:
                decision = "skip"; reason = "no_draft"
            ctx.terminal_action = decision
            ctx.skip_reason = reason if decision == "skip" else ""
            logger.info("[proactive] finish_turn: %s reason=%s cited=%d",
                        decision, reason, len(ctx.cited_ids))
            return f"ok: turn finished ({decision})"

        elif tool_name == "get_recent_chat":
            return ctx.recent_chat or "（无近期对话）"

        elif tool_name == "recall_memory":
            query = str(arguments.get("query", "") or "")
            if not query:
                return "（无查询词）"
            if self._memory_engine is not None:
                try:
                    from raven_agent.memory.engine import MemoryQuery
                    result = await self._memory_engine.query(
                        MemoryQuery(text=query, intent="interest", limit=3)
                    )
                    if not result.records:
                        return f"（无与 '{query}' 相关的记忆）"
                    return "\n".join(
                        f"- {r.summary[:300]}"
                        for r in result.records if r.summary
                    )
                except Exception:
                    logger.warning(
                        "[proactive] recall_memory engine 查询失败: %s",
                        query, exc_info=True,
                    )
                    return f"（recall_memory 查询失败: {query}）"
            # 降级：memory engine 不可用时从 Markdown 文件中搜索
            if self._memory is not None:
                try:
                    mem_text = ""
                    if hasattr(self._memory, "read_long_term"):
                        mem_text = str(self._memory.read_long_term() or "")
                    if not mem_text:
                        return "（无长期记忆）"
                    lower_q = query.lower()
                    lines = [l for l in mem_text.splitlines() if lower_q in l.lower()]
                    if not lines:
                        return f"（无与 '{query}' 相关的记忆）"
                    return "\n".join(lines[:5])
                except Exception:
                    logger.warning(
                        "[proactive] recall_memory 降级搜索失败: %s",
                        query, exc_info=True,
                    )
                    return f"（recall_memory 查询失败: {query}）"
            return "（无 memory engine 也无 memory store）"

        elif tool_name == "web_fetch":
            url = str(arguments.get("url", "") or "").strip()
            if not url:
                return '{"error": "url 不能为空"}'
            wf = self._get_web_fetch_tool()
            if wf is None:
                return '{"error": "web_fetch 不可用（初始化失败）"}'
            try:
                result = await wf.execute(
                    url=url,
                    format="text",
                    max_chars=self._cfg.proactive_web_fetch_max_chars,
                )
                return result.text
            except Exception as exc:
                logger.warning("[proactive] web_fetch 失败 url=%s: %s", url, exc)
                return f'{{"error": "web_fetch 请求失败: {exc}"}}'

        elif tool_name == "get_content":
            item_ids = self._parse_item_ids(arguments)
            if not item_ids:
                return '{"error": "item_ids 不能为空"}'
            result = {}
            for iid in item_ids:
                result[iid] = ctx.content_store.get(iid, "")
                if iid not in ctx.content_store:
                    logger.debug("[proactive] get_content: %s 不在预取缓存中", iid)
            return json.dumps(result, ensure_ascii=False)

        else:
            return f"Unknown tool: {tool_name}"

    @staticmethod
    def _parse_item_ids(arguments: dict) -> list[str]:
        """从工具参数中解析 item_ids 列表。

        兼容两种调用方式：
          - item_ids: ["a:1", "b:2"]  （批量，推荐）
          - item_id: "a:1"            （单条，向下兼容）

        输入:
            arguments: LLM 传入的工具参数字典。

        输出:
            去空后的 item_id 字符串列表。
        """
        raw = arguments.get("item_ids") or arguments.get("item_id") or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(i).strip() for i in raw if str(i).strip()]

    # ── ACK ─────────────────────────────────────────────────────────────

    async def _ack_and_mark_all_seen(
        self, ctx: "AgentTickContext", sensor: "Sensor",
    ) -> None:
        """skip/异常路径下对全部条目做 ACK + 本地标记。

        无论 LLM 判定 skip 还是发生异常，本轮拉取到的外部条目都已完成评估，
        应当在 MCP 源层 ACK 并在本地 seen_items 表写入记录，
        避免下次 tick 重复拉取和判断，浪费 token。

        输入:
            ctx: AgentTickContext。
            sensor: Sensor 实例。
        """
        items = ctx.alerts + ctx.contents
        if not items:
            return

        # 1. ACK 到 MCP 源
        fetcher = getattr(sensor, "_source_fetcher", None)
        if fetcher is not None:
            try:
                await fetcher.ack_events(items)
                logger.info("[proactive] skip-ACK %d 外部事件", len(items))
            except Exception as exc:
                logger.warning("[proactive] skip-ACK 失败: %s", exc)

        # 2. 本地 seen_items 兜底
        state_store = getattr(sensor, "_state_store", None)
        if state_store is not None:
            entries: list[tuple[str, str]] = []
            for item in items:
                parts = item.item_id.split(":", 1)
                source_key = parts[0] if len(parts) >= 2 else ""
                entries.append((source_key, item.item_id))
            try:
                state_store.mark_items_seen(entries)
                logger.info("[proactive] skip-mark_seen %d 条目", len(entries))
            except Exception as exc:
                logger.warning("[proactive] skip-mark_seen 失败: %s", exc)
    
    # ── Agent Loop Prompt 构建 ─────────────────────────────────────────

    def _build_agent_system_prompt(self) -> str:
        """构建 Agent Loop 的 system prompt。

        对齐 akashic-agent proactive_turn.py:_build_system_prompt() 的完整模板。

        输出:
            system 消息文本。
        """
        return (
            "你现在处于主动推送决策模式：判断现在是否该给用户发一条消息，以及发什么。\n"
            "数据已预取完毕，会在后续 user 消息里提供；基于那些数据直接决策。\n\n"
            "【优先级】Alert > Content > Context-fallback（本轮是否允许以 user prompt 中的说明为准）\n\n"
            "【你的任务】\n"
            "⚡ 如果本轮有 Alert：把本轮所有 Alert 整合成一条消息，调用 message_push "
            "并填写本轮全部 Alert 的 id 作为 evidence，然后 finish_turn(decision=reply) 结束。\n"
            "Alert 是系统触发的高优先级通知，不走内容筛选流程。\n"
            "1. 对本轮 Content 逐条判断：这条内容是否可能让用户不感兴趣，是否可能不符合规则，"
            "是否值得进入 interesting。\n"
            "2. 你的主工作是分类，不是主动研究新题材，不是主动扩展候选池。\n"
            "3. 你要基于规则和用户偏好，把本轮 Content 分成 interesting 和 not_interesting。\n\n"
            "【你的输出】\n"
            "1. 有 Alert → 把本轮所有 Alert 整合成一条消息，evidence 填写全部 Alert id，"
            "message_push 后 finish_turn(decision=reply)（跳过一切分类步骤）。\n"
            "2. 无 Alert：对每条 Content 给出最终分类：mark_interesting 或 mark_not_interesting。\n"
            "3. 如果最终没有 interesting，调用 finish_turn(decision=skip, reason=no_content)。\n"
            "4. 如果最终有 interesting，生成一条最终消息并按 message_push + finish_turn(decision=reply) 收尾。\n\n"
            "【工具职责】\n"
            "1. 规则面板（PROACTIVE_CONTEXT.md）：这是用户当前明确提出并要求你遵守的规则集合。"
            "它定义你该怎么筛、哪些要先验证、哪些必须过滤；它不提供新闻事实。\n"
            "2. recall_memory：仅用于 Content 评估——判断单条内容是否可能是用户雷点，"
            "或是否可能让用户感兴趣。Alert 不需要调用此工具。\n"
            "3. get_recent_chat：只用于最后判断现在是否适合打扰用户。\n"
            "4. get_content：批量从预取缓存获取候选条目正文。"
            "对你想深入评估的条目调用，返回 {id: text} 映射。"
            "text 为空表示预取失败，可用 web_fetch 降级。\n"
            "5. web_fetch：抓取指定 URL 的网页全文。"
            "当你需要核实细节、补全正文、校验规则中提到的来源时使用。\n"
            "6. mark_interesting / mark_not_interesting：写入最终分类结果，"
            "支持批量传入 item_ids 数组。\n"
            "7. message_push：暂存草稿，不终止 loop。\n"
            "8. finish_turn(decision=reply) 或 finish_turn(decision=skip, reason=...)："
            "提交或放弃，终止 loop。\n\n"
            "【规则优先级】\n"
            "1. 规则面板代表用户当前对主动推送的明确要求，应视为规则而不是建议。\n"
            "2. 当规则面板规定了过滤条件、白名单、黑名单、必须先验证的步骤时，你必须遵守，"
            "不要凭常识跳过。\n"
            "3. recall_memory 只能帮助你判断用户兴趣和雷点，不能替代规则校验。\n"
            "4. 如果规则判断和你的常识直觉冲突，以规则面板为准。\n\n"
            "【信息源规则】\n"
            "1. 主信息源只有本轮已提供的 Alerts / Content / Context。"
            "只有这些来源里的事实才能进入最终发送内容。\n"
            "2. 用户长期记忆、规则面板、recent_chat 只用于过滤、排序、同步规则、判断是否打扰；"
            "它们不是新的事实来源，也不是新的候选主题列表。\n"
            "3. 当本轮 alert 和 content 都为空时，你有三条路：\n"
            "   a. finish_turn(decision=skip, reason=no_content)（默认，大多数情况选这条）\n"
            "   b. get_recent_chat → 若最近对话有自然延伸的未完成话题，"
            "可先 message_push 再 finish_turn(decision=reply) 轻松挑起对话；\n"
            "      此时 evidence 必须为空 []，消息里不得引用任何外部事件或可验证事实。\n"
            "   c. 查看上方的背景上下文（Context）——若温度、健康指标、天气等有明显值得说的亮点，"
            "可以 message_push + finish_turn(reply) 推送；\n"
            "      此时 evidence 必须为空 []。"
            "若没有亮点，必须选 a。\n"
            "   禁止在这三条路之外做任何事。\n\n"
            "【决策流程】\n\n"
            "【Alert 快速路径】本轮如有 Alert：\n"
            "  → get_recent_chat 确认用户不在忙\n"
            "  → 把本轮所有 Alert 的内容整合成一条消息，evidence 必须填写本轮全部 Alert 的 id\n"
            "  → message_push → finish_turn(decision=reply) 结束\n"
            "  → 结束，可以不调用 recall_memory / mark_*\n\n"
            "【Content 路径】本轮无 Alert 时，Content 的主要任务不是做研究，"
            "而是把本轮候选逐条分成 interesting 或 not_interesting。\n"
            "Content 评估必须逐条进行，不能把不同主题的多条内容打包成一次统一判断。\n"
            "每条 Content 必须单独给出 mark_interesting 或 mark_not_interesting 结论，"
            "不能因为先评估的条目不感兴趣就跳过剩余条目直接 skip。\n"
            "只有当某一条内容本身与你已知的用户兴趣明显匹配时，才能把这一条标记为 interesting。\n"
            "如果一批条目里只有部分相关，必须只标记相关的那几条，"
            "其他条目继续判断或标记为 not_interesting。\n"
            "严禁因为其中 1-2 条命中兴趣，就把整批 item_ids 一次性 mark_interesting。\n"
            "调用 mark_interesting / mark_not_interesting 时，尽量附带一句简短 reason，"
            "说明是规则过滤、用户雷点、明显相关或其他哪一种原因。\n\n"
            "推荐的最小流程（仅适用于 Content 路径，Alert 路径见上）：\n"
            "  1. 先看标题和来源，做快速初筛。\n"
            "  2. 对初筛通过的条目，用 get_content 批量获取正文（若已预取），"
            "或 web_fetch 抓取原文。\n"
            "  3. 用 recall_memory 判断这条内容是否可能是用户雷点，或是否可能让用户感兴趣。\n"
            "  4. 最终把每条内容分类为 mark_interesting（批量 item_ids）或 mark_not_interesting（批量 item_ids）。\n"
            "  5. 所有条目分类完毕后：有 interesting → get_recent_chat 判断是否打扰 → "
            "message_push + finish_turn(decision=reply)；"
            "全部不感兴趣 → finish_turn(decision=skip, reason=no_content)\n"
            "  ⚠️ mark_* 不是终止动作，之后必须调 finish_turn\n\n"
            "【发送要求】\n"
            "- 语气自然，像朋友分享，不是推送通知\n"
            "- message_push 必须带非空 message；"
            "finish_turn(decision=skip, reason=...) 不要在之前调用 message_push\n"
            "- 消息里出现的具体数字、事实，必须来自本轮已提供的 Alerts/Content 数据；"
            "严禁基于训练知识或记忆脑补任何可验证事实。\n"
            "- 有链接附链接，没有不硬编\n"
            "- evidence 格式：\"{ack_server}:{event_id}\"，如 \"rss:rss_123\"\n"
            "- 当本轮 content 和 alerts 均为空时，evidence 必须为 []\n"
            "- 没有实质内容时 finish_turn(decision=skip, reason=no_content) 是正确选择\n\n"
            "【finish_turn.reason】no_content | user_busy | all_not_interesting | other"
        )

    def _build_agent_user_prompt(self, ctx: "AgentTickContext") -> str:
        """构建 Agent Loop 的 user prompt（注入本轮数据）。

        输入:
            ctx: AgentTickContext。

        输出:
            包含所有本轮采集数据的 user 消息文本。
        """
        from datetime import datetime, timezone

        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y年%m月%d日 %H:%M")
        parts = [f"当前时间：{now_str}"]

        # ── Alerts ──
        if ctx.alerts:
            lines = [f"## Alerts（{len(ctx.alerts)} 条，优先处理）"]
            for i, a in enumerate(ctx.alerts, 1):
                lines.append(a.to_prompt_line(i))
            parts.append("\n".join(lines))

        # ── Content ──
        if ctx.contents:
            lines = [f"## Content 候选列表（{len(ctx.contents)} 条，逐条判断）"]
            for i, c in enumerate(ctx.contents, 1):
                body = ctx.content_store.get(c.item_id, "")
                has_body = bool(body)
                line = c.to_prompt_line(i, has_content=has_body)
                if has_body:
                    snippet = body[:300].replace("\n", " ")
                    line += f"\n       【正文预览】{snippet}..."
                lines.append(line)
            parts.append("\n".join(lines))

        # ── Context ──
        if ctx.contexts:
            lines = ["## 背景上下文"]
            for c in ctx.contexts:
                lines.append(json.dumps(c.to_prompt_item(), ensure_ascii=False))
            parts.append("\n".join(lines))

        # ── 近期对话 ──
        parts.append(f"## 近期对话\n{ctx.recent_chat or '（无）'}")

        # ── 长期记忆 ──
        if ctx.memory_text:
            parts.append(f"## 用户长期记忆\n{ctx.memory_text[:1500]}")

        # ── 规则面板 ──
        if ctx.context_rules:
            parts.append(f"## 推送规则（硬规则，必须遵守）\n{ctx.context_rules[:2000]}")

        # ── 近期主动消息 ──
        if ctx.recent_proactive_text:
            parts.append(f"## 近期已推\n{ctx.recent_proactive_text[:1500]}")

        # ── 任务指令 ──
        if ctx.alerts:
            parts.append(
                "## 任务\n"
                "本轮有 Alert。走 Alert 快速路径——跳过一切 Content 分类步骤：\n"
                "1. get_recent_chat\n"
                "2. 整合所有 Alert → message_push → finish_turn(reply)"
            )
        elif ctx.contents:
            parts.append(
                "## 任务\n"
                "本轮无 Alert。逐条查看 Content 候选，对每条调 mark_interesting "
                "或 mark_not_interesting。全部分类完毕后：\n"
                "- 有 interesting → get_recent_chat → message_push → finish_turn(reply)\n"
                "- 没有 → finish_turn(skip, reason=all_not_interesting)"
            )
        else:
            parts.append(
                "## 任务\n"
                "本轮无 Alert/Content 候选。你有三条路：\n"
                "a. finish_turn(skip, reason=no_content) — 默认，选这条\n"
                "b. get_recent_chat → 若最近对话有自然延伸的未完成话题，"
                "可 message_push + finish_turn(reply)（evidence 必须为空）\n"
                "c. 查看上方背景上下文——若有明显亮点（天气突变、健康异常等），"
                "可 message_push + finish_turn(reply)（evidence 必须为空）\n"
                "禁止编造任何 item_id，禁止调用 recall_memory"
            )

        return "\n\n".join(parts)
    
    async def _maybe_run_drift(
        self,
        session_key: str,
        now_utc: "datetime",
    ) -> None:
        """检查 Drift 条件，满足时执行 DriftTurnPipeline。

        条件：
        1. drift_store 已配置
        2. 距离上次 drift 超过 min_interval_hours
        3. 有可用的 skill

        输入:
            session_key: 目标会话 key。
            now_utc: 当前 UTC 时间。

        输出:
            None。
        """
        if self._drift_store is None:
            return

        # 条件 1: min_interval 约束
        min_hours = float(getattr(self._cfg, "drift_min_interval_hours", 2))
        last = self._drift_store.get_last_drift_at()
        if last is not None:
            since_last = (now_utc - last).total_seconds() / 3600.0
            if since_last < min_hours:
                logger.info(
                    "[proactive] drift 跳过: 距上次 drift 仅 %.1f 小时 "
                    "(min=%.1f)", since_last, min_hours,
                )
                return

        # 条件 2: 有可用 skill
        if not self._drift_store.scan_skills():
            logger.info("[proactive] drift 跳过: 无可用 skill")
            return

        # 构建并运行 DriftTurnPipeline
        ctx = DriftAgentTickContext(
            now_utc=now_utc,
            session_key=session_key,
        )

        pipeline = DriftTurnPipeline(
            store=self._drift_store,
            provider=self._provider,
            model=self._model,
            max_steps=self._cfg.drift_max_steps,
            max_web_fetch_chars=self._cfg.drift_web_fetch_max_chars,
            memory=self._memory,
            shared_tools=getattr(self, "_tools", None),
            send_message_fn=lambda text, media: (
                self._drift_send_message(session_key, text, media)
            ),
            sessions=self._sessions,
            tool_hooks=getattr(self, "_tool_hooks", None),
        )

        try:
            entered = await pipeline.run(ctx)
            if entered:
                logger.info(
                    "[proactive] drift 已完成: tick_id=%s finished=%s "
                    "message_sent=%s steps=%d",
                    ctx.tick_id, ctx.drift_finished,
                    ctx.drift_message_sent, ctx.steps_taken,
                )
        except Exception:
            logger.exception("[proactive] drift 异常")

    
    async def _drift_send_message(
        self,
        session_key: str,
        text: str,
        media_paths: list[str],
    ) -> bool:
        """将 push_tool 包装为 Drift 需要的 (text, media) → bool 签名。

        从 _maybe_run_drift 通过 lambda 桥接传入 session_key，
        保持方法签名独立、不闭包捕获外部变量。

        输入:
            session_key: 目标会话 key。
            text: 消息文本。
            media_paths: 媒体文件路径列表。

        输出:
            True 表示发送成功。
        """
        if self._push_tool is None:
            return False
        try:
            parts = session_key.split(":", 1)
            channel = parts[0]
            chat_id = parts[1] if len(parts) > 1 else ""
            await self._push_tool.execute(
                channel=channel or self._cfg.default_channel,
                chat_id=chat_id or str(
                    getattr(self._cfg, "default_chat_id", "")
                ),
                message=text,
                proactive=True,
            )
            return True
        except Exception:
            logger.warning(
                "[proactive] drift send_message 失败: session=%s", session_key,
            )
            return False
    
    def _maybe_cleanup_state(self) -> None:
        """每 24 小时触发一次 ProactiveStateStore.cleanup()。"""
        if self._state_store is None:
            return
        now_ts = _utcnow().timestamp()
        last = getattr(self, "_last_cleanup_ts", 0.0)
        if now_ts - last > 86400:
            self._last_cleanup_ts = now_ts
            asyncio.create_task(
                asyncio.to_thread(self._state_store.cleanup),
                name="proactive_state_cleanup",
            )
    
    
    @staticmethod
    def _format_recent_chat(
        recent: list[dict[str, Any]],
    ) -> str:
        """将近期对话消息列表格式化为可读文本。

        输入:
            recent: Sensor.collect_recent_chat() 的返回值。

        输出:
            格式化后的多行文本；空列表返回空字符串。
        """
        if not recent:
            return ""
        lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:300]
            prefix = "用户" if role == "user" else "助手"
            ts = msg.get("timestamp", "")
            if ts:
                try:
                    t = datetime.fromisoformat(ts).astimezone()
                    ts_str = t.strftime('%m-%d %H:%M')
                    lines.append(f"[{ts_str}] {prefix}: {content}")
                except (ValueError, TypeError):
                    lines.append(f"{prefix}: {content}")
            else:
                lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    # _format_recent_proactive() 在第 33 章 MCP Sources 就位后激活——
    # 从 ProactiveStateStore 读最近 5 条已推消息，格式化后喂进 Judge prompt
    # "近期主动消息"段，LLM 在评分时自然判断语义重复。
    
    
    
    
    def _next_interval(self, base_score: float | None = None) -> int:
        """根据 base_score 返回自适应等待秒数。

        无 presence 时回退到config.toml 配置的绝对固定间隔；首次启动时依据电量估算初始值。

        输入:
            base_score: 上次 tick 的综合评分；None 表示首次启动或 tick 异常。

        输出:
            下次等待秒数。
        """
        # 无 presence → 固定间隔
        if not self._presence:
            return self._interval_seconds

        # 首次启动：用电量估算初始 base_score
        if base_score is None:
            session_key = self._target_session_key
            last_user_at = self._presence.get_last_user_at(session_key)
            energy = compute_energy(last_user_at)
            base_score = d_energy(energy) * self._w_e

        return next_tick_from_score(
            base_score,
            tick_s3=self._tick_s3,
            tick_s2=self._tick_s2,
            tick_s1=self._tick_s1,
            tick_s0=self._tick_s0,
            tick_jitter=self._tick_jitter,
            rng=self._rng,
        )

    # ── 规则面板 ───────────────────────────────────────────────────

    def _proactive_context_path(self) -> Path:
        """返回 PROACTIVE_CONTEXT.md 的完整路径。

        PROACTIVE_CONTEXT.md 放在 workspace 根目录下，
        与 schedules.json、mcp_servers.json 等其他单文件制品同级。

        输出:
            Path 对象。
        """
        return self._workspace_root / _PROACTIVE_CONTEXT_FILE

    def _ensure_proactive_context_file(self) -> None:
        """确保规则面板文件存在——不存在则用模板创建。

        workspace 根目录已由 Workspace.ensure() 创建，
        此处只需写文件。

        输出:
            None。
        """
        path = self._proactive_context_path()
        if path.exists():
            return
        path.write_text(_PROACTIVE_CONTEXT_TEMPLATE, encoding="utf-8")

    def _read_proactive_context(self) -> str:
        """读取规则面板文件内容。

        输出:
            文件文本内容（去首尾空白）；读取失败返回空字符串。
        """
        path = self._proactive_context_path()
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("[proactive] 读取 proactive context 失败: %s", e)
            return ""