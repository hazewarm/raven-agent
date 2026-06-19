"""
proactive/sensor.py —— 数据采集层。

在每个 Proactive tick 中收集当前可用的所有数据：
  - 近期对话消息（从 SessionManager 读取 target session 的历史）
  - 长期记忆文本（从 MarkdownMemoryStore 读取 MEMORY.md）
  - 主动上下文规则（从 workspace 读取 PROACTIVE_CONTEXT.md）
  - 打断系数 compute_interruptibility()：三维分量加权 + 随机探索
    · f_reply：用户对上次推送的回复情况
    · f_activity：用户最近的全局活跃度
    · f_fatigue：近期推送疲劳度
  - 近期主动消息列表（用于 Judge 去重参考）

当前为"上下文感知模式"——数据源限于本地已有信息。
第 33 章将通过 MCP 数据源注入外部内容（Alerts / Content / Context），
届时 Sensor.collect_external() 方法作为扩展点接入。
"""

from __future__ import annotations

import logging
import math
import random as _random_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import asyncio

from raven_agent.proactive.energy import compute_energy, d_recent
from raven_agent.proactive.presence import PresenceStore

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """返回当前 UTC 时间（aware）。

    输出:
        datetime 对象。
    """
    return datetime.now(timezone.utc)


class Sensor:
    """Proactive 数据采集器。

    在每个 tick 中收集本地可用的所有上下文数据。
    当前为上下文感知模式——数据源限于 session 历史 + memory + 规则面板。
    第 33 章通过 MCP 注入外部内容源时，可扩展 collect_external() 方法。

    参数:
        sessions: SessionManager 实例（读取近期对话）。
        presence: PresenceStore 实例（读取心跳数据）。
        memory: MarkdownMemoryStore 实例（读取长期记忆）。
        workspace_root: workspace 根目录（读取 PROACTIVE_CONTEXT.md）。
        cfg: ProactiveConfig 实例（算法参数）。
        rng: 可选的 random.Random 实例（确定性测试用）。
    """

    def __init__(
        self,
        *,
        sessions: Any,
        presence: PresenceStore | None,
        memory: Any | None,
        workspace_root: Path,
        cfg: Any,
        rng: _random_module.Random | None = None,
        source_fetcher: Any = None,
        state_store: Any = None,
    ) -> None:
        self._sessions = sessions
        self._presence = presence
        self._memory = memory
        self._workspace_root = workspace_root
        self._cfg = cfg
        self._rng = rng or _random_module.Random()
        self._source_fetcher = source_fetcher
        self._state_store = state_store

    # ── 近期对话 ───────────────────────────────────────────────────

    def collect_recent_chat(
        self,
        session_key: str,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """采集目标 session 的近期对话消息。

        过滤掉 role 非 user/assistant 的消息和空 content，
        每条消息截取前 200 字符以避免 prompt 过长。

        输入:
            session_key: 目标会话 key。
            n: 采集条数，默认使用 cfg.recent_chat_messages。

        输出:
            [{"role": "user", "content": "...", "timestamp": "..."}, ...] 列表。
        """
        max_n = n if n is not None else getattr(self._cfg, "recent_chat_messages", 20)
        if not session_key:
            return []
        try:
            session = self._sessions.get_or_create(session_key)
        except Exception:
            return []
        messages = session.messages[-max_n:]
        results: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "role", "")
            content = str(getattr(msg, "content", ""))
            if role not in ("user", "assistant"):
                continue
            if not content:
                continue
            # 跳过上下文帧（被动链路注入的系统信息块）
            if content.startswith("[系统上下文]"):
                continue
            results.append({
                "role": role,
                "content": content[:200],
                "timestamp": getattr(msg, "timestamp", ""),
            })
        return results

    # ── 长期记忆 ───────────────────────────────────────────────────

    def read_long_term_memory(self) -> str:
        """读取长期记忆文本（MEMORY.md 内容）。

        用于 Judge 了解用户偏好、兴趣、长期约定。

        输出:
            MEMORY.md 文本内容；无 memory store 或读取失败返回空字符串。
        """
        if self._memory is None:
            return ""
        try:
            return str(self._memory.read_long_term() or "").strip()
        except Exception:
            return ""

    def has_memory(self) -> bool:
        """检查是否有可用的长期记忆。

        输出:
            True 表示 memory store 存在且 MEMORY.md 有内容。
        """
        return bool(self.read_long_term_memory())

    # ── 规则面板 ───────────────────────────────────────────────────

    def read_proactive_context(self) -> str:
        """读取用户主动推送规则面板（PROACTIVE_CONTEXT.md）。

        这个文件由主 Agent 维护，定义白名单/黑名单/过滤条件/优先级。
        Proactive Agent 将其视为必须遵守的规则，而非参考建议。

        输出:
            PROACTIVE_CONTEXT.md 文本内容；文件不存在或读取失败返回空字符串。
        """
        path = self._workspace_root / "PROACTIVE_CONTEXT.md"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    
    # ── 外部数据源 ─────────────────────────────────────────────────

    async def collect_external(
        self,
        session_key: str,
    ) -> dict[str, Any]:
        """从 MCP source 拉取三代外部数据。

        返回的字典包含：
        alert_items：AlertContract 列表（去重后）。
        content_items：ContentContract 列表（去重后）。
        context_items：ContextContract 列表。
        total_after_dedup：去重后的 alert + content 总数，用于 D_content 计算。

        输入:
            session_key: 目标 session key。

        输出:
            含 alert_items / content_items / context_items / 统计字段的字典。
        """
        fetcher = self._source_fetcher
        if fetcher is None:
            return {
                "alert_items": [], "content_items": [], "context_items": [],
                "total_after_dedup": 0,
            }

        # 一次读盘，三次复用——alert / content / context 各自调用
        # _fetch_by_channel 时不再各自触发 load_sources()。
        # 在树莓派 SD 卡等低速介质上，避免单次 tick 内并发 3 次
        # 对同一个几百字节小文件的同步磁盘 I/O。
        sources = fetcher._source_store.load_sources()

        async def _fetch_safely(coro, name: str) -> list:
            """包装单个 fetch 调用，独立超时 + 异常隔离。

            一个 source 的网络故障不应丢弃另外两个已成功返回的数据。
            """
            try:
                return await asyncio.wait_for(coro, timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("[sensor] 外部采集超时（15s）: %s", name)
            except Exception as exc:
                logger.warning("[sensor] 外部采集失败: %s err=%s", name, exc)
            return []

        alerts, content, context = await asyncio.gather(
            _fetch_safely(fetcher.fetch_alerts(sources=sources), "alerts"),
            _fetch_safely(fetcher.fetch_content(sources=sources), "content"),
            _fetch_safely(fetcher.fetch_context(sources=sources), "context"),
        )

        # 去重（基于 ProactiveStateStore.seen_items）
        if self._state_store is not None:
            source_configs = {src["server"]: src for src in sources}
            # alert 去重
            new_alerts: list = []
            for a in alerts:
                parts = a.item_id.split(":", 1)
                sk = parts[0] if len(parts) >= 2 else ""
                ttl = source_configs.get(sk, {}).get("dedupe_ttl_hours", 72)
                if not self._state_store.is_item_seen(sk, a.item_id, int(ttl)):
                    new_alerts.append(a)
            alerts = new_alerts

            # content 去重
            new_content: list = []
            for c in content:
                src_key = c.raw.get("ack_server", "")
                ttl = source_configs.get(src_key, {}).get("dedupe_ttl_hours", 72)
                if not self._state_store.is_item_seen(src_key, c.item_id, int(ttl)):
                    new_content.append(c)
            content = new_content

        return {
            "alert_items": alerts,
            "content_items": content,
            "context_items": context,
            "total_after_dedup": len(alerts) + len(content),
        }
    
    
    # ── 打断系数 ───────────────────────────────────────────────────

    def compute_interruptibility(
        self,
        session_key: str,
        *,
        now_utc: datetime | None = None,
        recent_msg_count: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """计算当前时刻对目标 session 的打断适宜度。

        三维分量：
          f_reply   — 用户在上次推送后是否回复过？回复越快越高。
          f_activity — 用户最近是否全局活跃？越活跃越高。
          f_fatigue  — 过去 24h 推送了多少次？越多越低。

        三个分量按配置权重加权，外加随机探索扰动防止行为僵化。

        输入:
            session_key: 目标会话 key。
            now_utc: 当前时间（aware）；默认 UTC now。
            recent_msg_count: 近期消息条数（用于 f_activity 的对话丰富度分量）。

        输出:
            (interruptibility, detail) 元组。
            interruptibility: [0, 1] 的打断系数。
            detail: {"f_reply": ..., "f_activity": ..., "f_fatigue": ...,
                     "random_delta": ...}。
        """
        now = now_utc or _utcnow()
        presence = self._presence
        if presence is None or not session_key:
            return 1.0, {
                "f_reply": 1.0, "f_activity": 1.0,
                "f_fatigue": 1.0, "random_delta": 0.0,
            }

        # 1. 计算三个确定性分量。
        f_reply = self._reply_factor(session_key, now)
        f_activity = self._activity_factor(session_key, now, recent_msg_count)
        f_fatigue = self._fatigue_factor(session_key, now)

        # 2. 按配置权重聚合。
        raw = self._weighted_interruptibility(f_reply, f_activity, f_fatigue)

        # 3. 追加随机探索扰动（±random_strength 的均匀噪声）。
        random_strength = float(
            getattr(self._cfg, "interrupt_random_strength", 0.12)
        )
        random_delta = self._rng.uniform(-random_strength, random_strength)

        # 4. 裁剪到 [min_floor, 1.0]。
        min_floor = float(getattr(self._cfg, "interrupt_min_floor", 0.08))
        score = max(min_floor, min(1.0, raw + random_delta))

        return score, {
            "f_reply": f_reply,
            "f_activity": f_activity,
            "f_fatigue": f_fatigue,
            "random_delta": random_delta,
        }

    def _reply_factor(self, session_key: str, now_utc: datetime) -> float:
        """计算用户对上次推送的回复意愿分量。

        如果上次推送后用户回复了，回复延迟越短 → f_reply 越高（用户接受推送）。
        如果上次推送后用户没回复，静默越久 → f_reply 越低（用户可能在忙/不感兴趣）。

        输入:
            session_key: 目标会话 key。
            now_utc: 当前时间。

        输出:
            [0, 1] 的回复意愿分量。
        """
        presence = self._presence
        if presence is None:
            return 0.6

        last_user = presence.get_last_user_at(session_key)
        last_proactive = presence.get_last_proactive_at(session_key)

        # 从未推送过 → 中性值
        if last_proactive is None:
            return 0.6

        # 用户在上次推送后回复了 → 回复延迟越短，f_reply 越高
        if last_user is not None and last_user > last_proactive:
            lag_min = max(0.0, (last_user - last_proactive).total_seconds() / 60.0)
            decay = max(
                float(getattr(self._cfg, "interrupt_reply_decay_minutes", 120.0)), 1.0
            )
            # 核心数学模型：指数衰减函数 y = e^(-x/k)
            # 假设 decay 为 120：
            # - 如果 lag_min = 0 分钟 (秒回)   -> exp(0) = 1.00 (满分)
            # - 如果 lag_min = 60 分钟         -> exp(-0.5) ≈ 0.60
            # - 如果 lag_min = 120 分钟        -> exp(-1) ≈ 0.36
            # - 如果 lag_min = 240 分钟        -> exp(-2) ≈ 0.13
            return math.exp(-lag_min / decay)

        # 用户在上次推送后未回复 → 静默越久，f_reply 越低
        silence_min = max(0.0, (now_utc - last_proactive).total_seconds() / 60.0)
        # 静默衰减参数（默认 360 分钟 / 6小时）
        decay = max(
            float(getattr(self._cfg, "interrupt_no_reply_decay_minutes", 360.0)), 1.0
        )
        # 核心数学模型：带底线的指数衰减函数 y = 0.15 + 0.35 * e^(-x/k)
        # 注意：最高分只有 0.15 + 0.35 = 0.50（因为既然用户没回复，初始意愿分就不该太高）。
        # 最低分会被托底在 0.15。
        # 假设 decay 为 360：
        # - 如果 silence_min = 0 分钟 (刚刚推送完) -> 0.15 + 0.35 * exp(0) = 0.50
        # - 如果 silence_min = 360 分钟 (静默6小时) -> 0.15 + 0.35 * exp(-1) ≈ 0.27
        # - 如果 silence_min = 720 分钟 (静默12小时) -> 0.15 + 0.35 * exp(-2) ≈ 0.19
        return 0.15 + 0.35 * math.exp(-silence_min / decay)

    def _activity_factor(
        self,
        session_key: str,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> float:
        """计算用户活跃度分量。

        综合两个子分量：
          f_live：距上次全局活跃越久 → 越低（指数衰减）。
          f_recent：近期对话条数越多 → 越高（对数归一化）。
        各占 50% 权重。

        输入:
            session_key: 目标会话 key。
            now_utc: 当前时间。
            recent_msg_count: 近期消息条数。

        输出:
            [0, 1] 的活跃度分量。
        """
        presence = self._presence
        if presence is None:
            return 0.2

        # f_live：基于全局最近活跃时间
        last_global_user = presence.most_recent_user_at()
        if last_global_user is None:
            f_live = 0.2
        else:
            idle_min = max(0.0, (now_utc - last_global_user).total_seconds() / 60.0)
            decay = max(
                float(getattr(self._cfg, "interrupt_activity_decay_minutes", 180.0)),
                1.0,
            )
            f_live = math.exp(-idle_min / decay)

        # f_recent：基于近期对话条数
        recent_scale = float(getattr(self._cfg, "score_recent_scale", 10.0))
        f_recent = d_recent(recent_msg_count, recent_scale)

        return 0.5 * f_live + 0.5 * f_recent

    def _fatigue_factor(self, session_key: str, now_utc: datetime) -> float:
        window_hours = int(
            getattr(self._cfg, "interrupt_fatigue_window_hours", 24)
        )
        soft_cap = max(
            float(getattr(self._cfg, "interrupt_fatigue_soft_cap", 6.0)), 0.1
        )

        if self._state_store is not None:
            sent_count = self._state_store.count_deliveries_in_window(
                session_key, window_hours, now=now_utc,
            )
        else:
            sent_count = self._count_deliveries_in_window(
                session_key, window_hours, now_utc,
            )

        return 1.0 / (1.0 + sent_count / soft_cap)

    def _weighted_interruptibility(
        self,
        f_reply: float,
        f_activity: float,
        f_fatigue: float,
    ) -> float:
        """按配置权重加权聚合三个打断分量。

        输入:
            f_reply: 回复意愿分量。
            f_activity: 活跃度分量。
            f_fatigue: 疲劳度分量。

        输出:
            [0, 1] 的原始打断系数（加权平均）。
        """
        w_reply = float(getattr(self._cfg, "interrupt_weight_reply", 0.35))
        w_activity = float(getattr(self._cfg, "interrupt_weight_activity", 0.25))
        w_fatigue = float(getattr(self._cfg, "interrupt_weight_fatigue", 0.15))
        w_sum = w_reply + w_activity + w_fatigue
        if w_sum <= 0:
            return 0.0
        return (w_reply * f_reply + w_activity * f_activity + w_fatigue * f_fatigue) / w_sum

    # ── 推送频率统计（基于内存 delivery 记录）──────────────────────

    def _count_deliveries_in_window(
        self,
        session_key: str,
        window_hours: int,
        now_utc: datetime,
    ) -> int:
        """统计过去 window_hours 小时内对目标 session 的推送次数。

        基于内存中的 delivery 时间戳列表统计。
        当前实现为简化版——使用 Sensor 实例上的 _delivery_log 字典。
        后续可升级为 ProactiveStateStore 的 SQLite 持久化版本。

        输入:
            session_key: 目标会话 key。
            window_hours: 统计窗口（小时）。
            now_utc: 当前时间。

        输出:
            推送次数的整数计数。
        """

        # 1. 懒加载 (Lazy Initialization) 字典
        # 为什么不在 __init__ 里声明？因为这是一个轻量级的内存态实现。
        # 只有当真正发生过推送，或者第一次查询时，才挂载这个字典，节省内存开销。
        if not hasattr(self, "_delivery_log"):
            self._delivery_log: dict[str, list[datetime]] = {}
        
        # 获取目标用户的历史推送时间戳列表，如果没有则返回空列表
        timestamps = self._delivery_log.get(session_key, [])
        count = 0
        window_s = window_hours * 3600.0
        for ts in timestamps:
            delta = (now_utc - ts).total_seconds()

            if 0 <= delta <= window_s:
                count += 1
        return count

    def record_delivery(self, session_key: str, when: datetime | None = None) -> None:
        """记录一次推送发送（供疲劳度计算使用）。

        输入:
            session_key: 目标会话 key。
            when: 推送时间；默认 UTC now。

        输出:
            None。
        """
        if not hasattr(self, "_delivery_log"):
            self._delivery_log = {}
        ts = when or _utcnow()
        self._delivery_log.setdefault(session_key, []).append(ts)

    # ── 近期主动消息 ───────────────────────────────────────────────

    def collect_recent_proactive(
        self,
        session_key: str,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """采集近期主动推送消息（用于 Judge 去重参考）。

        输入:
            session_key: 目标 session key。
            n: 采集条数，默认 5。

        输出:
            [{"content": "...", "sent_at": "..."}, ...] 列表。
        """
        if self._state_store is None:
            return []
        return self._state_store.recent_deliveries(session_key, n)

    @staticmethod
    def _parse_timestamp(raw: Any) -> datetime | None:
        """将原始时间戳解析为 aware datetime。

        输入:
            raw: ISO 字符串或其他可解析格式。

        输出:
            aware datetime；解析失败返回 None。
        """
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            ts = datetime.fromisoformat(text)
        except Exception:
            return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return ts