"""
Scheduler: 定时任务核心模块。

组件:
  LatencyTracker     — 自适应 P90 延迟估算（软实时预触发）
  parse_duration     — "30s" / "5m" / "2h" 等时长解析
  parse_when_at      — "14:30" / ISO datetime 解析
  is_cron_expr       — 判断是否是 cron 表达式
  compute_fire_at    — 计算首次触发时间（含 request_time 延迟补偿）
  compute_actual_trigger — 计算实际触发时间（SOFT 提前 P90）
  ScheduledJob       — 任务数据类
  JobStore           — JSON 持久化
  SchedulerService   — 主调度服务（asyncio tick 循环）
"""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from raven_agent.persistence import load_json, save_json

logger = logging.getLogger(__name__)


# ── 延迟跟踪模块 ───────────────────────────────────────────────


class LatencyTracker:
    """滑动窗口 P90 延迟追踪，用于 SOFT tier 预触发偏移量自适应。

    每次 SOFT 任务执行完成后记录 AI 推理耗时。
    lead 属性返回 P90 估算值，作为 SOFT 预触发的提前量。

    参数:
        default: 样本不足时的默认 P90 值（秒），默认 25.0。
        window: 滑动窗口最大样本数，默认 20。即保留最近 20 次的 AI 推理耗时记录用于 P90 计算。
    """

    def __init__(self, default: float = 25.0, window: int = 20) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self.default = default

    def record(self, elapsed: float) -> None:
        """记录一次 AI 执行耗时。

        输入:
            elapsed: 本次 AI 推理耗时，单位秒。

        输出:
            None。
        """
        self._samples.append(elapsed)

    @property
    def lead(self) -> float:
        """返回 P90 估算值（秒）；样本不足 3 个时返回 default。

        输出:
            float，预触发提前量（秒）。
        """
        if len(self._samples) < 3:
            return self.default
        # 返回 n-1 个分位点中的第9个，即P90
        return statistics.quantiles(list(self._samples), n=10)[8]


# ── 时间解析模块 ─────────────────────────────────────────────────

_DURATION_RE = re.compile(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")

# 只能匹配 "1d2h30m15s" 这种格式，必须dhms顺序且中间只能是整数（1.5h非法）
def parse_duration(s: str) -> timedelta:
    """解析时长字符串。

    支持格式: '30s', '5m', '2h', '1h30m', '1d2h'。

    输入:
        s: 时长字符串。

    输出:
        timedelta 对象。

    异常:
        ValueError: 格式无效时抛出。
    """
    s = s.strip()
    m = _DURATION_RE.match(s)
    if not m or not any(m.groups()):
        raise ValueError(f"无效的时间间隔: {s!r}，示例: '30s', '5m', '2h', '1h30m'")
    days, hours, minutes, seconds = (int(x or 0) for x in m.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)



def parse_when_at(
    s: str,
    tz: str = "UTC",
    _now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """解析 'at' 时间：HH:MM（自动判断今天/明天）或 ISO datetime。

    输入:
        s: 时间字符串，如 "14:30" 或 "2025-06-01T09:00"。
        tz: 时区名称，如 "Asia/Shanghai"，默认 "UTC"。
        _now_fn: 可注入的 now 函数，仅用于测试。正常使用时应保持为 None。

    输出:
        timezone-aware datetime 对象。

    异常:
        ValueError: 格式无效时抛出。
    """
    tzinfo = ZoneInfo(tz)
    now_fn = _now_fn or (lambda: datetime.now(tzinfo))
    s = s.strip()

    # HH:MM 格式
    if re.match(r"^\d{1,2}:\d{2}$", s):
        now = now_fn()
        t = datetime.strptime(s, "%H:%M").time()
        dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        if dt <= now:
            # 如果时间已过，则安排在明天
            dt += timedelta(days=1)
        return dt

    # ISO datetime 格式
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        return dt
    except ValueError:
        pass

    raise ValueError(f"无法解析时间: {s!r}，示例: '14:30', '2025-06-01T09:00'")


def is_cron_expr(s: str) -> bool:
    """判断字符串是否是 cron 表达式（5 或 6 字段）。

    输入:
        s: 待判断字符串。

    输出:
        True 表示是 cron 表达式。
    """
    parts = s.strip().split()
    # 只要是 5 或 6 个字段的字符串，我们就暂定它是一个 cron 表达式（更严格的验证在之后）
    return len(parts) in (5, 6)


# ── Cron 语法辅助函数 ─────────────────────────────────────────────────

# 解析单个cron字段
def _parse_cron_field(field: str, minimum: int, maximum: int) -> set[int]:
    """解析单个 cron 字段（分/时/日/月/周），返回匹配值的集合。

    输入:
        field: 单个 cron 字段，如 "*/5"、"1,3,5"、"9-17"。
        minimum: 字段最小值。
        maximum: 字段最大值。

    输出:
        匹配值的集合。

    异常:
        ValueError: 字段无效时抛出。
    """
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"无效 cron step: {field!r}")
        if part == "*":
            start, end = minimum, maximum
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(part)
        if start < minimum or end > maximum or start > end:
            raise ValueError(f"无效 cron 字段: {field!r}")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError(f"无效 cron 字段: {field!r}")
    return values

# 解析 cron 表达式的下次触发时间（apscheduler 不可用时的回退方案）
def _next_cron_fire_fallback(cron_expr: str, tz: str, after: datetime) -> datetime:
    """自实现的 cron 下次触发时间计算（apscheduler 不可用时的回退方案）。

    输入:
        cron_expr: 5 或 6 字段的 cron 表达式。
        tz: 时区名称。
        after: 从此时间之后开始寻找下次触发。

    输出:
        UTC-aware datetime。

    异常:
        ValueError: cron 表达式无效或无法在合理范围内找到时抛出。
    """
    parts = cron_expr.strip().split()
    if len(parts) == 5:
        # 传统 5 字段 cron，精度到“分钟”
        second_values = {0}
        minute_s, hour_s, dom_s, month_s, dow_s = parts
        step = timedelta(minutes=1)
        current = after.astimezone(ZoneInfo(tz)).replace(second=0, microsecond=0)
        if current <= after.astimezone(ZoneInfo(tz)):
            current += step
    elif len(parts) == 6:
        second_s, minute_s, hour_s, dom_s, month_s, dow_s = parts
        second_values = _parse_cron_field(second_s, 0, 59)
        step = timedelta(seconds=1)
        current = after.astimezone(ZoneInfo(tz)).replace(microsecond=0) + step
    else:
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")

    # 将剩下的字符串片段全部解析成 Python Set（集合），方便后续 O(1) 的极速比对
    minute_values = _parse_cron_field(minute_s, 0, 59)
    hour_values = _parse_cron_field(hour_s, 0, 23)
    dom_values = _parse_cron_field(dom_s, 1, 31)
    month_values = _parse_cron_field(month_s, 1, 12)
    # cron DOW: 0=Sunday, Python weekday(): 0=Monday → 转换
    dow_values = _parse_cron_field(dow_s.replace("7", "0"), 0, 6)

    # 设置防死循环的最大推演上限：最多推演一年零一天 (366天)。
    # 循环次数取决于步长是分钟还是秒。
    for _ in range(366 * 24 * 60 * (60 if len(parts) == 6 else 1)):
        cron_dow = (current.weekday() + 1) % 7
        if (
            current.second in second_values
            and current.minute in minute_values
            and current.hour in hour_values
            and current.day in dom_values
            and current.month in month_values
            and cron_dow in dow_values
        ):
            # 下一个触发时间
            return current.astimezone(timezone.utc)
        current += step
    raise ValueError(f"无法在合理范围内解析 cron 表达式: {cron_expr!r}")


def next_cron_fire(cron_expr: str, tz: str, after: datetime) -> datetime:
    """计算 cron 表达式下次触发时间。

    优先使用 apscheduler CronTrigger 精确计算，
    apscheduler 不可用时回退到自实现 fallback。

    输入:
        cron_expr: 5 或 6 字段的 cron 表达式。
        tz: 时区名称。
        after: 从此时间之后开始寻找。

    输出:
        UTC-aware datetime。
    """
    try:
        from apscheduler.triggers.cron import CronTrigger
    except ModuleNotFoundError:
        return _next_cron_fire_fallback(cron_expr, tz, after)

    # APScheduler 3.x 兼容：优先 pytz，回退 ZoneInfo
    try:
        pytz = import_module("pytz")
        tzinfo = pytz.timezone(tz)
    except Exception:
        tzinfo = ZoneInfo(tz)

    trigger = CronTrigger.from_crontab(cron_expr, timezone=tzinfo)
    result = trigger.get_next_fire_time(None, after)
    if result is None:
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


# ── 计算首次触发时间（含 request_time 延迟补偿） ──────────────────────────────────────────


def compute_fire_at(
    trigger: str,
    when: str,
    tz: str = "UTC",
    request_time: str | None = None,
    _now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """计算首次触发时间。

    after 模式：以 request_time（用户消息到达时间）为基准，
    补偿 AI 推理延迟，确保 fire_at 从用户视角算起。

    输入:
        trigger: "at" | "after" | "every"。
        when: 触发时间描述，如 "14:30" / "5m" / "0 9 * * *"。
        tz: 时区名称。
        request_time: trigger=after 时可选的用户消息到达时间（ISO 格式）。
        _now_fn: 可注入的 now 函数。

    输出:
        UTC-aware datetime 首次触发时间。
    """
    tzinfo = ZoneInfo(tz)
    now_fn = _now_fn or (lambda: datetime.now(tzinfo))

    # ==========================================
    # 场景 1：绝对时间模式 ("at")
    # 例如：when="14:30" 或 "2025-06-01T09:00"
    # ==========================================
    if trigger == "at":
        return parse_when_at(when, tz, _now_fn)

    # ==========================================
    # 场景 2：相对延迟模式 ("after")
    # 例如：when="30m" (30分钟后)
    # 核心设计：消除 AI 思考和网络延迟带来的起始时间误差（需要传入信息实际创建的时间戳）
    # ==========================================
    if trigger == "after":
        duration = parse_duration(when)
        if request_time:
            base = datetime.fromisoformat(request_time)
            if base.tzinfo is None:
                base = base.replace(tzinfo=tzinfo)
        else:
            base = now_fn()
        return base + duration

    # ==========================================
    # 场景 3：循环模式 ("every")
    # ==========================================
    if trigger == "every":
        # 细分场景 A：复杂的 Cron 表达式 (如 "0 9 * * *")
        if is_cron_expr(when):
            return next_cron_fire(when, tz, now_fn())
        # 细分场景 B：简单的固定时间间隔 (如 "1h"，表示每小时一次)
        interval = parse_duration(when)
        # 下一次触发点 = 当前时间 + 间隔（不考虑延迟补偿）
        return now_fn() + interval

    raise ValueError(f"未知触发类型: {trigger!r}，须为 at/after/every")


def compute_actual_trigger(
    fire_at: datetime,
    tier: str,
    tracker: LatencyTracker,
) -> datetime:
    """计算实际触发时刻。

    INSTANT: 等于 fire_at（直接推送，无 AI 延迟）。
    SOFT:    fire_at - P90（提前触发 AI，让 AI 在 fire_at 前完成处理）。

    输入:
        fire_at: 名义触发时间。
        tier: "instant" 或 "soft"。
        tracker: 延迟追踪器。

    输出:
        实际应触发的时间。
    """
    if tier == "instant":
        return fire_at
    return fire_at - timedelta(seconds=tracker.lead)


# ── ScheduledJob ─────────────────────────────────────────────────


@dataclass
class ScheduledJob:
    """定时任务数据类。"""

    # ── 核心调度规则 ──
    trigger: str      # 触发模式："at" (定点), "after" (延迟), "every" (循环)
    tier: str         # 执行模式："instant" (直接发文本), "soft" (调用AI生成)
    fire_at: datetime # 系统的下一次目标触发时间（必须是带时区的 UTC-aware 对象）
    
    # ── 目标收件人 ──
    channel: str      # 发送渠道，例如 "telegram", "cli"
    chat_id: str      # 具体的用户或群组 ID

    # ── 循环模式 (every) 专用字段 ──
    # 这两个字段互斥，如果是简单间隔就是 interval_seconds，如果是复杂历法就是 cron_expr
    interval_seconds: int | None = None
    cron_expr: str | None = None

    # ── 载荷内容 (Payload) ──
    message: str | None = None # tier="instant" 时，直接推送的固定死文本
    prompt: str | None = None  # tier="soft" 时，发给大模型让它执行的指令

    # ── 辅助与元数据 ──
    name: str | None = None    # 用户给任务起的别名（如 "喝水提醒"），方便后续按名字取消
    timezone: str = "UTC"      # 任务所属的时区，影响 cron 的计算

    # ── 运行时状态（系统自动维护） ──
    # default_factory 确保每次实例化时自动生成当前时间和全新的 UUID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_count: int = 0         # 记录这个任务已经执行了多少次
    enabled: bool = True       # 软删除/禁用开关
    id: str = field(default_factory=lambda: str(uuid.uuid4())) # 全局唯一标识符


# ── JobStore 持久化存储─────────────────────────────────────────────────────


class JobStore:
    """JSON 文件持久化，读写 ScheduledJob 列表。

    委托给 raven_agent.persistence 做原子读写——避免进程中途崩溃
    导致 schedules.json 变成半截文件，定时任务全部丢失。

    参数:
        path: JSON 持久化文件的完整路径。
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[ScheduledJob]:
        """从 JSON 文件加载已持久化的任务列表。

        输出:
            ScheduledJob 列表。文件不存在或格式损坏时返回空列表。
        """
        raw = load_json(self.path, default=[])
        if not raw:
            return []
        try:
            return [self._from_dict(d) for d in raw]
        except Exception as e:
            logger.warning("[job_store] 反序列化失败: %s", e)
            return []

    def save(self, jobs: dict[str, ScheduledJob]) -> None:
        """将当前任务字典原子写入 JSON 文件。

        输入:
            jobs: job_id → ScheduledJob 的映射。
        """
        data = [self._to_dict(j) for j in jobs.values()]
        save_json(self.path, data)

    # ── 私有转换方法 (序列化/反序列化) ──

    def _to_dict(self, job: ScheduledJob) -> dict[str, Any]:
        d = asdict(job)
        d["fire_at"] = job.fire_at.isoformat()
        d["created_at"] = job.created_at.isoformat()
        return d

    def _from_dict(self, d: dict[str, Any]) -> ScheduledJob:
        d = dict(d)
        d["fire_at"] = self._parse_dt(d["fire_at"])
        d["created_at"] = self._parse_dt(d["created_at"])
        return ScheduledJob(**d)

    @staticmethod
    def _parse_dt(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return datetime.now(timezone.utc)


# ── SchedulerService ─────────────────────────────────────────────


class SchedulerService:
    """asyncio 定时任务服务。

    - 每秒 tick 一次，检查 actual_trigger <= now 的 job
    - INSTANT: 直接 message_push
    - SOFT: 调用 agent_loop 生成内容 → 记录延迟 → push 响应
    - 持久化到 JSON，重启后自动恢复

    参数:
        store_path: JobStore JSON 文件的路径。
        push_tool: MessagePushTool 实例或兼容接口（需有 async execute 方法）。
        agent_loop_provider: 返回 agent loop 的可调用对象。
            agent loop 需有 async process_direct(content, channel, chat_id,
            session_key, omit_user_turn, skip_post_memory, disabled_tools) 方法。
        tracker: LatencyTracker 实例，默认创建新实例。
        _now_fn: 可注入的 now 函数，仅用于测试。
    """

    # 容忍度（宽限期）：如果系统宕机了，重启后发现某个一次性任务过期了。
    # 如果过期时间在 5 分钟（300秒）内，系统还会“补发”；如果过期太久，就算了。
    GRACE_SECONDS = 300  # 5分钟内的 misfire 仍执行

    def __init__(self, store_path, push_tool, agent_loop_provider=None, tracker=None, _now_fn=None):
        self.store = JobStore(store_path)
        self.push_tool = push_tool       # 发消息的工具 (如 Telegram Bot)
        
        # 为什么是 provider (函数) 而不是直接传 agent_loop 对象？
        # 解决循环依赖：启动时 Scheduler 先建好了，但 AI Agent 管线可能还没初始化完成。
        # 用函数包装一下，等真正需要触发 SOFT 任务时再去拉取真实的 Agent。
        self._agent_loop_provider = agent_loop_provider 
        
        self.tracker = tracker or LatencyTracker() # P90 延迟追踪器
        self._now = _now_fn or (lambda: datetime.now(timezone.utc))
        
        self._jobs: dict[str, ScheduledJob] = {}
        self._in_flight: set[str] = set() # 正在执行中的任务 ID 集合，防止重复触发
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ── Public API ───────────────────────────────────────────────

    async def run(self) -> None:
        """启动调度循环（后台 asyncio Task）。

        输入: 无。
        输出: None。
        """
        self.load_and_recover()
        self._running = True
        logger.info("SchedulerService 已启动")
        while self._running:
            await asyncio.sleep(1)  # 每 1 秒苏醒一次
            await self._tick()      # 检查并触发到期任务

    def start(self) -> asyncio.Task[None]:
        """在后台启动调度器并返回 Task。

        输入: 无。
        输出: 运行调度循环的 asyncio.Task。
        """
        self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self) -> None:
        """停止调度循环。

        输入: 无。
        输出: None。
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()

    def add_job(self, job: ScheduledJob) -> None:
        """注册一个定时任务并持久化。

        输入:
            job: ScheduledJob 实例。

        输出: None。
        """
        if job.fire_at.tzinfo is None:
            job.fire_at = job.fire_at.replace(tzinfo=timezone.utc)
        self._jobs[job.id] = job
        self.store.save(self._jobs)
        logger.info(
            "Job added: %s tier=%s trigger=%s fire_at=%s",
            job.id[:8], job.tier, job.trigger, job.fire_at.isoformat(),
        )

    def cancel_job(self, job_id: str) -> bool:
        """按 ID 取消任务。

        输入:
            job_id: 任务 UUID。

        输出:
            True 表示成功取消，False 表示任务不存在。
        """
        if job_id not in self._jobs:
            return False
        del self._jobs[job_id]
        self.store.save(self._jobs)
        return True

    def cancel_job_by_name(self, name: str) -> list[str]:
        """按名称取消任务（可能取消多个同名任务）。

        输入:
            name: 任务名称。

        输出:
            被取消的任务 ID 列表。
        """
        cancelled = [jid for jid, j in self._jobs.items() if j.name == name]
        for jid in cancelled:
            del self._jobs[jid]
        if cancelled:
            self.store.save(self._jobs)
        return cancelled

    def list_jobs(self) -> list[ScheduledJob]:
        """列出所有待执行任务。

        输出:
            ScheduledJob 列表。
        """
        return list(self._jobs.values())

    def load_and_recover(self) -> None:
        """启动时加载持久化 jobs，处理 misfire。

        - every 任务: 推进到下一个未来时间
        - at/after 在宽限期内: 保留并立即执行
        - at/after 超出宽限期: 丢弃
        """
        now = self._now()
        jobs = self.store.load()
        count_loaded = 0

        for job in jobs:
            if not job.enabled:
                continue

            if job.fire_at.tzinfo is None:
                job.fire_at = job.fire_at.replace(tzinfo=timezone.utc)

            # 核心判断：如果任务的目标时间在“过去”（即系统宕机期间错过了）
            if job.fire_at <= now:
                age = (now - job.fire_at).total_seconds()
                if job.trigger == "every":
                    # 循环任务错过了，直接把它的下一次触发时间拨到未来
                    job.fire_at = self._advance_every(job, now)
                    self._jobs[job.id] = job
                    count_loaded += 1
                elif age <= self.GRACE_SECONDS:
                    # 一次性任务错过了，但在 5 分钟宽限期内则保留
                    self._jobs[job.id] = job
                    count_loaded += 1
                else:
                    logger.info(
                        "Job %s (%s) expired %.0fs ago — discarded",
                        job.id[:8], (job.name or "unnamed"), age,
                    )
            else:
                self._jobs[job.id] = job
                count_loaded += 1

        logger.info("SchedulerService 恢复了 %d 个任务", count_loaded)

    # ── Internal ─────────────────────────────────────────────────

    async def _tick(self) -> None:
        """每秒执行一次：检查到期任务并触发。"""
        now = self._now()
        for job in list(self._jobs.values()):
            if not job.enabled or job.id in self._in_flight:
                continue
            actual_trigger = compute_actual_trigger(job.fire_at, job.tier, self.tracker)
            if actual_trigger <= now:
                label = job.name or job.id[:8]
                logger.info(
                    "[scheduler] 触发 %r  tier=%s  channel=%s:%s",
                    label, job.tier, job.channel, job.chat_id,
                )
                self._in_flight.add(job.id)
                asyncio.create_task(self._execute_and_reschedule(job))

    async def _execute_and_reschedule(self, job: ScheduledJob) -> None:
        """执行任务并处理重调度或移除。"""
        try:
            await self._execute(job)
            job.run_count += 1
        except Exception as e:
            logger.error("Job %s 执行失败: %s", job.id[:8], e, exc_info=True)
        finally:
            self._in_flight.discard(job.id) # 执行完毕，解除锁定
            now = self._now()
            if job.trigger == "every":
                # 【循环任务 (every)】：它还需要活下去，计算下一次的触发时间
            
                # 【神来之笔：+ 1微秒】
                # 为什么要加 1 微秒？假设这是一个 cron 任务 "0 8 * * *" (每天8点)。
                # 如果 SOFT 模式在 7:59:40 提前执行完毕，此时的 now 是 7:59:40。
                # 如果不加限制，系统计算下一个 8点，算出来的还是今天的 8点！
                # 就会导致在 8点之前，这个任务被疯狂重复触发。
                # 加了 1 微秒，强制把计算的基准点推过了名义触发时间，保证计算出的是“明天的8点”。
                reschedule_after = max(now, job.fire_at) + timedelta(microseconds=1)
                # 更新任务的下一次触发时间
                job.fire_at = self._advance_every(job, reschedule_after)
                self._jobs[job.id] = job
            else:
                # 一次性任务 (at / after)：执行完直接从内存中删掉
                # 紧接着store.save，确保持久化状态一致
                self._jobs.pop(job.id, None)
            self.store.save(self._jobs)

    async def _execute(self, job: ScheduledJob) -> None:
        """执行单个任务（INSTANT 或 SOFT）。"""
        label = job.name or job.id[:8]
        # 直接推送固定信息
        if job.tier == "instant":
            result = await self.push_tool.execute(
                channel=job.channel,
                chat_id=job.chat_id,
                message=job.message,
            )
            logger.info("[scheduler] instant 推送完成 %r: %s", label, result)
        else:
            # 调用大模型
            loop = self._get_agent_loop()
            t0 = time.monotonic()
            content = await loop.process_direct(
                content=job.prompt,
                channel=job.channel,
                chat_id=job.chat_id,
                session_key=f"scheduler:{job.id}",
                omit_user_turn=True,
                skip_post_memory=True,
                disabled_tools=["message_push"],
            )
            elapsed = time.monotonic() - t0
            self.tracker.record(elapsed)
            logger.info(
                "[scheduler] soft AI 完成 %r  耗时=%.1fs  P90=%.1fs",
                label, elapsed, self.tracker.lead,
            )
            if content:
                result = await self.push_tool.execute(
                    channel=job.channel,
                    chat_id=job.chat_id,
                    message=content,
                )
                logger.info("[scheduler] soft 推送完成 %r: %s", label, result)
            else:
                logger.warning("[scheduler] soft AI 返回空内容 %r，跳过推送", label)

    def _get_agent_loop(self) -> Any:
        """惰性获取 agent loop。"""
        if self._agent_loop_provider is None:
            raise RuntimeError("scheduler soft job requires agent_loop_provider")
        return self._agent_loop_provider()

    def _advance_every(self, job: ScheduledJob, after: datetime) -> datetime:
        """将 every job 的 fire_at 推进到 after 之后的下一个触发时间。

        输入:
            job: ScheduledJob（trigger=every）。
            after: 推进的参考时间。

        输出:
            下一个 fire_at（UTC-aware）。
        """
        if job.cron_expr:
            return next_cron_fire(job.cron_expr, job.timezone, after)
        interval = timedelta(seconds=job.interval_seconds or 3600)
        next_fire = job.fire_at + interval
        while next_fire <= after:
            next_fire += interval
        return next_fire


