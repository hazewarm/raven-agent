"""Tests for ProactiveLoop: adaptive tick scheduling, context file, lifecycle."""

import asyncio
import random as _random_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raven_agent.proactive.loop import ProactiveLoop
from raven_agent.proactive.presence import PresenceStore


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_presence() -> MagicMock:
    """返回 mock PresenceStore。"""
    m = MagicMock(spec=PresenceStore)
    m.get_last_user_at.return_value = None
    m.get_last_proactive_at.return_value = None
    m.most_recent_user_at.return_value = None
    return m


@pytest.fixture
def loop(mock_presence: MagicMock, tmp_path: Path) -> ProactiveLoop:
    """创建测试用 ProactiveLoop（短 tick 间隔 + 确定性 RNG）。"""
    cfg = SimpleNamespace(
        proactive_max_steps=50,
        quiet_hours_start=0,
        quiet_hours_end=8,
        quiet_hours_drift=True,
        drift_enabled=False,
    )
    return ProactiveLoop(
        presence=mock_presence,
        target_session_key="tg:test123",
        workspace_root=tmp_path,
        tick_s0=1,
        tick_s1=1,
        tick_s2=1,
        tick_s3=1,
        tick_jitter=0.0,      # 关闭抖动，便于断言
        rng=_random_module.Random(42),
        cfg=cfg,
    )


# ── Context file ────────────────────────────────────────────────────


def test_creates_context_file_on_init(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """初始化时自动创建 PROACTIVE_CONTEXT.md。"""
    loop = ProactiveLoop(
        presence=mock_presence,
        target_session_key="tg:test",
        workspace_root=tmp_path,
    )
    context_path = tmp_path / "PROACTIVE_CONTEXT.md"
    assert context_path.exists()
    content = context_path.read_text(encoding="utf-8")
    assert "Proactive Context" in content


def test_context_file_idempotent(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """重复初始化不会覆盖已有内容。"""
    context_path = tmp_path / "PROACTIVE_CONTEXT.md"
    tmp_path.mkdir(parents=True, exist_ok=True)
    context_path.write_text("custom rules", encoding="utf-8")

    loop = ProactiveLoop(
        presence=mock_presence,
        target_session_key="tg:test",
        workspace_root=tmp_path,
    )
    assert context_path.read_text(encoding="utf-8") == "custom rules"


# ── _next_interval ──────────────────────────────────────────────────


def test_next_interval_no_base_score_uses_energy_fallback(
    loop: ProactiveLoop, mock_presence: MagicMock
) -> None:
    """首次启动（base_score=None）时用 energy 估算初始间隔。"""
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now - timedelta(minutes=10)

    interval = loop._next_interval(None)
    # base_score ≈ d_energy(energy) * 0.40，energy ≈ 0.72（10 分钟前）
    # d_energy ≈ 0.28，base_score ≈ 0.11 → ≤ 0.20 → tick_s0 = 1
    assert interval >= 1


def test_next_interval_high_score_short(
    loop: ProactiveLoop, mock_presence: MagicMock
) -> None:
    """高 base_score 返回较短间隔。"""
    # 临时改 tick 间隔以区分
    loop._tick_s3 = 5
    loop._tick_s0 = 100
    assert loop._next_interval(0.85) == 5
    assert loop._next_interval(0.05) == 100


def test_next_interval_no_presence_fallback(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """无 presence 时回退到固定间隔。"""
    loop = ProactiveLoop(
        presence=None,
        target_session_key="tg:test",
        workspace_root=tmp_path,
        interval_seconds=900,
    )
    assert loop._next_interval(0.85) == 900


# ── Lifecycle ───────────────────────────────────────────────────────


async def test_start_stop_lifecycle(
    loop: ProactiveLoop, mock_presence: MagicMock
) -> None:
    """start → 短暂运行 → stop 不报错。"""
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now

    loop.start()
    # 等待足够时间让至少一次 tick 执行
    await asyncio.sleep(0.2)
    loop.stop()
    # 等待 loop 退出
    await asyncio.sleep(0.2)

    # tick 应该被调用了（至少一次日志输出）
    # 不好直接断言日志，但可以验证 get_last_user_at 被调用了
    # （tick 中会读取 presence）
    assert mock_presence.get_last_user_at.call_count >= 1


async def test_tick_executes_and_returns_score(
    loop: ProactiveLoop, mock_presence: MagicMock
) -> None:
    """_tick() 执行后返回 base_score。"""
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now

    score = await loop._tick()
    assert score is not None
    assert 0.0 <= score <= 1.0


async def test_tick_no_presence_returns_none(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """无 presence 时 _tick 返回 None。"""
    loop = ProactiveLoop(
        presence=None,
        target_session_key="tg:test",
        workspace_root=tmp_path,
    )
    score = await loop._tick()
    assert score is None


# ── Adaptive rhythm ─────────────────────────────────────────────────


async def test_frequent_ticks_when_energy_low(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """电量低（很久没说话）→ tick 间隔短。

    骨架阶段只有 D_energy 有效（D_content / D_recent 为 0），
    所以设 w_e=1.0 让 D_energy 单独驱动 base_score 跨越所有档位。
    实际运行中 w_e=0.40，其余来自 Sensor 输入——这里只验证方向性。
    """
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now - timedelta(hours=48)

    loop = ProactiveLoop(
        presence=mock_presence,
        target_session_key="tg:test",
        workspace_root=tmp_path,
        tick_s0=100,   # ≤ 0.20
        tick_s1=200,   # > 0.20
        tick_s2=300,   # > 0.40
        tick_s3=5,     # > 0.70
        tick_jitter=0.0,
        w_e=1.0,       # 让 D_energy 单独驱动，跨越所有档位
        rng=_random_module.Random(42),
    )
    # energy ≈ 0.055 (48h) → d_energy ≈ 0.945 → base_score = 0.945 > 0.70 → tick_s3
    interval = loop._next_interval(None)
    assert interval == 5


async def test_infrequent_ticks_when_energy_high(
    mock_presence: MagicMock, tmp_path: Path
) -> None:
    """电量高（刚聊完）→ tick 间隔长。"""
    now = datetime.now(timezone.utc)
    mock_presence.get_last_user_at.return_value = now - timedelta(seconds=10)

    loop = ProactiveLoop(
        presence=mock_presence,
        target_session_key="tg:test",
        workspace_root=tmp_path,
        tick_s0=100,   # ≤ 0.20
        tick_s1=200,
        tick_s2=300,
        tick_s3=5,     # > 0.70
        tick_jitter=0.0,
        w_e=1.0,
        rng=_random_module.Random(42),
    )
    # energy ≈ 0.997 (10s) → d_energy ≈ 0.003 → base_score = 0.003 ≤ 0.20 → tick_s0
    interval = loop._next_interval(None)
    assert interval == 100