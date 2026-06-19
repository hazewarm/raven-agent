"""Tests for proactive energy model: compute_energy, d_*, composite_score, next_tick."""

import math
import random as _random_module
from datetime import datetime, timedelta, timezone

import pytest

from raven_agent.proactive.energy import (
    composite_score,
    compute_energy,
    d_content,
    d_energy,
    d_recent,
    next_tick_from_score,
    random_weight,
)


# ── compute_energy ─────────────────────────────────────────────────


def test_energy_none_returns_zero() -> None:
    """从未收到用户消息时电量为 0。"""
    assert compute_energy(None) == 0.0


def test_energy_just_now_near_one() -> None:
    """刚收到消息时电量接近 1。"""
    now = datetime.now(timezone.utc)
    e = compute_energy(now, now)
    assert e > 0.95  # alpha+beta+gamma = 1.0，t=0 时恰好为 1


def test_energy_decays_over_time() -> None:
    """电量随时间单调递减。"""
    now = datetime.now(timezone.utc)
    e1 = compute_energy(now - timedelta(minutes=1), now)
    e2 = compute_energy(now - timedelta(minutes=30), now)
    e3 = compute_energy(now - timedelta(minutes=240), now)

    assert e1 > e2 > e3
    assert e1 > 0.9   # 1 min: 几乎满电
    assert e2 < 0.7   # 30 min: τ₁ 已经衰减过半
    assert e3 < 0.3   # 240 min: τ₁ + τ₂ 都大幅衰减


def test_energy_after_two_days_near_zero() -> None:
    """48 小时后电量接近 0（仅 τ₃ 长尾残留约 0.055）。"""
    now = datetime.now(timezone.utc)
    e = compute_energy(now - timedelta(minutes=2880), now)
    assert e < 0.06


def test_energy_explicit_now() -> None:
    """显式传入 now 参数正确计算。"""
    last = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    now = datetime(2025, 6, 1, 12, 30, 0, tzinfo=timezone.utc)
    e = compute_energy(last, now)
    expected = 0.50 * math.exp(-30 / 30) + 0.35 * math.exp(-30 / 240) + 0.15 * math.exp(-30 / 2880)
    assert abs(e - expected) < 0.001


# ── d_energy ────────────────────────────────────────────────────────


def test_d_energy_inverse() -> None:
    """d_energy = 1 - energy。"""
    assert d_energy(1.0) == 0.0
    assert d_energy(0.0) == 1.0
    assert d_energy(0.3) == pytest.approx(0.7)


def test_d_energy_clamped() -> None:
    """输入超出 [0, 1] 时被裁剪。"""
    assert d_energy(1.5) == 0.0
    assert d_energy(-0.5) == 1.0


# ── d_content ───────────────────────────────────────────────────────


def test_d_content_zero_items() -> None:
    """0 条新内容时新鲜度为 0。"""
    assert d_content(0) == 0.0
    assert d_content(-1) == 0.0


def test_d_content_saturation() -> None:
    """条数增多时趋于 1。"""
    assert d_content(1, halfsat=3.0) < 0.3
    assert d_content(10, halfsat=3.0) > 0.9
    assert d_content(3, halfsat=3.0) == pytest.approx(1 - math.exp(-1), rel=0.01)


# ── d_recent ────────────────────────────────────────────────────────


def test_d_recent_zero_messages() -> None:
    """0 条消息时语境丰富度为 0。"""
    assert d_recent(0) == 0.0


def test_d_recent_log_scale() -> None:
    """对数尺度下渐进增长。"""
    assert d_recent(5, scale=10.0) > 0.5
    assert d_recent(20, scale=10.0) > 0.9
    assert d_recent(100, scale=10.0) <= 1.0  # 上限 1.0


# ── composite_score ────────────────────────────────────────────────


def test_composite_score_weighted() -> None:
    """加权合成正确。"""
    score = composite_score(1.0, 0.5, 0.0, w_e=0.5, w_c=0.3, w_r=0.2)
    expected = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.0
    assert score == pytest.approx(expected)


def test_composite_score_clamped() -> None:
    """结果裁剪到 [0, 1]。"""
    assert composite_score(2.0, 2.0, 2.0, w_e=1.0, w_c=1.0, w_r=1.0) == 1.0
    assert composite_score(-1.0, -1.0, -1.0) == 0.0


# ── next_tick_from_score ───────────────────────────────────────────


def test_high_score_short_interval() -> None:
    """高 base_score → 短间隔。"""
    rng = _random_module.Random(42)
    interval = next_tick_from_score(0.85, rng=rng, tick_jitter=0.0)
    assert interval == 420  # tick_s3


def test_mid_score_mid_interval() -> None:
    """中等 base_score → 中等间隔。"""
    rng = _random_module.Random(42)
    interval = next_tick_from_score(0.50, rng=rng, tick_jitter=0.0)
    assert interval == 1080  # tick_s2


def test_low_score_long_interval() -> None:
    """低 base_score → 长间隔。"""
    rng = _random_module.Random(42)
    interval = next_tick_from_score(0.10, rng=rng, tick_jitter=0.0)
    assert interval == 4800  # tick_s0


def test_boundary_scores() -> None:
    """边界值正确映射到对应档位。"""
    rng = _random_module.Random(42)
    assert next_tick_from_score(0.71, rng=rng, tick_jitter=0.0) == 420   # > 0.70 → s3
    assert next_tick_from_score(0.70, rng=rng, tick_jitter=0.0) == 1080  # ≤ 0.70 → s2
    assert next_tick_from_score(0.41, rng=rng, tick_jitter=0.0) == 1080  # > 0.40 → s2
    assert next_tick_from_score(0.40, rng=rng, tick_jitter=0.0) == 2400  # ≤ 0.40 → s1
    assert next_tick_from_score(0.21, rng=rng, tick_jitter=0.0) == 2400  # > 0.20 → s1
    assert next_tick_from_score(0.20, rng=rng, tick_jitter=0.0) == 4800  # ≤ 0.20 → s0


def test_jitter_adds_variation() -> None:
    """随机抖动产生变化。"""
    rng1 = _random_module.Random(1)
    rng2 = _random_module.Random(2)
    i1 = next_tick_from_score(0.50, rng=rng1)
    i2 = next_tick_from_score(0.50, rng=rng2)
    # 由于随机种子不同，间隔大概率不同
    assert i1 != i2


def test_jitter_zero_no_variation() -> None:
    """关闭抖动时返回精确值。"""
    rng = _random_module.Random(42)
    interval = next_tick_from_score(0.50, rng=rng, tick_jitter=0.0)
    assert interval == 1080


def test_jitter_range() -> None:
    """抖动在 ±tick_jitter 范围内。"""
    rng = _random_module.Random(42)
    # 跑 100 次，验证都在合理范围内
    for _ in range(100):
        interval = next_tick_from_score(0.50, rng=rng, tick_jitter=0.3)
        assert 1080 * 0.7 <= interval <= 1080 * 1.3


# ── random_weight ──────────────────────────────────────────────────


def test_random_weight_range() -> None:
    """random_weight 始终在 [0.5, 1.5] 范围内。"""
    rng = _random_module.Random(42)
    for _ in range(100):
        w = random_weight(rng)
        assert 0.5 <= w <= 1.5


def test_random_weight_mean_near_one() -> None:
    """大量采样均值接近 1.0。"""
    rng = _random_module.Random(42)
    samples = [random_weight(rng) for _ in range(1000)]
    mean = sum(samples) / len(samples)
    assert 0.95 <= mean <= 1.05