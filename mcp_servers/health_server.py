import json
import requests
import tomllib  # Python 3.12 原生支持，零额外依赖
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 1. 初始化并读取 config.toml
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.toml"
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"找不到配置文件: {CONFIG_FILE.resolve()}")

with open(CONFIG_FILE, "rb") as f:
    config = tomllib.load(f)

# 提取 URL 和 代理配置
WORKER_URL = config.get("cloudflare", {}).get("worker_url")
if not WORKER_URL:
    raise ValueError("配置文件 config.toml 中缺少 [cloudflare] worker_url 设置！")

# 读取代理配置，组装给 requests 使用
PROXY_URL = config.get("network", {}).get("proxy_url")
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None


# 2. 初始化 MCP 和指标映射表
mcp = FastMCP("AppleHealthContext")

NAME_MAPPING = {
    "step_count": "今日步数",
    "resting_heart_rate": "静息心率(次/分)",
    "heart_rate": "全天心率情况",
    "sleep_analysis": "昨夜睡眠质量",
    "heart_rate_variability": "心率变异性HRV(毫秒)",
    "environmental_audio_exposure": "环境噪音暴露(分贝)",
    "apple_exercise_time": "运动时间(分钟)",
    "time_in_daylight": "日照时间(分钟)",
}

def _fetch_metrics() -> list[dict]:
    """从 Cloudflare Worker 拉取原始指标列表，失败返回空列表。"""
    try:
        response = requests.get(WORKER_URL, timeout=10, proxies=PROXIES)
        response.raise_for_status()
        return response.json().get("data", {}).get("metrics", [])
    except Exception:
        return []


def _get_latest(metrics: list[dict], name: str) -> dict | None:
    """从指标列表中取指定 name 的最新数据点。"""
    for item in metrics:
        if item.get("name") == name:
            data_points = item.get("data", [])
            if data_points:
                return data_points[0]
    return None


@mcp.tool()
def get_health_context() -> str:
    """
    获取用户当天最新的身体状态和健康快照。
    当用户询问以下信息时，必须调用此工具：
    1. 今天的运动量或步数。
    2. 昨晚的睡眠质量（包括总睡眠时长、深睡、快速眼动）。
    3. 心率情况（静息心率、全天平均心率、心率变异性 HRV）。
    4. 当前身处的环境噪音暴露度、日照时长。
    """
    metrics = _fetch_metrics()
    if not metrics:
        return "云端信箱暂无健康数据。"

    summary = []

    for item in metrics:
        raw_name = item.get("name", "未知指标")
        cn_name = NAME_MAPPING.get(raw_name, raw_name)

        data_points = item.get("data", [])
        if not data_points:
            continue

        latest_data = data_points[0]

        if raw_name == "sleep_analysis":
            total = round(latest_data.get("totalSleep", 0), 1)
            deep = round(latest_data.get("deep", 0), 1)
            rem = round(latest_data.get("rem", 0), 1)
            core = round(latest_data.get("core", 0), 1)
            summary.append(f"- {cn_name}: 总计 {total} 小时 (深度睡眠 {deep}h, 核心睡眠 {core}h, 快速动眼睡眠 {rem}h)")

        elif "Avg" in latest_data:
            avg = int(float(latest_data["Avg"]))
            max_hr = int(float(latest_data["Max"]))
            min_hr = int(float(latest_data["Min"]))
            summary.append(f"- {cn_name}: 平均 {avg} (范围: {min_hr}-{max_hr})")

        elif "qty" in latest_data:
            qty = float(latest_data["qty"])
            if raw_name in ["step_count", "resting_heart_rate", "apple_exercise_time", "time_in_daylight"]:
                val = int(qty)
            else:
                val = round(qty, 1)
            summary.append(f"- {cn_name}: {val}")

    if not summary:
        return "云端信箱暂无健康数据。"

    final_context = "主人的最新身体状态数据：\n" + "\n".join(summary)
    return final_context


# ═══════════════════════════════════════════════════════════════════════
# Alert 工具（供 Proactive 主动通道使用）
# ═══════════════════════════════════════════════════════════════════════

# ── 基线配置 ──

RHR_BASELINE = 62                   # 个人静息心率基线（次/分），按实际长期数据调整
RHR_ELEVATED_PCT = 0.15             # 超基线 15% → medium
RHR_HIGH_PCT = 0.30                 # 超基线 30% → high
SLEEP_SHORT_THRESHOLD_HOURS = 6     # 睡眠 < 6 小时提醒
DAYLIGHT_HIGH_THRESHOLD_MIN = 30    # 日照 > 30 分钟视为暴晒过度

# ── get_proactive_events ──────────────────────────────────────────────

@mcp.tool()
def get_proactive_events() -> str:
    """
    返回健康告警事件列表，供 Proactive 主动推送使用。

    同时检查静息心率、睡眠时长、日照时长三个维度，
    将异常结果以 proactive alert 格式返回。
    无异常时返回空列表。
    """
    metrics = _fetch_metrics()
    if not metrics:
        return "[]"

    events: list[dict] = []

    hr_alert = _check_resting_heart_rate(metrics)
    if hr_alert:
        events.append(hr_alert)

    sleep_alert = _check_sleep_duration(metrics)
    if sleep_alert:
        events.append(sleep_alert)

    daylight_alert = _check_daylight(metrics)
    if daylight_alert:
        events.append(daylight_alert)

    return json.dumps(events, ensure_ascii=False)


# ── 单项检查 ──────────────────────────────────────────────────────────

def _check_resting_heart_rate(metrics: list[dict]) -> dict | None:
    """静息心率异常检测——以个人基线为基准，用相对阈值判断。"""
    latest = _get_latest(metrics, "resting_heart_rate")
    if latest is None:
        return None

    rhr = int(float(latest.get("qty", 0)))
    elevated_line = int(RHR_BASELINE * (1 + RHR_ELEVATED_PCT))
    high_line = int(RHR_BASELINE * (1 + RHR_HIGH_PCT))

    if rhr <= elevated_line:
        return None

    if rhr > high_line:
        severity = "high"
        tone = "关切但不过度紧张"
        title = "静息心率异常偏高"
    else:
        severity = "medium"
        tone = "关切并温和提醒"
        title = "静息心率偏高"

    return {
        "kind": "alert",
        "event_id": f"health_rhr_{rhr}",
        "source_type": "health_event",
        "source_name": "Apple Health",
        "title": title,
        "content": (
            f"今日静息心率 {rhr} 次/分，高于个人基线 {RHR_BASELINE} 次/分的"
            f"{'30%' if severity == 'high' else '15%'}以上。"
            f"建议留意是否有压力过大、睡眠不足或咖啡因摄入过多的情况。"
        ),
        "severity": severity,
        "suggested_tone": tone,
        "metrics": {
            "current_bpm": rhr,
            "baseline_bpm": RHR_BASELINE,
            "elevated_line_bpm": elevated_line,
            "high_line_bpm": high_line,
        },
    }


def _check_sleep_duration(metrics: list[dict]) -> dict | None:
    """睡眠时长不足检测。"""
    latest = _get_latest(metrics, "sleep_analysis")
    if latest is None:
        return None

    total = latest.get("totalSleep", 0)
    if total >= SLEEP_SHORT_THRESHOLD_HOURS:
        return None

    deep = round(latest.get("deep", 0), 1)
    rem = round(latest.get("rem", 0), 1)
    core = round(latest.get("core", 0), 1)

    return {
        "kind": "alert",
        "event_id": f"health_sleep_{int(total * 10)}",
        "source_type": "health_event",
        "source_name": "Apple Health",
        "title": "睡眠时长不足",
        "content": (
            f"昨晚仅睡 {total} 小时（深度 {deep}h + 核心 {core}h + REM {rem}h），"
            f"低于建议的 {SLEEP_SHORT_THRESHOLD_HOURS} 小时底线。"
            f"长期睡眠不足会影响免疫力、情绪和认知表现。今天尽量早些休息。"
        ),
        "severity": "medium",
        "suggested_tone": "关切但温和，不要制造焦虑",
        "metrics": {
            "total_hours": total,
            "deep_hours": deep,
            "rem_hours": rem,
            "core_hours": core,
            "threshold_hours": SLEEP_SHORT_THRESHOLD_HOURS,
        },
    }


def _check_daylight(metrics: list[dict]) -> dict | None:
    """日照过量检测——户外暴晒过久才提醒。"""
    latest = _get_latest(metrics, "time_in_daylight")
    if latest is None:
        return None

    minutes = int(float(latest.get("qty", 0)))
    if minutes <= DAYLIGHT_HIGH_THRESHOLD_MIN:
        return None

    return {
        "kind": "alert",
        "event_id": f"health_daylight_{minutes}",
        "source_type": "health_event",
        "source_name": "Apple Health",
        "title": "户外暴晒时间较长",
        "content": (
            f"今日户外日照时间已达 {minutes} 分钟，超过 {DAYLIGHT_HIGH_THRESHOLD_MIN} 分钟。"
            f"长时间暴晒可能加速皮肤老化、增加晒伤风险。建议适当遮阳、补涂防晒。"
        ),
        "severity": "low",
        "suggested_tone": "轻松提醒，像朋友关心",
        "metrics": {"daylight_minutes": minutes, "threshold_min": DAYLIGHT_HIGH_THRESHOLD_MIN},
    }


if __name__ == "__main__":
    mcp.run()