"""
测试 A: 单模板渲染验证

用指定的 TripPlan JSON + 指定风格，走一遍 Pydantic 校验 + Jinja2 渲染。

用法:
    cd raven-agent
    uv run python plugins/travel/test_render.py [风格] [JSON路径]

    不传参数 → 默认 art_deco + 最近生成的 trip_*.json
    传一个参数 → 指定风格
    传两个参数 → 指定风格 + 指定 JSON 路径

示例:
    uv run python plugins/travel/test_render.py
    uv run python plugins/travel/test_render.py cyberpunk
    uv run python plugins/travel/test_render.py bauhaus data/trip_xxx.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from plugins.travel.schema import TripPlan
from plugins.travel.renderer import TripRenderer, DEFAULT_STYLE, STYLE_TEMPLATE_MAP

OUTPUT_DIR = _PROJECT_ROOT / "plugins" / "travel" / "output"
TEMPLATE_DIR = _PROJECT_ROOT / "plugins" / "travel" / "templates"

# 默认 JSON 搜索路径
DEFAULT_JSON_DIRS = [
    _PROJECT_ROOT / "peerAgent" / "travel-planner" / "outputs",
    OUTPUT_DIR,
]

AMAP_JS_KEY = os.getenv("AMAP_JS_KEY", "")
AMAP_SECURITY_CODE = os.getenv("AMAP_SECURITY_CODE", "")
AMAP_WEB_SERVICE_KEY = os.getenv("AMAP_WEB_SERVICE_KEY", "")
# 渲染模式：dynamic 或 static。dynamic 会生成包含 JS 的交互式 HTML，static 则生成纯静态 HTML。
RENDER_MODE = "static"


def _find_latest_json() -> Path | None:
    """在默认目录中搜索最新的 trip_*.json。"""
    candidates: list[Path] = []
    for d in DEFAULT_JSON_DIRS:
        if d.exists():
            candidates.extend(d.glob("trip_*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    # ── 解析参数 ──
    style = sys.argv[1] if len(sys.argv) >= 2 else "art_deco"
    json_path = None
    if len(sys.argv) >= 3:
        json_path = Path(sys.argv[2])
    else:
        found = _find_latest_json()
        if found:
            json_path = found

    if style not in STYLE_TEMPLATE_MAP and style != "art_deco":
        print(f"❌ 未知风格: {style}")
        print(f"   可用: {', '.join(sorted(STYLE_TEMPLATE_MAP.keys()))}")
        sys.exit(1)

    if json_path is None:
        print("❌ 找不到 JSON 文件")
        print(f"   请指定路径或确保以下目录有 trip_*.json:")
        for d in DEFAULT_JSON_DIRS:
            print(f"     {d}")
        sys.exit(1)
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        sys.exit(1)

    print(f"📄 JSON: {json_path}")
    print(f"🎨 风格: {style}")

    # ── 1. 读 JSON + Pydantic 校验 ──
    raw = json_path.read_text(encoding="utf-8")
    print(f"✅ 读取 JSON ({len(raw)} 字符)")
    try:
        plan = TripPlan.model_validate_json(raw)
    except Exception as exc:
        print(f"❌ Pydantic 校验失败:\n{exc}")
        sys.exit(1)

    print(f"✅ Pydantic 校验通过")
    print(f"   城市: {plan.city} | {plan.total_days}天 | {plan.start_date} ~ {plan.end_date}")
    for d in plan.days:
        print(f"   Day{d.day}: {d.theme} ({len(d.activities)} activités)")
    print(f"   预算: {plan.budget.currency}{plan.budget.total:.0f}")
    print(f"   小红书: {len(plan.xhs_notes)} 篇 | 行李: {len(plan.packing_list)} 项 | 建议: {len(plan.suggestions)} 条")

    verif_counts: dict[str, int] = {}
    for day in plan.days:
        for act in day.activities:
            if act.location:
                v = act.location.verification
                verif_counts[v] = verif_counts.get(v, 0) + 1
    print(f"   verification: {verif_counts}")

    # ── 2. 渲染 ──
    if not TEMPLATE_DIR.exists():
        print(f"❌ 模板目录不存在: {TEMPLATE_DIR}")
        sys.exit(1)

    renderer = TripRenderer(
        template_dir=TEMPLATE_DIR,
        output_dir=OUTPUT_DIR,
        amap_js_key=AMAP_JS_KEY,
        amap_security_code=AMAP_SECURITY_CODE,
        render_mode=RENDER_MODE,
        amap_web_service_key=AMAP_WEB_SERVICE_KEY,
    )

    try:
        path = renderer.render(plan, trip_id=f"test_{style}", style=style)
    except Exception as exc:
        print(f"❌ 渲染失败: {exc}")
        sys.exit(1)

    size = path.stat().st_size
    print(f"\n✅ HTML 已生成")
    print(f"   文件: {path}")
    print(f"   大小: {size} bytes ({size/1024:.1f} KB)")

    # ── 3. 完整性检查 ──
    html = path.read_text(encoding="utf-8")
    checks = [
        ("<!DOCTYPE html>", "HTML5 doctype"),
        ("<title>", "title 标签"),
        ("_AMapSecurityConfig", "安全密钥"),
        ("verification-badge", "verification CSS"),
        ("fontawesome-free", "Font Awesome CDN"),
        ("tailwindcss", "Tailwind CSS CDN"),
        ("</html>", "闭合标签"),
    ]
    print(f"\n📋 完整性: {' '.join('✅' + l for k, l in checks if k in html)}")
    for keyword, label in checks:
        if keyword not in html:
            print(f"   ⚠️ {label} 缺失")


if __name__ == "__main__":
    main()
