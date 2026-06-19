"""
批量重新渲染所有模板样式。
遍历所有风格并生成 HTML 输出。

用法:
    cd raven-agent
    uv run python plugins/travel/batch_render.py [JSON路径]

    不传参数 → 自动搜索最新的 trip_*.json
    传一个参数 → 指定 JSON 路径
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from plugins.travel.schema import TripPlan
from plugins.travel.renderer import TripRenderer, STYLE_TEMPLATE_MAP

OUTPUT_DIR = _PROJECT_ROOT / "plugins" / "travel" / "output"
TEMPLATE_DIR = _PROJECT_ROOT / "plugins" / "travel" / "templates"

AMAP_JS_KEY = os.getenv("AMAP_JS_KEY", "")
AMAP_SECURITY_CODE = os.getenv("AMAP_SECURITY_CODE", "")
AMAP_WEB_SERVICE_KEY = os.getenv("AMAP_WEB_SERVICE_KEY", "")
RENDER_MODE = "static"

JSON_SEARCH_DIRS = [
    _PROJECT_ROOT / "peerAgent" / "travel-planner" / "outputs",
    OUTPUT_DIR,
]


def find_latest_json() -> Path | None:
    candidates: list[Path] = []
    for d in JSON_SEARCH_DIRS:
        if d.exists():
            candidates.extend(d.glob("trip_*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    # 找 JSON
    if len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
    else:
        json_path = find_latest_json()

    if json_path is None:
        print("❌ 找不到 trip_*.json")
        sys.exit(1)

    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        sys.exit(1)

    print(f"📄 JSON: {json_path}")

    # 加载并校验
    raw = json_path.read_text(encoding="utf-8")
    try:
        plan = TripPlan.model_validate_json(raw)
    except Exception as exc:
        print(f"❌ Pydantic 校验失败:\n{exc}")
        sys.exit(1)

    print(f"✅ 数据校验通过: {plan.city} {plan.total_days}天\n")

    # 渲染器
    renderer = TripRenderer(
        template_dir=TEMPLATE_DIR,
        output_dir=OUTPUT_DIR,
        amap_js_key=AMAP_JS_KEY,
        amap_security_code=AMAP_SECURITY_CODE,
        render_mode=RENDER_MODE,
        amap_web_service_key=AMAP_WEB_SERVICE_KEY,
    )

    styles = sorted(STYLE_TEMPLATE_MAP.keys())
    total = len(styles)
    ok = 0
    fail = 0

    for i, style in enumerate(styles, 1):
        print(f"[{i:2d}/{total}] {style:30s} ... ", end="", flush=True)
        try:
            # 为每个风格重置状态，确保 TripPlan 不保留上次渲染的数据
            plan.trip_id = ""
            plan.generated_at = ""
            plan.style = ""
            path = renderer.render(plan, trip_id=f"test_{style}", style=style)
            size_kb = path.stat().st_size / 1024
            print(f"✅ {size_kb:5.0f} KB")
            ok += 1
        except Exception as exc:
            print(f"❌ {exc}")
            fail += 1
        # 避免高德静态地图 API 限流，渲染间隔 1 秒
        if i < total:
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"完成: {ok} 成功, {fail} 失败, 共 {total} 个风格")


if __name__ == "__main__":
    main()
