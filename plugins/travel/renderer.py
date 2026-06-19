"""
plugins/travel/renderer.py —— Jinja2 旅行规划 HTML 渲染引擎。

职责:
  1. 加载与缓存 Jinja2 模板
  2. 根据用户选择的风格（或默认）选取对应模板
  3. 将 TripPlan 对象渲染为完整的单文件 HTML
  4. 注入高德地图 JS API Key 用于交互地图
  5. 输出到 plugins/travel/output/ 目录

设计原则:
  - 模板继承: base.html 定义结构骨架，29 个风格模板只覆盖 CSS 变量和装饰
  - 单文件输出: CSS/JS 全部内联，无外部依赖（除 CDN 的 Font Awesome/Tailwind/高德 JS API）
  - 类型安全: 通过 Pydantic 模型访问数据，避免 Jinja2 中拼写错误

使用方式:
  >>> from pathlib import Path
  >>> from schema import TripPlan
  >>> from renderer import TripRenderer
  >>>
  >>> renderer = TripRenderer(
  ...     template_dir=Path("plugins/travel/templates"),
  ...     output_dir=Path("plugins/travel/output"),
  ...     amap_js_key="your-amap-js-api-key"
  ... )
  >>> html_path = renderer.render(plan, trip_id="abc123", style="art_deco")
  >>> print(html_path)
  Path("plugins/travel/output/trip_abc123.html")
"""

from __future__ import annotations

import base64
import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from .schema import TripPlan

logger = logging.getLogger(__name__)

# 风格名称 → 模板文件名的映射表
# key:   用户可选的风格 slug（传入 renderer.render(style=...)）
# value: 对应的 Jinja2 模板文件名
STYLE_TEMPLATE_MAP: dict[str, str] = {
    "minimalist":              "style__minimalist.html",
    "bold_modern":             "style__bold_modern.html",
    "elegant_vintage":         "style__elegant_vintage.html",
    "futuristic_tech":         "style__futuristic_tech.html",
    "scandinavian":            "style__scandinavian.html",
    "art_deco":                "style__art_deco.html",
    "japanese_minimalism":     "style__japanese_minimalism.html",
    "postmodern_deconstruct":  "style__postmodern_deconstruct.html",
    "punk":                    "style__punk.html",
    "british_rock":            "style__british_rock.html",
    "black_metal":             "style__black_metal.html",
    "memphis":                 "style__memphis.html",
    "cyberpunk":               "style__cyberpunk.html",
    "pop_art":                 "style__pop_art.html",
    "deconstructed_swiss":     "style__deconstructed_swiss.html",
    "vaporwave":               "style__vaporwave.html",
    "neo_expressionism":       "style__neo_expressionism.html",
    "extreme_minimalism":      "style__extreme_minimalism.html",
    "neo_futurism":            "style__neo_futurism.html",
    "surrealist_collage":      "style__surrealist_collage.html",
    "neo_baroque":             "style__neo_baroque.html",
    "liquid_digital":          "style__liquid_digital.html",
    "hypersensory_minimalism": "style__hypersensory_minimalism.html",
    "neo_expressionist_data":  "style__neo_expressionist_data.html",
    "victorian":               "style__victorian.html",
    "bauhaus":                 "style__bauhaus.html",
    "constructivism":          "style__constructivism.html",
    "memphis_design":          "style__memphis_design.html",
    "german_expressionism":    "style__german_expressionism.html",
}

# 默认风格（当用户未指定或不存在的风格时使用）
DEFAULT_STYLE = "art_deco"


def _make_static_markers(coords: list[str]) -> str:
    """将坐标列表转为高德静态地图 markers 参数。

    每个坐标生成一个标记点，用数字 1, 2, 3, ... 作为 label。
    标记尺寸 large、颜色 0x1890ff（与动态地图蓝色一致）。

    输入:
        coords: ["lng,lat", ...] 格式的坐标字符串列表。

    输出:
        高德静态地图 markers 参数值，例如 "large,0x1890ff,1:116.39,39.91|large,0x1890ff,2:116.40,39.92"。
    """
    markers = []
    for i, coord in enumerate(coords):
        label = str(i + 1)  # 1, 2, 3, ... 与动态地图的彩色圆圈序号一致
        markers.append(f"large,0x1890ff,{label}:{coord}")
    return "|".join(markers)


def _map_center(coords: list[str]) -> str:
    """计算坐标列表的中心点（centroid）。

    用于高德静态地图的 center 参数，确保一天内所有景点都在视野范围内。

    输入:
        coords: ["lng,lat", ...] 格式的坐标字符串列表。

    输出:
        中心点坐标，例如 "116.40,39.93"。
    """
    if not coords:
        return "116.3972,39.9163"  # fallback: 天安门
    lngs = []
    lats = []
    for coord in coords:
        parts = coord.split(",")
        if len(parts) == 2:
            lngs.append(float(parts[0]))
            lats.append(float(parts[1]))
    if not lngs:
        return "116.3972,39.9163"
    center_lng = sum(lngs) / len(lngs)
    center_lat = sum(lats) / len(lats)
    return f"{center_lng:.6f},{center_lat:.6f}"


def _map_zoom(coords: list[str]) -> int:
    """计算包含所有坐标（带边距）的最佳缩放级别。

    基于高德静态地图的 Mercator 投影，根据坐标跨度和图片尺寸
    （800×500 像素）反算 zoom。zoom 越小视野越广。

    输入:
        coords: ["lng,lat", ...] 格式的坐标字符串列表。

    输出:
        整数 zoom 级别（8-17）。

    """
    if not coords or len(coords) < 2:
        return 15  # 单个点默认放大

    lngs = []
    lats = []
    for coord in coords:
        parts = coord.split(",")
        if len(parts) == 2:
            lngs.append(float(parts[0]))
            lats.append(float(parts[1]))

    if len(lngs) < 2:
        return 15

    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)

    # 边距系数：带 40% 容差确保标记点在图片内
    padding = 1.4
    span_lng = (max_lng - min_lng) * padding or 0.002
    span_lat = (max_lat - min_lat) * padding or 0.002

    # 图片尺寸（与模板中 size 参数一致）
    MAP_W, MAP_H = 800, 500

    # Mercator 投影：zoom = log2(360 * W / (256 * span_lng))
    zoom_lng = math.log2(360 * MAP_W / (256 * span_lng))
    zoom_lat = math.log2(180 * MAP_H / (256 * span_lat))

    # 取较小的 zoom（视野更广），确保两个方向都容下所有点
    zoom = int(min(zoom_lng, zoom_lat))
    return max(8, min(zoom, 17))


def _build_static_map_url(coords: list[str], key: str, size: str = "800*500") -> str:
    """构造高德静态地图 API URL。

    复用 _map_zoom / _map_center / _make_static_markers。

    输入:
        coords: ["lng,lat", ...] 格式的坐标字符串列表。
        key: 高德 Web 服务 Key。
        size: 图片尺寸，默认 "800*500"。

    输出:
        完整的高德静态地图 API URL。
    """
    zoom = _map_zoom(coords)
    center = _map_center(coords)
    markers = _make_static_markers(coords)
    return (
        f"https://restapi.amap.com/v3/staticmap"
        f"?key={key}&size={size}&zoom={zoom}&center={center}&markers={markers}"
    )


def _fetch_as_data_uri(url: str, timeout: int = 10) -> str:
    """下载 URL 内容并转为 base64 data URI。

    输入:
        url: 图片 URL。
        timeout: 超时秒数。

    输出:
        "data:image/png;base64,..." 格式的 data URI。
    """
    with urlopen(url, timeout=timeout) as resp:
        data = base64.b64encode(resp.read()).decode("ascii")
        return f"data:image/png;base64,{data}"


class TripRenderer:
    """旅行规划 HTML 渲染器。

    加载 Jinja2 模板目录，提供 render() 方法将 TripPlan 转换为 HTML 文件。

    参数:
        template_dir: Jinja2 模板目录路径（plugins/travel/templates/）。
        output_dir: HTML 输出目录路径（plugins/travel/output/）。
        amap_js_key: 高德地图 JS API Key（用于网页内嵌交互地图）。

    示例:
        renderer = TripRenderer(
            template_dir=Path("plugins/travel/templates"),
            output_dir=Path("plugins/travel/output"),
            amap_js_key="your_key_here"
        )
        path = renderer.render(trip_plan, trip_id="abc", style="cyberpunk")
    """

    def __init__(
        self,
        template_dir: Path,
        output_dir: Path,
        amap_js_key: str = "",
        amap_security_code: str = "",
        render_mode: str = "static",
        amap_web_service_key: str = "",
    ) -> None:
        """初始化渲染器。

        输入:
            template_dir: 模板目录路径，必须存在且包含 base.html。
            output_dir: 输出目录路径，不存在则自动创建。
            amap_js_key: 高德地图 JS API Key（可选，render_mode=dynamic 时使用）。
            amap_security_code: 高德安全密钥 jscode（2021-12-02 后申请的 Key 必须，
                否则地图白色/标注异常）。
            render_mode: "static" 生成静态地图（<img>），"dynamic" 生成 JS API 交互地图。
            amap_web_service_key: 高德 Web 服务 Key（render_mode=static 时必需）。

        异常:
            FileNotFoundError: 模板目录不存在时抛出。
        """
        if not template_dir.exists():
            raise FileNotFoundError(f"模板目录不存在: {template_dir}")

        # 创建 Jinja2 环境
        # FileSystemLoader: 从文件系统加载模板
        # select_autoescape: 对 .html 文件启用自动转义（防 XSS）
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
        )

        # ★ 注册自定义过滤器：静态地图参数计算
        self._env.filters["static_markers"] = _make_static_markers
        self._env.filters["map_center"] = _map_center
        self._env.filters["map_zoom"] = _map_zoom

        # 模板缓存: {文件名: 编译后的 Template 对象}
        # 避免每次 render() 都重新编译模板
        self._template_cache: dict[str, Template] = {}

        # 输出目录
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 高德地图 JS API Key + 安全密钥（dynamic 模式）
        self._amap_js_key = amap_js_key
        self._amap_security_code = amap_security_code

        # ★ 渲染模式 + 静态地图 Key
        self._render_mode = render_mode
        self._amap_web_service_key = amap_web_service_key

    def render(
        self,
        plan: TripPlan,
        trip_id: str = "",
        style: str = "",
    ) -> Path:
        """将 TripPlan 渲染为单文件 HTML 并写入磁盘。

        渲染流程:
          1. 生成 trip_id（如果未提供）
          2. 填充元数据（generated_at, data_sources）
          3. 解析风格 → 获取模板
          4. 预取静态地图（render_mode=static 时下载图片转 base64）
          5. 构造 Jinja2 上下文数据
          6. 渲染 HTML 字符串
          7. 写入文件
          8. 返回输出文件路径

        输入:
            plan: TripPlan 对象（已通过 Pydantic 校验）。
            trip_id: 行程唯一 ID（为空时自动生成 "trp_" + 12位hex）。
            style: 风格 slug（如 "cyberpunk"、"art_deco"）。
                   为空时使用默认风格。
                   不存在时回退到默认风格并 log warning。

        输出:
            生成的 HTML 文件的绝对路径 Path 对象。

        示例:
            >>> path = renderer.render(plan, style="bauhaus")
            >>> print(path)
            Path("/.../output/trip_trp_a1b2c3d4e5f6.html")
        """
        # 1. 生成 trip_id
        if not trip_id:
            trip_id = f"trp_{uuid.uuid4().hex[:12]}"

        # 2. 填充元数据（不覆盖已有值）
        if not plan.trip_id:
            plan.trip_id = trip_id
        if not plan.generated_at:
            plan.generated_at = datetime.now().isoformat()
        if not plan.data_sources:
            plan.data_sources = ["amap", "xiaohongshu"]
        if style and not plan.style:
            plan.style = style

        # 3. 解析风格 → 选择模板
        template = self._get_template(style)

        # 4. 构造 Jinja2 上下文
        # 将 Pydantic 模型转为 dict，方便模板中使用
        plan_dict = plan.model_dump(mode="json")

        context = {
            # plan: 完整 TripPlan 对象（模板可直接 plan.city, plan.days 等）
            "plan": plan,
            # trip_json: JSON 字符串，用于前端 JavaScript（动态地图时需要）
            "trip_json": json.dumps(plan_dict, ensure_ascii=False),
            # amap_js_key: 高德地图 JS API Key（dynamic 模式交互地图）
            "amap_js_key": self._amap_js_key,
            # amap_security_code: 高德安全密钥（2021-12-02 后的 Key 必须）
            "amap_security_code": self._amap_security_code,
            # ★ 渲染模式：True 表示使用静态地图（<img>），False 表示交互地图
            "use_static_map": self._render_mode == "static",
            # ★ 高德 Web 服务 Key（static 模式静态地图 REST API）
            "amap_web_service_key": self._amap_web_service_key,
            # current_year: 用于 footer 版权年份
            "current_year": datetime.now().year,
        }

        # ★ 静态地图预取（render_mode=static 且有 Web 服务 Key 时）
        # 前端不再直接请求 restapi.amap.com，改为渲染时下载图片转 base64
        # 内嵌到 HTML 中。Telegram WebView 不再拦截外部请求，离线可用。
        day_map_uris: dict[int, str] = {}
        if self._render_mode == "static" and self._amap_web_service_key:
            for day in plan.days:
                coords: list[str] = []
                for act in day.activities:
                    if act.location and act.location.coordinates:
                        coords.append(
                            f"{act.location.coordinates.lng},"
                            f"{act.location.coordinates.lat}"
                        )
                if not coords:
                    continue
                url = _build_static_map_url(coords, self._amap_web_service_key)
                try:
                    day_map_uris[day.day] = _fetch_as_data_uri(url)
                    logger.info(
                        "[TripRenderer] 静态地图预取成功 day=%d markers=%d",
                        day.day, len(coords),
                    )
                except Exception as exc:
                    # 下载失败回退到原始 URL（浏览器仍可加载）
                    logger.warning(
                        "[TripRenderer] 静态地图预取失败 day=%d: %s，回退至 URL",
                        day.day, exc,
                    )
                    day_map_uris[day.day] = url
        context["day_map_uris"] = day_map_uris

        # 5. 渲染 HTML
        html_content = template.render(**context)

        # 6. 写入文件
        output_filename = f"trip_{trip_id}.html"
        output_path = self._output_dir / output_filename
        output_path.write_text(html_content, encoding="utf-8")

        logger.info(
            "[TripRenderer] HTML 已生成: %s (风格=%s, 大小=%d bytes)",
            output_path, style or DEFAULT_STYLE, len(html_content.encode("utf-8")),
        )

        # 7. 返回路径
        return output_path

    def _get_template(self, style: str) -> Template:
        """根据风格 slug 获取编译后的 Jinja2 模板。

        模板选择逻辑:
          1. style 为空 → 使用默认风格
          2. 查找 STYLE_TEMPLATE_MAP[style] → 获取模板文件名
          3. style 不存在于 map → 使用默认风格 + 记录 warning
          4. 模板文件不存在 → 使用默认风格 + 记录 warning
          5. 模板已缓存 → 直接返回缓存对象
          6. 模板未缓存 → 编译并缓存

        输入:
            style: 风格 slug（如 "cyberpunk"）。

        输出:
            编译后的 Jinja2 Template 对象。

        异常:
            不抛出异常 —— 所有错误都回退到默认风格。
        """
        # 解析模板文件名
        template_filename = STYLE_TEMPLATE_MAP.get(style)
        if template_filename is None:
            logger.warning(
                "[TripRenderer] 未知风格 '%s'，回退到默认风格 '%s'。"
                "可用风格: %s",
                style, DEFAULT_STYLE, list(STYLE_TEMPLATE_MAP.keys()),
            )
            template_filename = STYLE_TEMPLATE_MAP[DEFAULT_STYLE]

        # 检查缓存
        if template_filename in self._template_cache:
            return self._template_cache[template_filename]

        # 编译模板
        try:
            template = self._env.get_template(template_filename)
        except Exception as exc:
            logger.warning(
                "[TripRenderer] 加载模板 '%s' 失败: %s，回退到默认风格",
                template_filename, exc,
            )
            template_filename = STYLE_TEMPLATE_MAP[DEFAULT_STYLE]
            if template_filename in self._template_cache:
                return self._template_cache[template_filename]
            template = self._env.get_template(template_filename)

        # 缓存
        self._template_cache[template_filename] = template
        return template

    def list_styles(self) -> list[str]:
        """列出所有可用的风格 slug。

        输出:
            风格 slug 的排序列表。

        示例:
            >>> renderer.list_styles()
            ['art_deco', 'bauhaus', 'black_metal', ...]
        """
        return sorted(STYLE_TEMPLATE_MAP.keys())
