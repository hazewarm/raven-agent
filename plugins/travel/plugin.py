"""
plugins/travel/plugin.py —— 旅行规划插件。

为 raven-agent 的 LLM 提供 render_trip_html 工具。
此工具接收 TripPlan JSON 文件路径，调用 Jinja2 渲染器生成
精美的杂志风 HTML。

此插件使用 plugins/travel/ 下的共享组件:
  - schema.py    → TripPlan Pydantic 校验
  - renderer.py  → Jinja2 HTML 渲染
  - templates/   → 29 种风格的 Jinja2 模板

配置 (_conf_schema.json):
  - amap_js_key: 高德地图 JS API Key（用于网页内嵌交互地图），默认 ""。
  - default_style: 默认渲染风格 slug，默认 "art_deco"。
"""

from __future__ import annotations

import logging
from pathlib import Path

from raven_agent.plugins import Plugin, on_after_reasoning, on_tool_pre, tool

from .renderer import DEFAULT_STYLE, STYLE_TEMPLATE_MAP, TripRenderer
from .schema import TripPlan

logger = logging.getLogger(__name__)


class TravelPlugin(Plugin):
    """旅行规划插件 —— 提供 render_trip_html 工具。

    输入:
        无（通过 Plugin 基类自动加载）。

    输出:
        TravelPlugin 实例。

    上下文注入（由 PluginManager 自动注入 self.context）:
        - plugin_dir: 插件目录 Path → 用于定位 templates/ 和 output/
        - config: 插件配置（amap_js_key, default_style）

    工具:
        render_trip_html: 将 TripPlan JSON 文件渲染为 HTML 网页。
    """

    # 插件名（必需）
    name = "travel"

    # ── 私有属性（在 initialize() 中初始化） ──
    _renderer: TripRenderer | None = None

    async def initialize(self) -> None:
        """初始化渲染器和输出目录。

        在 PluginManager 加载插件后自动调用。
        此时 self.context 已注入，可访问 plugin_dir、config 等。

        输入:
            无。

        输出:
            None。

        异常:
            初始化失败时 PluginManager 会回滚整个插件注册。
        """
        # 获取模板目录（plugins/travel/templates/）
        template_dir = self.context.plugin_dir / "templates"

        # 获取输出目录（plugins/travel/output/）
        output_dir = self.context.plugin_dir / "output"

        # 获取高德 JS API Key 和安全密钥（从插件配置读取）
        amap_js_key = ""
        amap_security_code = ""
        if self.context.config is not None:
            amap_js_key = self.context.config.get("amap_js_key", default="")
            amap_security_code = self.context.config.get("amap_security_code", default="")

        # ★ 读取渲染模式和静态地图 Key
        self._render_mode = "static"
        self._amap_web_service_key = ""
        self._dashboard_base_url = ""
        if self.context.config is not None:
            self._render_mode = self.context.config.get("render_mode", default="static")
            self._amap_web_service_key = self.context.config.get("amap_web_service_key", default="")
            self._dashboard_base_url = self.context.config.get("dashboard_base_url", default="")

        # 创建渲染器
        self._renderer = TripRenderer(
            template_dir=template_dir,
            output_dir=output_dir,
            amap_js_key=amap_js_key,
            amap_security_code=amap_security_code,
            render_mode=self._render_mode,
            amap_web_service_key=self._amap_web_service_key,
        )

        logger.info(
            "[TravelPlugin] 初始化完成 模板目录=%s 输出目录=%s render_mode=%s",
            template_dir, output_dir, self._render_mode,
        )

    @tool(
        "render_trip_html",
        risk="write",
        search_hint="渲染旅行计划 HTML 网页 预览 攻略",
    )
    async def render_trip_html(
        self,
        event,  # PluginToolEvent —— 必须前两个参数 self, event
        trip_json_path: str,
        style: str = "",
    ) -> str:
        """将已有的 TripPlan JSON 文件渲染为精美的杂志风单文件 HTML。

        ★ 设计原则：此工具接收 JSON 文件的绝对路径而非 JSON 字符串。
        原因：TripPlan JSON 体积较大（5-15KB），传入 LLM 上下文浪费 token，
        且存在 LLM 意外篡改数据的风险。JSON 在上游（Peer Agent 或 MCP 调研）
        生成后直接落盘，此处通过文件路径读取，数据零损失。

        Args:
            trip_json_path: TripPlan JSON 文件的绝对路径。
                           文件必须存在且包含合法的 TripPlan JSON。
                           JSON 由上游生成并落盘：
                           - Peer Agent 路径 → plugins/travel/output/trip_{id}.json
                           - MCP 调研路径 → 由 LLM 先 write 落盘再传路径
            style: 渲染风格 slug，默认为空（使用默认 art_deco 风格）。
                   可选值见下方可用风格列表。

        Returns:
            成功时返回:
              ✅ 旅行攻略网页已生成！
              📄 文件路径: <绝对路径>
              🎨 渲染风格: <风格名>
              📊 行程摘要: <城市> <N>日游 | <日期范围>
              📅 每日: Day1 主题 | Day2 主题 | ...

            文件不存在时返回:
              ❌ 文件不存在: <路径>，请检查路径是否正确。

            校验失败时返回:
              ❌ TripPlan JSON 校验失败: <错误详情>

        可用风格 (共 29 种):
            minimalist, bold_modern, elegant_vintage, futuristic_tech,
            scandinavian, art_deco, japanese_minimalism,
            postmodern_deconstruct, punk, british_rock, black_metal,
            memphis, cyberpunk, pop_art, deconstructed_swiss,
            vaporwave, neo_expressionism, extreme_minimalism,
            neo_futurism, surrealist_collage, neo_baroque,
            liquid_digital, hypersensory_minimalism,
            neo_expressionist_data, victorian, bauhaus,
            constructivism, memphis_design, german_expressionism
        """
        if self._renderer is None:
            return "❌ 插件未初始化，请联系管理员检查插件配置。"

        # ── 0. 检查文件是否存在 ──
        json_path = Path(trip_json_path)
        if not json_path.exists():
            return (
                f"❌ 文件不存在: {trip_json_path}\n"
                f"请检查路径是否正确。通常 JSON 文件位于:\n"
                f"  - Peer Agent 生成: plugins/travel/output/trip_<id>.json\n"
                f"  - 手动落盘: 请先用 write 工具保存 JSON 再传入路径"
            )

        # ── 1. 从磁盘读取 JSON 并 Pydantic 校验 ──
        # ★ JSON 内容从头到尾不经过 LLM 上下文
        # 读取 + 校验全部在插件后端完成
        try:
            raw_json = json_path.read_text(encoding="utf-8")
            plan = TripPlan.model_validate_json(raw_json)
        except Exception as exc:
            logger.warning(
                "[TravelPlugin] TripPlan 校验失败 path=%s: %s",
                trip_json_path, exc,
            )
            return (
                f"❌ TripPlan JSON 校验失败:\n"
                f"文件: {trip_json_path}\n"
                f"错误: {exc}\n\n"
                f"常见问题:\n"
                f"  - 日期格式必须为 YYYY-MM-DD\n"
                f"  - coordinates 必须来自工具数据，禁止编造\n"
                f"  - temp_high/temp_low 为纯数字（不含°C）\n"
                f"  - 每天至少 1 个 activity\n"
                f"  - 必填字段不能为空字符串"
            )

        # ── 2. 解析风格 ──
        # 如果 LLM 传了 style 参数，验证其是否有效
        style = style.strip().lower() if style else ""
        if style and style not in STYLE_TEMPLATE_MAP:
            logger.warning(
                "[TravelPlugin] LLM 请求了未知风格 '%s'，回退到默认风格",
                style,
            )
            style = ""

        # ── 3. 渲染 HTML ──
        try:
            output_path = self._renderer.render(
                plan=plan,
                trip_id="",  # 自动生成
                style=style,
            )
            self._last_rendered_html = str(output_path)
        except Exception as exc:
            logger.error("[TravelPlugin] 渲染失败: %s", exc)
            return f"❌ HTML 渲染失败: {exc}"

        # ── 4. 构造返回信息 ──
        style_name = style or DEFAULT_STYLE
        day_summaries = "\n".join(
            f"  📅 Day{d.day} ({d.date}): {d.theme}"
            for d in plan.days
        )

        return (
            f"✅ 旅行攻略网页已生成！\n\n"
            f"📄 文件路径: {output_path}\n"
            f"🎨 渲染风格: {style_name}\n"
            f"📊 行程摘要: {plan.city} {plan.total_days}日游 "
            f"| {plan.start_date} ~ {plan.end_date}\n"
            f"💰 预算总计: {plan.budget.currency}{plan.budget.total:.0f}\n\n"
            f"{day_summaries}\n\n"
            f"💡 提示: 用浏览器打开 HTML 文件即可查看完整攻略。\n"
            f"   如需更换风格，使用 render_trip_html 重新渲染并指定 style 参数。"
        )

    @on_after_reasoning()
    async def attach_html_media(self, ctx):
        """将 render_trip_html 生成的 HTML 输出注入出站消息。

        static 模式：注入 outbound_metadata["media"] → HTML 文件作为 Telegram 附件发送。
        dynamic 模式：将回复中的本地文件路径替换为 Dashboard 公网 URL
                    → 用户通过链接在线查看交互地图。

        输入:
            ctx: AfterReasoningCtx。

        输出:
            修改后的 AfterReasoningCtx。
        """
        html_path = getattr(self, '_last_rendered_html', None)
        if not html_path:
            return ctx

        if self._render_mode == "dynamic" and self._dashboard_base_url:
            # ★ dynamic 模式：替换回复中的本地路径为 Dashboard URL
            filename = Path(html_path).name
            dashboard_url = self._dashboard_base_url.rstrip("/")
            public_url = f"{dashboard_url}/trips/{filename}"
            ctx.reply = ctx.reply.replace(str(html_path), public_url)
            # 把「用浏览器打开 HTML 文件」改为「点击链接在线查看」
            ctx.reply = ctx.reply.replace(
                "💡 提示: 用浏览器打开 HTML 文件即可查看完整攻略。",
                f"🔗 在线查看: {public_url}",
            )
        else:
            # static 模式 / dynamic 模式未配置 dashboard_base_url
            # → HTML 文件作为 Telegram 附件发送
            media = list(ctx.outbound_metadata.get("media", []))
            media.append(html_path)
            ctx.outbound_metadata["media"] = media
            if self._render_mode == "dynamic" and not self._dashboard_base_url:
                ctx.reply += (
                    "\n\n⚠️ 已启用 dynamic 渲染模式但未配置 dashboard_base_url，"
                    "自动回退为附件发送。请在 plugin_config.json 中设置 dashboard_base_url。"
                )

        self._last_rendered_html = None  # 清理，防止下轮误注入
        return ctx

    @on_tool_pre(tool_name="delegate_travel_planner")
    async def inject_channel_for_peer(self, event) -> dict | None:
        """向 delegate_travel_planner 注入 channel/chat_id。

        PeerAgentTool 默认从 kwargs 取 channel/chat_id，
        但 LLM 调用时不会传这两个参数 → 永远是 "unknown"。
        Poller 完成后的系统通知也因此无法路由到正确用户。

        这里从 turn pipeline 的 metadata 中提取真实的 channel/chat_id，
        注入到工具参数中。与 SpawnToolContextHook 做法完全一致。

        输入:
            event: PluginToolHookEvent。

        输出:
            改写后的 arguments dict，channel/chat_id 已注入。
            None 表示无需改写（已经设置了值）。
        """
        args = dict(event.arguments)
        changed = False
        if event.metadata:
            for key in ("channel", "chat_id"):
                meta_val = event.metadata.get(key)
                if meta_val and not args.get(key):
                    args[key] = meta_val
                    changed = True
        return args if changed else None

    async def terminate(self) -> None:
        """插件终止时的清理工作。

        输入:
            无。

        输出:
            None。
        """
        logger.info("[TravelPlugin] 已终止")
