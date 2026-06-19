"""
peerAgent/travel-planner/planner.py —— 5-Agent 旅行规划流水线。

核心类 TripPlanner 实现：
  Step 1: 天气调研 → LLM 自主调用高德 MCP maps_weather
  Step 2: 景点调研 → LLM 自主调用高德 MCP maps_text_search 等
  Step 3: 酒店调研 → LLM 自主调用高德 MCP 周边搜索等
  Step 4: 小红书调研 → LLM 自主调用 XHS MCP search_xiaohongshu 等
  Step 5: 两阶段综合规划
    Pass 1: 交叉验证（LLM 带 Amap 工具，回查 XHS 推荐地点）
    Pass 2: 生成 TripPlan JSON（Pydantic 校验兜底，最多重试 3 次）

设计要点:
  - ★ 所有工具通过 fastmcp.Client 管理 MCP server 子进程获取
  - amap-mcp-server 和 stride28-search-mcp 各自作为独立子进程
  - 工具 schema 从 MCP server 动态获取，不手写
  - 统一的 _run_agent_step() ReAct 引擎驱动所有调研 Step
  - LLM 自主决定调哪个工具、传什么参数、搜几轮、何时结束
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import Config, load_config
from mcp_manager import McpManager
from prompts import PROMPTS

logger = logging.getLogger(__name__)


class TripPlanner:
    """旅行规划流水线 —— 5 个 Agent 按顺序执行。

    参数:
        config: Config 实例，包含 LLM/Amap/XHS/TripPlanner 配置。

    使用方式:
        >>> config = load_config()
        >>> planner = TripPlanner(config)
        >>> plan = await planner.plan("北京4日深度文化游 7/11-7/14")
    """

    def __init__(self, config: Config) -> None:
        """初始化规划器。

        输入:
            config: Config（从 Peer Agent 自身的 config.toml 读取）。
        """
        # ── LLM 客户端（OpenAI 兼容） ──
        self._llm = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        self._model = config.llm.model
        self._max_tokens = config.llm.max_tokens

        # ── MCP 管理器（高德 + 小红书） ──
        self._mcp = McpManager(amap_cfg=config.amap, xhs_cfg=config.xhs)

        # ── 规划参数 ──
        self._max_react_rounds = config.trip_planner.max_react_rounds
        self._max_synthesis_retries = config.trip_planner.max_synthesis_retries
        self._output_dir = Path(config.trip_planner.output_dir).resolve()

        logger.info(
            "[TripPlanner] 初始化完成 model=%s provider=%s",
            config.llm.model, config.llm.provider,
        )

    async def plan(self, goal: str) -> dict[str, Any]:
        """执行完整的 5-Agent 旅行规划流水线。

        输入:
            goal: 用户原始请求（如 "北京4日深度文化游 7/11-7/14 喜欢历史小众"）。

        输出:
            完整的 TripPlan dict（已通过 Pydantic 校验）。

        异常:
            RuntimeError: MCP 启动失败或综合规划重试耗尽。
        """
        # ── 启动 MCP server ──
        await self._mcp.start()
        try:
            # 1. 解析用户意图（从 goal 提取城市/日期/偏好）
            parsed = await self._parse_intent(goal)
            city = parsed["city"]
            dates = parsed["dates"]
            preference = parsed["preference"]

            logger.info(
                "[TripPlanner] 开始规划 city=%s dates=%s pref=%s",
                city, dates, preference,
            )

            # 2. 天气调研
            weather_data = await self._step_weather(city, dates)

            # 3. 景点调研
            attraction_data = await self._step_attractions(
                city, preference, weather_data,
            )

            # 4. 酒店调研
            hotel_data = await self._step_hotels(city, attraction_data)

            # 5. 小红书调研
            xhs_data = await self._step_xhs(city, preference)

            # 6. 综合规划（含交叉验证 + JSON 生成）
            plan = await self._step_synthesize(
                goal=goal,
                city=city,
                dates=dates,
                preference=preference,
                weather_data=weather_data,
                attraction_data=attraction_data,
                hotel_data=hotel_data,
                xhs_data=xhs_data,
            )

            logger.info("[TripPlanner] 规划完成 city=%s", city)
            return plan
        finally:
            await self._mcp.close()

    # ═══════════════════════════════════════════════════════════════
    # 意图解析
    # ═══════════════════════════════════════════════════════════════

    async def _parse_intent(self, goal: str) -> dict[str, Any]:
        """从用户自然语言请求中提取结构化信息。

        输入:
            goal: 用户原始请求文本。

        输出:
            {"city": "城市名",
             "dates": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
             "preference": "偏好描述"}
        """
        response = await self._llm.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "从用户的旅行请求中提取城市、日期和偏好。"
                        "输出纯 JSON: "
                        '{"city":"...", "dates":{"start":"YYYY-MM-DD",'
                        '"end":"YYYY-MM-DD"}, "preference":"..."}'
                        "如果日期不完整，合理推断（默认今年）。"
                        "如果偏好未提及，填 '综合体验'。"
                    ),
                },
                {"role": "user", "content": goal},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    # ═══════════════════════════════════════════════════════════════
    # 核心: ReAct 循环 —— 每个 Agent Step 的通用引擎
    # ═══════════════════════════════════════════════════════════════

    async def _run_agent_step(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_rounds: int | None = None,
    ) -> str:
        """ReAct 循环: LLM 自主决定调哪些工具、调几次、何时结束。

        工具 schema 由 MCP server 动态提供（list_tools），
        工具调用自动路由到对应的 MCP server（call_tool）。

        输入:
            system_prompt: 该 Step 的 system prompt（定义角色和输出格式）。
            user_prompt: 该 Step 的初始任务描述 + 上下文。
            tools: OpenAI tool schema 列表（从 self._mcp.list_tools() 获取）。
            max_rounds: 最大 ReAct 轮数，默认使用 config.toml 中的值。

        输出:
            LLM 的最终文本回复（调研结果）。

        异常:
            RuntimeError: 超过最大轮数仍未完成时抛出。
        """
        if max_rounds is None:
            max_rounds = self._max_react_rounds

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for round_idx in range(max_rounds):
            response = await self._llm.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )
            msg = response.choices[0].message

            # 转为 dict 追加到消息历史
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)

            # 如果 LLM 不再请求工具 → 完成
            if not msg.tool_calls:
                logger.info(
                    "[TripPlanner] ReAct 完成 round=%d content_len=%d",
                    round_idx + 1, len(msg.content or ""),
                )
                return msg.content or ""

            # ★ 执行工具调用 → 统一路由到 MCP server
            for tc in msg.tool_calls:
                tool_result = await self._mcp.call_tool(
                    tool_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        raise RuntimeError(
            f"ReAct 超过最大轮数 {max_rounds}，LLM 未能在规定轮数内完成任务"
        )

    # ═══════════════════════════════════════════════════════════════
    # Step 1-4: 调研 Agent
    #
    # 每个 Step 从对应的 MCP server 获取工具列表，
    # LLM 自主决定搜什么、搜几次、何时结束。
    # ═══════════════════════════════════════════════════════════════

    async def _step_weather(
        self, city: str, dates: dict[str, str],
    ) -> str:
        """Step 1: 天气调研（LLM + 高德 MCP maps_weather）。

        输入:
            city: 城市名。
            dates: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}。

        输出:
            LLM 分析后的天气文本（Markdown 格式）。
        """
        return await self._run_agent_step(
            system_prompt=PROMPTS["weather"],
            user_prompt=(
                f"请查询 {city} 的天气预报。\n"
                f"旅行日期: {dates['start']} ~ {dates['end']}。\n"
                f"请分析旅行期间每天的天气，并给出出行建议"
                f"(如: 哪几天适合户外、哪几天需备雨具等)。"
            ),
            tools=await self._mcp.list_tools("amap"),
        )

    async def _step_attractions(
        self, city: str, preference: str, weather_data: str,
    ) -> str:
        """Step 2: 景点调研（LLM + 高德 MCP 搜索工具）。

        输入:
            city: 城市名。
            preference: 用户偏好（如"历史文化+小众"）。
            weather_data: Step 1 天气输出（用于天气适配）。

        输出:
            LLM 筛选推荐后的景点列表文本。
        """
        return await self._run_agent_step(
            system_prompt=PROMPTS["attractions"],
            user_prompt=(
                f"目的地: {city}\n"
                f"用户偏好: {preference}\n\n"
                f"天气参考:\n{weather_data}\n\n"
                f"请根据偏好拆分关键词搜索景点，可以多轮搜索。\n"
                f"从搜索结果中筛选最合适的推荐，每景点给出推荐理由。\n"
                f"注意: 下雨/高温天优先推荐室内景点。"
            ),
            tools=await self._mcp.list_tools("amap"),
        )

    async def _step_hotels(
        self, city: str, attraction_data: str,
    ) -> str:
        """Step 3: 酒店调研（LLM + 高德 MCP 搜索 + 周边搜索）。

        输入:
            city: 城市名。
            attraction_data: Step 2 景点输出（用于判断最优住宿区域）。

        输出:
            LLM 筛选推荐后的酒店列表文本。
        """
        return await self._run_agent_step(
            system_prompt=PROMPTS["hotels"],
            user_prompt=(
                f"目的地: {city}\n\n"
                f"已选景点 (供位置参考):\n{attraction_data}\n\n"
                f"请从景点坐标中提取活动集中的区域，\n"
                f"用 maps_around_search 在景点周边搜索酒店，\n"
                f"也用 maps_text_search 搜索不同档次的酒店。\n"
                f"综合推荐 3-5 家位置便利、性价比高的住宿。"
            ),
            tools=await self._mcp.list_tools("amap"),
        )

    async def _step_xhs(
        self, city: str, preference: str,
    ) -> str:
        """Step 4: 小红书调研（LLM + XHS MCP 搜索 + 笔记全文）。

        输入:
            city: 城市名。
            preference: 用户偏好。

        输出:
            LLM 整理的小红书笔记摘要文本。
            包含笔记链接、关键信息提取、适用性说明。
        """
        return await self._run_agent_step(
            system_prompt=PROMPTS["xhs"],
            user_prompt=(
                f"请搜索小红书 {city} 的旅行攻略。\n"
                f"偏好: {preference}。\n"
                f"请从多个角度搜索（如 '{city} 深度游'、'{city} 小众景点'、'{city} 避坑指南'），\n"
                f"筛选高赞笔记并阅读全文，提取有价值的攻略信息。\n"
                f"输出每条笔记的标题、作者、链接、点赞数、关键信息和适用性。"
            ),
            tools=await self._mcp.list_tools("xhs"),
        )

    # ═══════════════════════════════════════════════════════════════
    # Step 5: 交叉验证 + 综合规划（两阶段）
    # ═══════════════════════════════════════════════════════════════

    async def _step_synthesize(
        self,
        goal: str,
        city: str,
        dates: dict[str, str],
        preference: str,
        weather_data: str,
        attraction_data: str,
        hotel_data: str,
        xhs_data: str,
    ) -> dict[str, Any]:
        """Step 5: 两阶段综合规划 → TripPlan JSON。

        Pass 1: 交叉验证 —— LLM 带 Amap 工具（maps_text_search, maps_geo），
                自主验证 XHS 推荐的每个地点。
                找到 → community_match，未找到 → community_only。

        Pass 2: 生成 JSON —— Pydantic 校验失败重试（最多 N 次）。

        输入:
            所有前序 Step 的输出文本。
        输出:
            完整的 TripPlan dict。
        """
        # ═════════════════════════════════════════════════════════
        # Pass 1: 交叉验证（LLM 带 Amap 工具，自主决策）
        # ═════════════════════════════════════════════════════════
        # 只给 maps_text_search 和 maps_geo —— 不需要天气/周边搜索
        all_amap_tools = await self._mcp.list_tools("amap")
        verification_tools = [
            t for t in all_amap_tools
            if t["function"]["name"] in ("maps_text_search", "maps_geo")
        ]

        verification_report = await self._run_agent_step(
            system_prompt=PROMPTS["planner"],
            user_prompt=(
                f"## 任务: 交叉验证\n\n"
                f"城市: {city}\n\n"
                f"## Amap 景点 (坐标已验证)\n{attraction_data}\n\n"
                f"## Amap 酒店 (坐标已验证)\n{hotel_data}\n\n"
                f"## XHS 小红书推荐 (无坐标)\n{xhs_data}\n\n"
                f"---\n"
                f"对 XHS 推荐的每个地点:\n"
                f"1. 在 Amap 列表中模糊匹配 → 匹配到则标记 community_match\n"
                f"2. 未匹配 → 调 maps_text_search 回查\n"
                f"3. 仍未找到 → 调 maps_geo 获取大致坐标 → community_only\n"
                f"4. 连地址都没有 → 说明无法验证，建议放入 suggestions\n\n"
                f"请输出验证报告，格式:\n"
                f"XHS推荐 | 匹配结果 | verification | coordinates | xhs_note_id\n"
            ),
            tools=verification_tools,
            max_rounds=None,  # 走 config.toml 的 max_react_rounds (12)，交叉验证需要更多轮次
        )

        # ═════════════════════════════════════════════════════════
        # Pass 2: 生成 JSON（无工具，纯生成，最多重试 N 次）
        # ═════════════════════════════════════════════════════════
        synthesis_prompt = (
            f"## 用户请求\n{goal}\n\n"
            f"## 城市\n{city}\n"
            f"日期\n{date_range_text(dates)}\n"
            f"偏好\n{preference}\n\n"
            f"## 天气\n{weather_data}\n\n"
            f"## Amap 景点\n{attraction_data}\n\n"
            f"## Amap 酒店\n{hotel_data}\n\n"
            f"## XHS 小红书\n{xhs_data}\n\n"
            f"## 交叉验证报告\n{verification_report}\n\n"
            f"---\n"
            f"综合以上所有信息（含验证报告），输出 TripPlan JSON。\n"
            f"XHS 推荐优先采用，坐标必须来自 Amap 或验证结果。\n"
            f"每个 Location 必须填写 verification 和 xhs_source_note_id。"
        )

        last_error = ""
        for attempt in range(1, self._max_synthesis_retries + 1):
            logger.info(
                "[TripPlanner] JSON 生成 第 %d/%d 次",
                attempt, self._max_synthesis_retries,
            )

            response = await self._llm.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": PROMPTS["planner"]},
                    {
                        "role": "user",
                        "content": (
                            synthesis_prompt
                            + (
                                f"\n\n⚠️ 上次校验失败: {last_error}\n"
                                f"请修正上述错误后重新输出。"
                                if last_error else ""
                            )
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )

            raw_json = response.choices[0].message.content

            try:
                # 从父项目的 plugins/travel/schema.py 加载 TripPlan
                _project_root = Path(__file__).resolve().parent.parent.parent
                sys.path.insert(0, str(_project_root))
                from plugins.travel.schema import TripPlan

                plan = TripPlan.model_validate_json(raw_json)
                logger.info("[TripPlanner] Pydantic 校验通过")
                return plan.model_dump(mode="json")
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[TripPlanner] 校验失败 (attempt=%d): %s",
                    attempt, last_error,
                )

        raise RuntimeError(
            f"综合规划重试 {self._max_synthesis_retries} 次仍失败: {last_error}"
        )


def date_range_text(dates: dict[str, str]) -> str:
    """将 dates dict 格式化为可读文本。

    输入:
        dates: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}。

    输出:
        "YYYY-MM-DD ~ YYYY-MM-DD"。
    """
    return f"{dates['start']} ~ {dates['end']}"
