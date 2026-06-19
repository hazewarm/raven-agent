"""
plugins/travel/schema.py —— TripPlan Pydantic 数据模型。

此文件定义了旅行规划中所有结构化数据的 Schema。
它是 plugins/travel/ 和 peerAgent/travel-planner/ 之间的
**唯一真相来源（Single Source of Truth）**。

两个模块都 import 此文件来保证数据结构一致：
  - plugins/travel/plugin.py  → 校验 LLM 输出的 JSON
  - plugins/travel/renderer.py → 类型安全的 Jinja2 渲染
  - peerAgent/travel-planner/planner.py → 校验综合规划 LLM 的输出
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
# 枚举类型
# ═══════════════════════════════════════════════════════════════════

class MealType(str, Enum):
    """餐食类型枚举。

    值:
        BREAKFAST: 早餐
        LUNCH:     午餐
        DINNER:    晚餐
        SNACK:     小吃/下午茶
    """
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class TransportMode(str, Enum):
    """交通方式枚举。

    值:
        WALK:   步行
        METRO:  地铁
        BUS:    公交
        TAXI:   出租车/网约车
        BIKE:   共享单车
        TRAIN:  火车/高铁
        FLIGHT: 飞机
    """
    WALK = "walk"
    METRO = "metro"
    BUS = "bus"
    TAXI = "taxi"
    BIKE = "bike"
    TRAIN = "train"
    FLIGHT = "flight"


# ═══════════════════════════════════════════════════════════════════
# 基础数据类型
# ═══════════════════════════════════════════════════════════════════

class Coordinates(BaseModel):
    """地理坐标。

    输入:
        lat: 纬度（-90 ~ 90），必填。
        lng: 经度（-180 ~ 180），必填。

    验证:
        lat 范围 [-90, 90]
        lng 范围 [-180, 180]

    示例:
        Coordinates(lat=39.9042, lng=116.4074)  # 北京
    """
    lat: float = Field(..., ge=-90, le=90, description="纬度")
    lng: float = Field(..., ge=-180, le=180, description="经度")


class WeatherInfo(BaseModel):
    """单日天气信息。

    输入:
        date: 日期（YYYY-MM-DD），必填。
        weather: 天气状况描述（如"晴"、"多云"、"小雨"），必填。
        temp_high: 最高温度（摄氏度），必填，纯数字不含°C。
        temp_low: 最低温度（摄氏度），必填，纯数字不含°C。
        wind: 风力描述（如"东风1-3级"），可选。
        humidity: 相对湿度百分比（0-100），可选。
        tips: 当日出行建议（如"注意防晒"、"带雨伞"），可选。

    约束:
        - temp_high 必须 ≥ temp_low
        - 温度值为纯数字，不含符号

    示例:
        WeatherInfo(
            date="2025-07-11", weather="晴",
            temp_high=34, temp_low=24,
            wind="南风2级", humidity=45,
            tips="注意防晒，建议上午户外活动"
        )
    """
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="日期")
    weather: str = Field(..., min_length=1, description="天气状况")
    temp_high: int = Field(..., description="最高温度（摄氏度，纯数字）")
    temp_low: int = Field(..., description="最低温度（摄氏度，纯数字）")
    wind: str = Field(default="", description="风力描述")
    humidity: int = Field(default=0, ge=0, le=100, description="湿度百分比")
    tips: str = Field(default="", description="当日出行建议")

    @field_validator("temp_high")
    @classmethod
    def temp_high_must_be_greater(cls, v: int, info: Any) -> int:
        """验证最高温度不低于最低温度。

        输入:
            v: 最高温度值。
            info: Pydantic validation info（含其他字段值）。

        输出:
            验证通过的最高温度值。

        异常:
            ValueError: 最高温度低于最低温度时抛出。
        """
        temp_low = info.data.get("temp_low")
        if temp_low is not None and v < temp_low:
            raise ValueError(f"最高温度({v})不能低于最低温度({temp_low})")
        return v


class Location(BaseModel):
    """地点信息（景点/餐厅/酒店通用）。

    输入:
        name: 地点名称，必填。
        address: 详细地址，必填。
        coordinates: 经纬度坐标（来自高德 API，禁止编造），必填。
        amap_url: 高德地图导航链接，可选。
        phone: 联系电话，可选。
        rating: 评分（0-5），可选。
        price_info: 价格信息文本（如"门票40元"、"人均80元"），可选。
        opening_hours: 营业/开放时间文本，可选。
        notes: 备注信息（如"需要预约"），可选。

        # ★ 数据可信度标记（Step 5 交叉验证后赋值）
        verification: 数据来源可信度:
            "amap_verified"  — 高德 Agent (Step 2/3) 直接搜索得到
            "community_match" — 小红书推荐 → Step 5 回查 Amap 匹配成功
            "community_only"  — 小红书推荐 → Step 5 回查也未找到，坐标近似
        xhs_source_note_id: 如果来自小红书，指向原始笔记 ID，可选。

    约束:
        - coordinates 必须来自真实 API 数据，禁止 LLM 编造
          即使是 community_only 级别，也要用 geo_code 获取大致坐标
        - amap_url 格式：https://uri.amap.com/marker?position={lng},{lat}

    示例:
        # 高德直接搜索的景点
        Location(
            name="故宫博物院", ..., verification="amap_verified"
        )
        # 小红书推荐 → Step 5 回查匹配成功
        Location(
            name="山也民宿(西湖景区店)", ..., verification="community_match",
            xhs_source_note_id="abc123"
        )
        # 小红书推荐 → 回查也未找到，坐标仅为大致区域
        Location(
            name="老王家的私房菜", ..., verification="community_only",
            xhs_source_note_id="def456",
            notes="此地点来自小红书博主推荐，建议导航前电话确认"
        )
    """
    name: str = Field(..., min_length=1, description="地点名称")
    address: str = Field(..., min_length=1, description="详细地址")
    coordinates: Coordinates = Field(
        ..., description="经纬度坐标（来自API，禁止编造）"
    )
    amap_url: str = Field(default="", description="高德地图导航链接")
    phone: str = Field(default="", description="联系电话")
    rating: float = Field(default=0.0, ge=0, le=5, description="评分（0-5）")
    price_info: str = Field(default="", description="价格信息")
    opening_hours: str = Field(default="", description="营业/开放时间")
    notes: str = Field(default="", description="备注")
    # ★ 数据可信度
    verification: str = Field(
        default="amap_verified",
        pattern=r"^(amap_verified|community_match|community_only)$",
        description="数据可信度: amap_verified / community_match / community_only",
    )
    xhs_source_note_id: str = Field(
        default="", description="小红书笔记 ID（如来自社区推荐）"
    )


class TransportSegment(BaseModel):
    """交通换乘段。

    描述两地之间的一段交通（如"从故宫坐地铁到颐和园"）。
    一段可能包含多个步骤（如"步行到地铁站→坐14号线→步行到景点"）。

    输入:
        from_place: 出发地点名称，必填。
        to_place: 到达地点名称，必填。
        mode: 主要交通方式，必填。
        duration_minutes: 预计耗时（分钟），必填。
        description: 详细描述（如"地铁14号线，豫园站→陆家嘴站"），必填。
        route: 具体路线（如"14号线往嘉定方向"），可选。
        cost: 交通费用（元），可选。

    示例:
        TransportSegment(
            from_place="故宫", to_place="颐和园",
            mode=TransportMode.METRO,
            duration_minutes=45,
            description="地铁4号线大兴线，西单站→北宫门站",
            route="4号线往安河桥北方向",
            cost=5.0
        )
    """
    from_place: str = Field(..., min_length=1, description="出发地点")
    to_place: str = Field(..., min_length=1, description="到达地点")
    mode: TransportMode = Field(..., description="主要交通方式")
    duration_minutes: int = Field(..., gt=0, description="预计耗时（分钟）")
    description: str = Field(..., min_length=1, description="详细交通描述")
    route: str = Field(default="", description="具体路线信息")
    cost: float = Field(default=0.0, ge=0, description="交通费用（元）")


class Meal(BaseModel):
    """一餐信息。

    输入:
        type: 餐食类型（早/午/晚/小吃），必填。
        restaurant: 餐厅地点信息，必填。
        must_try: 推荐菜品列表，可选。
        estimated_cost: 预估人均消费（元），可选。

    示例:
        Meal(
            type=MealType.LUNCH,
            restaurant=Location(
                name="四季民福烤鸭店(故宫店)",
                address="东城区南池子大街32号",
                coordinates=Coordinates(lat=39.9145, lng=116.4037),
                price_info="人均120元",
                rating=4.5
            ),
            must_try=["北京烤鸭", "芥末鸭掌", "小吊梨汤"],
            estimated_cost=120
        )
    """
    type: MealType = Field(..., description="餐食类型")
    restaurant: Location = Field(..., description="餐厅信息")
    must_try: list[str] = Field(default_factory=list, description="推荐菜品")
    estimated_cost: float = Field(default=0.0, ge=0, description="人均预估消费（元）")


class XHSNote(BaseModel):
    """小红书笔记引用。

    输入:
        title: 笔记标题，必填。
        author: 作者昵称，必填。
        url: 笔记链接（必须是具体帖子链接，非搜索链接），必填。
        likes: 点赞数文本（如"3.2k"），可选。
        snippet: 内容摘要（100字以内），可选。
        relevance: 与本次旅行的相关性说明，可选。

    约束:
        - url 必须是具体帖子链接，不是搜索结果页链接
        - snippet 不超过 100 字

    示例:
        XHSNote(
            title="北京4天3夜深度游！这些小众景点太绝了",
            author="旅行日记本",
            url="https://www.xiaohongshu.com/explore/abc123",
            likes="3.2k",
            snippet="故宫东华门进避开人流...颐和园坐船游览太惬意...",
            relevance="提供了故宫和颐和园的实用游览建议"
        )
    """
    title: str = Field(..., min_length=1, description="笔记标题")
    author: str = Field(..., min_length=1, description="作者昵称")
    url: str = Field(..., min_length=1, description="笔记链接（具体帖子链接）")
    likes: str = Field(default="", description="点赞数")
    snippet: str = Field(default="", max_length=100, description="内容摘要（≤100字）")
    relevance: str = Field(default="", description="与本次旅行的相关性")


class Activity(BaseModel):
    """每日行程中的一个活动（景点参观/用餐/交通）。

    输入:
        order: 当天顺序号（从 1 开始），必填。
        start_time: 开始时间（HH:MM），必填。
        end_time: 结束时间（HH:MM），必填。
        title: 活动标题（如"游览故宫"），必填。
        description: 活动描述（100-150字，带小红书"网感"），必填。
        location: 活动地点信息，可选（纯交通段可为空）。
        transport_to_next: 到下一个活动地点的交通方式，可选。
        category: 活动分类，必填。
            可选值: "attraction" | "meal" | "transport" | "rest" | "shopping" | "hotel"
        meal_detail: 餐食详情（仅 category="meal" 时使用），可选。
        notes: 注意事项，可选。

    约束:
        - end_time 必须晚于 start_time
        - description 应有"网感"，不枯燥

    示例:
        Activity(
            order=1,
            start_time="08:30", end_time="11:30",
            title="故宫深度游",
            description="从东华门入宫避开天安门人流，沿着中轴线一路向北..."
            location=Location(name="故宫博物院", ...),
            transport_to_next=TransportSegment(
                from_place="故宫", to_place="四季民福",
                mode=TransportMode.WALK,
                duration_minutes=10,
                description="步行约800米，沿途可欣赏东华门大街"
            ),
            category="attraction"
        )
    """
    order: int = Field(..., ge=1, description="当天顺序号")
    start_time: str = Field(
        ..., pattern=r"^\d{2}:\d{2}$", description="开始时间 HH:MM"
    )
    end_time: str = Field(
        ..., pattern=r"^\d{2}:\d{2}$", description="结束时间 HH:MM"
    )
    title: str = Field(..., min_length=1, description="活动标题")
    description: str = Field(
        ..., min_length=10, max_length=200,
        description="活动描述（100-150字，带网感）",
    )
    location: Location | None = Field(default=None, description="活动地点")
    transport_to_next: TransportSegment | None = Field(
        default=None, description="到下一地点的交通"
    )
    category: str = Field(
        ...,
        pattern=r"^(attraction|meal|transport|rest|shopping|hotel)$",
        description="活动分类",
    )
    meal_detail: Meal | None = Field(default=None, description="餐食详情")
    notes: str = Field(default="", description="注意事项")


class DayPlan(BaseModel):
    """单日行程计划。

    输入:
        day: 第几天（从 1 开始），必填。
        date: 日期（YYYY-MM-DD），必填。
        theme: 当天主题（如"皇城中轴线深度探索"），必填。
        weather: 当天天气信息，必填。
        activities: 当天活动列表（至少 1 个），必填。
        hotel: 当晚住宿酒店信息，可选。
        daily_budget: 当天预估花费（元），可选。
        day_notes: 当天特别提示，可选。

    约束:
        - 每天 2-4 个景点活动（不含用餐和交通）
        - 下雨/高温天优先安排室内活动

    示例:
        DayPlan(
            day=1, date="2025-07-11",
            theme="皇城中轴线深度探索",
            weather=WeatherInfo(date="2025-07-11", weather="晴",
                                temp_high=34, temp_low=24),
            activities=[...],
            hotel=Location(name="北京王府井希尔顿", ...),
            daily_budget=580
        )
    """
    day: int = Field(..., ge=1, description="第几天")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="日期")
    theme: str = Field(..., min_length=1, description="当天主题")
    weather: WeatherInfo = Field(..., description="当天天气")
    activities: list[Activity] = Field(..., min_length=1, description="活动列表")
    hotel: Location | None = Field(default=None, description="当晚住宿")
    daily_budget: float = Field(default=0.0, ge=0, description="当日预估花费")
    day_notes: str = Field(default="", description="当天特别提示")


class Budget(BaseModel):
    """总预算明细。

    输入:
        total: 总预算（元），必填。
        accommodation: 住宿总花费（元），必填。
        transport: 交通总花费（元），必填。
        food: 餐饮总花费（元），必填。
        tickets: 门票总花费（元），必填。
        shopping: 购物预算（元），可选。
        other: 其他花费（元），可选。
        currency: 货币符号，默认 "¥"。

    约束:
        - total 应约等于各分项之和（允许 ±5% 误差）
        - 各分项均为非负数

    示例:
        Budget(
            total=4200, accommodation=1800, transport=400,
            food=1200, tickets=500, shopping=300,
            currency="¥"
        )
    """
    total: float = Field(..., ge=0, description="总预算")
    accommodation: float = Field(..., ge=0, description="住宿总花费")
    transport: float = Field(..., ge=0, description="交通总花费")
    food: float = Field(..., ge=0, description="餐饮总花费")
    tickets: float = Field(..., ge=0, description="门票总花费")
    shopping: float = Field(default=0.0, ge=0, description="购物预算")
    other: float = Field(default=0.0, ge=0, description="其他花费")
    currency: str = Field(default="¥", description="货币符号")


class EmergencyInfo(BaseModel):
    """紧急实用信息。

    输入:
        police: 报警电话，默认 "110"。
        ambulance: 急救电话，默认 "120"。
        fire: 火警电话，默认 "119"。
        tourism_hotline: 旅游咨询热线，可选。
        hospital: 最近医院名称及地址，可选。
        embassy: 使领馆信息（境外旅行时），可选。
        custom_contacts: 自定义紧急联系人列表，可选。

    示例:
        EmergencyInfo(
            tourism_hotline="010-12301",
            hospital="北京协和医院 东城区帅府园1号"
        )
    """
    police: str = Field(default="110", description="报警电话")
    ambulance: str = Field(default="120", description="急救电话")
    fire: str = Field(default="119", description="火警电话")
    tourism_hotline: str = Field(default="", description="旅游咨询热线")
    hospital: str = Field(default="", description="最近医院")
    embassy: str = Field(default="", description="使领馆信息")
    custom_contacts: list[str] = Field(
        default_factory=list, description="自定义紧急联系人"
    )


class PackingItem(BaseModel):
    """行李清单项。

    输入:
        name: 物品名称，必填。
        category: 分类，必填。
            可选值: "clothing"|"toiletries"|"electronics"|"documents"|"medicine"|"other"
        essential: 是否必需品，默认 False。
        note: 补充说明，可选。

    示例:
        PackingItem(name="防晒霜", category="toiletries",
                    essential=True, note="SPF50+")
    """
    name: str = Field(..., min_length=1, description="物品名称")
    category: str = Field(
        ...,
        pattern=r"^(clothing|toiletries|electronics|documents|medicine|other)$",
        description="分类",
    )
    essential: bool = Field(default=False, description="是否必需品")
    note: str = Field(default="", description="补充说明")


class TripPlan(BaseModel):
    """旅行规划的根模型 —— 完整旅行计划。

    此模型是所有旅行数据的顶层容器。LLM 的输出必须符合此 Schema。
    Jinja2 模板通过此模型的字段渲染 HTML。

    输入:
        city: 目的地城市名，必填。
        start_date: 出发日期（YYYY-MM-DD），必填。
        end_date: 返回日期（YYYY-MM-DD），必填。
        total_days: 总天数（自动计算或手动指定），必填。
        travel_style: 旅行风格/偏好描述（如"深度文化体验"），必填。
        days: 每日行程计划列表，必填。
        weather_overview: 整体天气描述（1-2句），可选。
        budget: 总预算，必填。
        suggestions: 整体旅行建议/贴士列表，必填。
        emergency: 紧急实用信息，必填。
        packing_list: 行李清单，可选。
        xhs_notes: 参考的小红书笔记列表，可选。

    ----- 元数据（由渲染器自动填充，LLM 不需要输出）-----
        trip_id: 行程唯一 ID（渲染时自动生成）。
        generated_at: 生成时间 ISO 字符串（渲染时自动填充）。
        data_sources: 数据来源列表（渲染时自动填充）。
        style: 模板风格 slug（渲染时选择）。

    约束:
        - end_date 必须在 start_date 之后或同一天
        - total_days 应等于 days 列表长度
        - budget.total 约等于各分项之和
    """
    # —— 基础信息 ——
    city: str = Field(..., min_length=1, description="目的地城市")
    start_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="出发日期"
    )
    end_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="返回日期"
    )
    total_days: int = Field(..., ge=1, description="总天数")
    travel_style: str = Field(..., min_length=1, description="旅行风格/偏好")

    # —— 核心数据 ——
    days: list[DayPlan] = Field(..., min_length=1, description="每日行程")
    weather_overview: str = Field(default="", description="整体天气描述")
    budget: Budget = Field(..., description="预算明细")
    suggestions: list[str] = Field(default_factory=list, description="旅行建议")
    emergency: EmergencyInfo = Field(
        default_factory=EmergencyInfo, description="紧急信息"
    )
    packing_list: list[PackingItem] = Field(
        default_factory=list, description="行李清单"
    )
    xhs_notes: list[XHSNote] = Field(
        default_factory=list, description="小红书笔记引用"
    )

    # —— 元数据（渲染器填充） ——
    trip_id: str = Field(default="", description="行程ID（自动生成）")
    generated_at: str = Field(default="", description="生成时间（自动填充）")
    data_sources: list[str] = Field(
        default_factory=list, description="数据来源"
    )
    style: str = Field(default="", description="模板风格slug")

    @field_validator("end_date")
    @classmethod
    def end_date_must_be_after_start(cls, v: str, info: Any) -> str:
        """验证结束日期不早于开始日期。

        输入:
            v: 结束日期字符串。
            info: Pydantic validation info。

        输出:
            验证通过的结束日期。

        异常:
            ValueError: 结束日期早于开始日期时抛出。
        """
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError(f"结束日期({v})不能早于开始日期({start})")
        return v
