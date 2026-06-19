from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable


MemoryQueryIntent = Literal["context", "answer", "timeline", "interest", "procedure"]


# ——枚举类，定义能力形态常量————————————————————————————————————————
class EngineProfile(str, Enum):
    """Memory engine 的能力形态。

    参数:
        无。调用方通过枚举成员使用，例如 EngineProfile.RICH_MEMORY_ENGINE。

    返回:
        EngineProfile 枚举值。
    """

    # 定义四种记忆引擎
    RICH_MEMORY_ENGINE = "rich_memory_engine"
    CLASSIC_MEMORY_SERVICE = "classic_memory_service"
    WORKFLOW_MEMORY_ENGINE = "workflow_memory_engine"
    CONTEXT_RESOURCE_ENGINE = "context_resource_engine"


class MemoryCapability(str, Enum):
    """Memory engine 可声明的能力。

    参数:
        无。调用方通过枚举成员使用，例如 MemoryCapability.RETRIEVE_SEMANTIC。

    返回:
        MemoryCapability 枚举值。
    """

    # 数据录入
    INGEST_TEXT = "ingest.text"
    INGEST_MESSAGES = "ingest.messages"
    INGEST_RESOURCE = "ingest.resource"
    # 内容检索
    RETRIEVE_SEMANTIC = "retrieve.semantic"
    RETRIEVE_CONTEXT_BLOCK = "retrieve.context_block"
    RETRIEVE_STRUCTURED_HITS = "retrieve.structured_hits"
    # 数据管理
    MANAGE_HISTORY = "manage.history"
    MANAGE_UPDATE = "manage.update"
    MANAGE_DELETE = "manage.delete"
    # 图谱关系
    ENRICH_GRAPH_RELATIONS = "enrich.graph_relations"
    # 语义增强
    SEMANTICS_RICH_MEMORY = "semantics.rich_memory"



# ——数据载体类定义，在系统中传递标准化数据——————————————————————————————
@dataclass(frozen=True)
class MemoryScope:
    """一次 memory 操作的作用域。

    参数:
        session_key: 全局会话 key，例如 cli:default。
        channel: 消息来源渠道，例如 cli、telegram。
        chat_id: 渠道内的聊天或用户标识。

    返回:
        MemoryScope 实例。
    """

    session_key: str = ""
    channel: str = ""
    chat_id: str = ""


@dataclass(frozen=True)
class MemoryEngineDescriptor:
    """Memory engine 的描述信息。

    参数:
        name: engine 名称，例如 disabled、memory2。
        profile: engine 能力形态（EngineProfile 枚举值）。
        capabilities: engine 已声明支持的能力集合（MemoryCapability 枚举值）。
        notes: 额外说明，供调试和管理界面展示。

    返回:
        MemoryEngineDescriptor 实例。
    """

    name: str
    profile: EngineProfile
    capabilities: frozenset[MemoryCapability]
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryIngestRequest:
    """提交给 MemoryEngine 的摄入请求。

    参数:
        content: 要摄入的内容，可以是文本、消息列表或结构化 dict。
        source_kind: 内容来源类型，例如 conversation_turn、conversation_batch、external_resource。
        scope: 本次摄入所属作用域。
        hints: 调用方给 engine 的非强制提示。
        metadata: 调用方附带的元数据，例如 source_ref。

    返回:
        MemoryIngestRequest 实例。
    """

    content: object
    source_kind: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    hints: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryIngestResult:
    """MemoryEngine 摄入结果。

    参数:
        accepted: engine 是否接受本次摄入。
        created_ids: 本次摄入新建或影响的 memory id 列表。
        summary: 对本次摄入结果的简短说明。
        raw: engine 私有调试信息。

    返回:
        MemoryIngestResult 实例。
    """

    accepted: bool
    created_ids: list[str] = field(default_factory=list)
    summary: str = ""
    raw: dict[str, object] = field(default_factory=dict)


@dataclass
class EvidenceRef:
    """MemoryRecord 的证据引用。

    参数:
        kind: 证据类型，支持 message、message_range、turn、external。
        refs: 证据 id 列表，例如 message ids。
        resolver: 证据解析器名称，默认由 session 消息系统解析。
        source_ref: engine 内部使用的来源引用。
        metadata: 额外证据元数据。

    返回:
        EvidenceRef 实例。
    """

    kind: Literal["message", "message_range", "turn", "external"] = "message"
    refs: list[str] = field(default_factory=list)
    resolver: str = "session"
    source_ref: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """MemoryEngine 返回的一条标准记忆记录。

    参数:
        id: 记忆条目的稳定 id。
        kind: 记忆类型，例如 event、profile、preference、procedure。
        summary: 可直接展示或注入 prompt 的摘要。
        score: 本次查询中的相关性分数。
        engine_kind: 产生该记录的 engine 名称。
        evidence: 支撑该记忆的证据引用列表。
        signals: engine 附带的排序、热度、标签等信号。
        injected: 该记录是否已进入 prompt 注入块。

    返回:
        MemoryRecord 实例。
    """

    id: str
    kind: str
    summary: str
    score: float
    engine_kind: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    signals: dict[str, object] = field(default_factory=dict)
    injected: bool = False


@dataclass(frozen=True)
class MemoryQueryFilters:
    """MemoryQuery 的过滤条件。

    参数:
        kinds: 限定记忆类型；空 tuple 表示不限定。
        time_start: 时间范围起点，主要给 timeline / event 检索使用。
        time_end: 时间范围终点，主要给 timeline / event 检索使用。
        hints: 查询提示，例如 require_scope_match。

    返回:
        MemoryQueryFilters 实例。
    """

    kinds: tuple[str, ...] = ()
    time_start: datetime | None = None
    time_end: datetime | None = None
    hints: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    # @dataclass自动进行了__init__进行初始化，并会在最后自动去寻找类里有没有定义 __post_init__
    # 如果有，它就会调用这个方法，处理基础属性赋值完成后，立刻需要执行的额外逻辑
    def __post_init__(self) -> None:
        """清理 kinds 并冻结 hints。

        参数:
            无。dataclass 初始化后自动调用。

        返回:
            None。
        """

        # MappingProxyType 可以把 dict 包成只读 mapping，防止后续修改
        """
        from types import MappingProxyType

        data = MappingProxyType({"a": 1})
        assert data["a"] == 1
        data["b"] = 2  # TypeError
        """

        # frozen=True 已经禁止了整个实例的修改，直接 self.kinds = ... 会报错
        # 但在 __post_init__ 中我们可以使用 object.__setattr__ 来修改属性值
        # 而 object.__setattr__(self, "属性名", 值) 则是直接绕过了当前类的限制，
        # 强行调用最底层的基础机制来给属性赋值
        
        # 1. 清理 kinds，去掉空字符串并转换成 tuple
        object.__setattr__(
            self,
            "kinds",
            tuple(
                value
                for item in self.kinds
                if (value := str(item).strip())
            ),
        )
        # 2. 冻结 hints，转换成 MappingProxyType 包装的只读 mapping
        # 先 dict（字典拷贝）的理由：1. 即使原本的 hints 被修改了，也不会影响到我们存储的内容；2. 统一为标准 dict 类型，防止调用方传入的 Mapping 类型不兼容。
        object.__setattr__(self, "hints", MappingProxyType(dict(self.hints)))


@dataclass
class MemoryQuery:
    """提交给 MemoryEngine 的查询请求。

    参数:
        text: 查询文本。
        intent: 查询意图，决定 engine 选择 context、answer、timeline 等检索策略。
        scope: 查询作用域。
        filters: 类型、时间和 hint 过滤条件。
        context: 调用方附带的运行时上下文。
        limit: 最多返回多少条结构化记录。

    返回:
        MemoryQuery 实例。
    """

    text: str
    intent: MemoryQueryIntent = "answer"
    scope: MemoryScope = field(default_factory=MemoryScope)
    filters: MemoryQueryFilters = field(default_factory=MemoryQueryFilters)
    context: dict[str, object] = field(default_factory=dict)
    limit: int = 8


@dataclass
class MemoryQueryResult:
    """MemoryEngine 查询结果。

    参数:
        text_block: 可直接注入 prompt 的文本块。
        records: 结构化记忆记录列表。
        trace: 查询过程追踪信息。
        raw: engine 私有原始结果。

    返回:
        MemoryQueryResult 实例。
    """

    text_block: str = ""
    records: list[MemoryRecord] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryMutation:
    """提交给 MemoryEngine 的显式变更请求。

    参数:
        kind: 变更类型；remember 表示新增记忆，update 表示修正已有记忆，forget 表示语义遗忘，restore 表示恢复已遗忘记忆。
        scope: 变更所属作用域。
        summary: remember / update 时要写入的新记忆摘要。
        memory_kind: 调用方建议的记忆类型。
        source_ref: 记忆来源引用。
        ids: update / forget / restore 时目标 memory id 列表。
        metadata: 额外变更参数，例如 steps、tool_requirement、reason。

    返回:
        MemoryMutation 实例。
    """

    kind: Literal["remember", "update", "forget", "restore"]
    scope: MemoryScope = field(default_factory=MemoryScope)
    summary: str = ""
    memory_kind: str = ""
    source_ref: str = ""
    ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """清理 ids 并冻结 metadata。

        参数:
            无。dataclass 初始化后自动调用。

        返回:
            None。
        """
        
        # 1. 清理 ids，去掉空字符串并转换成 tuple
        object.__setattr__(
            self,
            "ids",
            tuple(item for raw in self.ids if (item := str(raw).strip())),
        )
        # 2. 冻结 metadata，转换成 MappingProxyType 包装的只读 mapping
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass
class MemoryMutationResult:
    """MemoryEngine 显式变更结果。

    参数:
        accepted: engine 是否接受本次变更。
        item_id: remember 时主要影响的 memory id。
        actual_kind: engine 最终采用的记忆类型。
        status: 写入状态，例如 new、merged、superseded、disabled。
        affected_ids: 本次变更影响到的现有 memory ids。
        missing_ids: forget 时没有找到的 ids。
        items: 本次变更涉及的条目简表。
        raw: engine 私有原始结果。

    返回:
        MemoryMutationResult 实例。
    """

    accepted: bool
    item_id: str = ""
    actual_kind: str = ""
    status: str = ""
    affected_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    items: list[dict[str, object]] = field(default_factory=list)
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryToolSpec:
    """MemoryEngine 建议注册给模型的工具描述。

    参数:
        description: 工具说明。
        parameters: 工具 JSON Schema 参数。
        risk: 工具风险等级。
        search_hint: deferred tool search 使用的搜索提示。
        tool_class: 可选自定义工具类；为空时后续使用内置标准工具。

    返回:
        MemoryToolSpec 实例。
    """

    description: str
    parameters: dict[str, object]
    risk: Literal["read-only", "write", "external-side-effect"] = "read-only"
    search_hint: str = ""
    tool_class: type | None = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class MemoryToolProfile:
    """MemoryEngine 对 remember / recall / forget 工具的建议配置。

    参数:
        recall: recall_memory 工具配置；None 表示不注册。
        memorize: remember/memorize 工具配置；None 表示不注册。
        forget: forget_memory 工具配置；None 表示不注册。

    返回:
        MemoryToolProfile 实例。
    """

    recall: MemoryToolSpec | None = None
    memorize: MemoryToolSpec | None = None
    forget: MemoryToolSpec | None = None



# ——接口协议类 (Protocol) - 定义引擎必须实现的方法契约————————————————————
# Protocol 是 Python 的一种接口定义方式，允许我们定义一个类应该有哪些方法，但不提供具体实现。
# 通过 @runtime_checkable 装饰器，我们可以在运行时使用 isinstance() 来检查一个对象是否符合这个协议。
"""
class HasName(Protocol):
    def name(self) -> str: ...

class User:
    def name(self) -> str:
        return "Raven"

assert isinstance(User(), HasName)

但默认情况下，isinstance(obj, ProtocolClass) 不能运行。
需要加：

@runtime_checkable
class HasName(Protocol):
    def name(self) -> str: ...

此时isinstance(obj, ProtocolClass) 可以运行。
"""

@runtime_checkable
class MemoryIngestApi(Protocol):
    """MemoryEngine 的摄入能力协议。

    参数:
        实现类需要提供 ingest(request) 方法。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        """摄入对话、文本或资源。

        参数:
            request: 摄入请求。

        返回:
            MemoryIngestResult。
        """

        ...


@runtime_checkable
class MemoryRetrievalApi(Protocol):
    """MemoryEngine 的查询能力协议。

    参数:
        实现类需要提供 query(request) 方法。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        """查询结构化或语义记忆。

        参数:
            request: 查询请求。

        返回:
            MemoryQueryResult。
        """

        ...


@runtime_checkable
class MemoryWriteApi(Protocol):
    """MemoryEngine 的显式写入能力协议。

    参数:
        实现类需要提供 mutate() 和 reinforce_items_batch()。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        """执行 remember / forget 等显式变更。

        参数:
            request: 变更请求。

        返回:
            MemoryMutationResult。
        """

        ...

    def reinforce_items_batch(self, ids: list[str]) -> None:
        """批量强化被使用过的记忆条目。

        参数:
            ids: 要强化的 memory id 列表。

        返回:
            None。
        """

        ...


@runtime_checkable
class MemoryAdminApi(Protocol):
    """MemoryEngine 的管理能力协议。

    参数:
        实现类需要提供 describe、tool_profile 和 Dashboard 管理方法。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    def describe(self) -> MemoryEngineDescriptor:
        """返回 engine 描述。

        参数:
            无。

        返回:
            MemoryEngineDescriptor。
        """

        ...

    def tool_profile(self) -> MemoryToolProfile:
        """返回 memory tools 的建议 profile。

        参数:
            无。

        返回:
            MemoryToolProfile。
        """

        ...

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        """按关键词匹配 procedure 记忆。

        参数:
            action_tokens: 从用户意图或工具名中切出的动作词。

        返回:
            匹配到的 procedure 简表列表。
        """

        ...

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """列出某个时间范围内的 event 记忆。

        参数:
            time_start: 时间范围起点。
            time_end: 时间范围终点。
            limit: 最多返回多少条。

        返回:
            event 记忆简表列表。
        """

        ...

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        """给 Dashboard 列出 memory items。

        参数:
            q: 文本搜索关键字。
            memory_type: 记忆类型过滤。
            status: 状态过滤，例如 active、superseded。
            source_ref: 来源引用过滤。
            scope_channel: 渠道过滤。
            scope_chat_id: 聊天标识过滤。
            has_embedding: 是否要求存在 embedding。
            page: 页码，从 1 开始。
            page_size: 每页条数。
            sort_by: 排序字段。
            sort_order: asc 或 desc。

        返回:
            二元组：(当前页 items, 总条数)。
        """

        ...

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        """给 Dashboard 读取单条 memory item。

        参数:
            item_id: memory id。
            include_embedding: 是否返回 embedding 原始向量。

        返回:
            memory item 字典；不存在时返回 None。
        """

        ...

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        """给 Dashboard 更新单条 memory item。

        参数:
            item_id: memory id。
            status: 新状态。
            extra_json: 新的额外 JSON 字段。
            source_ref: 新来源引用。
            happened_at: 新发生时间。
            emotional_weight: 新情绪权重。

        返回:
            更新后的 memory item；不存在时返回 None。
        """

        ...

    def delete_item(self, item_id: str) -> bool:
        """物理删除单条 memory item。

        参数:
            item_id: memory id。

        返回:
            删除成功返回 True，否则返回 False。
        """

        ...

    def delete_items_batch(self, ids: list[str]) -> int:
        """批量物理删除 memory items。

        参数:
            ids: memory id 列表。

        返回:
            实际删除条数。
        """

        ...

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        """给 Dashboard 查找相似 memory items。

        参数:
            item_id: 作为相似度查询基准的 memory id。
            top_k: 最多返回多少条。
            memory_type: 限定相似条目的记忆类型。
            score_threshold: 最低相似分数。
            include_superseded: 是否包含已失效条目。

        返回:
            相似 memory item 简表列表。
        """

        ...

# 组合协议，继承了上述四大 API 协议，是引擎的最终接口形态
@runtime_checkable
class MemoryEngine(
    MemoryIngestApi,
    MemoryRetrievalApi,
    MemoryWriteApi,
    MemoryAdminApi,
    Protocol,
):
    """raven-agent 的结构化语义记忆引擎协议。

    参数:
        实现类必须同时满足 ingest、query、mutate 和 admin 能力协议。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    pass


class DisabledMemoryEngine:
    """语义记忆关闭时使用的空 MemoryEngine。
    
    实现了 MemoryEngine 协议的所有方法，
    但所有的操作都是拒绝写入、返回空数据或直接返回 False，
    以确保系统在缺少记忆引擎时不会报错退出。

    参数:
        无。

    返回:
        DisabledMemoryEngine 实例。
    """

    # 本 Engine 的描述信息，供系统查询和管理界面展示
    DESCRIPTOR = MemoryEngineDescriptor(
        name="disabled",
        profile=EngineProfile.CONTEXT_RESOURCE_ENGINE,
        capabilities=frozenset(),
        notes={"reason": "semantic memory disabled"},
    )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        """拒绝摄入请求。

        参数:
            request: 摄入请求。

        返回:
            accepted=False 的 MemoryIngestResult。
        """

        return MemoryIngestResult(
            accepted=False,
            summary="semantic memory disabled",
            raw={"reason": "disabled", "source_kind": request.source_kind},
        )

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        """返回空查询结果。

        参数:
            request: 查询请求。

        返回:
            不含 records 的 MemoryQueryResult，并在 trace 标记 disabled。
        """

        return MemoryQueryResult(
            trace={
                "engine": self.DESCRIPTOR.name,
                "intent": request.intent,
                "reason": "disabled",
            }
        )

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        """拒绝 remember / update / forget / restore 请求。

        参数:
            request: 变更请求。

        返回:
            accepted=False 的 MemoryMutationResult。
        """

        if request.kind in {"forget", "restore", "update"}:
            return MemoryMutationResult(
                accepted=False,
                status="disabled",
                missing_ids=list(request.ids),
            )
        return MemoryMutationResult(accepted=False, status="disabled")

    def reinforce_items_batch(self, ids: list[str]) -> None:
        """disabled engine 不强化任何条目。

        参数:
            ids: 要强化的 memory id 列表。

        返回:
            None。
        """

        return None

    def describe(self) -> MemoryEngineDescriptor:
        """返回 disabled engine 描述。

        参数:
            无。

        返回:
            MemoryEngineDescriptor。
        """

        return self.DESCRIPTOR

    def tool_profile(self) -> MemoryToolProfile:
        """返回空 memory tool profile。

        参数:
            无。

        返回:
            recall、memorize、forget 均为 None 的 MemoryToolProfile。
        """

        return MemoryToolProfile()

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        """disabled engine 不返回 procedure 匹配。

        参数:
            action_tokens: 动作词列表。

        返回:
            空列表。
        """

        return []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """disabled engine 不返回 event。

        参数:
            time_start: 时间范围起点。
            time_end: 时间范围终点。
            limit: 最多返回多少条。

        返回:
            空列表。
        """

        return []

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        """disabled engine 不返回 Dashboard items。

        参数:
            q: 文本搜索关键字。
            memory_type: 记忆类型过滤。
            status: 状态过滤。
            source_ref: 来源引用过滤。
            scope_channel: 渠道过滤。
            scope_chat_id: 聊天标识过滤。
            has_embedding: 是否要求存在 embedding。
            page: 页码。
            page_size: 每页条数。
            sort_by: 排序字段。
            sort_order: 排序方向。

        返回:
            二元组：空列表和总数 0。
        """

        return [], 0

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        """disabled engine 不返回单条 item。

        参数:
            item_id: memory id。
            include_embedding: 是否包含 embedding。

        返回:
            None。
        """

        return None

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        """disabled engine 不更新 item。

        参数:
            item_id: memory id。
            status: 新状态。
            extra_json: 新的额外 JSON 字段。
            source_ref: 新来源引用。
            happened_at: 新发生时间。
            emotional_weight: 新情绪权重。

        返回:
            None。
        """

        return None

    def delete_item(self, item_id: str) -> bool:
        """disabled engine 不删除 item。

        参数:
            item_id: memory id。

        返回:
            False。
        """

        return False

    def delete_items_batch(self, ids: list[str]) -> int:
        """disabled engine 不批量删除 item。

        参数:
            ids: memory id 列表。

        返回:
            0。
        """

        return 0

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        """disabled engine 不返回相似 item。

        参数:
            item_id: 作为相似度查询基准的 memory id。
            top_k: 最多返回多少条。
            memory_type: 限定相似条目的记忆类型。
            score_threshold: 最低相似分数。
            include_superseded: 是否包含已失效条目。

        返回:
            空列表。
        """

        return []