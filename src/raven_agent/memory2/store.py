from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import sqlite_vec

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None
    _SQLITE_VEC_AVAILABLE = False

from raven_agent.memory2.models import (
    MemoryItem,
    clamp_emotional_weight,
    content_hash,
    normalize_memory_type,
    normalize_status,
    now_iso,
)

logger = logging.getLogger(__name__)

# ——数据库表结构设计——————————————————————————————————————————————————————————————————
"""
1. 核心记忆表 memory_items：存储所有的记忆条目和对应的向量
- id: 记忆的唯一ID， 主键
- memory_type: 记忆类型，字符串
- summary: 记忆摘要，字符串
- content_hash: 摘要内容哈希，字符串，memory_type + summary 的哈希值，用于去重
- embedding: 摘要向量，JSON 字符串
- reinforcement: 强化次数，整数，（被提取/回想起的次数），默认 1
- emotional_weight: 情绪权重，整数，（用于评估记忆的重要程度），默认 0
- extra_json: 扩展元数据，JSON 字符串，类型相关的额外信息
- source_ref: 来源引用，字符串（这条记忆从哪来的）
- happened_at: 事件发生时间，字符串 ISO 格式
- status: 条目状态，字符串，active 或 superseded，默认 active
- created_at: 创建时间，字符串 ISO 格式
- updated_at: 更新时间，字符串 ISO 格式

-ux_memory_items_hash_type：联合唯一索引，确保同一类型下不会有完全重复的记忆内容
-ix_memory_items_type_status：memory_type + status 索引，优化按类型和状态查询
-ix_memory_items_source_ref：source_ref 索引，优化按来源查询

2. 记忆巩固事件表 consolidation_events ：用于实现写入的“幂等性”（防止同一来源重复写入）

3. 记忆更迭表 memory_replacements ：记录记忆的演进历史（比如旧观念被新观念取代）
-为新旧 ID 建立索引 ix_memory_replacements_old_item 和 ix_memory_replacements_new_item，
 方便追溯记忆的演化图谱
"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id                TEXT PRIMARY KEY,
    memory_type       TEXT NOT NULL,
    summary           TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    embedding         TEXT,
    reinforcement     INTEGER NOT NULL DEFAULT 1,
    emotional_weight  INTEGER NOT NULL DEFAULT 0,
    extra_json        TEXT,
    source_ref        TEXT,
    happened_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_items_hash_type
    ON memory_items (content_hash, memory_type);
CREATE INDEX IF NOT EXISTS ix_memory_items_type_status
    ON memory_items (memory_type, status);
CREATE INDEX IF NOT EXISTS ix_memory_items_source_ref
    ON memory_items (source_ref);

CREATE TABLE IF NOT EXISTS consolidation_events (
    source_ref  TEXT PRIMARY KEY,
    item_id     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_replacements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    old_item_id       TEXT NOT NULL,
    old_memory_type   TEXT NOT NULL,
    old_summary       TEXT NOT NULL,
    old_source_ref    TEXT,
    old_happened_at   TEXT,
    old_extra_json    TEXT,
    new_item_id       TEXT NOT NULL,
    new_memory_type   TEXT NOT NULL,
    new_summary       TEXT NOT NULL,
    new_source_ref    TEXT,
    new_happened_at   TEXT,
    new_extra_json    TEXT,
    relation_type     TEXT NOT NULL DEFAULT 'supersede',
    source_ref        TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_replacements_old_item
    ON memory_replacements (old_item_id, created_at);
CREATE INDEX IF NOT EXISTS ix_memory_replacements_new_item
    ON memory_replacements (new_item_id, created_at);
"""

def _build_fts_schema(tokenizer: str) -> str:
    """生成 FTS5 虚拟表及触发器的 DDL。

    参数:
        tokenizer: FTS5 tokenizer 名称，如 'simple' 或 'trigram'。

    返回:
        完整的 FTS5 DDL 字符串。
    """
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    summary,
    content='memory_items',
    content_rowid='rowid',
    tokenize='{tokenizer}'
);

CREATE TRIGGER IF NOT EXISTS memory_items_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, summary)
    VALUES('delete', old.rowid, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_au AFTER UPDATE OF summary ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, summary)
    VALUES('delete', old.rowid, old.summary);
    INSERT INTO memory_items_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
"""

# ——辅助工具函数 (序列化与数学计算)————————————————————————————————————————————————————
def _json_dumps(value: dict[str, object] | None) -> str | None:
    """把 dict 序列化为 JSON 字符串。

    参数:
        value: 要序列化的字典；None 或空字典返回 None。

    返回:
        JSON 字符串或 None。
    """

    if not value:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_object(raw: object) -> dict[str, object]:
    """把 SQLite 中的 JSON 字符串解析为 dict。

    参数:
        raw: SQLite row 中的 extra_json 字段。

    返回:
        解析后的字典；解析失败或不是对象时返回空字典。
    """

    if not raw:
        return {}
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}


def _embedding_dumps(embedding: list[float] | None) -> str | None:
    """把 embedding 序列化为 JSON 字符串。

    参数:
        embedding: 浮点向量；None 或空列表返回 None。

    返回:
        JSON 字符串或 None。
    """

    if not embedding:
        return None
    return json.dumps([float(value) for value in embedding])


def _embedding_loads(raw: object) -> list[float] | None:
    """把 SQLite 中的 embedding 字符串解析为浮点向量。

    参数:
        raw: SQLite row 中的 embedding 字段。

    返回:
        浮点向量；解析失败或不是数组时返回 None。
    """

    if not raw:
        return None
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, list):
        return None
    return [float(value) for value in loaded if isinstance(value, int | float)]


# 向量相似度计算
def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的 cosine similarity。

    取值范围 [-1, 1]，值越接近 1 代表语义越相似

    参数:
        left: 左侧浮点向量。
        right: 右侧浮点向量。

    返回:
        cosine similarity；维度不一致或空向量时返回 0.0。
    """

    if not left or not right or len(left) != len(right):
        return 0.0
    left_arr = np.array(left, dtype=np.float32)
    right_arr = np.array(right, dtype=np.float32)
    denom = float(np.linalg.norm(left_arr)) * float(np.linalg.norm(right_arr)) + 1e-9
    return float(left_arr @ right_arr) / denom


def _normalize_embedding(embedding: list[float]) -> list[float]:
    """对 embedding 做 L2 归一化。

    参数:
        embedding: 原始浮点向量。

    返回:
        归一化后的浮点向量；零向量原样返回。
    """

    vector = np.array(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return [float(value) for value in embedding]
    return cast(list[float], (vector / norm).tolist())


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """把 embedding 打包为 sqlite-vec 使用的 float32 blob。

    参数:
        embedding: 原始浮点向量。

    返回:
        float32 二进制 blob。
    """

    normalized = _normalize_embedding(embedding)
    return struct.pack(f"{len(normalized)}f", *normalized)


def _l2_distance_to_cosine(distance: float) -> float:
    """把单位向量 L2 距离转换成 cosine similarity。

    参数:
        distance: sqlite-vec 返回的 L2 distance。

    返回:
        cosine similarity。
    """

    return 1.0 - (distance * distance) / 2.0


# 热度分
def _parse_datetime(value: object) -> datetime | None:
    """把 ISO 时间字符串解析为 datetime。

    参数:
        value: ISO 时间字符串或其他对象。

    返回:
        datetime；无法解析时返回 None。
    """

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hotness_score(
    *,
    reinforcement: int,
    updated_at: str,
    emotional_weight: int,
    half_life_days: float,
) -> float:
    """计算 memory item 的热度分。

    参数:
        reinforcement: 条目强化次数。
        updated_at: 条目最近更新时间 ISO 字符串。
        emotional_weight: 情绪权重，0 到 10。
        half_life_days: 时间衰减半衰期天数。

    返回:
        0 到 1 附近的热度分。
    """

    updated = _parse_datetime(updated_at)
    if updated is None:
        return 0.0
    now = datetime.now(timezone.utc)
    age_days = max((now - updated).total_seconds() / 86400.0, 0.0)
    effective_half_life = max(
        float(half_life_days) * (1.0 + 0.5 * clamp_emotional_weight(emotional_weight) / 10.0),
        0.1,
    )
    frequency = 1.0 / (1.0 + math.exp(-math.log1p(max(0, int(reinforcement)))))
    recency = math.exp(-math.log(2) / effective_half_life * age_days)
    return frequency * recency


def _blend_score(
    *,
    semantic: float,
    reinforcement: int,
    updated_at: str,
    emotional_weight: int,
    hotness_alpha: float,
    hotness_half_life_days: float,
) -> tuple[float, dict[str, float]]:
    """融合语义分和热度分。

    参数:
        semantic: cosine similarity。
        reinforcement: 条目强化次数。
        updated_at: 条目最近更新时间。
        emotional_weight: 情绪权重。
        hotness_alpha: 热度权重；0 表示只用 semantic。
        hotness_half_life_days: 热度半衰期。

    返回:
        二元组：(最终分数, 调试分数字典)。
    """

    alpha = max(0.0, min(1.0, float(hotness_alpha)))
    hotness = 0.0
    if alpha > 0.0:
        hotness = _hotness_score(
            reinforcement=reinforcement,
            updated_at=updated_at,
            emotional_weight=emotional_weight,
            half_life_days=hotness_half_life_days,
        )
    final = (1.0 - alpha) * semantic + alpha * hotness
    debug = {
        "semantic": round(semantic, 4),
        "hotness": round(hotness, 4),
        "final": round(final, 4),
    }
    return final, debug

def _new_item_id(summary: str, memory_type: str) -> str:
    """生成新的 memory item id。

    参数:
        summary: 记忆摘要。
        memory_type: 记忆类型。

    返回:
        12 位十六进制 id。
    """
    # 利用类型、摘要内容和当前时间戳作为种子，生成 MD5 哈希
    # 截取前 12 位作为记忆的简短 UUID
    seed = f"{memory_type}:{summary}:{time.time_ns()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    """把 sqlite3.Row 转换成 MemoryItem。

    参数:
        row: SELECT memory_items 得到的一行。

    返回:
        MemoryItem 实例。
    """

    # ORM 映射：将数据库查出来的 row 转换为 Python 数据类 MemoryItem
    return MemoryItem(
        id=str(row["id"]),
        memory_type=str(row["memory_type"]),
        summary=str(row["summary"]),
        content_hash=str(row["content_hash"]),
        embedding=_embedding_loads(row["embedding"]),
        reinforcement=int(row["reinforcement"]),
        extra_json=_json_object(row["extra_json"]),
        source_ref=cast(str | None, row["source_ref"]),
        happened_at=cast(str | None, row["happened_at"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        emotional_weight=int(row["emotional_weight"]),
    )


def _row_to_hit(
    row: sqlite3.Row,
    *,
    score: float,
    score_debug: dict[str, float] | None = None,
    keyword_score: float | None = None,
) -> dict[str, object]:
    """把 memory_items row 转换为检索 hit 字典。

    参数:
       row: memory_items 查询结果。
        score: 最终排序分数。
        score_debug: 可选分数拆解。
        keyword_score: 可选关键词命中分。

    返回:
        Retriever 使用的 hit 字典。
    """

    hit: dict[str, object] = {
        "id": str(row["id"]),
        "memory_type": str(row["memory_type"]),
        "summary": str(row["summary"]),
        "score": round(float(score), 4),
        "source_ref": str(row["source_ref"] or ""),
        "happened_at": str(row["happened_at"] or ""),
        "status": str(row["status"]),
        "extra_json": _json_object(row["extra_json"]),
        "reinforcement": int(row["reinforcement"]),
        "emotional_weight": int(row["emotional_weight"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }
    if score_debug is not None:
        hit["_score_debug"] = score_debug
    if keyword_score is not None:
        hit["keyword_score"] = round(float(keyword_score), 4)
    return hit

# ——类初始化与连接管理——————————————————————————————————————————————————————————————
class MemoryStore2:
    """Memory2 的 SQLite 存储层。

    参数:
        db_path: SQLite 数据库文件路径。
        vec_dim: sqlite-vec 使用的向量维度；0 表示运行时按首条 embedding 推断。

    返回:
        MemoryStore2 实例。
    """

    def __init__(self, db_path: str | Path, vec_dim: int = 0) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._vec_dim = max(0, int(vec_dim))
        self._vec_enabled = False
        self._vec_init_error = ""
        self._vec_fallback_logged = False
        self._has_fts = False
        self._simple_loaded = False

        # --- 新增：先加载外部扩展，再执行 schema 初始化 ---
        self._load_simple_extension()

        self.ensure_schema()

    
    def _load_simple_extension(self) -> None:
        """加载 sqlite-simple (jieba) 分词插件并动态设置词典路径。"""
        # 利用 pathlib 动态获取当前 store.py 文件所在目录
        current_dir = Path(__file__).parent
        ext_path = current_dir / "ext" / "libsimple.so"
        dict_path = current_dir / "ext" / "dict"

        if not ext_path.exists() or not dict_path.exists():
            logger.warning("未能找到 libsimple.so 或 dict 目录，FTS jieba 分词插件无法加载。")
            return

        try:
            self._db.enable_load_extension(True)
            self._db.load_extension(str(ext_path))
            
            # 【正确做法】：调用 simple 插件提供的 jieba_dict() 函数，传入词典的绝对路径
            self._db.execute("SELECT jieba_dict(?)", (str(dict_path.absolute()),))
            
            self._db.enable_load_extension(False)
            self._simple_loaded = True
            logger.info("已成功加载 sqlite-simple 分词插件。")
        except AttributeError:
            logger.error("当前 Python 环境的 sqlite3 模块禁用了 enable_load_extension。")
        except sqlite3.OperationalError as exc:
            logger.error("加载 sqlite-simple 插件失败: %s", exc)
    
    
    def ensure_schema(self) -> None:
        """确保 SQLite schema、FTS5 和 sqlite-vec 已初始化。

        参数:
            无。

        返回:
            None。
        """

        with self._lock:
            self._db.executescript(SCHEMA)
            self._ensure_status_indexes()
            self._ensure_fts()
            self._db.commit()
            if self._vec_dim > 0:
                self._ensure_vec_table(self._vec_dim)
            elif not self._vec_enabled and _SQLITE_VEC_AVAILABLE and sqlite_vec is not None:
                # 进程重启后 _vec_enabled 归零，但 vec_items 虚拟表可能已持久在 db 中。
                # 检查 sqlite_master 确认虚拟表存在，如有则恢复启用状态。
                self._recover_vec_if_exists()
    
    def _ensure_status_indexes(self) -> None:
        """确保 Memory2 查询需要的普通索引存在。

        参数:
            无。

        返回:
            None。
        """

        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_memory_items_status ON memory_items (status)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_memory_items_updated_at ON memory_items (updated_at)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_memory_items_happened_at ON memory_items (happened_at)"
        )
    
    def _ensure_fts(self) -> None:
        """尽力创建 memory_items 的 FTS5 索引。

        参数:
            无。

        返回:
            None。当前 SQLite 不支持 FTS5 时仅把 _has_fts 设为 False。
        """

        tokenizer = 'simple' if self._simple_loaded else 'trigram'
        try:
            self._db.executescript(_build_fts_schema(tokenizer))
            self._has_fts = True
        except sqlite3.OperationalError as exc:
            self._has_fts = False
            logger.warning("memory2 FTS5 初始化失败，keyword lane 将使用 LIKE fallback: %s", exc)
    
    def _ensure_vec_table(self, dim: int) -> None:
        """尽力初始化 sqlite-vec 虚拟表。

        参数:
            dim: embedding 维度。

        返回:
            None。初始化失败时自动禁用 sqlite-vec。
        """

        if dim <= 0 or self._vec_enabled:
            return
        if not _SQLITE_VEC_AVAILABLE or sqlite_vec is None:
            self._vec_init_error = "sqlite_vec 未安装"
            return
        try:
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{dim}])"
            )
            self._db.commit()
            self._vec_dim = dim
            self._vec_enabled = True
            self._migrate_existing_to_vec()
        except Exception as exc:
            self._vec_enabled = False
            self._vec_init_error = str(exc)
            logger.warning("sqlite-vec 初始化失败，回退到 numpy fullscan: %s", exc)

    def _recover_vec_if_exists(self) -> None:
        """进程重启后恢复 vec 启用状态。

        vec_items 虚拟表持久在 db 文件中，但 _vec_enabled 是内存标志位，
        重启后会归零。本方法检查虚拟表是否已存在，若存在则加载 sqlite-vec
        扩展并从建表语句解析维度，恢复 _vec_enabled / _vec_dim。

        参数:
            无。

        返回:
            None。
        """

        try:
            row = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_items'"
            ).fetchone()
        except Exception:
            row = None
        if row is None:
            return

        try:
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
            # 从已存虚拟表解析维度：从 vec_items 取一条 embedding 的 blob 长度反推
            dim_row = self._db.execute(
                "SELECT embedding FROM vec_items LIMIT 1"
            ).fetchone()
            if dim_row:
                blob = dim_row[0]
                if isinstance(blob, bytes):
                    self._vec_dim = len(blob) // 4  # float32 = 4 bytes
                elif isinstance(blob, list):
                    self._vec_dim = len(blob)
            if self._vec_dim <= 0:
                return
            self._vec_enabled = True
            logger.info("sqlite-vec 已从持久化虚拟表恢复，维度=%d", self._vec_dim)
        except Exception:
            pass  # 恢复失败静默跳过，走 fallback
    
    def _migrate_existing_to_vec(self) -> None:
        """把已有 memory_items.embedding 同步到 vec_items。

        参数:
            无。

        返回:
            None。
        """

        if not self._vec_enabled or self._vec_dim <= 0:
            return
        existing = {int(row[0]) for row in self._db.execute("SELECT rowid FROM vec_items").fetchall()}
        rows = self._db.execute(
            "SELECT rowid, embedding FROM memory_items WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            rowid = int(row["rowid"])
            if rowid in existing:
                continue
            embedding = _embedding_loads(row["embedding"])
            if embedding is None or len(embedding) != self._vec_dim:
                continue
            self._db.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (rowid, _embedding_to_blob(embedding)),
            )
        self._db.commit()
    
    def _ensure_vec_dim_for_embedding(self, embedding: list[float] | None) -> None:
        """根据写入 embedding 的维度初始化 sqlite-vec。

        参数:
            embedding: 即将写入的 embedding；为空时不处理。

        返回:
            None。
        """

        if not embedding or self._vec_enabled:
            return
        if self._vec_dim <= 0:
            self._vec_dim = len(embedding)
        if len(embedding) == self._vec_dim:
            self._ensure_vec_table(self._vec_dim)
    
    def _vec_insert(self, rowid: int, embedding: list[float] | None) -> None:
        """把一条 memory item embedding 同步到 vec_items。

        参数:
            rowid: memory_items 的 SQLite rowid。
            embedding: 要同步的 embedding。

        返回:
            None。
        """

        if not embedding:
            return
        self._ensure_vec_dim_for_embedding(embedding)
        if not self._vec_enabled or len(embedding) != self._vec_dim:
            return
        try:
            self._db.execute("DELETE FROM vec_items WHERE rowid=?", (rowid,))
            self._db.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (rowid, _embedding_to_blob(embedding)),
            )
        except Exception as exc:
            logger.warning("sqlite-vec 同步 rowid=%s 失败: %s", rowid, exc)
    
    def _vec_delete(self, rowids: list[int]) -> None:
        """从 vec_items 删除一批 rowid。

        参数:
            rowids: memory_items 的 SQLite rowid 列表。

        返回:
            None。
        """

        if not self._vec_enabled or not rowids:
            return
        try:
            self._db.executemany("DELETE FROM vec_items WHERE rowid=?", [(rowid,) for rowid in rowids])
        except Exception as exc:
            logger.warning("sqlite-vec 删除失败: %s", exc)

    def close(self) -> None:
        """关闭 SQLite 连接。

        参数:
            无。

        返回:
            None。
        """

        if self._closed:
            return
        self._db.close()
        self._closed = True


    # 核心摄入逻辑 (Upsert & 幂等控制)
    def upsert_item(
        self,
        *,
        memory_type: str,
        summary: str,
        embedding: list[float] | None,
        source_ref: str | None = None,
        extra_json: dict[str, object] | None = None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """写入或强化一条 memory item。

        参数:
            memory_type: 记忆类型。
            summary: 记忆摘要。
            embedding: 摘要 embedding；没有时传 None。
            source_ref: 来源引用。
            extra_json: 类型专用扩展字段。
            happened_at: 事件发生时间。
            emotional_weight: 情绪权重。

        返回:
            形如 new:{id} 或 reinforced:{id} 的结果字符串。
        """

        normalized_type = normalize_memory_type(memory_type)
        normalized_summary = summary.strip()
        if not normalized_summary:
            return "skipped:empty"
        item_hash = content_hash(normalized_summary, normalized_type)
        now = now_iso()
        weight = clamp_emotional_weight(emotional_weight)

        with self._lock:
            # 1. 查询数据库中是否已经有完全相同的内容
            existing = self._db.execute(
                "SELECT id, status FROM memory_items WHERE content_hash=? AND memory_type=?",
                (item_hash, normalized_type),
            ).fetchone()
            if existing is not None:
                # 2. 如果已存在，触发“记忆强化”逻辑 (Upsert 的 Update 阶段)
                item_id = str(existing["id"])
                # 状态改为 active，reinforcement 加 1，情绪权重取最大，更新时间改为 now，来源和发生时间如果有新值也更新
                self._db.execute(
                    """
                    UPDATE memory_items
                    SET status='active',
                        reinforcement=reinforcement + 1,
                        emotional_weight=MAX(emotional_weight, ?),
                        updated_at=?,
                        source_ref=COALESCE(NULLIF(?, ''), source_ref),
                        happened_at=COALESCE(NULLIF(?, ''), happened_at)
                    WHERE id=?
                    """,
                    (weight, now, source_ref or "", happened_at or "", item_id),
                )
                if embedding:
                    row = self._db.execute(
                        "SELECT rowid FROM memory_items WHERE id=?",
                        (item_id,),
                    ).fetchone()
                    if row is not None:
                        self._vec_insert(int(row["rowid"]), embedding)
                self._db.commit()
                return f"reinforced:{item_id}"

            # 3. 如果不存在，插入全新记忆 (Upsert 的 Insert 阶段)
            item_id = _new_item_id(normalized_summary, normalized_type)
            cursor = self._db.execute(
            """
            INSERT INTO memory_items (
                id, memory_type, summary, content_hash, embedding,
                reinforcement, emotional_weight, extra_json, source_ref,
                happened_at, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                normalized_type,
                normalized_summary,
                item_hash,
                _embedding_dumps(embedding),
                1,
                weight,
                _json_dumps(extra_json),
                source_ref,
                happened_at,
                "active",
                now,
                now,
            ),
        )
        rowid = int(cursor.lastrowid or 0)
        if rowid:
            self._vec_insert(rowid, embedding)
        self._db.commit()
        return f"new:{item_id}"


    def upsert_consolidation_event(
        self,
        *,
        source_ref: str,
        summary: str,
        embedding: list[float] | None,
        extra_json: dict[str, object] | None = None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """幂等写入一条 consolidation event。

        参数:
            source_ref: consolidation 来源引用；同一个 source_ref 只写一次。
            summary: event 摘要。
            embedding: event 摘要 embedding。
            extra_json: 扩展字段。
            happened_at: 事件发生时间。
            emotional_weight: 情绪权重。

        返回:
            写入结果字符串；重复 source_ref 返回 skipped:{item_id}。
        """

        clean_source = source_ref.strip()
        if not clean_source:
            return "skipped:empty_source"
        with self._lock:
            # 通过 source_ref 检查是否已经处理过这个事件（幂等性检查）
            existing = self._db.execute(
                "SELECT item_id FROM consolidation_events WHERE source_ref=?",
                (clean_source,),
            ).fetchone()
            if existing is not None:
                return f"skipped:{existing['item_id'] or clean_source}"
            
            # 如果没处理过，复用上面的 upsert_item 写入事件记忆
            result = self.upsert_item(
                memory_type="event",
                summary=summary,
                embedding=embedding,
                source_ref=clean_source,
                extra_json=extra_json,
                happened_at=happened_at,
                emotional_weight=emotional_weight,
            )
            item_id = result.split(":", 1)[1] if ":" in result else ""
            
            # 记录到 consolidation_events 表，打上标记，下次就不处理了
            self._db.execute(
                "INSERT INTO consolidation_events(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (clean_source, item_id, now_iso()),
            )
            self._db.commit()
            return result

    
    def has_consolidation_source_ref(self, source_ref: str) -> bool:
        """检查 consolidation source_ref 是否已经写入过。

        参数:
            source_ref: consolidation 来源引用。

        返回:
            已存在返回 True，否则返回 False。
        """
        # 简单查询是否打过 tag
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM consolidation_events WHERE source_ref=? LIMIT 1",
                (source_ref.strip(),),
            ).fetchone()
        return row is not None


    # 查询与批处理 (软删除与恢复)
    def get_item(self, item_id: str) -> MemoryItem | None:
        """按 id 读取一条 memory item。

        参数:
            item_id: memory item id。

        返回:
            MemoryItem；不存在时返回 None。
        """

        with self._lock:
            row = self._db.execute(
                "SELECT * FROM memory_items WHERE id=?",
                (item_id,),
            ).fetchone()
        return _row_to_item(row) if row is not None else None

    def get_items_by_ids(self, ids: list[str]) -> list[MemoryItem]:
        """按 id 列表读取 memory items。

        参数:
            ids: memory item id 列表。

        返回:
            按输入 ids 顺序返回存在的 MemoryItem 列表。
        """

        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memory_items WHERE id IN ({placeholders})",
                clean_ids,
            ).fetchall()
        by_id = {_row_to_item(row).id: _row_to_item(row) for row in rows}
        return [by_id[item_id] for item_id in clean_ids if item_id in by_id]

    def list_by_type(self, memory_type: str) -> list[MemoryItem]:
        """按 memory_type 列出 active items。

        参数:
            memory_type: 记忆类型。

        返回:
            MemoryItem 列表。
        """

        normalized_type = normalize_memory_type(memory_type)
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memory_items WHERE memory_type=? AND status='active' ORDER BY created_at ASC",
                (normalized_type,),
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    def mark_superseded_batch(self, ids: list[str]) -> int:
        """批量把 memory items 标记为 superseded。

        参数:
            ids: memory item id 列表。

        返回:
            实际更新条数。
        """
        # 软删除

        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return 0
        now = now_iso()
        with self._lock:
            cursor = self._db.executemany(
                "UPDATE memory_items SET status='superseded', updated_at=? WHERE id=? AND status!='superseded'",
                [(now, item_id) for item_id in clean_ids],
            )
            self._db.commit()
        return int(cursor.rowcount or 0)

    def restore_items_batch(self, ids: list[str]) -> int:
        """批量把 superseded memory items 恢复为 active。

        参数:
            ids: memory item id 列表。

        返回:
            实际恢复条数。
        """

        # 撤销软删除
        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return 0
        now = now_iso()
        with self._lock:
            cursor = self._db.executemany(
                "UPDATE memory_items SET status='active', updated_at=? WHERE id=? AND status='superseded'",
                [(now, item_id) for item_id in clean_ids],
            )
            self._db.commit()
        return int(cursor.rowcount or 0)

    # 记忆更迭与向量搜索
    def record_replacement(
        self,
        *,
        old_item: MemoryItem,
        new_item: MemoryItem,
        relation_type: str = "supersede",
        source_ref: str | None = None,
    ) -> None:
        """记录一条 memory replacement 关系。

        参数:
            old_item: 被替换的旧条目。
            new_item: 替换它的新条目。
            relation_type: 关系类型，默认 supersede。
            source_ref: 本次替换来源。

        返回:
            None。
        """

        with self._lock:
            self._db.execute(
                """
                INSERT INTO memory_replacements (
                    old_item_id, old_memory_type, old_summary, old_source_ref,
                    old_happened_at, old_extra_json, new_item_id, new_memory_type,
                    new_summary, new_source_ref, new_happened_at, new_extra_json,
                    relation_type, source_ref, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    old_item.id,
                    old_item.memory_type,
                    old_item.summary,
                    old_item.source_ref,
                    old_item.happened_at,
                    _json_dumps(old_item.extra_json),
                    new_item.id,
                    new_item.memory_type,
                    new_item.summary,
                    new_item.source_ref,
                    new_item.happened_at,
                    _json_dumps(new_item.extra_json),
                    relation_type,
                    source_ref,
                    now_iso(),
                ),
            )
            self._db.commit()

    def list_replacements(self) -> list[dict[str, object]]:
        """列出 memory replacement 关系。

        参数:
            无。

        返回:
            replacement 字典列表。
        """

        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memory_replacements ORDER BY id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def reinforce_items_batch(self, ids: list[str], *, emotional_weight: int = 0) -> int:
        """批量强化 memory items。

        参数:
            ids: memory item id 列表。
            emotional_weight: 可选情绪权重；会与旧 emotional_weight 取最大值。

        返回:
            实际更新条数。
        """

        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return 0
        now = now_iso()
        weight = clamp_emotional_weight(emotional_weight)
        with self._lock:
            cursor = self._db.executemany(
                """
                UPDATE memory_items
                SET reinforcement=reinforcement + 1,
                    emotional_weight=MAX(emotional_weight, ?),
                    updated_at=?
                WHERE id=?
                """,
                [(weight, now, item_id) for item_id in clean_ids],
            )
            self._db.commit()
        return int(cursor.rowcount or 0)

    def vector_search_batch(
        self,
        query_embeddings: list[list[float]],
        *,
        top_k: int = 8,
        memory_types: list[str] | None = None,
        score_threshold: float = 0.0,
        include_superseded: bool = False,
        hotness_alpha: float = 0.0,
        hotness_half_life_days: float = 14.0,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[list[dict[str, object]]]:
        """批量执行向量检索。

        参数:
            query_embeddings: 查询向量列表。
            top_k: 每条查询最多返回多少条。
            memory_types: 可选记忆类型过滤。
            score_threshold: 最低相似度。
            include_superseded: 是否包含 superseded 条目。
            hotness_alpha: 热度融合权重。
            hotness_half_life_days: 热度半衰期。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            与 query_embeddings 顺序一致的命中列表。
        """

        return [
            self.vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                memory_types=memory_types,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
                time_start=time_start,
                time_end=time_end,
            )
            for query_embedding in query_embeddings
        ]
    
    
    def vector_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 8,
        memory_types: list[str] | None = None,
        score_threshold: float = 0.0,
        include_superseded: bool = False,
        hotness_alpha: float = 0.0,
        hotness_half_life_days: float = 14.0,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[dict[str, object]]:
        """执行向量检索，sqlite-vec 优先，numpy fullscan fallback。

        参数:
            query_embedding: 查询向量。
            top_k: 最多返回多少条。
            memory_types: 可选记忆类型过滤。
            score_threshold: 最低 cosine similarity。
            include_superseded: 是否包含 superseded 条目。
            hotness_alpha: 热度融合权重。
            hotness_half_life_days: 热度半衰期。
            time_start: 可选时间过滤起点。
            time_end: 可选时间过滤终点。

        返回:
            命中字典列表。
        """

        if not query_embedding:
            return []
        if self._vec_enabled and time_start is None and time_end is None:
            return self._vector_search_vec(
                query_embedding=query_embedding,
                top_k=top_k,
                memory_types=memory_types,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )
        if not self._vec_fallback_logged:
            reason = self._vec_init_error or "sqlite-vec 未启用或当前查询需要 fallback"
            logger.warning("memory2 vector_search 使用 numpy fullscan fallback: %s", reason)
            self._vec_fallback_logged = True
        return self._vector_search_fullscan(
            query_embedding=query_embedding,
            top_k=top_k,
            memory_types=memory_types,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
            hotness_alpha=hotness_alpha,
            hotness_half_life_days=hotness_half_life_days,
            time_start=time_start,
            time_end=time_end,
        )
    
    def _vector_search_vec(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        memory_types: list[str] | None,
        score_threshold: float,
        include_superseded: bool,
        hotness_alpha: float,
        hotness_half_life_days: float,
    ) -> list[dict[str, object]]:
        """使用 sqlite-vec 执行 KNN 检索。

        参数:
            query_embedding: 查询向量。
            top_k: 最多返回多少条。
            memory_types: 可选记忆类型过滤。
            score_threshold: 最低 cosine similarity。
            include_superseded: 是否包含 superseded 条目。
            hotness_alpha: 热度融合权重。
            hotness_half_life_days: 热度半衰期。

        返回:
            命中字典列表；维度不匹配时回退 fullscan。
        """

        if len(query_embedding) != self._vec_dim:
            return self._vector_search_fullscan(
                query_embedding=query_embedding,
                top_k=top_k,
                memory_types=memory_types,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )

        fetch_k = max(top_k * 3, 20)
        params: list[object] = [_embedding_to_blob(query_embedding), fetch_k]
        where = ["1=1"]
        if not include_superseded:
            where.append("m.status='active'")
        normalized_types = [normalize_memory_type(item) for item in memory_types] if memory_types else []
        if normalized_types:
            placeholders = ",".join("?" for _ in normalized_types)
            where.append(f"m.memory_type IN ({placeholders})")
            params.extend(normalized_types)

        sql = f"""
            SELECT m.id, m.memory_type, m.summary, m.source_ref, m.happened_at,
                m.status, m.extra_json, m.reinforcement, m.emotional_weight,
                m.created_at, m.updated_at, v.distance
            FROM (
                SELECT rowid, distance
                FROM vec_items
                WHERE embedding MATCH ?
                AND k = ?
            ) v
            JOIN memory_items m ON m.rowid = v.rowid
            WHERE {' AND '.join(where)}
            ORDER BY v.distance ASC
        """

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            semantic = _l2_distance_to_cosine(float(row["distance"]))
            if semantic < score_threshold:
                continue
            final, debug = _blend_score(
                semantic=semantic,
                reinforcement=int(row["reinforcement"]),
                updated_at=str(row["updated_at"]),
                emotional_weight=int(row["emotional_weight"]),
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )
            results.append(_row_to_hit(row, score=final, score_debug=debug))
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[: max(1, int(top_k))]
    

    def _row_to_hit(
        row: sqlite3.Row,
        *,
        score: float,
        score_debug: dict[str, float] | None = None,
        keyword_score: float | None = None,
    ) -> dict[str, object]:
        """把 memory_items row 转换为检索 hit 字典。

        参数:
            row: memory_items 查询结果。
            score: 最终排序分数。
            score_debug: 可选分数拆解。
            keyword_score: 可选关键词命中分。

        返回:
            Retriever 使用的 hit 字典。
        """

        hit: dict[str, object] = {
            "id": str(row["id"]),
            "memory_type": str(row["memory_type"]),
            "summary": str(row["summary"]),
            "score": round(float(score), 4),
            "source_ref": str(row["source_ref"] or ""),
            "happened_at": str(row["happened_at"] or ""),
            "status": str(row["status"]),
            "extra_json": _json_object(row["extra_json"]),
            "reinforcement": int(row["reinforcement"]),
            "emotional_weight": int(row["emotional_weight"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        if score_debug is not None:
            hit["_score_debug"] = score_debug
        if keyword_score is not None:
            hit["keyword_score"] = round(float(keyword_score), 4)
        return hit
    
    def _vector_search_fullscan(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        memory_types: list[str] | None,
        score_threshold: float,
        include_superseded: bool,
        hotness_alpha: float,
        hotness_half_life_days: float,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[dict[str, object]]:
        """使用 numpy fullscan 执行向量检索。

        参数:
            query_embedding: 查询向量。
            top_k: 最多返回多少条。
            memory_types: 可选记忆类型过滤。
            score_threshold: 最低 cosine similarity。
            include_superseded: 是否包含 superseded 条目。
            hotness_alpha: 热度融合权重。
            hotness_half_life_days: 热度半衰期。
            time_start: 可选时间过滤起点。
            time_end: 可选时间过滤终点。

        返回:
            命中字典列表。
        """

        where = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if not include_superseded:
            where.append("status='active'")
        normalized_types = [normalize_memory_type(item) for item in memory_types] if memory_types else []
        if normalized_types:
            placeholders = ",".join("?" for _ in normalized_types)
            where.append(f"memory_type IN ({placeholders})")
            params.extend(normalized_types)
        if time_start is not None:
            where.append("happened_at IS NOT NULL AND happened_at >= ?")
            params.append(time_start.isoformat())
        if time_end is not None:
            where.append("happened_at IS NOT NULL AND happened_at < ?")
            params.append(time_end.isoformat())

        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memory_items WHERE {' AND '.join(where)}",
                params,
            ).fetchall()

        scored: list[dict[str, object]] = []
        for row in rows:
            item_embedding = _embedding_loads(row["embedding"])
            if item_embedding is None:
                continue
            semantic = _cosine_similarity(query_embedding, item_embedding)
            if semantic < score_threshold:
                continue
            final, debug = _blend_score(
                semantic=semantic,
                reinforcement=int(row["reinforcement"]),
                updated_at=str(row["updated_at"]),
                emotional_weight=int(row["emotional_weight"]),
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )
            scored.append(_row_to_hit(row, score=final, score_debug=debug))
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return scored[: max(1, int(top_k))]


    def keyword_search_summary(
        self,
        terms: list[str],
        *,
        memory_types: list[str] | None = None,
        limit: int = 20,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[dict[str, object]]:
        """对 memory summary 执行关键词检索。

        参数:
            terms: 关键词列表。
            memory_types: 可选记忆类型过滤。
            limit: 最多返回多少条。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            命中字典列表，每条包含 keyword_score。
        """

        clean_terms = [term.strip() for term in terms if len(term.strip()) >= 2]
        if not clean_terms:
            return []
        if self._has_fts:
            return self._keyword_search_fts(
                clean_terms,
                memory_types=memory_types,
                limit=limit,
                time_start=time_start,
                time_end=time_end,
            )
        return self._keyword_search_like(
            clean_terms,
            memory_types=memory_types,
            limit=limit,
            time_start=time_start,
            time_end=time_end,
        )
    
    def _keyword_search_fts(
        self,
        terms: list[str],
        *,
        memory_types: list[str] | None,
        limit: int,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[dict[str, object]]:
        """使用 FTS5 执行关键词检索。

        参数:
            terms: 关键词列表。
            memory_types: 可选记忆类型过滤。
            limit: 最多返回多少条。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            命中字典列表。
        """

        query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        params: list[Any] = [query]
        where = ["memory_items_fts MATCH ?", "m.status='active'"]
        normalized_types = [normalize_memory_type(item) for item in memory_types] if memory_types else []
        if normalized_types:
            placeholders = ",".join("?" for _ in normalized_types)
            where.append(f"m.memory_type IN ({placeholders})")
            params.extend(normalized_types)
        if time_start is not None:
            where.append("m.happened_at IS NOT NULL AND m.happened_at >= ?")
            params.append(time_start.isoformat())
        if time_end is not None:
            where.append("m.happened_at IS NOT NULL AND m.happened_at < ?")
            params.append(time_end.isoformat())
        params.append(max(1, int(limit)))

        sql = f"""
            SELECT m.*, bm25(memory_items_fts) AS rank_score
            FROM memory_items_fts
            JOIN memory_items m ON m.rowid = memory_items_fts.rowid
            WHERE {' AND '.join(where)}
            ORDER BY rank_score ASC, m.reinforcement DESC, m.id ASC
            LIMIT ?
        """
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            rank_score = abs(float(row["rank_score"]))
            keyword_score = 1.0 / (1.0 + rank_score)
            results.append(_row_to_hit(row, score=keyword_score, keyword_score=keyword_score))
        return results
    
    
    def _keyword_search_like(
        self,
        terms: list[str],
        *,
        memory_types: list[str] | None,
        limit: int,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[dict[str, object]]:
        """使用 LIKE fallback 执行关键词检索。

        参数:
            terms: 关键词列表。
            memory_types: 可选记忆类型过滤。
            limit: 最多返回多少条。
            time_start: 可选时间范围起点。
            time_end: 可选时间范围终点。

        返回:
            命中字典列表。
        """

        where = ["status='active'"]
        params: list[Any] = []
        like_conditions = " OR ".join("summary LIKE ?" for _ in terms)
        where.append(f"({like_conditions})")
        like_values = [f"%{term}%" for term in terms]
        params.extend(like_values)
        normalized_types = [normalize_memory_type(item) for item in memory_types] if memory_types else []
        if normalized_types:
            placeholders = ",".join("?" for _ in normalized_types)
            where.append(f"memory_type IN ({placeholders})")
            params.extend(normalized_types)
        if time_start is not None:
            where.append("happened_at IS NOT NULL AND happened_at >= ?")
            params.append(time_start.isoformat())
        if time_end is not None:
            where.append("happened_at IS NOT NULL AND happened_at < ?")
            params.append(time_end.isoformat())

        params.append(max(1, int(limit)))
        sql = f"SELECT * FROM memory_items WHERE {' AND '.join(where)} LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            summary = str(row["summary"])
            matched = sum(1 for term in terms if term in summary)
            keyword_score = matched / max(1, len(terms))
            if keyword_score <= 0:
                continue
            results.append(_row_to_hit(row, score=keyword_score, keyword_score=keyword_score))
        results.sort(
            key=lambda item: (float(item.get("keyword_score", 0.0)), int(item.get("reinforcement", 0))),
            reverse=True,
        )
        return results[: max(1, int(limit))]

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """列出时间范围内的 active event 记忆。

        参数:
            time_start: 时间范围起点。
            time_end: 时间范围终点。
            limit: 最多返回多少条。

        返回:
            event hit 字典列表。
        """

        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM memory_items
                WHERE memory_type='event'
                AND status='active'
                AND happened_at IS NOT NULL
                AND happened_at >= ?
                AND happened_at < ?
                ORDER BY happened_at ASC, id ASC
                LIMIT ?
                """,
                (time_start.isoformat(), time_end.isoformat(), max(1, min(int(limit), 500))),
            ).fetchall()
        return [_row_to_hit(row, score=1.0) for row in rows]

    
    # 后续 Dashboard 会用的物理删除能力
    def delete_item(self, item_id: str) -> bool:
        """物理删除单条 memory item。

        参数:
            item_id: memory item id。

        返回:
            删除成功返回 True，否则返回 False。
        """

        with self._lock:
            row = self._db.execute("SELECT rowid FROM memory_items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                return False
            cursor = self._db.execute("DELETE FROM memory_items WHERE id=?", (item_id,))
            self._vec_delete([int(row["rowid"])])
            self._db.commit()
        return int(cursor.rowcount or 0) > 0


    def delete_items_batch(self, ids: list[str]) -> int:
        """物理删除多条 memory items。

        参数:
            ids: memory item id 列表。

        返回:
            实际删除条数。
        """

        clean_ids = [item.strip() for item in ids if item.strip()]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            rowids = [
                int(row["rowid"])
                for row in self._db.execute(
                    f"SELECT rowid FROM memory_items WHERE id IN ({placeholders})",
                    clean_ids,
                ).fetchall()
            ]
            cursor = self._db.execute(
                f"DELETE FROM memory_items WHERE id IN ({placeholders})",
                clean_ids,
            )
            self._vec_delete(rowids)
            self._db.commit()
        return int(cursor.rowcount or 0)
    
    def merge_item_raw(
        self,
        *,
        item_id: str,
        new_summary: str,
        new_hash: str,
        new_embedding: list[float],
        new_extra_json: dict[str, object] | None = None,
    ) -> None:
        """原子更新 merge 目标 item。

        参数:
            item_id: 要被原地合并更新的 memory item id。
            new_summary: 合并后的 summary。
            new_hash: 合并后 summary 对应的 content_hash。
            new_embedding: 合并后 summary 对应的 embedding。
            new_extra_json: 合并后的 extra_json；None 表示不更新 extra_json。

        返回:
            None。
        """

        clean_summary = new_summary.strip()
        if not item_id.strip() or not clean_summary:
            return
        now = now_iso()
        try:
            with self._lock:
                if new_extra_json is None:
                    self._db.execute(
                        """
                        UPDATE memory_items
                        SET summary=?, content_hash=?, embedding=?,
                            reinforcement=reinforcement + 1, updated_at=?
                        WHERE id=?
                        """,
                        (
                            clean_summary,
                            new_hash,
                            _embedding_dumps(new_embedding),
                            now,
                            item_id,
                        ),
                    )
                else:
                    self._db.execute(
                        """
                        UPDATE memory_items
                        SET summary=?, content_hash=?, embedding=?, extra_json=?,
                            reinforcement=reinforcement + 1, updated_at=?
                        WHERE id=?
                        """,
                        (
                            clean_summary,
                            new_hash,
                            _embedding_dumps(new_embedding),
                            _json_dumps(new_extra_json),
                            now,
                            item_id,
                        ),
                    )
                row = self._db.execute(
                    "SELECT rowid FROM memory_items WHERE id=?",
                    (item_id,),
                ).fetchone()
                if row is not None:
                    self._vec_insert(int(row["rowid"]), new_embedding)
                self._db.commit()
        except sqlite3.IntegrityError:
            logger.warning("merge_item_raw content_hash collision for item %s", item_id)
            with self._lock:
                self._db.rollback()
    
    def find_similar_recent_events(
        self,
        embedding: list[float],
        *,
        days_back: int = 7,
        threshold: float = 0.92,
        top_k: int = 3,
    ) -> list[str]:
        """查找近期语义近似的 active event。

        参数:
            embedding: 新 event 的 embedding。
            days_back: 只查最近多少天创建的 event。
            threshold: cosine similarity 阈值。
            top_k: 最多返回多少个相似 event id。

        返回:
            相似 event 的 memory item id 列表。
        """

        if not embedding:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days_back)))).isoformat()
        with self._lock:
            rows = self._db.execute(
                """
                SELECT id, embedding
                FROM memory_items
                WHERE memory_type='event'
                AND status='active'
                AND embedding IS NOT NULL
                AND created_at >= ?
                """,
                (cutoff,),
            ).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            item_embedding = _embedding_loads(row["embedding"])
            if item_embedding is None:
                continue
            score = _cosine_similarity(embedding, item_embedding)
            if score >= threshold:
                scored.append((str(row["id"]), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item_id for item_id, _score in scored[: max(1, int(top_k))]]
    

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict[str, object]]:
        """按 trigger_tags 对 procedure 做关键词匹配。

        参数:
            action_tokens: 工具名、技能名、命令词、路径词等动作 token。

        返回:
            命中的 procedure hit 字典列表。
        """

        if not action_tokens:
            return []
        token_set = {token.lower() for token in action_tokens if token.strip()}
        action_text = " ".join(action_tokens).lower()

        with self._lock:
            rows = self._db.execute(
                """
                SELECT id, memory_type, summary, extra_json, source_ref, happened_at,
                    status, reinforcement, emotional_weight, created_at, updated_at
                FROM memory_items
                WHERE memory_type='procedure'
                AND status='active'
                AND extra_json IS NOT NULL
                """
            ).fetchall()

        matched: list[dict[str, object]] = []
        for row in rows:
            extra = _json_object(row["extra_json"])
            tags = extra.get("trigger_tags")
            if not isinstance(tags, dict):
                continue
            if tags.get("scope") != "tool_triggered":
                continue

            keywords = [str(item).strip() for item in tags.get("keywords", []) if len(str(item).strip()) >= 2]
            if keywords:
                hit = any(keyword.lower() in action_text for keyword in keywords)
            else:
                tag_tokens = {str(item).lower() for item in tags.get("tools", []) if str(item).strip()}
                tag_tokens |= {str(item).lower() for item in tags.get("skills", []) if str(item).strip()}
                if len(tag_tokens) > 4:
                    continue
                hit = bool(token_set & tag_tokens)

            if not hit:
                continue
            matched.append(
                _row_to_hit(
                    row,
                    score=1.0,
                    score_debug={"semantic": 0.0, "hotness": 0.0, "final": 1.0},
                )
            )
        return matched