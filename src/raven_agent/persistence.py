"""
统一 JSON 文件持久化基础工具。

替代散落在各模块的 _load() / _save() 重复实现，提供：
- 原子写（先写 .tmp 再 rename，防止写一半崩溃损坏文件）
- 读取容错（文件不存在或损坏不崩溃，返回 default）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["load_json", "save_json"]


def load_json(path: Path, default: Any = None) -> Any:
    """从文件读取 JSON，失败时返回 default。

    输入:
        path: JSON 文件路径。
        default: 文件不存在或解析失败时的返回值。

    输出:
        反序列化后的 Python 对象；文件不存在/损坏时返回 default。
    """
    if not path.exists():
        return default

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("load_json 失败: path=%s err=%s", path, e)
        return default


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """将数据以 JSON 原子写入文件。

    原子写流程：先写 .tmp 临时文件，再 os.replace 原子替换，
    避免进程在写中途崩溃导致目标文件变成半截 JSON。

    输入:
        path: 目标文件路径，父目录不存在时自动创建。
        data: 可 JSON 序列化的 Python 对象。
        indent: JSON 缩进空格数。

    异常:
        OSError / json.JSONEncodeError: 写入失败时上抛。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    tmp.replace(path)

    logger.debug("save_json 完成: %s", path)