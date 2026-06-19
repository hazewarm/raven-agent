from __future__ import annotations

from pathlib import Path
import logging

import jieba

jieba.setLogLevel(logging.INFO)

_DICT_DIR = Path(__file__).parent / "ext" / "dict"
_JIEBA_READY = False
_PUNCTUATION = set("\t\r\n .,!?;:'\"()[]{}<>，。！？；：、“”‘’（）【】《》…—·")


def extract_terms(query: str, limit: int = 20) -> list[str]:
    """使用 jieba 从 query 中提取关键词。

    参数:
        query: 原始查询文本。
        limit: 最多返回多少个关键词。

    返回:
        去重后的关键词列表。
    """

    global _JIEBA_READY

    text = str(query or "").strip()
    if not text:
        return []

    if not _JIEBA_READY:
        user_dict = _DICT_DIR / "user.dict.utf8"
        if user_dict.exists():
            jieba.load_userdict(str(user_dict))
        _JIEBA_READY = True

    terms: list[str] = []
    seen: set[str] = set()
    max_terms = max(1, int(limit))

    for token in jieba.cut(text, HMM=True):
        term = str(token or "").strip()
        if len(term) < 2:
            continue
        if all(char in _PUNCTUATION for char in term):
            continue
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break

    return terms