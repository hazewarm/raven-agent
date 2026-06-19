from __future__ import annotations

import re
from typing import Any

# 匹配标准工具名或变量名的正则：以字母开头，后接字母、数字或下划线
# 通常用于从一段中文文本中精准“抠出”英文的工具名（例如 "web_search"）
_ASCII_ALIAS_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# 定义禁止使用某工具的中文前缀触发词（用于自然语言推断）
_NEGATIVE_TOOL_PREFIXES = (
    "不能直接使用",
    "不能直接用",
    "不要直接使用",
    "不要直接用",
    "别直接使用",
    "别直接用",
    "不能先使用",
    "不能先用",
    "不要先使用",
    "不要先用",
    "禁止使用",
    "禁止用",
    "不能使用",
    "不能用",
    "不要使用",
    "不要用",
)

# 定义强制/优先使用某工具的中文前缀触发词
_POSITIVE_TOOL_PREFIXES = (
    "必须先使用",
    "必须先用",
    "必须使用",
    "必须用",
    "先使用",
    "先用",
    "优先使用",
    "优先用",
    "应该使用",
    "应该用",
    "直接使用",
    "直接用",
)


def build_procedure_rule_schema(
    summary: str,
    tool_requirement: str | None = None,
    steps: list[str] | None = None,
    rule_schema: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """构造 procedure 的工具约束 schema。

    参数:
        summary: procedure 摘要文本。
        tool_requirement: 调用方显式指定的必需工具名；为空则只从文本推断。
        steps: procedure 的步骤列表。
        rule_schema: 调用方传入的已有 schema；函数会在此基础上补全缺失槽位。

    返回:
        包含 required_tools、forbidden_tools、mentioned_tools 的字典。
    """

    # 1. 初始化并清洗已有的工具集合（转小写、去重）
    required = set(_normalize_schema_list((rule_schema or {}).get("required_tools")))
    forbidden = set(_normalize_schema_list((rule_schema or {}).get("forbidden_tools")))
    mentioned = set(_normalize_schema_list((rule_schema or {}).get("mentioned_tools")))
    
    # 2. 从摘要和步骤文本中提取所有出现的英文字符串（极大概率是工具名），加入 mentioned
    mentioned.update(_extract_ascii_aliases(summary))
    for step in steps or []:
        mentioned.update(_extract_ascii_aliases(step))

    # 3. 如果调用方没有明确给定规则，则尝试从自然语言中推断规则
    if not required or not forbidden:
        inferred_required, inferred_forbidden = _infer_rule_constraints(summary, steps)
        if not required:
            required.update(inferred_required)
        if not forbidden:
            forbidden.update(inferred_forbidden)

    # 4. 合并调用方强行指定的要求（tool_requirement）
    if tool_requirement:
        normalized = str(tool_requirement).strip().lower()
        if normalized:
            required.add(normalized)
            mentioned.add(normalized)

    # 5. 冲突解决：一个工具如果被要求“必须使用”，则自动从“禁止使用”名单中剔除
    forbidden.difference_update(required)

    # 6. 返回标准化后的字典，按字母排序以保证输出的一致性
    return {
        "required_tools": sorted(required),
        "forbidden_tools": sorted(forbidden),
        "mentioned_tools": sorted(mentioned),
    }


def resolve_procedure_rule_schema(summary: str, extra: dict[str, Any] | None) -> dict[str, list[str]]:
    """从 procedure extra_json 中解析并补全 rule_schema。

    参数:
        summary: procedure 摘要文本。
        extra: procedure 的 extra_json 字典。

    返回:
        标准 rule_schema 字典。
    """

    payload = extra or {}
    return build_procedure_rule_schema(
        summary=summary,
        tool_requirement=payload.get("tool_requirement"),
        steps=payload.get("steps") or [],
        rule_schema=payload.get("rule_schema"),
    )


def procedure_rules_conflict(
    new_schema: dict[str, list[str]],
    old_schema: dict[str, list[str]],
) -> bool:
    """判断两条 procedure 规则是否在工具约束上冲突。

    参数:
        new_schema: 新 procedure 的 rule_schema。
        old_schema: 旧 procedure 的 rule_schema。

    返回:
        如果一个要求的工具被另一个禁止，返回 True；否则返回 False。
    """

    new_terms = _schema_terms(new_schema)
    old_terms = _schema_terms(old_schema)
    if not new_terms or not old_terms or not (new_terms & old_terms):
        return False
    new_required = set(new_schema.get("required_tools") or [])
    new_forbidden = set(new_schema.get("forbidden_tools") or [])
    old_required = set(old_schema.get("required_tools") or [])
    old_forbidden = set(old_schema.get("forbidden_tools") or [])
    return bool((new_required & old_forbidden) or (new_forbidden & old_required))


def _extract_ascii_aliases(text: str) -> set[str]:
    """从文本中提取 ASCII 工具别名。

    智能拼接被空格断开的 snake_case 命名

    参数:
        text: procedure 摘要或步骤文本。

    返回:
        小写后的工具候选词集合。
    """

    aliases: set[str] = set()
    matches = list(_ASCII_ALIAS_PATTERN.finditer(text or ""))
    
    # 提取所有长度 >= 2 的单个英文 token
    for match in matches:
        token = match.group(0).lower()
        if len(token) >= 2:
            aliases.add(token)
    
    # 滑动窗口拼接：处理例如 "web search" 会被拼成 "web_search"
    for index in range(len(matches) - 1):
        left = matches[index]
        right = matches[index + 1]
        
        # 检查两个英文 token 之间是否只有空白符，如果没有其他字符干扰，则用下划线拼接
        if text[left.end() : right.start()].strip() != "":
            continue
        phrase = f"{left.group(0).lower()}_{right.group(0).lower()}"
        if len(phrase) >= 2:
            aliases.add(phrase)
    return aliases


def _normalize_schema_list(value: Any) -> list[str]:
    """把外部 schema 列表归一化为小写字符串列表。

    参数:
        value: 外部传入的任意对象。

    返回:
        去重排序后的字符串列表。
    """

    if not isinstance(value, list):
        return []
    return sorted(
        {
            str(item).strip().lower()
            for item in value
            if isinstance(item, str) and str(item).strip()
        }
    )


def _schema_terms(schema: dict[str, list[str]]) -> set[str]:
    """收集 rule_schema 中出现过的全部工具词。

    参数:
        schema: procedure rule_schema。

    返回:
        required、forbidden、mentioned 三类词的并集。
    """

    return set(schema.get("mentioned_tools") or []) | set(schema.get("required_tools") or []) | set(schema.get("forbidden_tools") or [])


def _infer_rule_constraints(
    summary: str,
    steps: list[str] | None,
) -> tuple[set[str], set[str]]:
    """从中文前缀中推断 required / forbidden 工具。

    防止前半句的否定词错误地作用于后半句的工具名上

    参数:
        summary: procedure 摘要。
        steps: procedure 步骤列表。

    返回:
        二元组：(required_tools, forbidden_tools)。
    """

    required: set[str] = set()
    forbidden: set[str] = set()
    for text in [summary, *(steps or [])]:
        for clause in re.split(r"[，。！？；;\n]", text or ""):
            for alias, prefix in _iter_alias_prefixes(clause):
                if any(prefix.endswith(cue) for cue in _NEGATIVE_TOOL_PREFIXES):
                    forbidden.add(alias)
                    continue
                if any(prefix.endswith(cue) for cue in _POSITIVE_TOOL_PREFIXES):
                    required.add(alias)
    return required, forbidden


def _iter_alias_prefixes(clause: str) -> list[tuple[str, str]]:
    """枚举工具候选词和它前面的中文约束前缀。

    参数:
        clause: 单个中文分句。

    返回:
        形如 (alias, prefix) 的列表。
    """

    matches = list(_ASCII_ALIAS_PATTERN.finditer(clause or ""))
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(matches):
        match = matches[index]
        prefix = _normalize_prefix(clause[max(0, match.start() - 12) : match.start()])
        if index < len(matches) - 1:
            next_match = matches[index + 1]
            if clause[match.end() : next_match.start()].strip() == "":
                alias = f"{match.group(0).lower()}_{next_match.group(0).lower()}"
                pairs.append((alias, prefix))
                index += 2
                continue
        pairs.append((match.group(0).lower(), prefix))
        index += 1
    return pairs


def _normalize_prefix(text: str) -> str:
    """去掉前缀中的空白字符。

    参数:
        text: 工具名之前的一小段文本。

    返回:
        无空白前缀。
    """

    return re.sub(r"\s+", "", text or "")