from __future__ import annotations

import logging
from collections.abc import Callable

from raven_agent.llm import LLMProvider
from raven_agent.llm_json import load_json_object_loose
from raven_agent.messages import system_message, user_message

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "你是一个记忆标注助手，输出严格 JSON object，不加任何额外文字。"

_USER_PROMPT_TEMPLATE = """\
你的任务是分析一条操作规范（procedure），为检索系统生成触发关键字标签。

## 系统注册的工具列表
{tool_list}

## 系统可用的技能列表
{skill_list}

## 待标注的操作规范
{summary}

## 输出格式
只输出 JSON object：
{{
  "tools": [],
  "skills": [],
  "keywords": [],
  "scope": "tool_triggered"
}}

字段规则：
- tools：触发该规范的工具名，必须严格来自工具列表，没有则为 []
- skills：触发该规范的技能名，必须严格来自技能列表，没有则为 []
- keywords：shell 命令关键词、路径关键词、站点关键词或任务关键词，不与 tools/skills 重复
- scope：tool_triggered 或 global
- global 表示不依赖具体工具、所有任务都应遵守的通用规范
- tool_triggered 表示只有工具/技能/关键词命中时才触发

示例：
规范：用户发送 B 站视频链接时，应先使用 web_fetch 读取页面，再判断是否需要搜索。
输出：{{"tools": ["web_fetch"], "skills": [], "keywords": ["B站", "视频"], "scope": "tool_triggered"}}

规范：运行 shell 命令时，工具尝试失败两次后必须收敛并反馈用户。
输出：{{"tools": ["shell"], "skills": [], "keywords": [], "scope": "tool_triggered"}}

规范：回答任何问题都要先给结论。
输出：{{"tools": [], "skills": [], "keywords": [], "scope": "global"}}
"""


class ProcedureTagger:
    """为 procedure 条目生成 trigger_tags。

    参数:
        provider: LLMProvider，用于根据 procedure 摘要生成标签。
        tools_fn: 返回当前已注册工具名的函数。
        skills_fn: 返回当前可用技能名的函数；不传则表示当前没有技能列表。

    返回:
        ProcedureTagger 实例。
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools_fn: Callable[[], list[str]],
        skills_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self._provider = provider
        self._tools_fn = tools_fn
        self._skills_fn = skills_fn or (lambda: [])

    async def tag(self, summary: str) -> dict[str, object] | None:
        """为一条 procedure summary 生成 trigger_tags。

        参数:
            summary: procedure 摘要。

        返回:
            合法 trigger_tags 字典；LLM 失败或输出无效时返回 None。
        """

        clean_summary = summary.strip()
        if not clean_summary:
            return None

        tools = [tool for tool in self._tools_fn() if tool.strip()]
        skills = [skill for skill in self._skills_fn() if skill.strip()]
        prompt = _USER_PROMPT_TEMPLATE.format(
            tool_list="\n".join(f"- {tool}" for tool in tools) or "（暂无）",
            skill_list="\n".join(f"- {skill}" for skill in skills) or "（暂无）",
            summary=clean_summary,
        )

        try:
            response = await self._provider.chat(
                messages=[
                    system_message(_SYSTEM_PROMPT),
                    user_message(prompt),
                ],
                tools=[],
                tool_choice="none",
            )
        except Exception as exc:
            logger.warning("procedure tagger failed: %s", exc)
            return None

        payload = load_json_object_loose(response.content)
        if payload is None:
            return None
        return validate_trigger_tags(payload, valid_tools=set(tools), valid_skills=set(skills))


def validate_trigger_tags(
    tag: dict[str, object],
    *,
    valid_tools: set[str],
    valid_skills: set[str],
) -> dict[str, object]:
    """校验并清洗 procedure trigger_tags。

    参数:
        tag: LLM 输出的原始 JSON object。
        valid_tools: 当前系统允许出现的工具名集合。
        valid_skills: 当前系统允许出现的技能名集合。

    返回:
        清洗后的 trigger_tags 字典。
    """

    tools = _filter_allowed_strings(tag.get("tools"), valid_tools)
    skills = _filter_allowed_strings(tag.get("skills"), valid_skills)
    keywords = _filter_keywords(tag.get("keywords"))
    scope = str(tag.get("scope") or "tool_triggered").strip()
    if scope not in {"tool_triggered", "global"}:
        scope = "global" if not tools and not skills and not keywords else "tool_triggered"
    if scope == "global" and (tools or skills or keywords):
        scope = "tool_triggered"
    return {
        "tools": tools,
        "skills": skills,
        "keywords": keywords,
        "scope": scope,
    }


def _filter_allowed_strings(value: object, allowed: set[str]) -> list[str]:
    """过滤出允许列表内的字符串。

    参数:
        value: LLM 输出的任意值。
        allowed: 允许出现的字符串集合。

    返回:
        去重且保序的合法字符串列表。
    """

    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip() if isinstance(item, str) else ""
        if not text or text not in allowed or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _filter_keywords(value: object) -> list[str]:
    """过滤 trigger keyword。

    参数:
        value: LLM 输出的 keywords 字段。

    返回:
        去重且保序的关键词列表；长度小于 2 的关键词会被丢弃。
    """

    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip() if isinstance(item, str) else ""
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result