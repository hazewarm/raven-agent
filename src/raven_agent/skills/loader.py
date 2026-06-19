"""
skills/loader.py —— 主 Agent 技能加载器。

扫描项目根目录 skills/ 文件夹（类似 mcp_servers/），
列举所有包含 SKILL.md 的子目录作为可用技能，
生成 XML 摘要注入被动通道 system prompt。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

# ── 工具函数 ─────────────────────────────────────────────────────────


def _strip_frontmatter(content: str) -> str:
    """剥除 SKILL.md 头部的 YAML frontmatter，只保留正文。

    输入:
        content: SKILL.md 完整文本（可能含 frontmatter）。

    输出:
        剥除 frontmatter 后的正文。无 frontmatter 时返回原文。
    """
    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n?", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
    return content


# ── SkillsLoader ─────────────────────────────────────────────────────


class SkillsLoader:
    """主 Agent 技能加载器。

    扫描指定技能目录，列举、加载、生成 system prompt 中的技能摘要。

    参数:
        skills_dir: 技能目录的绝对路径（如项目根目录下的 skills/）。

    使用示例:
        loader = SkillsLoader(skills_dir=Path("skills"))
        summary = loader.build_skills_summary()
        # → "<skills><skill ...>...</skills>" 或空字符串
    """

    def __init__(self, skills_dir: Path) -> None:
        """初始化技能加载器。

        输入:
            skills_dir: 技能目录的绝对路径。
                该目录下的每个子目录如果包含 SKILL.md，即被视为一个可用技能。

        输出:
            None。
        """
        self._skills_dir = Path(skills_dir)

    # ── 公共 API：列出所有技能 ──────────────────────────────────────

    def list_skills(self) -> list[dict[str, str]]:
        """列举所有可用技能。

        扫描技能目录，对每个包含 SKILL.md 的子目录解析 frontmatter。

        输出:
            技能信息列表，每项为字典：
            {
                "name": "feed-manage",                     # 技能目录名
                "path": "skills/feed-manage/SKILL.md",     # 文件路径（供 LLM read_file 用）
                "description": "管理 RSS 订阅源",           # 一句话描述
            }
            无技能时返回空列表。
        """
        skills: list[dict[str, str]] = []

        if not self._skills_dir.exists() or not self._skills_dir.is_dir():
            return skills

        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            content = self._read_file(skill_file)
            if not content:
                continue
            meta = self._parse_frontmatter(content)
            description = str(meta.get("description", skill_dir.name)).strip()

            skills.append({
                "name": skill_dir.name,
                "path": str(skill_file),
                "description": description,
            })

        logger.info("[skills] 已加载 %d 个 skill: %s",
            len(skills),
            [s["name"] for s in skills],
        )
        return skills

    # ── 公共 API：always 技能 ────────────────────────────────────────

    def get_always_skills(self) -> list[str]:
        """返回所有标记了 always=true 的技能名称列表。

        always 技能在每轮对话都会把完整内容注入 system prompt，
        适合需要无条件生效的核心指令（如全局行为规则）。

        输出:
            技能名称列表。无 always 技能时返回空列表。

        判定规则:
            frontmatter 中 `always: true` 即视为 always 技能。
            例:
                ---
                name: core-rules
                description: 核心行为规则
                always: true
                ---
        """
        result: list[str] = []
        for s in self.list_skills():
            content = self._load_skill_content(s["name"])
            if content is None:
                continue
            meta = self._parse_frontmatter(content)
            if meta.get("always") in (True, "true"):
                result.append(s["name"])
        return result

    # ── 公共 API：生成 prompt 片段 ───────────────────────────────────

    def build_skills_summary(self) -> str:
        """生成所有技能的 XML 摘要，用于注入 system prompt。

        LLM 看到摘要后，如果用户意图匹配某个 skill，
        可以用已有的 read_file 工具读取对应 location 路径。

        输出:
            XML 格式的技能列表字符串。无技能时返回空字符串。

        格式:
            <skills>
              <skill>
                <name>feed-manage</name>
                <description>管理 RSS 订阅源</description>
                <location>skills/feed-manage/SKILL.md</location>
              </skill>
              ...
            </skills>
        """
        all_skills = self.list_skills()
        if not all_skills:
            return ""

        def _escape(s: str) -> str:
            """XML 转义特殊字符。"""
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = _escape(s["name"])
            desc = _escape(s.get("description", name))

            lines.append("  <skill>")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{s['path']}</location>")
            lines.append("  </skill>")

        lines.append("</skills>")
        return "\n".join(lines)

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """将指定技能的完整内容加载为 prompt 片段（剥除 frontmatter）。

        用于 always 技能——每轮自动注入，不依赖 LLM 显式匹配。

        输入:
            skill_names: 要加载的技能名称列表。

        输出:
            格式化后的技能内容字符串，多个技能之间以分隔线隔开。
            格式:
                ### Skill: feed-manage

                ## 目标
                ...

                ---

                ### Skill: summarize

                ## 工作流程
                ...
            所有 skill 都加载失败时返回空字符串。
        """
        parts: list[str] = []
        for name in skill_names:
            content = self._load_skill_content(name)
            if content is None:
                continue
            body = _strip_frontmatter(content)
            if not body:
                continue
            parts.append(f"### Skill: {name}\n\n{body}")

        return "\n\n---\n\n".join(parts) if parts else ""

    # ── 内部方法 ────────────────────────────────────────────────────

    def _load_skill_content(self, name: str) -> str | None:
        """按名称读取 SKILL.md 的完整文本。

        输入:
            name: 技能名称（即子目录名）。

        输出:
            SKILL.md 文本内容。未找到返回 None。
        """
        path = self._skills_dir / name / "SKILL.md"
        return self._read_file(path)

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """解析 SKILL.md 的 YAML frontmatter。

        frontmatter 位于文件头部，以 --- 起止。
        支持简单 key: value 和列表（以 - 开头）。

        输入:
            content: SKILL.md 的完整文本。

        输出:
            frontmatter 键值对字典。无 frontmatter 返回空字典。

        示例:
            ---
            name: audit-memory
            description: 审计长期记忆的准确性
            requires_mcp:
              - memory-server
              - search-server
            ---
            → {"name": "audit-memory", "description": "审计长期记忆的准确性",
               "requires_mcp": ["memory-server", "search-server"]}
        """
        if not content.startswith("---"):
            return {}
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if match is None:
            return {}
        metadata: dict[str, Any] = {}
        lines = match.group(1).split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if ":" not in line:
                i += 1
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if value:
                metadata[key] = value
                i += 1
                continue
            # value 为空 → 检查后续行是否为 YAML 列表项（  - item）
            list_items: list[str] = []
            j = i + 1
            while j < len(lines):
                item_match = re.match(r"^\s+-\s+(.+)", lines[j])
                if item_match is None:
                    break
                list_items.append(item_match.group(1).strip().strip("\"'"))
                j += 1
            if list_items:
                metadata[key] = list_items
                i = j
            else:
                metadata[key] = value
                i += 1
        return metadata

    @staticmethod
    def _read_file(path: Path) -> str | None:
        """读取文件内容，文件不存在或读取失败返回 None。

        输入:
            path: 文件路径。

        输出:
            文件文本内容；文件不存在或读取失败返回 None。
        """
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            pass
        return None
