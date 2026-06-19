"""
proactive/drift_state.py —— Drift Skill 管理与状态持久化。

Drift 任务以 Skill 目录形式组织，每个 skill 的结构：
  drift/skills/<skill-name>/
  ├── SKILL.md       # skill 定义（YAML frontmatter + Markdown 正文）
  ├── state.json      # 自动维护的运行状态
  └── *.md            # skill 的工作文件

SKILL.md frontmatter 格式：
  ---
  name: <skill-name>
  description: <一句话描述>
  requires_mcp: []      # 可选：需要的 MCP server 列表
  ---

  正文：skill 的分步操作指南。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raven_agent.persistence import load_json, save_json

logger = logging.getLogger(__name__)


def _clip(text: str, limit: int) -> str:
    """截断字符串到指定长度。

    输入:
        text: 原始字符串。
        limit: 最大字符数。

    输出:
        截断后的字符串。
    """
    return str(text or "").strip()[:limit]


def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
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


@dataclass
class SkillMeta:
    """单个 Drift Skill 的扫描结果。

    字段:
        name: skill 名称（与目录名一致）。
        description: skill 的一句话描述。
        last_run_at: 上次运行时间（aware datetime）；从未运行则为 None。
        run_count: 累计运行次数。
        status: 当前状态："idle" | "in_progress"。
        next: 上次运行记录的下一步动作。
        requires_mcp: 依赖的 MCP server 名称列表。
    """

    name: str
    description: str
    last_run_at: datetime | None
    run_count: int
    status: str
    next: str
    requires_mcp: list[str]


class DriftStateStore:
    """Drift Skill 状态管理器。

    参数:
        source_dir: 技能源文件目录的绝对路径（如 drift/skills/）。
            该目录下每个包含 SKILL.md 的子目录被视为一个可用 skill。
            由用户维护，可放入版本控制。
        state_dir: 运行时状态目录的绝对路径（如 .raven/drift/）。
            系统在此自动创建 skills/ 子目录，存放 state.json 和 working files。
    """

    def __init__(self, source_dir: Path, state_dir: Path) -> None:
        self.source_dir = Path(source_dir).expanduser()
        self.state_dir = Path(state_dir).expanduser()
        self.state_skills_dir = self.state_dir / "skills"
        self._drift_file = self.state_dir / "drift.json"
        self.state_skills_dir.mkdir(parents=True, exist_ok=True)

    # ── Skill 扫描 ──────────────────────────────────────────────────

    def scan_skills(self) -> list[SkillMeta]:
        """扫描源目录下所有有效 skill。

        从 source_dir 读取 SKILL.md（frontmatter 解析），
        从 state_dir 读取 state.json（运行次数、next 等），
        合并构建 SkillMeta 列表。

        输出:
            SkillMeta 列表。无 skill 时返回空列表。
        """
        skills: list[SkillMeta] = []
        seen_names: set[str] = set()

        if not self.source_dir.exists():
            logger.info("[drift_state] source 目录不存在: %s", self.source_dir)
            return []

        for skill_dir in sorted(self.source_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill = self._load_skill_meta(skill_dir)
            if skill is None:
                continue
            if skill.name in seen_names:
                continue
            seen_names.add(skill.name)
            skills.append(skill)

        skills.sort(
            key=lambda item: item.last_run_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        logger.info(
            "[drift_state] scan_skills: 发现 %d 个 skill: %s",
            len(skills), [s.name for s in skills],
        )
        return skills

    def valid_skill_names(self) -> set[str]:
        """返回当前所有有效 skill 的名称集合。

        输出:
            set[str]。
        """
        return {s.name for s in self.scan_skills()}

    # ── Drift 运行记录 ──────────────────────────────────────────────

    def load_drift(self) -> dict[str, Any]:
        """加载 drift.json 中的运行记录。

        返回结构:
            {
                "version": 1,
                "recent_runs": [
                    {"skill": "...", "run_at": "...", "one_line": "...",
                     "message_result": "sent|silent"},
                    ...
                ],
                "note": "...",
                "last_drift_at": "..."  # ISO 时间戳
            }

        输出:
            运行记录字典。文件不存在或损坏时返回空结构。
        """
        raw = load_json(self._drift_file, default=None) or {}

        recent_runs = raw.get("recent_runs")
        if not isinstance(recent_runs, list):
            recent_runs = []

        rows: list[dict[str, str]] = []
        for row in recent_runs:
            if not isinstance(row, dict):
                continue
            skill = _clip(row.get("skill", ""), 80)
            run_at = _clip(row.get("run_at", ""), 80)
            one_line = _clip(row.get("one_line", ""), 150)
            message_result = _clip(row.get("message_result", ""), 20)
            if message_result not in {"sent", "silent"}:
                message_result = "silent"
            if not skill or not run_at or not one_line:
                continue
            rows.append({
                "skill": skill,
                "run_at": run_at,
                "one_line": one_line,
                "message_result": message_result,
            })

        return {
            "version": 1,
            "recent_runs": rows[-10:],
            "note": _clip(raw.get("note", ""), 150),
            "last_drift_at": raw.get("last_drift_at", ""),
        }

    def get_last_drift_at(self) -> datetime | None:
        """读取上次 Drift 运行时间（用于 min_interval 约束）。

        输出:
            aware datetime；从未运行返回 None。
        """
        raw = self.load_drift().get("last_drift_at", "")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return None

    # ── Skill 状态保存 ─────────────────────────────────────────────

    def save_finish(
        self,
        *,
        skill_used: str,
        one_line: str,
        next_action: str,
        message_result: str,
        note: str | None,
        now_utc: datetime,
    ) -> None:
        """保存一次 Drift 运行的结果。

        同时更新两个文件（都在 state_dir 下）：
        1. state_dir/skills/<skill_used>/state.json — skill 级运行状态
        2. state_dir/drift.json — 全局运行记录（recent_runs + last_drift_at + note）

        输入:
            skill_used: 本次运行的 skill 名称。
            one_line: 一句话运行摘要，最多 150 字符。
            next_action: 下一步动作描述，最多 100 字符。
            message_result: "sent"（本轮已推送消息）或 "silent"（本轮静默）。
            note: 可选的跨轮次笔记（写入 drift.json）。
            now_utc: 本次运行时间（aware datetime）。

        输出:
            None。
        """
        skill_name = str(skill_used or "").strip()
        skill_state_dir = self.state_skills_dir / skill_name
        skill_state_dir.mkdir(parents=True, exist_ok=True)


        state = self._load_skill_state(skill_state_dir)
        save_json(
            skill_state_dir / "state.json",
            {
                "version": 1,
                "last_run_at": now_utc.isoformat(),
                "run_count": max(0, int(state.get("run_count", 0) or 0)) + 1,
                "status": "in_progress",
                "next": _clip(next_action, 100),
            },
        )
        logger.info(
            "[drift_state] save_finish: skill=%s next=%s note=%s",
            skill_name, _clip(next_action, 100), bool(note),
        )

        # 2. 更新 drift.json
        drift = self.load_drift()
        recent_runs = list(drift.get("recent_runs", []))
        recent_runs.append({
            "skill": skill_name,
            "run_at": now_utc.isoformat(),
            "one_line": _clip(one_line, 150),
            "message_result": message_result,
        })

        payload: dict[str, Any] = {
            "version": 1,
            "recent_runs": recent_runs[-10:],
            "note": drift.get("note", ""),
            "last_drift_at": now_utc.isoformat(),
        }
        if note is not None:
            payload["note"] = _clip(note, 150)
        save_json(self._drift_file, payload)
    # ── 内部方法 ────────────────────────────────────────────────────

    def _load_skill_meta(self, skill_dir: Path) -> SkillMeta | None:
        """从源目录加载单个 skill 的元数据。

        SKILL.md 从 source_dir/<name>/ 读取，
        state.json 从 state_dir/skills/<name>/ 读取。

        输入:
            skill_dir: source_dir 下子目录的 Path。

        输出:
            SkillMeta 实例；SKILL.md 不存在或 name 不匹配返回 None。
        """
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        frontmatter = _parse_skill_frontmatter(skill_file.read_text(encoding="utf-8"))
        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()

        if not name or not description or name != skill_dir.name:
            logger.info("[drift_state] 跳过无效 skill dir=%s name=%r", skill_dir, name)
            return None

        # 状态从 state_dir 读取
        state = self._load_skill_state(self.state_skills_dir / name)
        last_run_at = None
        raw_last = state.get("last_run_at")
        if raw_last:
            try:
                last_run_at = datetime.fromisoformat(str(raw_last))
            except ValueError:
                pass

        return SkillMeta(
            name=name,
            description=description,
            last_run_at=last_run_at,
            run_count=max(0, int(state.get("run_count", 0) or 0)),
            status=self._normalize_status(state.get("status")),
            next=_clip(state.get("next", ""), 100),
            requires_mcp=(
                frontmatter.get("requires_mcp", [])
                if isinstance(frontmatter.get("requires_mcp"), list) else []
            ),
        )

    def _load_skill_state(self, skill_dir: Path) -> dict[str, Any]:
        """加载 skill 目录下的 state.json。

        输入:
            skill_dir: state_dir/skills/<name>/ 的 Path。

        输出:
            状态字典。文件不存在或损坏返回空字典。
        """
        raw = load_json(skill_dir / "state.json", default=None) or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalize_status(raw: Any) -> str:
        """规范化 status 字段。

        输入:
            raw: 原始值。

        输出:
            "idle" 或 "in_progress"。非法值回退为 "idle"。
        """
        status = str(raw or "").strip()
        return status if status in {"idle", "in_progress"} else "idle"