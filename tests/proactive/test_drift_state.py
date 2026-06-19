"""
test_drift_state.py —— DriftStateStore 单元测试。

覆盖：
  - scan_skills(): 空目录 / 单个 skill / 多个 skill / 无效 SKILL.md
  - save_finish(): state.json 写入 / drift.json 写入 / recent_runs 截断
  - load_drift(): 空文件 / 正常文件 / 损坏文件
  - get_last_drift_at(): 无记录 / 有记录 / 错误格式
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from raven_agent.proactive.drift_state import (
    DriftStateStore,
    SkillMeta,
    _parse_skill_frontmatter,
)


def _skill_md(name: str, description: str, **extra) -> str:
    """生成 SKILL.md 内容。"""
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    for k, v in extra.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append("## 目标")
    lines.append("")
    lines.append("这是一个测试 skill。")
    return "\n".join(lines)


class TestParseSkillFrontmatter:
    """_parse_skill_frontmatter 基础解析测试。"""

    def test_empty_content(self):
        assert _parse_skill_frontmatter("") == {}

    def test_no_frontmatter(self):
        assert _parse_skill_frontmatter("# 标题\n正文") == {}

    def test_simple_kv(self):
        result = _parse_skill_frontmatter(
            "---\nname: test\ndescription: 测试用\n---\n正文"
        )
        assert result["name"] == "test"
        assert result["description"] == "测试用"

    def test_list_value(self):
        result = _parse_skill_frontmatter(
            "---\nname: test\ndescription: 测试\nrequires_mcp:\n  - srv1\n  - srv2\n---\n"
        )
        assert result["requires_mcp"] == ["srv1", "srv2"]

    def test_empty_list_value(self):
        result = _parse_skill_frontmatter(
            "---\nname: test\ndescription: 测试\nrequires_mcp:\n---\n"
        )
        assert result["requires_mcp"] == ""


def _make_store(source_dir: Path, state_dir: Path) -> DriftStateStore:
    """创建 DriftStateStore，同时确保 source_dir 存在（即使是空目录）。"""
    source_dir.mkdir(parents=True, exist_ok=True)
    return DriftStateStore(source_dir=source_dir, state_dir=state_dir)


def _make_skill(source_dir: Path, state_dir: Path, name: str) -> DriftStateStore:
    """在 source_dir 下创建单个 skill 的 SKILL.md。"""
    skill_dir = source_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        _skill_md(name, f"{name} 的描述"),
        encoding="utf-8",
    )
    return _make_store(source_dir, state_dir)


class TestDriftStateStore:
    """DriftStateStore 核心行为测试。"""

    def test_empty_dir_scan_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(
                source_dir=Path(tmp) / "src",
                state_dir=Path(tmp) / "drift",
            )
            skills = store.scan_skills()
            assert skills == []

    def test_scan_single_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            skills = store.scan_skills()
            assert len(skills) == 1
            s = skills[0]
            assert s.name == "my-skill"
            assert s.description == "my-skill 的描述"
            assert s.last_run_at is None
            assert s.run_count == 0
            assert s.status == "idle"
            assert s.requires_mcp == []

    def test_scan_multiple_skills_sorted_by_last_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            for name in ("skill-a", "skill-b", "skill-c"):
                _make_skill(source_dir, state_dir, name)
            store = _make_store(source_dir, state_dir)
            skills = store.scan_skills()
            assert len(skills) == 3

            # 没有运行记录时，按名称字母序排列（sorted by last_run_at=None）
            names = [s.name for s in skills]
            assert set(names) == {"skill-a", "skill-b", "skill-c"}

    def test_skill_name_mismatch_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            skill_dir = source_dir / "real-name"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                _skill_md("wrong-name", "描述不匹配"),
                encoding="utf-8",
            )

            store = _make_store(source_dir, state_dir)
            skills = store.scan_skills()
            assert len(skills) == 0  # 目录名 "real-name" ≠ frontmatter name "wrong-name"

    def test_skill_without_frontmatter_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            skill_dir = source_dir / "bad-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "没有 frontmatter 的 SKILL.md",
                encoding="utf-8",
            )

            store = _make_store(source_dir, state_dir)
            skills = store.scan_skills()
            assert len(skills) == 0

    def test_valid_skill_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            for name in ("skill-a", "skill-b"):
                _make_skill(source_dir, state_dir, name)
            store = _make_store(source_dir, state_dir)
            names = store.valid_skill_names()
            assert names == {"skill-a", "skill-b"}

    def test_save_finish_writes_state_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")
            now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
            store.save_finish(
                skill_used="my-skill",
                one_line="完成了一次测试运行",
                next_action="继续下一个步骤",
                message_result="silent",
                note=None,
                now_utc=now,
            )

            # 验证 state.json（写入 state_dir）
            import json
            state = json.loads(
                (state_dir / "skills" / "my-skill" / "state.json").read_text(encoding="utf-8")
            )
            assert state["last_run_at"] == "2026-06-03T12:00:00+00:00"
            assert state["run_count"] == 1
            assert state["next"] == "继续下一个步骤"

            # 验证 drift.json（写入 state_dir）
            drift = json.loads(
                (state_dir / "drift.json").read_text(encoding="utf-8")
            )
            assert len(drift["recent_runs"]) == 1
            assert drift["recent_runs"][0]["skill"] == "my-skill"
            assert drift["recent_runs"][0]["message_result"] == "silent"
            assert drift["last_drift_at"] == "2026-06-03T12:00:00+00:00"

    def test_save_finish_truncates_recent_runs_to_10(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")
            for i in range(15):
                store.save_finish(
                    skill_used="my-skill",
                    one_line=f"第 {i + 1} 次运行",
                    next_action="继续",
                    message_result="silent",
                    note=None,
                    now_utc=datetime(2026, 6, 3, i, 0, 0, tzinfo=timezone.utc),
                )

            drift = store.load_drift()
            assert len(drift["recent_runs"]) == 10  # 截断到 10

    def test_get_last_drift_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")
            # 无记录
            assert store.get_last_drift_at() is None

            # 有记录
            now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
            store.save_finish(
                skill_used="my-skill",
                one_line="测试运行",
                next_action="继续",
                message_result="silent",
                note=None,
                now_utc=now,
            )
            last = store.get_last_drift_at()
            assert last == now

    def test_load_drift_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            source_dir.mkdir(parents=True, exist_ok=True)
            state_dir = Path(tmp) / "drift"
            state_dir.mkdir(parents=True, exist_ok=True)

            store = DriftStateStore(source_dir=source_dir, state_dir=state_dir)
            drift = store.load_drift()
            assert drift["version"] == 1
            assert drift["recent_runs"] == []

    def test_scan_rescans_after_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "src"
            state_dir = Path(tmp) / "drift"
            store = _make_skill(source_dir, state_dir, "my-skill")

            # 首次扫描：run_count = 0
            skills = store.scan_skills()
            assert skills[0].run_count == 0

            # 保存一次运行
            store.save_finish(
                skill_used="my-skill",
                one_line="运行完成",
                next_action="下一步",
                message_result="silent",
                note=None,
                now_utc=datetime.now(timezone.utc),
            )

            # 再次扫描：run_count 更新
            skills = store.scan_skills()
            assert skills[0].run_count == 1
            assert skills[0].next == "下一步"