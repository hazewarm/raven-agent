"""test_subagent_profiles.py —— 三套 profile 工具权限与任务目录测试。

覆盖：
  - research profile 不含写入/shell 工具
  - scripting profile 不含网络工具
  - scripting 写入严格落在 task_dir
  - prompt 包含任务目录路径约束
"""

from pathlib import Path

import pytest

from raven_agent.background.subagent_profiles import (
    SubagentRuntime,
    build_spawn_spec,
    build_spawn_subagent_prompt,
)
from raven_agent.config import LLMConfig
from raven_agent.llm import LLMProvider


def _runtime() -> SubagentRuntime:
    """创建测试用 SubagentRuntime，provider 不会被这些测试实际调用。"""
    provider = LLMProvider(
        LLMConfig(
            provider="test",
            model="dummy",
            api_key="dummy",
            base_url="http://127.0.0.1:1",
        )
    )
    return SubagentRuntime(provider=provider, model="dummy")


def test_research_has_no_write_or_shell(tmp_path: Path) -> None:
    """research 只包含只读和网络工具，不含写入或 shell。"""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        runtime=_runtime(),
        system_prompt="test",
        profile="research",
    )
    names = {tool.name for tool in spec.tools}

    assert "read_text_file" in names
    assert "list_directory" in names
    assert "web_fetch" in names
    assert "web_search" in names
    assert "write_text_file" not in names
    assert "edit_file" not in names
    assert "shell" not in names
    assert "spawn" not in names


def test_scripting_has_no_web_tools(tmp_path: Path) -> None:
    """scripting 可执行/可写任务目录，但没有网络访问工具。"""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        runtime=_runtime(),
        system_prompt="test",
        profile="scripting",
    )
    names = {tool.name for tool in spec.tools}

    assert "read_text_file" in names
    assert "list_directory" in names
    assert "write_text_file" in names
    assert "edit_file" in names
    assert "shell" in names
    assert "web_fetch" not in names
    assert "web_search" not in names
    assert "spawn" not in names


@pytest.mark.asyncio
async def test_scripting_write_scoped_to_task_dir(tmp_path: Path) -> None:
    """scripting 的 write_text_file 只能写入 task_dir，不能写入 workspace。"""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"
    task_dir.mkdir(parents=True)

    spec = build_spawn_spec(
        workspace=workspace,
        task_dir=task_dir,
        runtime=_runtime(),
        system_prompt="test",
        profile="scripting",
    )
    write_tool = next(t for t in spec.tools if t.name == "write_text_file")

    result = await write_tool.execute(path="final_report.md", content="done")

    assert result.metadata["ok"] is True
    assert (task_dir / "final_report.md").read_text(encoding="utf-8") == "done"
    # workspace 根目录不受影响
    assert not (workspace / "final_report.md").exists()


def test_scripting_prompt_mentions_task_dir(tmp_path: Path) -> None:
    """scripting system prompt 包含 task_dir 路径和写入约束。"""
    workspace = tmp_path / "workspace"
    task_dir = workspace / "subagent-runs" / "job-1"

    prompt = build_spawn_subagent_prompt(workspace, task_dir, profile="scripting")

    assert str(task_dir.resolve()) in prompt
    assert "当前任务目录" in prompt
    assert "final_report.md" in prompt