from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from raven_agent.llm import LLMProvider
from raven_agent.tools.base import Tool
from raven_agent.tools.filesystem import EditFileTool, WriteTextFileTool
from raven_agent.tools.hooks import ToolHook
from raven_agent.tools.readonly import ListDirectoryTool, ReadTextFileTool
from raven_agent.tools.shell import ShellTool
from raven_agent.tools.web_fetch import WebFetchTool
from raven_agent.tools.web_search import WebSearchTool

PROFILE_RESEARCH = "research"
PROFILE_SCRIPTING = "scripting"
PROFILE_GENERAL = "general"


@dataclass(frozen=True)
class SubagentRuntime:
    """构建 SubAgent 时复用的运行时依赖。

    输入:
        provider: LLMProvider 实例。
        model: SubAgent 使用的模型名；为空时 provider 会使用主配置模型。
        web_search_api_key: WebSearchTool 使用的 SerpAPI Key。
        web_search_gl: WebSearchTool 默认国家代码。
        web_search_hl: WebSearchTool 默认语言。
        tool_hooks: 传给子 Agent 的工具 hook 列表。

    输出:
        SubagentRuntime 实例。
    """

    provider: LLMProvider
    model: str = ""
    web_search_api_key: str = ""
    web_search_gl: str = "cn"
    web_search_hl: str = "zh-cn"
    tool_hooks: list[ToolHook] = field(default_factory=list)


@dataclass(frozen=True)
class SubagentSpec:
    """一个 SubAgent 的构造规格。

    输入:
        tools: 子 Agent 可用工具列表。
        system_prompt: 子 Agent system prompt。
        max_iterations: 子 Agent 最大 ReAct 轮数。

    输出:
        SubagentSpec 实例。调用 build(runtime) 后得到 SubAgent。
    """

    tools: list[Tool]
    system_prompt: str = ""
    max_iterations: int = 30

    def build(self, runtime: SubagentRuntime) -> "SubAgent":
        """按规格构建 SubAgent。

        输入:
            runtime: SubagentRuntime，提供 provider/model/hooks。

        输出:
            SubAgent 实例。
        """
        # 惰性导入打破跨模块循环依赖：
        #   subagent.py → tools/__init__.py → spawn.py → subagent_profiles.py
        #   → (顶层) subagent.py  ← 循环，模块尚未完成初始化
        from raven_agent.background.subagent import SubAgent

        return SubAgent(
            provider=runtime.provider,
            model=runtime.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            tool_hooks=runtime.tool_hooks,
        )


def build_research_subagent_prompt(workspace: Path, task_dir: Path) -> str:
    """构建 research profile 的 system prompt。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录；research 模式只读，不写入。

    输出:
        system prompt 字符串。
    """
    workspace_path = str(workspace.expanduser().resolve())
    return f"""\
你是主 agent 派生的调研型子 agent。你擅长跨大量来源检索信息、分析文件、综合结论。

=== 关键约束：只读模式，禁止修改任何文件 ===
你被严格禁止：
- 创建或写入文件（禁止 write_text_file、edit_file，以及任何形式的文件创建）
- 执行 shell 命令
- 修改工作区内任何现有文件
- 再次创建子任务（你没有 spawn 工具）

你的角色是：搜索、阅读、抓取、分析，最终以文本形式输出报告。

=== 可用工具提示 ===
- 读取文件：read_text_file
- 列目录：list_directory
- 抓取网页：web_fetch
- 搜索网页：web_search

=== 输出要求 ===
- 直接输出文本报告，不要写入文件
- 若任务未完成，必须说明：已完成什么 / 未完成什么 / 建议下一步
- 不要直接与用户对话；你的结果会回传给主 agent

工作区根目录：{workspace_path}
"""


def build_scripting_subagent_prompt(workspace: Path, task_dir: Path) -> str:
    """构建 scripting profile 的 system prompt。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录；写入只允许发生在这里。

    输出:
        system prompt 字符串。
    """
    workspace_path = str(workspace.expanduser().resolve())
    task_dir_path = str(task_dir.expanduser().resolve())
    return f"""\
你是主 agent 派生的执行型子 agent。你负责运行脚本、处理数据、生成文件。

=== 关键约束：写入仅限任务目录 ===
你被严格禁止：
- 向任务目录之外写入任何文件
- 删除 workspace 中的已有文件
- 再次创建子任务（你没有 spawn 工具）

=== 关于网络访问 ===
你当前的工具集中不包含 web_fetch / web_search。如果需要获取外部信息，
请通过 read_text_file 读取 worktree 中已有的文件，或将需求写进最终报告的
"未完成项"交还给主 agent 处理。
- 删除 workspace 中的已有文件
- 再次创建子任务（你没有 spawn 工具）

=== 工作指引 ===
- 所有产出文件只能写入当前任务目录：{task_dir_path}
- shell 的工作目录也是当前任务目录
- 可以读取工作区文件，但不要修改工作区原文件
- 最终报告默认写成 final_report.md 放在任务目录；若不需要持久化则直接输出文本

=== 可用工具提示 ===
- 读取文件：read_text_file
- 列目录：list_directory
- 写文件：write_text_file
- 编辑文件：edit_file
- 执行命令：shell

工作区根目录（只读）：{workspace_path}
当前任务目录（可写）：{task_dir_path}
"""


def build_general_subagent_prompt(workspace: Path, task_dir: Path) -> str:
    """构建 general profile 的 system prompt。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录；写入只允许发生在这里。

    输出:
        system prompt 字符串。
    """
    workspace_path = str(workspace.expanduser().resolve())
    task_dir_path = str(task_dir.expanduser().resolve())
    return f"""\
你是主 agent 派生的通用型子 agent。你可以调研信息、执行命令、读写任务目录内文件。

=== 关键约束 ===
- 禁止再次创建后台子任务（你没有 spawn 工具）
- 不直接与用户对话；你的结果会回传给主 agent
- 写入操作只能发生在当前任务目录，禁止修改工作区根目录的已有文件
- 不要把产物散落到任务目录之外

=== 工作指引 ===
- 先明确任务边界，避免过度延伸
- 调研和执行按需切换，不要同时打开过多方向
- 若创建或修改文件，最终结果必须列出每个文件的完整路径
- 最终报告若需持久化，写成 final_report.md 放在任务目录

工作区根目录（可读取）：{workspace_path}
当前任务目录（可写）：{task_dir_path}
"""


def build_research_spec(
    *,
    workspace: Path,
    task_dir: Path,
    runtime: SubagentRuntime,
    system_prompt: str,
    max_iterations: int = 20,
) -> SubagentSpec:
    """构建 research profile 的 SubagentSpec。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录。
        runtime: SubagentRuntime，提供 WebSearch 配置。
        system_prompt: 子 Agent system prompt。
        max_iterations: 最大 ReAct 轮数。

    输出:
        SubagentSpec。工具只包含只读文件和网页工具。
    """
    return SubagentSpec(
        tools=[
            ReadTextFileTool(allowed_dir=workspace),
            ListDirectoryTool(allowed_dir=workspace),
            WebFetchTool(),
            WebSearchTool(
                api_key=runtime.web_search_api_key,
                default_gl=runtime.web_search_gl,
                default_hl=runtime.web_search_hl,
            ),
        ],
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def build_scripting_spec(
    *,
    workspace: Path,
    task_dir: Path,
    runtime: SubagentRuntime,
    system_prompt: str,
    max_iterations: int = 20,
) -> SubagentSpec:
    """构建 scripting profile 的 SubagentSpec。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录。
        runtime: SubagentRuntime；当前只为接口一致保留。
        system_prompt: 子 Agent system prompt。
        max_iterations: 最大 ReAct 轮数。

    输出:
        SubagentSpec。工具包含读取、目录、任务目录写入和 shell，不包含网络工具。
    """
    return SubagentSpec(
        tools=[
            ReadTextFileTool(allowed_dir=workspace),
            ListDirectoryTool(allowed_dir=workspace),
            WriteTextFileTool(allowed_dir=task_dir),
            EditFileTool(allowed_dir=task_dir),
            ShellTool(working_dir=task_dir),
        ],
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def build_general_spec(
    *,
    workspace: Path,
    task_dir: Path,
    runtime: SubagentRuntime,
    system_prompt: str,
    max_iterations: int = 20,
) -> SubagentSpec:
    """构建 general profile 的 SubagentSpec。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录。
        runtime: SubagentRuntime，提供 WebSearch 配置。
        system_prompt: 子 Agent system prompt。
        max_iterations: 最大 ReAct 轮数。

    输出:
        SubagentSpec。工具包含调研和执行能力，但写入仍限制在 task_dir。
    """
    return SubagentSpec(
        tools=[
            ReadTextFileTool(allowed_dir=workspace),
            ListDirectoryTool(allowed_dir=workspace),
            WebFetchTool(),
            WebSearchTool(
                api_key=runtime.web_search_api_key,
                default_gl=runtime.web_search_gl,
                default_hl=runtime.web_search_hl,
            ),
            WriteTextFileTool(allowed_dir=task_dir),
            EditFileTool(allowed_dir=task_dir),
            ShellTool(working_dir=task_dir),
        ],
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


_PROFILE_PROMPT_BUILDERS = {
    PROFILE_RESEARCH: build_research_subagent_prompt,
    PROFILE_SCRIPTING: build_scripting_subagent_prompt,
    PROFILE_GENERAL: build_general_subagent_prompt,
}

_PROFILE_SPEC_BUILDERS = {
    PROFILE_RESEARCH: build_research_spec,
    PROFILE_SCRIPTING: build_scripting_spec,
    PROFILE_GENERAL: build_general_spec,
}


def build_spawn_subagent_prompt(
    workspace: Path,
    task_dir: Path,
    profile: str = PROFILE_RESEARCH,
) -> str:
    """根据 profile 选择对应的 subagent system prompt。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录。
        profile: research / scripting / general；未知值回退到 research。

    输出:
        system prompt 字符串。
    """
    builder = _PROFILE_PROMPT_BUILDERS.get(profile, build_research_subagent_prompt)
    return builder(workspace, task_dir)


def build_spawn_spec(
    *,
    workspace: Path,
    task_dir: Path,
    runtime: SubagentRuntime,
    system_prompt: str,
    max_iterations: int = 20,
    profile: str = PROFILE_RESEARCH,
) -> SubagentSpec:
    """根据 profile 构建 SubagentSpec。

    输入:
        workspace: 工作区根目录。
        task_dir: 当前任务目录。
        runtime: SubagentRuntime。
        system_prompt: 子 Agent system prompt。
        max_iterations: 最大 ReAct 轮数。
        profile: research / scripting / general；未知值回退到 research。

    输出:
        SubagentSpec。
    """
    builder = _PROFILE_SPEC_BUILDERS.get(profile, build_research_spec)
    return builder(
        workspace=workspace,
        task_dir=task_dir,
        runtime=runtime,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )