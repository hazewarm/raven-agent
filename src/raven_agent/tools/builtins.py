from __future__ import annotations

from pathlib import Path

from raven_agent.tools.filesystem import EditFileTool, ReadImageInfoTool, WriteTextFileTool
from raven_agent.tools.readonly import ListDirectoryTool, ReadTextFileTool
from raven_agent.tools.registry import ToolRegistry
from raven_agent.tools.search import ToolSearchTool
from raven_agent.tools.shell import ShellTool
from raven_agent.tools.web_fetch import WebFetchTool
from raven_agent.tools.web_search import WebSearchTool
from raven_agent.tools.message_push import MessagePushTool
from raven_agent.scheduler import SchedulerService
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from raven_agent.llm import LLMProvider

from raven_agent.tools.vision import ReadImageVisionTool
from raven_agent.tools.audio import TranscribeAudioTool


def build_default_tools(
    allowed_dir: Path | None = None,
    *,
    web_search_api_key: str = "",
    web_search_gl: str = "cn",
    web_search_hl: str = "zh-cn",
    scheduler: SchedulerService | None = None,
    scheduler_tz: str = "UTC",
    vl_provider: "LLMProvider | None" = None,
    vl_model: str = "",
    audio_model: str = "small",
    audio_enabled: bool = True,
) -> ToolRegistry:
    """创建默认内置工具注册表。

    输入:
        allowed_dir: 默认文件和 shell 工具允许访问或运行的根目录。
        web_search_api_key: SerpAPI API Key，由 config.toml 加载后注入。
        web_search_gl: Web Search 默认国家代码。
        web_search_hl: Web Search 默认语言。
        scheduler: SchedulerService 实例。
        scheduler_tz: 调度器时区。
        vl_provider: VL 视觉模型 Provider；不为 None 时注册 read_image_vision 工具。
        vl_model: VL 视觉模型名称。

    输出:
        ToolRegistry。包含 always-on 基础工具和 deferred 高风险/扩展工具。
    """

    registry = ToolRegistry()
    registry.register(ToolSearchTool(registry), risk="read-only", always_on=True, search_hint="工具搜索 工具发现 解锁工具")
    registry.register(ReadTextFileTool(allowed_dir=allowed_dir), risk="read-only", always_on=True, search_hint="读取文件 查看文本 查看源码 查看配置")
    registry.register(ListDirectoryTool(allowed_dir=allowed_dir), risk="read-only", always_on=True, search_hint="列目录 查看目录 项目结构 文件列表")
    registry.register(WriteTextFileTool(allowed_dir=allowed_dir), risk="write", always_on=False, search_hint="写文件 创建文件 保存文本 覆盖文件")
    registry.register(EditFileTool(allowed_dir=allowed_dir), risk="write", always_on=False, search_hint="编辑文件 修改文件 替换文本 代码修改")
    registry.register(ReadImageInfoTool(allowed_dir=allowed_dir), risk="read-only", always_on=True, search_hint="图片信息 图片尺寸 图片格式 image metadata")
    registry.register(WebFetchTool(), risk="read-only", always_on=True, search_hint="抓取网页 读取URL 文档 网页内容 HTTP fetch")
    registry.register(WebSearchTool(api_key=web_search_api_key, default_gl=web_search_gl, default_hl=web_search_hl), risk="read-only", always_on=True, search_hint="搜索互联网 搜索网页 Google SerpAPI 时效信息 新闻 查询资料")
    registry.register(ShellTool(working_dir=allowed_dir), risk="external-side-effect", always_on=False, search_hint="shell bash 命令 运行测试 执行命令 诊断环境")
    registry.register(MessagePushTool(), risk="external-side-effect", always_on=True, search_hint="发送消息 推送通知 主动发消息 push message send",)
    if scheduler is not None:
        from raven_agent.tools.schedule import (
            CancelScheduleTool,
            ListSchedulesTool,
            ScheduleTool,
        )
        registry.register(
            ScheduleTool(scheduler, default_tz=scheduler_tz),
            risk="write",
            search_hint="定时 日程 闹钟 延时执行 提醒 每天",
        )
        registry.register(
            ListSchedulesTool(scheduler),
            risk="read-only",
            search_hint="提醒列表 已有计划 查看定时任务 日程查询",
        )
        registry.register(
            CancelScheduleTool(scheduler),
            risk="write",
            search_hint="删除提醒 取消任务 删除日程 取消闹钟",
        )

    if vl_provider is not None and vl_model:
        registry.register(
            ReadImageVisionTool(vl_provider=vl_provider, vl_model=vl_model, allowed_dir=allowed_dir),
            risk="read-only",
            always_on=True,
            search_hint="看图 识图 图片内容 视觉识别 VL vision",
        )
    # ── 音频转录工具 ──
    if audio_enabled:
        registry.register(
            TranscribeAudioTool(allowed_dir=allowed_dir, model=audio_model),
            risk="read-only",
            always_on=True,
            search_hint="语音转文字 音频转录 STT 听语音 whisper 语音识别",
        )

    return registry