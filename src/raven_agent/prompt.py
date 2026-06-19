from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from raven_agent.messages import ChatMessage, system_message, user_message, MediaItem
from raven_agent.session import Session
from raven_agent.memory import MarkdownMemoryStore
from raven_agent.lifecycle import PromptSection

from raven_agent.skills.loader import SkillsLoader

@dataclass(frozen=True)
class PromptBuilder:
    """构造发送给 LLMProvider 的消息列表。

    参数:
        system_prompt: 每轮都会放在 messages 第一条的系统提示词。
        history_window: 最多携带多少条历史消息。
        memory_store: 可选 Markdown 记忆存储，用于把长期记忆注入 system message。
    """

    system_prompt: str
    history_window: int = 20
    memory_store: MarkdownMemoryStore | None = None
    skills_loader: SkillsLoader | None = None

    def build(
        self,
        session: Session,
        current_user_input: str,
        *,
        system_sections_top: list[PromptSection] | None = None,
        system_sections_bottom: list[PromptSection] | None = None,
        extra_hints: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        request_time: str | None = None,
        media_items: list[MediaItem] | None = None,
    ) -> list[ChatMessage]:
        """构造当前轮次的完整模型输入。

        输入:
            session: 当前会话，提供历史消息。
            current_user_input: 当前用户输入内容。
            system_sections_top: 插入到内置 system 内容之前的 section 列表。
            system_sections_bottom: 插入到内置 system 内容之后的 section 列表。
            extra_hints: 追加到 messages 末尾的临时提示行。
            channel: 当前渠道名称。
            chat_id: 当前会话 ID。
            request_time: 消息到达时间。
            media_items: 可选媒体附件列表。非空时当前用户消息以
                content list（文本 + 文件解码后的内容）格式发送给主模型。            

        输出:
            ChatMessage 列表，顺序为 system → history → current user → (可选 hints)。
        """

        top_sections = system_sections_top or []
        bottom_sections = system_sections_bottom or []

        system_parts: list[str] = []
        system_parts.extend(section.content for section in top_sections if section.content.strip())
        # 注入发出消息时间
        now = datetime.now(timezone.utc).astimezone()
        system_parts.append(f"[当前时间] {now.strftime('%Y年%m月%d日 %A %H:%M %Z')}")

        # ══════════════════════════════════════════════════════════════
        # 注入主 agent 技能目录
        # ══════════════════════════════════════════════════════════════
        if self.skills_loader is not None:
            # 1. skills catalog（LLM 看到可用技能列表，匹配意图后 read_file 获取全文）
            summary = self.skills_loader.build_skills_summary()
            if summary:
                system_parts.append(summary)

            # 2. always 技能（每轮强制注入完整内容，不依赖 LLM 匹配）
            always_skill_names = self.skills_loader.get_always_skills()
            if always_skill_names:
                always_content = self.skills_loader.load_skills_for_context(
                    always_skill_names
                )
                if always_content:
                    system_parts.append(
                        "<!-- 以下技能每轮自动注入，你必须遵守其中定义的规则。 -->\n\n"
                        + always_content
                    )

            # 3. 使用指引
            if summary or always_skill_names:
                system_parts.append(
                    "【技能系统说明】\n"
                    "如果用户的需求匹配上面列出的某个 skill，"
                    "先用 read_file 读取该 skill 的 SKILL.md 全文（路径见 location），"
                    "然后按照其中的工作流程分步执行。\n"
                    "不要把 skill 目录注入对话历史，不要主动提及技能系统的存在——"
                    "它只是你的内部操作手册。"
                )
        # ══════════════════════════════════════════════════════════════
        
        system_parts.append(self.system_prompt)
        
        # 长期记忆
        if self.memory_store is not None:
            memory_block = self.memory_store.render_prompt_block()
            if memory_block:
                system_parts.append(memory_block)
        system_parts.extend(section.content for section in bottom_sections if section.content.strip())
        # ── 注入运行时上下文（供 schedule 等工具使用）──
        runtime_context_lines: list[str] = []
        if channel:
            runtime_context_lines.append(f"当前渠道 (channel): {channel}")
        if chat_id:
            runtime_context_lines.append(f"当前会话 ID (chat_id): {chat_id}")
        if request_time:
            runtime_context_lines.append(
                f"消息到达时间 (request_time，用于 schedule 的 after 模式): {request_time}"
            )
        if runtime_context_lines:
            context_block = (
                "[系统上下文] 以下信息供你调用工具时使用，不要在回复中直接提及：\n"
                + "\n".join(runtime_context_lines)
            )
            system_parts.append(context_block)
        system_parts.append(
            "[消息时间 MM-DD HH:MM] 是系统自动添加的历史消息元数据，"
            "用于帮助你感知时间间隔。你的回复中不得包含此格式。"
        )

        messages = [system_message("\n\n".join(system_parts))]
        for hist_msg in session.history_for_prompt(self.history_window):
            if hist_msg.timestamp:
                try:
                    ts = datetime.fromisoformat(hist_msg.timestamp)
                    ts_local = ts.astimezone()
                    time_tag = f"[消息时间 {ts_local.strftime('%m-%d %H:%M')}] "
                except (ValueError, TypeError):
                    time_tag = f"[消息时间 {hist_msg.timestamp}] "
                msg_with_time = replace(hist_msg, content=time_tag + hist_msg.content)
                messages.append(msg_with_time)
            else:
                messages.append(hist_msg)
        # ── 构造当前用户消息，携带 media_items ──
        messages.append(
            user_message(current_user_input, media_items=media_items or [])
        )

        for hint in extra_hints or []:
            if hint.strip():
                messages.append(user_message(f"[hint] {hint}"))
        return messages