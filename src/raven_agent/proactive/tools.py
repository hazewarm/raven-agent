"""
proactive/tools.py —— Agent Loop 模式的工具 Schema 定义。

Agent Loop 中 LLM 通过调用这些工具完成内容分类、消息撰写和决策提交。
"""

from __future__ import annotations

from typing import Any

TOOL_MARK_INTERESTING: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mark_interesting",
        "description": (
            "将一条或多条候选内容标记为 interesting。可批量传入 item_ids 数组。"
            "调用后这些条目进入本轮推送候选池。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "候选条目的唯一 ID 列表，格式 '{ack_server}:{event_id}'。支持批量传入。",
                },
                "reason": {
                    "type": "string",
                    "description": "标记原因，20 字以内",
                },
            },
            "required": ["item_ids"],
        },
    },
}

TOOL_MARK_NOT_INTERESTING: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mark_not_interesting",
        "description": (
            "将一条或多条候选内容标记为不感兴趣。可批量传入 item_ids 数组。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "候选条目的唯一 ID 列表，格式 '{ack_server}:{event_id}'。支持批量传入。",
                },
                "reason": {
                    "type": "string",
                    "description": "不感兴趣的原因，20 字以内。如 '与用户偏好不符'、'规则命中过滤'",
                },
            },
            "required": ["item_ids"],
        },
    },
}

TOOL_MESSAGE_PUSH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "message_push",
        "description": (
            "暂存一条推送消息草稿。可多次调用修改。"
            "evidence 填写你整合进这条消息的所有 interesting 条目 ID。"
            "调用后还需调 finish_turn 提交。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要发送的消息文本。语气自然像朋友分享，不是系统通知。",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "本条消息引用的条目 ID 列表。格式 '{ack_server}:{event_id}'。",
                },
            },
            "required": ["message", "evidence"],
        },
    },
}

TOOL_FINISH_TURN: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish_turn",
        "description": "结束本轮 proactive 处理。调用后 loop 终止。",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["reply", "skip"],
                    "description": "reply=发送已暂存的消息并结束，skip=不发送直接结束",
                },
                "reason": {
                    "type": "string",
                    "description": "skip 时必填。可选: no_content / user_busy / all_not_interesting",
                },
            },
            "required": ["decision"],
        },
    },
}

TOOL_GET_RECENT_CHAT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_recent_chat",
        "description": "获取用户最近对话记录，用于判断是否适合此时打扰。",
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_RECALL_MEMORY: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": "查询用户长期记忆，判断某条内容是否匹配用户偏好或雷点。仅用于 Content 评估。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词。建议包含来源名以精确匹配。",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_WEB_FETCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "【优先工具】抓取指定 URL 的网页正文。"
            "当你需要核实某条候选内容的细节、补全正文、或校验规则中提到的来源时，优先使用此工具。"
            "返回 JSON 含 text 字段（markdown 格式，最多 8000 字符）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头。",
                },
            },
            "required": ["url"],
        },
    },
}

TOOL_GET_CONTENT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_content",
        "description": (
            "从预取缓存中批量获取候选内容正文。传入 item_ids 列表，返回 {id: text} 映射。"
            "text 为空表示预取失败，此时可考虑用 web_fetch 降级获取。"
            "仅对你想深入了解的条目调用，无需每条都取。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "候选条目 ID 列表，格式 '{ack_server}:{event_id}'",
                },
            },
            "required": ["item_ids"],
        },
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    TOOL_MARK_INTERESTING,
    TOOL_MARK_NOT_INTERESTING,
    TOOL_MESSAGE_PUSH,
    TOOL_FINISH_TURN,
    TOOL_GET_RECENT_CHAT,
    TOOL_RECALL_MEMORY,
    TOOL_WEB_FETCH,
    TOOL_GET_CONTENT,
]