from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果。

    参数:
        text: 给模型或用户阅读的文本结果。
        metadata: 结构化附加信息，供测试、日志或后续系统使用。
    """

    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_tool_result(result: str | ToolResult) -> ToolResult:
    """把工具返回值统一转换为 ToolResult。

    参数:
        result: 工具返回的字符串或 ToolResult。

    返回:
        ToolResult 对象。
    """

    if isinstance(result, ToolResult):
        return result
    return ToolResult(text=result)


class Tool(ABC):
    """所有工具必须实现的抽象基类。

    参数:
        无。具体工具可以在子类 __init__ 中定义自己的初始化参数。
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI tool calling 使用的 schema。

        返回:
            OpenAI Chat Completions API 接收的 tool schema 字典。
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | ToolResult:
        """执行工具。

        参数:
            **kwargs: 模型或调用方传入的工具参数。

        返回:
            字符串或 ToolResult，最终会被规范化为 ToolResult。
        """