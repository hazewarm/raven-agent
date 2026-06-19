from __future__ import annotations

from pathlib import Path


def resolve_tool_path(path: str, allowed_dir: Path | None = None) -> Path:
    """解析工具传入路径，并可选限制在 allowed_dir 内。

    输入:
        path: 用户或模型传入的文件路径，可以是相对路径、绝对路径或带 ~ 的路径。
        allowed_dir: 允许访问的根目录；为 None 时不做目录限制。

    输出:
        解析后的绝对 Path。

    异常:
        PermissionError: 当解析后的路径不在 allowed_dir 内时抛出。
    """

    raw_path = Path(path).expanduser()
    base_dir = allowed_dir.resolve() if allowed_dir is not None else None

    if base_dir is not None and not raw_path.is_absolute():
        cwd_resolved = raw_path.resolve()
        if cwd_resolved.is_relative_to(base_dir):
            resolved = cwd_resolved
        else:
            resolved = (base_dir / raw_path).resolve()
    else:
        resolved = raw_path.resolve()

    if base_dir is not None and not resolved.is_relative_to(base_dir):
        raise PermissionError(f"路径超出允许目录: {path}")

    return resolved