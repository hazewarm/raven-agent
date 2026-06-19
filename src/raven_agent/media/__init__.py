"""raven-agent 多媒体基础设施包。

本包提供图片编码、MIME 检测、音频处理等纯函数基础设施。
不包含面向 LLM 的 Tool 子类——Tool 子类在 raven_agent.tools 中实现，
它们从本包导入基础设施函数。
"""

from raven_agent.media.image import (
    IMAGE_MAX_DATA_URI_BYTES,
    IMAGE_MAX_EDGE,
    detect_image_mime,
    encode_image_to_data_uri,
)

from raven_agent.media.audio import transcribe_audio