from __future__ import annotations

from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本 embedding 提供者协议。

    参数:
        实现类需要提供 embed_text、embed_batch 和 close 方法。

    返回:
        结构化协议类型；自身不直接实例化。
    """

    async def embed_text(self, text: str) -> list[float]:
        """把单条文本转换为 embedding。

        参数:
            text: 要编码的文本。

        返回:
            浮点向量；无法生成时返回空列表。
        """

        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量把文本转换为 embedding。

        参数:
            texts: 要编码的文本列表。

        返回:
            与 texts 顺序一致的向量列表。
        """

        ...

    async def close(self) -> None:
        """关闭 provider 持有的资源。

        参数:
            无。

        返回:
            None。
        """

        ...


class DisabledEmbeddingProvider:
    """未配置 embedding 时使用的空 provider。

    参数:
        无。

    返回:
        DisabledEmbeddingProvider 实例。
    """

    async def embed_text(self, text: str) -> list[float]:
        """返回空向量。

        参数:
            text: 要编码的文本；本实现会忽略。

        返回:
            空列表。
        """

        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """为每条文本返回空向量。

        参数:
            texts: 要编码的文本列表；本实现会忽略内容。

        返回:
            与 texts 等长的空向量列表。
        """

        return [[] for _ in texts]

    async def close(self) -> None:
        """关闭空 provider。

        参数:
            无。

        返回:
            None。
        """

        return None


def _clean_texts(texts: list[str], max_chars: int) -> list[str]:
    """清理并截断 embedding 输入文本。

    参数:
        texts: 原始文本列表。
        max_chars: 每条文本最多保留的字符数。

    返回:
        清理后的文本列表，空文本会变成单个空格。
    """

    cleaned: list[str] = []
    for text in texts:
        value = str(text).strip()[:max_chars]
        cleaned.append(value or " ")
    return cleaned


class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible embedding provider。

    参数:
        api_key: embedding API key。
        base_url: OpenAI-compatible API base URL。
        model: embedding 模型名。
        dimensions: 可选输出维度；0 表示不传 dimensions。
        max_batch_size: 每批最多请求多少条文本。
        max_chars: 每条文本最多保留多少字符。

    返回:
        OpenAICompatibleEmbeddingProvider 实例。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int = 0,
        max_batch_size: int = 10,
        max_chars: int = 2000,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = max(0, int(dimensions))
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_chars = max(1, int(max_chars))

    async def embed_text(self, text: str) -> list[float]:
        """把单条文本转换为 embedding。

        参数:
            text: 要编码的文本。

        返回:
            浮点向量；无法生成时抛出底层 SDK 异常。
        """

        vectors = await self.embed_batch([text])
        return vectors[0] if vectors else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量把文本转换为 embedding。

        参数:
            texts: 要编码的文本列表。

        返回:
            与 texts 顺序一致的浮点向量列表。
        """

        if not texts:
            return []

        results: list[list[float]] = []
        cleaned = _clean_texts(texts, self._max_chars)
        for index in range(0, len(cleaned), self._max_batch_size):
            batch = cleaned[index : index + self._max_batch_size]
            request: dict[str, object] = {
                "model": self._model,
                "input": batch,
            }
            if self._dimensions > 0:
                request["dimensions"] = self._dimensions
            response = await self._client.embeddings.create(**request)
            ordered = sorted(response.data, key=lambda item: item.index)
            results.extend([list(item.embedding) for item in ordered])
        return results

    async def close(self) -> None:
        """关闭 OpenAI SDK 底层 HTTP client。

        参数:
            无。

        返回:
            None。
        """

        await self._client.close()