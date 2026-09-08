"""Async OpenAI-compatible chat completion provider.

The provider returns plain Markdown deltas.  JSON response parsing belongs to
neither the provider nor the assistant, so partial output is always renderable.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol


class LLMProvider(Protocol):
    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60,
        client: Any | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("httpx is required for the async LLM provider") from exc
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout, connect=min(2.5, self.timeout))
                )
        return self._client

    async def aclose(self) -> None:
        """Close the one shared HTTP client owned by this provider."""

        if not self._owns_client:
            return
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> AsyncIterator[str]:
        # Missing credentials are a supported offline mode.  Returning before
        # importing/constructing httpx guarantees that no network request is
        # attempted and lets the assistant emit its local fallback.
        if not self.api_key:
            return
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.model.startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        client = await self._get_client()
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = _content_delta(data)
                if delta:
                    yield delta


def _content_delta(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    delta = choice.get("delta") or choice.get("message") or {}
    if not isinstance(delta, Mapping):
        return ""
    content = delta.get("content", "")
    return content if isinstance(content, str) else ""
