"""Unified HTTP client for all external API calls using httpx."""

import logging
import textwrap
from functools import lru_cache
from typing import Any

import httpx

from ..config import settings
from ..text import normalize_text

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_http_client() -> httpx.Client:
    """Return a cached httpx.Client with connection pooling."""
    return httpx.Client(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1000,
    extra_options: dict[str, Any] | None = None,
) -> str:
    """Send a chat completion request. Auto-routes zhipuai SDK for bigmodel.cn."""
    if _should_use_zhipu_sdk():
        return _zhipu_completion(messages, temperature, max_tokens, extra_options)

    payload: dict[str, Any] = {
        "model": settings.ai_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_options:
        payload.update(extra_options)

    client = get_http_client()
    url = settings.ai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.ai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Chat API HTTP {exc.response.status_code}: {_truncate_body(exc.response)}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Chat API request failed: {exc}") from exc


def embedding_request(texts: list[str]) -> list[list[float]]:
    """Send an embedding request."""
    if not settings.embedding_api_key:
        raise RuntimeError("EMBEDDING_API_KEY is not configured.")
    if not texts:
        return []

    client = get_http_client()
    url = settings.embedding_base_url.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.embedding_model, "input": texts}
    try:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        rows = sorted(body["data"], key=lambda row: int(row.get("index", 0)))
        return [list(map(float, row["embedding"])) for row in rows]
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Embedding API HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Embedding API request failed: {exc}") from exc


def describe_error(exc: Exception, api_key: str = "") -> str:
    """Describe an HTTP/request error for user-facing messages, sanitizing secrets."""
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTPError {exc.response.status_code}"
        body = _truncate_body(exc.response, max_chars=200)
        if body:
            detail += f": {body}"
    elif isinstance(exc, (httpx.RequestError, OSError)):
        detail = f"RequestError: {exc}"
    else:
        detail = str(exc) or type(exc).__name__

    text = _normalize_str(detail)
    if api_key:
        text = text.replace(api_key, "***")
    return textwrap.shorten(text, width=220, placeholder="...")


def _should_use_zhipu_sdk() -> bool:
    from urllib.parse import urlparse

    host = urlparse(settings.ai_base_url).hostname or ""
    return host.endswith("bigmodel.cn")


def _zhipu_completion(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    extra_options: dict[str, Any] | None = None,
) -> str:
    from zhipuai import ZhipuAI

    model = settings.ai_model.lower()
    extra: dict[str, Any] = {}
    if any(name in model for name in ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")):
        extra["thinking"] = {"type": "disabled"}
    if extra_options:
        extra.update(extra_options)

    client_zhipu = ZhipuAI(
        api_key=settings.ai_api_key,
        base_url=settings.ai_base_url.rstrip("/"),
        timeout=int(settings.ai_timeout),
        max_retries=0,
    )
    response = client_zhipu.chat.completions.create(
        model=settings.ai_model,
        messages=messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=max_tokens,
        **extra,
        stream=False,
    )
    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected model response: {response}") from exc

    text = (getattr(message, "content", "") or "").strip()
    if text:
        return text

    finish_reason = getattr(choice, "finish_reason", "")
    reasoning = getattr(message, "reasoning_content", "")
    detail = "Empty model response"
    if finish_reason:
        detail += f"; finish_reason={finish_reason}"
    if reasoning:
        detail += "; reasoning_content was returned without final content"
    raise RuntimeError(detail)


def _truncate_body(response: httpx.Response, max_chars: int = 200) -> str:
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - best effort diagnostics only.
        return ""
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _normalize_str(value: str) -> str:
    return normalize_text(value)
