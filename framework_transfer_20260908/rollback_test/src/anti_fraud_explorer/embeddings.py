"""OpenAI-compatible embedding indexing and semantic retrieval for case evidence."""

from __future__ import annotations

import json
import math
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import config
from .dataset import FraudCase, KnowledgeBase, normalize_text


class EmbeddingUnavailable(RuntimeError):
    """Raised when semantic retrieval is not configured or cannot be used."""


class _EmbeddingHTTPClient(Protocol):
    """Small subset of ``httpx.Client`` needed by the embedding boundary."""

    def post(
        self,
        url: str,
        *,
        json: Any,
        headers: Any,
        timeout: httpx.Timeout,
    ) -> httpx.Response: ...


@dataclass(frozen=True)
class EmbeddingRecord:
    item_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingIndex:
    model: str
    base_url: str
    dimensions: int
    records: tuple[EmbeddingRecord, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EmbeddingIndex:
        records = []
        dimensions = int(payload.get("dimensions") or 0)
        for raw in payload.get("embeddings", []):
            item_id = str(raw.get("id") or "")
            vector = normalize_vector(raw.get("embedding") or [])
            if not item_id or not vector:
                continue
            dimensions = dimensions or len(vector)
            records.append(EmbeddingRecord(item_id=item_id, vector=tuple(vector)))
        return cls(
            model=str(payload.get("model") or ""),
            base_url=str(payload.get("base_url") or ""),
            dimensions=dimensions,
            records=tuple(records),
        )


class EmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        http_client: _EmbeddingHTTPClient | None = None,
    ):
        self.api_key = config.EMBEDDING_API_KEY if api_key is None else api_key
        self.base_url = (config.EMBEDDING_BASE_URL if base_url is None else base_url).rstrip("/")
        self.model = config.EMBEDDING_MODEL if model is None else model
        self.timeout = config.EMBEDDING_TIMEOUT if timeout is None else timeout
        self.max_retries = config.EMBEDDING_MAX_RETRIES if max_retries is None else max_retries
        self.retry_backoff = config.EMBEDDING_RETRY_BACKOFF if retry_backoff is None else retry_backoff
        self._http_client = http_client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingUnavailable("EMBEDDING_API_KEY is not configured.")
        if not texts:
            return []

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                return self._embed_texts_once(texts)
            except (EmbeddingUnavailable, TimeoutError) as exc:
                last_error = str(exc) or type(exc).__name__
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay(exc, attempt))
        raise EmbeddingUnavailable(last_error or "Embedding request failed.")

    def retry_delay(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, EmbeddingUnavailable) and isinstance(exc.__cause__, httpx.HTTPStatusError):
            retry_after = exc.__cause__.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self.retry_backoff * (attempt + 1)

    def _embed_texts_once(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._http_client
            if client is None:
                with httpx.Client() as owned_client:
                    return self._request_embeddings(owned_client, texts)
            return self._request_embeddings(client, texts)
        except Exception as exc:
            raise EmbeddingUnavailable(describe_embedding_error(exc, self.api_key)) from exc

    def _request_embeddings(
        self,
        client: _EmbeddingHTTPClient,
        texts: list[str],
    ) -> list[list[float]]:
        response = client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(float(self.timeout)),
        )
        response.raise_for_status()
        body = response.json()
        rows = sorted(body["data"], key=lambda row: int(row.get("index", 0)))
        return [list(map(float, row["embedding"])) for row in rows]


def build_embedding_text(item: FraudCase, max_chars: int | None = None) -> str:
    max_chars = config.EMBEDDING_TEXT_MAX_CHARS if max_chars is None else max_chars
    parts = [
        f"案例：{item.title}",
        f"类别：{item.ccl2023_category or item.custom_subcategory}",
        f"入口渠道：{'、'.join(item.entry_channels)}" if item.entry_channels else "",
        f"受害人群：{item.victim_group}" if item.victim_group else "",
        f"关键手法：{'、'.join(item.key_methods)}" if item.key_methods else "",
        f"风险等级：{item.risk_level}" if item.risk_level else "",
        f"风险信号：{item.risk_signals}" if item.risk_signals else "",
        f"防范建议：{item.prevention_advice}" if item.prevention_advice else "",
        f"事实：{item.content}",
    ]
    text = normalize_text("\n".join(part for part in parts if part))
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def build_index_payload(
    kb: KnowledgeBase,
    client: EmbeddingClient,
    batch_size: int | None = None,
) -> dict[str, Any]:
    batch_size = config.EMBEDDING_BATCH_SIZE if batch_size is None else batch_size
    rows: list[dict[str, Any]] = []
    dimensions = 0
    for start in range(0, len(kb.items), batch_size):
        batch = kb.items[start : start + batch_size]
        texts = [build_embedding_text(item) for item in batch]
        vectors = client.embed_texts(texts)
        if len(vectors) != len(batch):
            raise EmbeddingUnavailable("Embedding API returned a mismatched batch length.")
        for item, vector in zip(batch, vectors, strict=True):
            normalized = normalize_vector(vector)
            dimensions = dimensions or len(normalized)
            rows.append({"id": item.id, "embedding": normalized})

    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "schema_version": kb.schema_version,
            "generated_at": kb.generated_at,
            "item_count": len(kb.items),
        },
        "base_url": client.base_url,
        "model": client.model,
        "dimensions": dimensions,
        "embeddings": rows,
    }


def write_index(payload: dict[str, Any], path: Path | None = None) -> None:
    path = config.EMBEDDING_INDEX_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _load_embedding_index.cache_clear()


@lru_cache(maxsize=1)
def _load_embedding_index(path_text: str) -> EmbeddingIndex | None:
    path = Path(path_text)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return EmbeddingIndex.from_payload(json.load(f))


def load_embedding_index(path: Path | None = None) -> EmbeddingIndex | None:
    path = config.EMBEDDING_INDEX_PATH if path is None else path
    return _load_embedding_index(str(path))


def embedding_scores(
    kb: KnowledgeBase,
    query: str,
    candidates: Iterable[FraudCase],
    client: EmbeddingClient | None = None,
    min_score: float | None = None,
) -> dict[str, float]:
    query = normalize_text(query)
    if not query:
        return {}

    index = load_embedding_index()
    if index is None or not index.records:
        raise EmbeddingUnavailable("Embedding index does not exist.")

    # Interactive retrieval has a hard latency budget. Index-building callers
    # pass their own retrying client; live queries get one short attempt and
    # immediately fall back to lexical search on failure.
    client = client or EmbeddingClient(
        timeout=config.EMBEDDING_REQUEST_TIMEOUT,
        max_retries=0,
    )
    query_vector = normalize_vector(client.embed_texts([query])[0])
    if not query_vector:
        return {}

    records = {record.item_id: record.vector for record in index.records}
    candidate_ids = {item.id for item in candidates}
    known_ids = {item.id for item in kb.items}
    threshold = config.EMBEDDING_MIN_SCORE if min_score is None else min_score
    scores: dict[str, float] = {}
    for item_id, vector in records.items():
        if item_id not in candidate_ids or item_id not in known_ids:
            continue
        score = dot(query_vector, vector)
        if score >= threshold:
            scores[item_id] = score
    return scores


def normalize_vector(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return []
    return [value / norm for value in values]


def dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def describe_embedding_error(exc: Exception, api_key: str = "") -> str:
    """Describe an embedding failure without exposing credentials or response noise."""
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTPError {exc.response.status_code}"
        body = _truncate_response_body(exc.response, max_chars=200, api_key=api_key)
        if body:
            detail += f": {body}"
    elif isinstance(exc, (httpx.RequestError, OSError)):
        detail = f"RequestError: {exc}"
    else:
        detail = str(exc) or type(exc).__name__

    if api_key:
        detail = detail.replace(api_key, "***")
    return textwrap.shorten(_normalize_error_text(detail), width=220, placeholder="...")


def _truncate_response_body(
    response: httpx.Response,
    max_chars: int = 200,
    api_key: str = "",
) -> str:
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - best effort diagnostics only.
        return ""
    if api_key:
        text = text.replace(api_key, "***")
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _normalize_error_text(value: str) -> str:
    return " ".join(value.split())
