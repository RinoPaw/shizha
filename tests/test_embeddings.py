from __future__ import annotations

import json

import httpx
import pytest

from anti_fraud_explorer.embeddings import EmbeddingClient, EmbeddingUnavailable


def test_embedding_client_uses_injected_httpx_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client = EmbeddingClient(
            api_key="offline-test-key",
            base_url="https://embedding.example/v1/",
            model="test-embedding",
            timeout=3,
            max_retries=0,
            http_client=http_client,
        )

        assert client.embed_texts(["刷单返利", "冒充公检法"]) == [[3.0, 0.0], [0.0, 2.0]]
    finally:
        http_client.close()

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://embedding.example/v1/embeddings"
    assert request.headers["authorization"] == "Bearer offline-test-key"
    assert json.loads(request.content) == {
        "model": "test-embedding",
        "input": ["刷单返利", "冒充公检法"],
    }


def test_embedding_client_without_key_degrades_explicitly_offline() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(EmbeddingUnavailable, match="EMBEDDING_API_KEY is not configured"):
            EmbeddingClient(
                api_key="",
                max_retries=0,
                http_client=http_client,
            ).embed_texts(["离线查询"])
    finally:
        http_client.close()

    assert called is False


def test_embedding_error_does_not_leak_api_key() -> None:
    secret = "embedding-secret-123"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text=f"invalid token {secret}",
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(EmbeddingUnavailable) as error:
            EmbeddingClient(
                api_key=secret,
                max_retries=0,
                http_client=http_client,
            ).embed_texts(["刷单返利"])
    finally:
        http_client.close()

    assert secret not in str(error.value)
    assert "HTTPError 401" in str(error.value)
