from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import ClassVar

import httpx
from fastapi.testclient import TestClient

import anti_fraud_explorer.api as api_module
from anti_fraud_explorer.api import create_app
from anti_fraud_explorer.asr_normalization import NormalizedSpan
from anti_fraud_explorer.dataset import KnowledgeBase, item_to_dict
from anti_fraud_explorer.models import AssistantEvent, SearchResponse
from anti_fraud_explorer.sessions import SessionStore


def make_kb() -> KnowledgeBase:
    return KnowledgeBase(
        {
            "schema_version": 4,
            "generated_at": "2026-08-20T00:00:00Z",
            "source": {"name": "offline-fixture"},
            "categories": [
                {"id": 1, "name": "冒充电商物流客服类", "item_count": 1},
                {"id": 2, "name": "刷单返利类", "item_count": 1},
            ],
            "items": [
                {
                    "case_id": "FZ-001",
                    "title": "冒充客服退款并诱导共享屏幕",
                    "summary": "对方冒充客服，以退款为由诱导共享屏幕。",
                    "judgment_reason": "正规客服不会索要验证码。",
                    "entry_channels": ["电话"],
                    "key_methods": ["屏幕共享", "索要验证码"],
                    "risk_signals": "要求共享屏幕并提供验证码",
                    "risk_level": "极高",
                    "prevention_advice": "立即退出共享并通过官方渠道核验。",
                    "source_name": "公安机关反诈通报",
                    "ccl2023_category": "冒充电商物流客服类",
                    "victim_group": "网购用户",
                },
                {
                    "case_id": "FZ-002",
                    "title": "刷单返利后要求垫资",
                    "summary": "先用小额返利建立信任，随后要求连续垫资。",
                    "judgment_reason": "以任务返利诱导持续转账。",
                    "entry_channels": ["微信"],
                    "key_methods": ["小额返利", "虚假订单/任务"],
                    "risk_signals": "要求脱离平台垫资",
                    "risk_level": "高",
                    "prevention_advice": "停止垫资并保存聊天记录。",
                    "source_name": "公安机关反诈通报",
                    "ccl2023_category": "刷单返利类",
                    "victim_group": "学生",
                },
            ],
        }
    )


class RecordingSearch:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.calls: list[dict[str, object]] = []

    def search(self, query: str = "", **kwargs: object) -> SearchResponse:
        self.calls.append({"query": query, **kwargs})
        limit = int(kwargs.get("limit", 30))
        offset = int(kwargs.get("offset", 0))
        items = tuple(self.knowledge_base.items[offset : offset + limit])
        return SearchResponse(items=items, total=len(self.knowledge_base.items))


class RecordingAssistant:
    def __init__(self, search: RecordingSearch, events: tuple[AssistantEvent, ...] = ()) -> None:
        self.search = search
        self.events = events
        self.calls: list[dict[str, str | None]] = []

    async def stream_turn(
        self,
        question: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        category: str = "",
        locale_hint: str = "",
    ) -> AsyncIterator[AssistantEvent]:
        self.calls.append(
            {
                "question": question,
                "session_id": session_id,
                "turn_id": turn_id,
                "category": category,
                "locale_hint": locale_hint,
            }
        )
        for event in self.events:
            yield event


class ASGIClient:
    """Small synchronous facade over httpx's async ASGI transport."""

    def __init__(self, app: object) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: object) -> httpx.Response:
        return self.request("POST", url, **kwargs)


class FakeXfyunStream:
    """Deterministic auto-ASR stream used to exercise websocket orchestration."""

    AUTO = "auto"
    DIALECT = "dialect"
    MULTILINGUAL = "multilingual"
    LEGACY = "legacy"

    instances: ClassVar[list[FakeXfyunStream]] = []
    transcript = ""
    candidate_texts: tuple[str, ...] = ()
    partial = ""
    provider_language = "zh"
    block_finish = False
    block_start = False
    finish_gate = threading.Event()
    start_gate = threading.Event()

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.on_partial = kwargs.get("on_partial")
        self.started = threading.Event()
        self.finish_called = threading.Event()
        self.closed = threading.Event()
        self.block_finish = self.__class__.block_finish
        self.block_start = self.__class__.block_start
        self.transcript = self.__class__.transcript
        self.candidates = self.__class__.candidate_texts or (
            (self.transcript,) if self.transcript else ()
        )
        self.partial = self.__class__.partial
        self.selected_mode = str(kwargs.get("preferred_mode") or self.DIALECT)
        self.detected_language = self.__class__.provider_language
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.started.set()
        if self.block_start:
            await asyncio.to_thread(self.__class__.start_gate.wait, 2)

    async def send_audio(self, _data: bytes) -> None:
        return

    async def finish(self) -> str:
        self.finish_called.set()
        if self.block_finish:
            await asyncio.to_thread(self.__class__.finish_gate.wait, 2)
        if self.partial and self.on_partial is not None:
            await self.on_partial(self.partial)
        return self.transcript

    async def close(self) -> None:
        self.closed.set()


class VoiceRecordingAssistant:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.calls: list[dict[str, str | None]] = []
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()

    async def stream_turn(
        self,
        question: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        category: str = "",
        locale_hint: str = "",
    ) -> AsyncIterator[AssistantEvent]:
        assert turn_id is not None
        session = session_id or "voice-session"
        self.calls.append(
            {
                "question": question,
                "session_id": session_id,
                "turn_id": turn_id,
                "category": category,
                "locale_hint": locale_hint,
            }
        )
        self.started.set()
        try:
            yield AssistantEvent(
                "response.text.delta",
                session,
                turn_id,
                0,
                payload={"delta": "好的"},
            )
            while self.block and not self.release.is_set():
                await asyncio.sleep(0.01)
            yield AssistantEvent(
                "turn.completed",
                session,
                turn_id,
                1,
                payload={"answer": "好的"},
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def voice_test_client(monkeypatch, assistant: VoiceRecordingAssistant) -> TestClient:
    monkeypatch.setattr(api_module, "XF_APP_ID", "test-app")
    monkeypatch.setattr(api_module, "XF_API_KEY", "test-key")
    monkeypatch.setattr(api_module, "XF_API_SECRET", "test-secret")
    FakeXfyunStream.instances.clear()
    FakeXfyunStream.block_finish = False
    FakeXfyunStream.block_start = False
    FakeXfyunStream.partial = ""
    FakeXfyunStream.candidate_texts = ()
    FakeXfyunStream.provider_language = "zh"
    FakeXfyunStream.finish_gate.set()
    FakeXfyunStream.start_gate.set()
    monkeypatch.setattr(api_module, "AutoXfyunStream", FakeXfyunStream)
    search = RecordingSearch(make_kb())
    return TestClient(
        create_app(
            assistant=assistant,  # type: ignore[arg-type]
            search=search,  # type: ignore[arg-type]
            sessions=SessionStore(),
        )
    )


def receive_until(websocket, predicate, *, limit: int = 20) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for _ in range(limit):
        message = websocket.receive_json()
        messages.append(message)
        if predicate(message):
            return messages
    raise AssertionError(f"did not receive expected websocket message: {messages!r}")


def build_client(
    *,
    events: tuple[AssistantEvent, ...] = (),
) -> tuple[ASGIClient, KnowledgeBase, RecordingSearch, RecordingAssistant]:
    kb = make_kb()
    search = RecordingSearch(kb)
    assistant = RecordingAssistant(search, events)
    client = ASGIClient(
        create_app(
            assistant=assistant,  # type: ignore[arg-type]
            search=search,  # type: ignore[arg-type]
            sessions=SessionStore(),
        )
    )
    return client, kb, search, assistant


def test_health_aliases_and_meta_expose_stable_contract() -> None:
    client, _kb, _search, _assistant = build_client()

    for path in ("/healthz", "/api/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.2.0"}

    response = client.get("/api/meta")
    assert response.status_code == 200
    assert response.json() == {
        "app_version": "0.2.0",
        "schema_version": 4,
        "generated_at": "2026-08-20T00:00:00Z",
        "source": {"name": "offline-fixture"},
        "item_count": 2,
        "category_count": 2,
        "categories": ["冒充电商物流客服类", "刷单返利类"],
        "risk_levels": ["极高", "高"],
        "entry_channels": ["微信", "电话"],
        "capabilities": {
            "text_chat": True,
            "realtime_voice": False,
            "voice_provider": "",
            "voice_error": "voice_provider_unconfigured",
        },
    }


def test_tts_retries_xiaoshuang_with_available_neutral_voice(monkeypatch) -> None:
    voices: list[str] = []

    class FakeCommunicate:
        def __init__(self, _text: str, *, voice: str, **_kwargs: object) -> None:
            self.voice = voice
            voices.append(voice)

        async def stream(self) -> AsyncIterator[dict[str, object]]:
            if self.voice == "zh-CN-XiaoshuangNeural":
                raise RuntimeError("voice is not available on this endpoint")
            yield {"type": "audio", "data": b"fallback-audio"}

    monkeypatch.setattr(api_module.edge_tts, "Communicate", FakeCommunicate)
    client, _kb, _search, _assistant = build_client()

    response = client.get("/api/tts", params={"text": "你好", "locale": "zh-CN"})

    assert response.status_code == 200
    assert response.content == b"fallback-audio"
    assert voices == ["zh-CN-XiaoshuangNeural", "zh-CN-XiaoyiNeural"]


def test_frontend_static_routes_coexist_with_api_routes(tmp_path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body>offline ui</body></html>", encoding="utf-8"
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('offline');", encoding="utf-8")

    monkeypatch.setattr(api_module, "FRONTEND_DIR", tmp_path)
    client, _kb, _search, _assistant = build_client()

    index = client.get("/")
    assert index.status_code == 200
    assert index.text == "<!doctype html><html><body>offline ui</body></html>"

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert asset.text == "console.log('offline');"

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_items_and_detail_return_public_shapes_and_forward_filters() -> None:
    client, kb, search, _assistant = build_client()

    response = client.get(
        "/api/items",
        params={
            "q": "刷单垫资",
            "category": "刷单返利类",
            "risk_level": "高",
            "entry_channel": "微信",
            "victim_group": "学生",
            "limit": 1,
            "offset": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "limit": 1,
        "offset": 1,
        "items": [item_to_dict(kb.items[1])],
    }
    assert search.calls[-1] == {
        "query": "刷单垫资",
        "category": "刷单返利类",
        "risk_level": "高",
        "entry_channel": "微信",
        "victim_group": "学生",
        "limit": 1,
        "offset": 1,
    }

    detail = client.get("/api/items/FZ-001")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["id"] == "FZ-001"
    assert detail_body["content"].startswith("对方冒充客服")
    assert detail_body["risk_level"] == "极高"
    assert detail_body["key_methods"] == ["屏幕共享", "索要验证码"]
    assert "search_text" not in detail_body

    missing = client.get("/api/items/does-not-exist")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "item_not_found"}


def parse_sse(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append(
            {
                "event": lines["event"],
                "id": int(lines["id"]),
                "data": json.loads(lines["data"]),
            }
        )
    return events


def test_chat_sse_preserves_order_and_nested_event_payload() -> None:
    events = (
        AssistantEvent("turn.started", "session-1", "turn-1", 0, payload={"question": "介绍"}),
        AssistantEvent("response.sources", "session-1", "turn-1", 1, payload={"sources": [{"id": "FZ-001"}]}),
        AssistantEvent("turn.completed", "session-1", "turn-1", 2, payload={"answer": "完成"}),
    )
    client, _kb, _search, assistant = build_client(events=events)

    response = client.post(
        "/api/chat",
        json={"question": "  介绍  ", "session_id": " session-1 ", "category": " 刷单返利类 "},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = parse_sse(response.text)

    assert [item["event"] for item in parsed] == [event.type for event in events]
    assert [item["id"] for item in parsed] == [0, 1, 2]
    for item, expected in zip(parsed, events):
        data = item["data"]
        assert isinstance(data, dict)
        assert set(data) == {"type", "session_id", "turn_id", "seq", "timestamp", "payload"}
        assert data["type"] == expected.type
        assert data["seq"] == expected.seq
        assert data["payload"] == expected.payload
    assert assistant.calls == [
        {
            "question": "介绍",
            "session_id": "session-1",
                "turn_id": None,
                "category": "刷单返利类",
                "locale_hint": "",
            }
    ]


def test_voice_empty_vad_does_not_cancel_active_answer(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant(block=True)
    FakeXfyunStream.transcript = ""
    with (
        voice_test_client(monkeypatch, assistant) as client,
        client.websocket_connect("/api/voice") as websocket,
    ):
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "text", "text": "已有回答"})
            assert assistant.started.wait(2)
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "thinking"
                ),
            )

            websocket.send_json({"type": "utterance.start", "interrupt": True})
            assert not assistant.cancelled.is_set()

            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "utterance.rejected",
            )
            assert not any(
                message.get("type") == "status" and message.get("status") == "user_speaking"
                for message in messages
            )
            assert not assistant.cancelled.is_set()

            assistant.release.set()
            receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )
            assert not assistant.cancelled.is_set()


def test_voice_empty_final_is_rejected_without_starting_new_answer(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    FakeXfyunStream.transcript = ""
    with (
        voice_test_client(monkeypatch, assistant) as client,
        client.websocket_connect("/api/voice") as websocket,
    ):
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "utterance.rejected",
            )
            assert messages[-1]["utterance_id"] == 1
            assert assistant.calls == []


def test_voice_partial_snapshot_precedes_and_is_calibrated_by_final(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    FakeXfyunStream.transcript = "你好"
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.partial = "你"
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )

            partial = next(message for message in messages if message.get("type") == "user.partial")
            final = next(
                message for message in messages if message.get("type") == "user.transcript"
            )
            assert partial["utterance_id"] == 1
            assert partial["revision"] == 1
            assert partial["text"] == "你"
            assert partial["final"] is False
            assert final["utterance_id"] == 1
            assert final["revision"] == 1
            assert final["text"] == "你好"
            assert final["final"] is True


def test_voice_completed_batch_does_not_leak_partial_into_next_turn(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    FakeXfyunStream.transcript = "你好"
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.partial = "你"
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            receive_until(websocket, lambda message: message.get("type") == "assistant.done")

            FakeXfyunStream.transcript = "刷单返利"
            FakeXfyunStream.partial = "苏"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "user_speaking"
                    and message.get("utterance_id") == 2
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            second_turn = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )

            partial = next(
                message for message in second_turn if message.get("type") == "user.partial"
            )
            final = next(
                message for message in second_turn if message.get("type") == "user.transcript"
            )
            assert partial["text"] == "苏"
            assert partial["revision"] == 1
            assert final["text"] == "刷单返利"
            assert [call["question"] for call in assistant.calls] == ["你好", "刷单返利"]


def test_voice_overlapping_finalizes_merge_in_id_order(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.transcript = "旧句"
        FakeXfyunStream.block_finish = True
        FakeXfyunStream.finish_gate.clear()
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "user_speaking"
                    and message.get("utterance_id") == 1
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "transcribing"
                    and message.get("utterance_id") == 1
                ),
            )
            first = FakeXfyunStream.instances[0]

            # The next utterance must not destroy the first provider final.
            FakeXfyunStream.block_finish = False
            FakeXfyunStream.transcript = "新句"
            websocket.send_json({"type": "utterance.start"})
            messages = receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "user_speaking"
                    and message.get("utterance_id") == 2
                ),
            )
            assert not first.closed.is_set()
            websocket.send_json({"type": "utterance.end"})
            messages.extend(
                receive_until(
                    websocket,
                    lambda message: message.get("type") == "assistant.done",
                )
            )

            transcripts = [
                message for message in messages if message.get("type") == "user.transcript"
            ]
            assert len(transcripts) == 1
            assert transcripts[0]["utterance_id"] == 2
            assert transcripts[0]["text"] == "旧句 新句"
            assert len(assistant.calls) == 1
            assert assistant.calls[0]["question"] == "旧句 新句"


def test_voice_new_start_implicitly_finishes_previous_asr(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.transcript = "旧句"
        FakeXfyunStream.block_finish = True
        FakeXfyunStream.finish_gate.clear()
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "user_speaking"
                    and message.get("utterance_id") == 1
                ),
            )

            # No utterance.end arrives for the first phrase. A new VAD onset
            # must finalize it instead of closing and losing its provider
            # result.
            FakeXfyunStream.block_finish = False
            FakeXfyunStream.transcript = "新句"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status"
                    and message.get("status") == "user_speaking"
                    and message.get("utterance_id") == 2
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            FakeXfyunStream.finish_gate.set()
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )

            transcripts = [
                message for message in messages if message.get("type") == "user.transcript"
            ]
            assert len(transcripts) == 1
            assert transcripts[0]["text"] == "旧句 新句"
            assert len(assistant.calls) == 1


def test_voice_text_invalidates_pending_asr_without_batch_lock_deadlock(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.transcript = "旧句"
        FakeXfyunStream.block_finish = True
        FakeXfyunStream.finish_gate.clear()
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "transcribing"
                ),
            )

            # Invalidating the generation before canceling the blocked final
            # must return promptly and start the typed turn exactly once.
            websocket.send_json({"type": "text", "text": "新的问题"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )
            assert any(
                message.get("type") == "status" and message.get("status") == "thinking"
                for message in messages
            )
            assert len(assistant.calls) == 1
            assert assistant.calls[0]["question"] == "新的问题"
            assert assistant.calls[0]["session_id"] is None
            assert assistant.calls[0]["turn_id"]
            assert assistant.calls[0]["category"] == ""
            assert not any(message.get("type") == "user.transcript" for message in messages)


def test_voice_nonempty_final_emits_transcript_and_ordered_assistant_events(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    FakeXfyunStream.transcript = "你好"
    with (
        voice_test_client(monkeypatch, assistant) as client,
        client.websocket_connect("/api/voice") as websocket,
    ):
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )

            assert any(
                message.get("type") == "user.transcript" and message.get("text") == "你好"
                for message in messages
            )
            assistant_messages = [
                message
                for message in messages
                if message.get("type") in {"assistant.delta", "assistant.done"}
            ]
            assert assistant_messages
            assert all(message.get("turn_id") for message in assistant_messages)
            assert len({message["turn_id"] for message in assistant_messages}) == 1
            assert all("connection_id" in message for message in messages)
            sequences = [message["sequence"] for message in messages]
            assert sequences == sorted(sequences)
            assert len(sequences) == len(set(sequences))


def test_voice_interrupt_delays_speaking_status_until_nonempty_partial(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    FakeXfyunStream.transcript = "你好"
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.partial = "你"
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start", "interrupt": True})
            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "user.partial",
            )

            speech_events = [
                message
                for message in messages
                if message.get("type") == "user.partial"
                or (message.get("type") == "status" and message.get("status") == "user_speaking")
            ]
            assert [message["type"] for message in speech_events] == [
                "status",
                "user.partial",
            ]
            assert speech_events[0]["utterance_id"] == speech_events[1]["utterance_id"] == 1


def test_voice_non_interrupt_still_publishes_speaking_status_immediately(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    with (
        voice_test_client(monkeypatch, assistant) as client,
        client.websocket_connect("/api/voice") as websocket,
    ):
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "utterance.start", "interrupt": False})
            messages = receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            assert messages[-1]["utterance_id"] == 1
            assert not any(message.get("type") == "user.partial" for message in messages)


def test_voice_barge_in_cancels_answer_without_closing_active_asr(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant(block=True)
    with (
        voice_test_client(monkeypatch, assistant) as client,
        client.websocket_connect("/api/voice") as websocket,
    ):
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "text", "text": "正在回答"})
            assert assistant.started.wait(2)
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "thinking"
                ),
            )
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            asr_stream = FakeXfyunStream.instances[-1]
            assert not asr_stream.closed.is_set()

            websocket.send_json({"type": "barge_in"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            assert assistant.cancelled.wait(2)
            assert not asr_stream.closed.is_set()


def test_voice_provider_handshake_does_not_block_immediate_barge_in(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant(block=True)
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.block_start = True
        FakeXfyunStream.start_gate.clear()
        try:
            with client.websocket_connect("/api/voice") as websocket:
                assert websocket.receive_json()["type"] == "ready"
                websocket.send_json({"type": "text", "text": "正在回答"})
                assert assistant.started.wait(2)
                receive_until(
                    websocket,
                    lambda message: (
                        message.get("type") == "status" and message.get("status") == "thinking"
                    ),
                )

                websocket.send_json({"type": "utterance.start", "interrupt": True})
                deadline = time.monotonic() + 2
                while not FakeXfyunStream.instances and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert FakeXfyunStream.instances[-1].started.wait(2)
                assert not FakeXfyunStream.start_gate.is_set()

                websocket.send_json({"type": "barge_in"})
                receive_until(
                    websocket,
                    lambda message: (
                        message.get("type") == "status" and message.get("status") == "user_speaking"
                    ),
                )
                assert assistant.cancelled.wait(2)
        finally:
            FakeXfyunStream.start_gate.set()


def test_voice_context_is_used_once_at_final_boundary(monkeypatch) -> None:
    assistant = VoiceRecordingAssistant()
    normalized_calls: list[dict[str, object]] = []

    def fake_normalize(raw_text: str, **kwargs: object) -> object:
        normalized_calls.append({"raw_text": raw_text, **kwargs})
        return SimpleNamespace(
            raw_text=raw_text,
            canonical_text="刷单返利",
            spans=(NormalizedSpan(0, 4, "刷单反利", "刷单返利", 0.9, "test"),),
        )

    monkeypatch.setattr(api_module, "normalize_asr_final", fake_normalize)
    FakeXfyunStream.transcript = "刷单反利"
    with voice_test_client(monkeypatch, assistant) as client:
        FakeXfyunStream.candidate_texts = ("刷单反利", "刷单返利")
        with client.websocket_connect("/api/voice") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({
                "type": "context",
                "category": "刷单返利类",
                "titles": ["刷单返利后要求垫资", "冒充客服退款并诱导共享屏幕"],
            })
            websocket.send_json({"type": "utterance.start"})
            receive_until(
                websocket,
                lambda message: (
                    message.get("type") == "status" and message.get("status") == "user_speaking"
                ),
            )
            websocket.send_json({"type": "utterance.end"})
            messages = receive_until(
                websocket,
                lambda message: message.get("type") == "assistant.done",
            )

            final = next(message for message in messages if message.get("type") == "user.transcript")
            assert final["text"] == "刷单返利"
            assert final["raw_text"] == "刷单反利"
            assert final["normalizations"][0]["canonical"] == "刷单返利"
            assert normalized_calls[0]["category"] == "刷单返利类"
            assert normalized_calls[0]["asr_candidates"] == ("刷单反利", "刷单返利")
            assert FakeXfyunStream.instances[0].kwargs["hotwords"] == (
                "刷单返利类",
                "刷单返利后要求垫资",
                "冒充客服退款并诱导共享屏幕",
            )
            assert assistant.calls[0]["question"] == "刷单返利"
            assert assistant.calls[0]["category"] == "刷单返利类"
