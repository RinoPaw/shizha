"""FastAPI/ASGI transport for the assistant core."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import edge_tts
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi import Path as ApiPath
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from . import __version__
from .asr_normalization import (
    NormalizedTranscript,
    normalize_asr_final,
    prepare_asr_normalization,
)
from .assistant import AssistantService, SearchService
from .config import (
    FRONTEND_DIR,
    TTS_PITCH,
    TTS_RATE,
    TTS_VOICE,
    XF_API_KEY,
    XF_API_SECRET,
    XF_APP_ID,
    XF_ASR_HOST,
    XF_ASR_RES_ID,
    XF_LEGACY_ASR_HOST,
    XF_MULTILINGUAL_API_KEY,
    XF_MULTILINGUAL_API_SECRET,
    XF_MULTILINGUAL_APP_ID,
    XF_MULTILINGUAL_ASR_HOST,
    XF_MULTILINGUAL_LANGUAGE_HINT,
)
from .dataset import item_to_dict
from .language import (
    detect_locale,
    get_language_profile,
    is_chinese_locale,
    locale_from_provider_language,
    normalize_locale_hint,
)
from .sessions import SessionStore
from .voice import AutoXfyunStream, VoiceProviderError

MAX_CHAT_CHARS = 4000
MAX_SESSION_ID_CHARS = 128
MAX_TURN_ID_CHARS = 128
MAX_TTS_CHARS = 4000
MAX_VOICE_CONTEXT_TITLES = 8
MAX_VOICE_RECENT_ITEMS = 8
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _case_to_dict(item: Any, *, detail: bool = False) -> dict[str, Any]:
    """Serialize one source-backed fraud case without leaking search internals."""

    return item_to_dict(item, include_content=detail)


def _contains_spoken_text(value: object) -> bool:
    """Punctuation-only ASR hypotheses are not evidence of human speech."""

    return any(character.isalnum() for character in str(value or ""))


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_CHAT_CHARS)
    session_id: str | None = Field(default=None, max_length=MAX_SESSION_ID_CHARS)
    category: str = Field(default="", max_length=200)
    locale_hint: str = Field(default="", max_length=64)


def create_app(
    *,
    assistant: AssistantService | None = None,
    search: SearchService | None = None,
    sessions: SessionStore | None = None,
) -> FastAPI:
    search = (
        search or (getattr(assistant, "search", None) if assistant else None) or SearchService()
    )
    sessions = (
        sessions or (getattr(assistant, "sessions", None) if assistant else None) or SessionStore()
    )
    assistant = assistant or AssistantService(search=search, sessions=sessions)
    kb = search.knowledge_base
    prepare_asr_normalization(kb)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(assistant, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(title="识诈", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        items = getattr(kb, "items", ())
        categories = [category.name for category in getattr(kb, "categories", ()) if category.name]
        risk_order = {"极高": 0, "高": 1, "中": 2, "低": 3, "极低": 4}
        risk_levels = sorted(
            {str(getattr(item, "risk_level", "") or "") for item in items if getattr(item, "risk_level", "")},
            key=lambda value: (risk_order.get(value, 99), value),
        )
        entry_channels = sorted({
            str(channel)
            for item in items
            for channel in (getattr(item, "entry_channels", ()) or ())
            if str(channel).strip()
        })
        xfyun_ready = bool(XF_APP_ID.strip() and XF_API_KEY.strip() and XF_API_SECRET.strip())
        return {
            "app_version": __version__,
            "schema_version": getattr(kb, "schema_version", 1),
            "generated_at": getattr(kb, "generated_at", ""),
            "source": getattr(kb, "source", {}),
            "item_count": len(items),
            "category_count": len(categories),
            "categories": categories,
            "risk_levels": risk_levels,
            "entry_channels": entry_channels,
            "capabilities": {
                "text_chat": True,
                "realtime_voice": xfyun_ready,
                "voice_provider": "xfyun" if xfyun_ready else "",
                "voice_error": "voice_provider_unconfigured" if not xfyun_ready else "",
            },
        }

    @app.get("/api/categories")
    async def categories() -> list[dict[str, Any]]:
        return [
            {"id": category.id, "name": category.name, "item_count": category.item_count}
            for category in kb.categories
        ]

    @app.get("/api/tts")
    async def synthesize_speech(
        text: str = Query(min_length=1, max_length=MAX_TTS_CHARS),
        locale: str = Query(default="", max_length=64),
        trace_id: str = Query(default="", max_length=128),
        segment: int = Query(default=0, ge=0, le=999),
        reason: str = Query(default="", max_length=40),
    ) -> StreamingResponse:
        text = text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="empty_text")
        # The locale attached to an assistant turn is authoritative for every
        # segment in that turn.  Inspect text only when an older client did
        # not send a resolved locale; otherwise a canonical Chinese project
        # name inside an English sentence could unexpectedly change voices.
        requested_locale = normalize_locale_hint(locale)
        language_profile = get_language_profile(
            requested_locale or detect_locale(text),
        )
        started = time.perf_counter()
        LOGGER.info(
            "[trace=%s segment=%s] tts.request.start reason=%s chars=%s locale=%s",
            trace_id or "-",
            segment,
            reason or "unspecified",
            len(text),
            language_profile.code,
        )

        async def audio_stream() -> AsyncIterator[bytes]:
            communicate = edge_tts.Communicate(
                text,
                voice=TTS_VOICE,
                rate=TTS_RATE,
                pitch=TTS_PITCH,
            )
            first_chunk = True
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    if first_chunk:
                        first_chunk = False
                        LOGGER.info(
                            "[trace=%s segment=%s] tts.first_audio_chunk +%.3fs",
                            trace_id or "-",
                            segment,
                            time.perf_counter() - started,
                        )
                    yield chunk["data"]
            LOGGER.info(
                "[trace=%s segment=%s] tts.stream.complete +%.3fs",
                trace_id or "-",
                segment,
                time.perf_counter() - started,
            )

        return StreamingResponse(
            audio_stream(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Speech-Locale": language_profile.code,
            },
        )

    @app.get("/api/items")
    async def items(
        q: str = Query(default="", max_length=4000),
        category: str = Query(default="", max_length=200),
        risk_level: str = Query(default="", max_length=100),
        entry_channel: str = Query(default="", max_length=200),
        victim_group: str = Query(default="", max_length=200),
        limit: int = Query(default=30, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=100000),
    ) -> dict[str, Any]:
        result = search.search(
            q,
            category=category,
            risk_level=risk_level,
            entry_channel=entry_channel,
            victim_group=victim_group,
            limit=limit,
            offset=offset,
        )
        result_items = tuple(getattr(result, "items", result or ()))
        return {
            "total": int(getattr(result, "total", len(result_items))),
            "limit": min(max(limit, 1), 100),
            "offset": max(offset, 0),
            "items": [_case_to_dict(item) for item in result_items],
        }

    @app.get("/api/items/{item_id}")
    async def item_detail(
        item_id: str = ApiPath(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        get_item = getattr(kb, "get", None)
        item = get_item(item_id) if get_item is not None else None
        if item is None:
            item = next(
                (
                    candidate for candidate in getattr(kb, "items", ())
                    if str(getattr(candidate, "case_id", "")) == item_id
                    or str(getattr(candidate, "id", "")) == item_id
                ),
                None,
            )
        if item is None:
            raise HTTPException(status_code=404, detail="item_not_found")
        return _case_to_dict(item, detail=True)

    @app.post("/api/chat", response_class=EventSourceResponse)
    async def chat(body: ChatRequest) -> AsyncIterator[ServerSentEvent]:
        question = body.question.strip()
        session_id = body.session_id.strip() if body.session_id else None
        category = body.category.strip()
        locale_hint = body.locale_hint.strip()

        async for assistant_event in assistant.stream_turn(
            question,
            session_id=session_id,
            category=category,
            locale_hint=locale_hint,
        ):
            yield ServerSentEvent(
                data=assistant_event.to_dict(),
                event=assistant_event.type,
                id=str(assistant_event.seq),
            )

    @app.post("/api/chat/{session_id}/turn/{turn_id}/cancel")
    async def cancel_turn(
        session_id: str = ApiPath(..., min_length=1, max_length=MAX_SESSION_ID_CHARS),
        turn_id: str = ApiPath(..., min_length=1, max_length=MAX_TURN_ID_CHARS),
    ) -> dict[str, Any]:
        cancelled = sessions.cancel_turn(session_id, turn_id)
        return {"cancelled": cancelled, "session_id": session_id, "turn_id": turn_id}

    @app.websocket("/api/voice")
    async def browser_voice(websocket: WebSocket) -> None:
        """Continuous browser VAD + Xunfei ASR + shared assistant pipeline."""

        if not (XF_APP_ID.strip() and XF_API_KEY.strip() and XF_API_SECRET.strip()):
            # Accept first so clients receive a machine-readable explanation;
            # text chat remains fully available on the same application.
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "code": "voice_provider_unconfigured",
                "message": "语音识别未配置讯飞凭据，文本功能仍可使用。",
            })
            await websocket.close(code=1013, reason="voice_unavailable")
            return
        await websocket.accept()
        connection_id = uuid.uuid4().hex
        asr_stream: AutoXfyunStream | None = None
        asr_start_task: asyncio.Task[None] | None = None
        answer_task: asyncio.Task[None] | None = None
        # Finalizing ASR is independent for each ended utterance.  Keeping a
        # single task here meant that a second VAD start cancelled the first
        # provider final, so a perfectly valid sentence simply disappeared.
        finalize_tasks: set[asyncio.Task[None]] = set()
        batch_results: dict[int, str] = {}
        batch_candidates: dict[int, tuple[str, ...]] = {}
        batch_languages: dict[int, str] = {}
        batch_asr_modes: dict[int, str] = {}
        # The browser may split one spoken turn into several VAD utterances.
        # Keep each provider hypothesis until its final arrives, then publish
        # the ordered concatenation so a late partial from the first segment
        # cannot erase the prefix while the second segment is being recognized.
        batch_partials: dict[int, str] = {}
        pending_user_speaking: set[int] = set()
        partial_revision = 0
        batch_pending: set[int] = set()
        batch_generation = 0
        batch_lock = asyncio.Lock()
        session_id: str | None = None
        voice_category = ""
        voice_context_titles: list[str] = []
        voice_locale_hint = ""
        active_turn_id: str | None = None
        utterance_sequence = 0
        event_sequence = 0
        send_lock = asyncio.Lock()

        def update_voice_context(event: dict[str, Any]) -> None:
            """Keep only bounded, non-sensitive recognition context from the browser."""

            nonlocal session_id, voice_category, voice_context_titles, voice_locale_hint
            category = str(event.get("category") or "").strip()
            voice_category = category[:200]
            incoming_session = str(event.get("session_id") or "").strip()
            if incoming_session:
                session_id = incoming_session[:MAX_SESSION_ID_CHARS]
            incoming_locale = normalize_locale_hint(event.get("locale_hint") or event.get("locale"))
            if incoming_locale:
                voice_locale_hint = incoming_locale

            values: list[Any] = []
            for key in ("selected_title", "selected_item", "selected"):
                value = event.get(key)
                if value:
                    values.append(value)
            for key in ("titles", "visible_titles", "visible_items", "items"):
                value = event.get(key)
                if isinstance(value, (list, tuple)):
                    values.extend(value)
            titles: list[str] = []
            seen: set[str] = set()
            for value in values:
                if isinstance(value, dict):
                    value = value.get("title", "")
                title = str(value or "").strip()
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title[:200])
                if len(titles) >= MAX_VOICE_CONTEXT_TITLES:
                    break
            voice_context_titles = titles

        def recent_voice_items() -> tuple[Any, ...]:
            if not session_id:
                return ()
            recent: list[Any] = []
            seen: set[str] = set()
            for turn in reversed(sessions.history(session_id)):
                for source_id in reversed(turn.source_ids):
                    if source_id in seen:
                        continue
                    item = kb.get(source_id)
                    if item is not None:
                        seen.add(source_id)
                        recent.append(item)
                    if len(recent) >= MAX_VOICE_RECENT_ITEMS:
                        return tuple(recent)
            return tuple(recent)

        def make_asr_stream(on_partial: Any) -> AutoXfyunStream:
            recent = recent_voice_items()
            recent_titles = [str(getattr(item, "title", "") or "") for item in recent]
            hotwords: list[str] = []
            seen: set[str] = set()
            for title in [voice_category, *voice_context_titles, *recent_titles]:
                title = title.strip()
                if title and title not in seen:
                    seen.add(title)
                    hotwords.append(title)
                if len(hotwords) >= MAX_VOICE_CONTEXT_TITLES + MAX_VOICE_RECENT_ITEMS:
                    break
            preferred_mode = (
                AutoXfyunStream.DIALECT
                if is_chinese_locale(voice_locale_hint)
                else AutoXfyunStream.MULTILINGUAL
            )
            return AutoXfyunStream(
                app_id=XF_APP_ID,
                api_key=XF_API_KEY,
                api_secret=XF_API_SECRET,
                multilingual_app_id=XF_MULTILINGUAL_APP_ID,
                multilingual_api_key=XF_MULTILINGUAL_API_KEY,
                multilingual_api_secret=XF_MULTILINGUAL_API_SECRET,
                host=XF_ASR_HOST,
                multilingual_host=XF_MULTILINGUAL_ASR_HOST,
                multilingual_language_hint=XF_MULTILINGUAL_LANGUAGE_HINT,
                legacy_host=XF_LEGACY_ASR_HOST,
                on_partial=on_partial,
                hotwords=tuple(hotwords),
                resource_id=XF_ASR_RES_ID,
                # Probe on every utterance so one conversation can naturally
                # switch between a Chinese dialect and English/Japanese/Korean.
                mode=AutoXfyunStream.AUTO,
                preferred_mode=preferred_mode,
            )

        async def send(payload: dict[str, Any]) -> None:
            nonlocal event_sequence
            try:
                async with send_lock:
                    event_sequence += 1
                    await websocket.send_json(
                        {
                            "connection_id": connection_id,
                            "sequence": event_sequence,
                            **payload,
                        }
                    )
            except (RuntimeError, WebSocketDisconnect):
                pass

        async def stop_task(task: asyncio.Task[None] | None) -> None:
            if task is None or task.done():
                return
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def stop_finalize_tasks() -> None:
            tasks = tuple(finalize_tasks)
            finalize_tasks.clear()
            if tasks:
                await asyncio.gather(*(stop_task(task) for task in tasks), return_exceptions=True)

        async def close_active_asr() -> None:
            nonlocal asr_stream, asr_start_task
            stream, asr_stream = asr_stream, None
            start_task, asr_start_task = asr_start_task, None
            if start_task is not None:
                if not start_task.done():
                    start_task.cancel()
                await asyncio.gather(start_task, return_exceptions=True)
            if stream is not None:
                await stream.close()

        def clear_voice_batch() -> None:
            nonlocal batch_generation, partial_revision
            batch_generation += 1
            batch_results.clear()
            batch_candidates.clear()
            batch_languages.clear()
            batch_asr_modes.clear()
            batch_partials.clear()
            pending_user_speaking.clear()
            batch_pending.clear()
            partial_revision = 0

        def combined_partial_text() -> str:
            return " ".join(
                batch_partials[item_id].strip()
                for item_id in sorted(batch_partials)
                if batch_partials[item_id].strip()
            ).strip()

        async def send_batch_partial(utterance_id: int, text: str) -> None:
            """Publish the current ordered ASR hypothesis for this batch."""

            nonlocal partial_revision
            if utterance_id not in batch_partials:
                return
            batch_partials[utterance_id] = str(text or "")
            combined = combined_partial_text()
            if not combined:
                return
            partial_revision += 1
            await send(
                {
                    "type": "user.partial",
                    # The latest id owns the visible bubble; the text includes
                    # all earlier utterance hypotheses in order.
                    "utterance_id": max(batch_partials),
                    "revision": partial_revision,
                    "text": combined,
                    "final": False,
                }
            )

        async def stop_answer(reason: str) -> None:
            nonlocal answer_task, active_turn_id
            task, turn_id = answer_task, active_turn_id
            answer_task = None
            active_turn_id = None
            if task is not None and not task.done():
                LOGGER.info(
                    "voice.answer.cancel connection=%s turn=%s reason=%s",
                    connection_id,
                    turn_id or "-",
                    reason,
                )
            await stop_task(task)

        async def answer(question: str, turn_id: str, locale_hint: str = "") -> None:
            nonlocal session_id
            answer_locale = detect_locale(question, hint=locale_hint or voice_locale_hint)
            LOGGER.info(
                "voice.agent.thinking connection=%s turn=%s question_chars=%s",
                connection_id,
                turn_id,
                len(question),
            )
            await send(
                {
                    "type": "status",
                    "status": "thinking",
                    "turn_id": turn_id,
                    "locale": answer_locale,
                }
            )
            try:
                async for event in assistant.stream_turn(
                    question,
                    session_id=session_id,
                    turn_id=turn_id,
                    category=voice_category,
                    locale_hint=answer_locale,
                ):
                    if active_turn_id != turn_id:
                        return
                    session_id = event.session_id
                    if event.type == "response.text.delta":
                        event_locale = str(event.payload.get("locale") or answer_locale)
                        await send(
                            {
                                "type": "assistant.delta",
                                "session_id": event.session_id,
                                "turn_id": event.turn_id,
                                "text": event.payload.get("delta", ""),
                                "locale": event_locale,
                            }
                        )
                    elif event.type == "response.sources":
                        await send(
                            {
                                "type": "sources",
                                "session_id": event.session_id,
                                "turn_id": event.turn_id,
                                "items": event.payload.get("sources", []),
                            }
                        )
                    elif event.type == "turn.completed":
                        event_locale = str(event.payload.get("locale") or answer_locale)
                        await send(
                            {
                                "type": "assistant.done",
                                "session_id": event.session_id,
                                "turn_id": event.turn_id,
                                "text": event.payload.get("answer", ""),
                                "locale": event_locale,
                            }
                        )
                    elif event.type == "turn.failed":
                        code = str(event.payload.get("code") or "llm_unavailable")
                        message = (
                            "回答生成等待过久，请再试一次"
                            if code == "llm_first_token_timeout"
                            else "回答服务暂时不可用，请再试一次"
                        )
                        await send(
                            {
                                "type": "error",
                                "turn_id": turn_id,
                                "code": code,
                                "message": message,
                            }
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception(
                    "voice.answer.failed connection=%s turn=%s",
                    connection_id,
                    turn_id,
                )
                await send({"type": "error", "turn_id": turn_id, "message": "回答服务暂时不可用"})

        async def start_answer(
            question: str,
            reason: str,
            locale_hint: str = "",
        ) -> None:
            nonlocal answer_task, active_turn_id
            await stop_answer(reason)
            turn_id = uuid.uuid4().hex
            active_turn_id = turn_id
            answer_task = asyncio.create_task(answer(question, turn_id, locale_hint))

        async def commit_voice_batch(generation: int) -> None:
            """Commit one settled group of overlapping VAD utterances."""

            nonlocal partial_revision

            # A text/interrupt/connection cleanup invalidates the generation
            # before canceling finalize tasks.  The canceled task can reach
            # this function from its finally block, so reject it before
            # waiting for the batch lock (which the cleanup may hold).
            if generation != batch_generation:
                return
            async with batch_lock:
                if (
                    generation != batch_generation
                    or asr_stream is not None
                    or batch_pending
                    or not batch_results
                ):
                    return
                results = dict(batch_results)
                batch_results.clear()
                ordered_ids = sorted(results)
                asr_candidates = tuple(
                    candidate
                    for item_id in ordered_ids
                    for candidate in batch_candidates.pop(item_id, ())
                    if candidate
                )
                provider_languages = [batch_languages.pop(item_id, "") for item_id in ordered_ids]
                asr_modes = [batch_asr_modes.pop(item_id, "") for item_id in ordered_ids]
                raw_text = " ".join(
                    results[item_id].strip() for item_id in ordered_ids if results[item_id].strip()
                )
                committed_id = ordered_ids[-1]
                if not raw_text:
                    batch_partials.clear()
                    partial_revision = 0
                    await send({"type": "utterance.rejected", "utterance_id": committed_id})
                    return
                provider_locale = next(
                    (
                        locale
                        for language in reversed(provider_languages)
                        if (locale := locale_from_provider_language(language))
                    ),
                    None,
                )
                resolved_locale = detect_locale(
                    raw_text,
                    hint=provider_locale or voice_locale_hint,
                )
                normalized = (
                    normalize_asr_final(
                        raw_text,
                        kb=kb,
                        category=voice_category,
                        recent_items=recent_voice_items(),
                        asr_candidates=asr_candidates,
                    )
                    if is_chinese_locale(resolved_locale)
                    else NormalizedTranscript(raw_text, raw_text, ())
                )
                canonical_text = normalized.canonical_text
                normalizations = [asdict(span) for span in normalized.spans]
                LOGGER.info(
                    "voice.asr.batch_complete connection=%s utterances=%s raw_chars=%s canonical_chars=%s replacements=%s",
                    connection_id,
                    ",".join(str(item_id) for item_id in ordered_ids),
                    len(raw_text),
                    len(canonical_text),
                    len(normalizations),
                )
                await stop_answer("asr_batch")
                await send(
                    {
                        "type": "user.transcript",
                        "utterance_id": committed_id,
                        "revision": partial_revision,
                        "final": True,
                        "text": canonical_text,
                        "raw_text": raw_text,
                        "normalizations": normalizations,
                        "locale": resolved_locale,
                        "asr_engine": next((mode for mode in reversed(asr_modes) if mode), "auto"),
                    }
                )
                # This batch is complete. Do not let its partial prefix leak
                # into the next VAD turn; there are no pending finalizers to
                # invalidate, so the generation remains unchanged.
                batch_partials.clear()
                partial_revision = 0
                await start_answer(canonical_text, "new_utterance", resolved_locale)

        async def finish_utterance(
            stream: AutoXfyunStream,
            start_task: asyncio.Task[None],
            utterance_id: int,
            generation: int,
        ) -> None:
            try:
                await send(
                    {
                        "type": "status",
                        "status": "transcribing",
                        "utterance_id": utterance_id,
                    }
                )
                await start_task
                transcript = await stream.finish()
                LOGGER.info(
                    "voice.asr.finished connection=%s utterance=%s chars=%s",
                    connection_id,
                    utterance_id,
                    len(transcript),
                )
                if transcript:
                    LOGGER.info(
                        "voice.asr.complete connection=%s utterance=%s chars=%s",
                        connection_id,
                        utterance_id,
                        len(transcript),
                    )
                if generation == batch_generation:
                    batch_results[utterance_id] = transcript
                    batch_candidates[utterance_id] = stream.candidates
                    selected_mode = str(getattr(stream, "selected_mode", "") or "")
                    detected_language = str(getattr(stream, "detected_language", "") or "")
                    batch_asr_modes[utterance_id] = selected_mode
                    batch_languages[utterance_id] = detected_language
                    # The provider final is authoritative for this segment;
                    # keep it in the batch until commit so a late partial
                    # cannot replace the calibrated text.
                    batch_partials[utterance_id] = transcript
            except asyncio.CancelledError:
                raise
            except VoiceProviderError:
                LOGGER.warning(
                    "voice.asr.failed connection=%s utterance=%s",
                    connection_id,
                    utterance_id,
                    exc_info=True,
                )
                if generation == batch_generation:
                    batch_results[utterance_id] = ""
                    batch_candidates[utterance_id] = ()
                    batch_asr_modes[utterance_id] = ""
                    batch_languages[utterance_id] = ""
                await send({"type": "error", "message": "语音识别暂时不可用"})
            finally:
                await stream.close()
                pending_user_speaking.discard(utterance_id)
                batch_pending.discard(utterance_id)
                await commit_voice_batch(generation)

        async def start_asr(stream: AutoXfyunStream, utterance_id: int) -> None:
            started_at = time.perf_counter()
            LOGGER.info(
                "voice.asr.connect.start connection=%s utterance=%s",
                connection_id,
                utterance_id,
            )
            await stream.start()
            LOGGER.info(
                "voice.asr.connect.ready connection=%s utterance=%s +%.3fs",
                connection_id,
                utterance_id,
                time.perf_counter() - started_at,
            )

        def schedule_finalize(
            stream: AutoXfyunStream,
            start_task: asyncio.Task[None],
            utterance_id: int,
        ) -> None:
            batch_pending.add(utterance_id)
            task = asyncio.create_task(
                finish_utterance(stream, start_task, utterance_id, batch_generation)
            )
            finalize_tasks.add(task)

            def clear_finalize(done_task: asyncio.Task[None]) -> None:
                finalize_tasks.discard(done_task)

            task.add_done_callback(clear_finalize)

        try:
            await send({"type": "ready"})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    if asr_stream is not None:
                        await asr_stream.send_audio(data)
                    continue
                raw = message.get("text")
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "utterance.start":
                    async with batch_lock:
                        # VAD only proves that the microphone heard energy.
                        # Keep the current answer alive until ASR produces
                        # speech (or the client explicitly confirms barge-in).
                        utterance_sequence += 1
                        utterance_id = utterance_sequence
                        batch_partials[utterance_id] = ""
                        LOGGER.info(
                            "voice.vad.utterance_start connection=%s utterance=%s interrupt=%s level=%s threshold=%s",
                            connection_id,
                            utterance_id,
                            bool(event.get("interrupt")),
                            event.get("level", "-"),
                            event.get("threshold", "-"),
                        )
                        if asr_stream is not None:
                            previous_stream, asr_stream = asr_stream, None
                            previous_start, asr_start_task = asr_start_task, None
                            previous_id = utterance_id - 1
                            LOGGER.info(
                                "voice.vad.utterance_implicit_end connection=%s utterance=%s",
                                connection_id,
                                previous_id,
                            )
                            if previous_start is not None:
                                schedule_finalize(previous_stream, previous_start, previous_id)
                            else:
                                await previous_stream.close()

                        partial_logged = False

                        async def send_partial(text: str, current: int = utterance_id) -> None:
                            nonlocal partial_logged
                            # Do not drop an older segment's hypothesis when a
                            # new VAD onset takes ownership. It remains part of
                            # the same batch and is sent with the newer prefix.
                            if current not in batch_partials:
                                return
                            if current in pending_user_speaking and _contains_spoken_text(text):
                                await send(
                                    {
                                        "type": "status",
                                        "status": "user_speaking",
                                        "utterance_id": current,
                                    }
                                )
                                pending_user_speaking.discard(current)
                            if not partial_logged:
                                LOGGER.info(
                                    "voice.asr.first_partial connection=%s utterance=%s chars=%s",
                                    connection_id,
                                    current,
                                    len(str(text or "")),
                                )
                                partial_logged = True
                            await send_batch_partial(current, text)

                        if event.get("interrupt"):
                            pending_user_speaking.add(utterance_id)
                        else:
                            pending_user_speaking.discard(utterance_id)
                        asr_stream = make_asr_stream(send_partial)
                        # Provider setup is deliberately concurrent with the
                        # browser receive loop. XfyunStream buffers incoming
                        # PCM until the socket opens, so the user can interrupt
                        # immediately and the first syllable is still flushed
                        # in order after the handshake.
                        asr_start_task = asyncio.create_task(start_asr(asr_stream, utterance_id))
                        if utterance_id not in pending_user_speaking:
                            await send(
                                {
                                    "type": "status",
                                    "status": "user_speaking",
                                    "utterance_id": utterance_id,
                                }
                            )
                elif event_type == "utterance.end" and asr_stream is not None:
                    async with batch_lock:
                        utterance_id = utterance_sequence
                        LOGGER.info(
                            "voice.vad.utterance_end connection=%s utterance=%s",
                            connection_id,
                            utterance_id,
                        )
                        completed_stream, asr_stream = asr_stream, None
                        completed_start, asr_start_task = asr_start_task, None
                        if completed_start is not None:
                            schedule_finalize(completed_stream, completed_start, utterance_id)
                        else:
                            await completed_stream.close()
                elif event_type == "utterance.cancel":
                    async with batch_lock:
                        clear_voice_batch()
                        await stop_finalize_tasks()
                        await close_active_asr()
                        await send(
                            {"type": "utterance.rejected", "utterance_id": utterance_sequence}
                        )
                elif event_type == "barge_in":
                    # Batch commit and confirmed barge-in both replace the
                    # active answer. Serialize them so a concurrently
                    # committing ASR batch cannot create an answer just after
                    # the interruption has already been handled.
                    async with batch_lock:
                        LOGGER.info(
                            "voice.barge_in connection=%s utterance=%s",
                            connection_id,
                            utterance_sequence,
                        )
                        await stop_answer("barge_in")
                    await send(
                        {
                            "type": "status",
                            "status": "user_speaking",
                            "utterance_id": utterance_sequence,
                        }
                    )
                elif event_type == "text":
                    text = str(event.get("text") or "").strip()
                    if text:
                        async with batch_lock:
                            clear_voice_batch()
                            await stop_finalize_tasks()
                            await close_active_asr()
                            await start_answer(text, "text_input")
                elif event_type == "context":
                    update_voice_context(event)
                elif event_type == "interrupt":
                    async with batch_lock:
                        await stop_answer("client_interrupt")
                        clear_voice_batch()
                        await stop_finalize_tasks()
                        await close_active_asr()
                        await send(
                            {
                                "type": "status",
                                "status": "listening",
                                "utterance_id": utterance_sequence,
                            }
                        )
        except WebSocketDisconnect:
            pass
        finally:
            clear_voice_batch()
            await stop_finalize_tasks()
            await stop_answer("connection_closed")
            await close_active_asr()

    if FRONTEND_DIR.is_dir():
        app.frontend("/", directory=FRONTEND_DIR, fallback="index.html")

    return app


app = create_app()


def main() -> None:
    """Run the local ASGI server."""

    import uvicorn

    from .config import DEBUG, HOST, PORT

    uvicorn.run("anti_fraud_explorer.api:app", host=HOST, port=PORT, reload=DEBUG)


__all__ = ["app", "create_app", "main"]
