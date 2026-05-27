"""Flask web app for the anti-fraud case knowledge base."""

import uuid

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from .. import __version__
from ..agent import Agent, task_type_label
from ..config import PROJECT_ROOT, settings
from ..domain.dataset import get_knowledge_base, item_to_dict
from ..service.conversation import store as conv_store
from ..service.search import search_items
from ..service.asr import (
    asr_available,
    recognize_speech,
    VolcASRError,
)
from ..service.tts import (
    openai_tts_available,
    server_tts_engine,
    speech_audio_payload,
    stream_speech_audio,
    valid_tts_filename,
    volc_tts_available,
)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    app.logger.info(
        "TTS: engine=%s volc=%s openai=%s",
        server_tts_engine(),
        volc_tts_available(),
        openai_tts_available(),
    )

    @app.after_request
    def prevent_dev_cache(response):
        if request.path == "/" or request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/")
    def index():
        response = app.make_response(render_template("index.html"))
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/api/meta")
    def meta():
        kb = get_knowledge_base()
        risk_level_order = ["低", "中", "高", "极高", "无法判断"]
        risk_levels = risk_level_order
        entry_channels: list[str] = []
        for item in kb.items:
            for channel in item.entry_channels:
                if channel and channel not in entry_channels:
                    entry_channels.append(channel)
        return jsonify(
            {
                "app_version": __version__,
                "schema_version": kb.schema_version,
                "generated_at": kb.generated_at,
                "source": kb.source,
                "item_count": len(kb.items),
                "category_count": len(kb.categories),
                "risk_levels": risk_levels,
                "entry_channels": entry_channels,
                "categories": [
                    {"id": category.id, "name": category.name, "item_count": category.item_count}
                    for category in kb.categories
                ],
            }
        )

    @app.get("/api/items")
    def items():
        kb = get_knowledge_base()
        query = request.args.get("q", "")
        category = request.args.get("category", "")
        risk_level = request.args.get("risk_level", "")
        entry_channel = request.args.get("entry_channel", "")
        limit = max(int(request.args.get("limit", "30")), 1)
        offset = max(int(request.args.get("offset", "0")), 0)

        result, total = search_items(
            kb,
            query=query,
            category=category,
            risk_level=risk_level,
            entry_channel=entry_channel,
            limit=limit,
            offset=offset,
        )
        return jsonify(
            {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [_item_payload(item) for item in result],
            }
        )

    @app.get("/api/items/<item_id>")
    def item_detail(item_id: str):
        kb = get_knowledge_base()
        item = kb.get(item_id)
        if item is None:
            abort(404)
        return jsonify(_item_payload(item, include_content=True))

    @app.post("/api/ask")
    def ask():
        kb = get_knowledge_base()
        payload = request.get_json(silent=True) or {}
        question, category, session_id, include_speech, context = _parse_ask_payload(payload)

        agent = Agent(kb)
        try:
            result = agent.dispatch(
                query=question,
                category=category,
                include_speech=include_speech,
                context=context,
            )
        except Exception as exc:
            from ..ai import describe_model_error

            app.logger.exception("Ask request failed")
            warning = describe_model_error(exc)
            return jsonify(_ask_error_payload(session_id, warning))

        if result is None:
            return jsonify(_ask_empty_payload(session_id))

        return jsonify(_ask_success_payload(result, session_id, kb))

    @app.post("/api/tts")
    def create_tts_audio():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            return jsonify({"speech_engine": "browser", "error": "empty_text"})
        return jsonify(speech_audio_payload(text))

    @app.get("/api/tts/stream")
    def stream_tts_audio():
        text = str(request.args.get("text") or "").strip()
        audio_stream = stream_speech_audio(text)
        if audio_stream is None:
            abort(503)
        encoding = settings.volc_tts_encoding.lower()
        mimetype = {
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "opus": "audio/ogg",
            "wav": "audio/wav",
        }.get(encoding, "application/octet-stream")
        return Response(
            stream_with_context(audio_stream),
            mimetype=mimetype,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/tts/<filename>")
    def tts_audio(filename: str):
        if not valid_tts_filename(filename):
            abort(404)
        path = settings.tts_cache_dir / filename
        if not path.is_file():
            abort(404)
        return send_file(path, conditional=True, max_age=3600)

    @app.post("/api/asr")
    def asr_recognize():
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "no_audio_data"}), 400

        content_type = request.content_type or ""
        format_hint = "webm"
        if "audio/webm" in content_type:
            format_hint = "webm"
        elif "audio/ogg" in content_type:
            format_hint = "ogg"
        elif "audio/wav" in content_type:
            format_hint = "wav"
        elif "audio/mp3" in content_type or "audio/mpeg" in content_type:
            format_hint = "mp3"

        try:
            text = recognize_speech(audio_data, format=format_hint)
            return jsonify({"text": text})
        except VolcASRError as exc:
            app.logger.warning("ASR failed: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except Exception:
            app.logger.exception("ASR unexpected error")
            return jsonify({"error": "asr_unavailable"}), 503

    return app


def _item_payload(item, include_content: bool = False) -> dict:
    return item_to_dict(item, include_content=include_content)


def _parse_ask_payload(payload: dict) -> tuple[str, str, str, bool, dict | None]:
    """Extract and normalize ask request parameters."""
    question = str(payload.get("question") or "")
    category = str(payload.get("category") or "")
    session_id = str(payload.get("session_id") or "")
    voice_enabled = payload.get("voice_enabled", True)
    if isinstance(voice_enabled, str):
        include_speech = voice_enabled.lower() not in {"0", "false", "no", "off"}
    else:
        include_speech = bool(voice_enabled)

    if not session_id:
        session_id = uuid.uuid4().hex[:12]

    first_turn = conv_store.is_first_turn(session_id)
    context = conv_store.format_context(session_id) if not first_turn else None
    if context is None and isinstance(payload.get("context"), dict):
        context = payload.get("context")

    return question, category, session_id, include_speech, context


def _ask_error_payload(session_id: str, warning: str) -> dict:
    return {
        "type": "result",
        "session_id": session_id,
        "answer": f"问答暂时失败：{warning}",
        "speech": "",
        "mode": "fallback",
        "task_type": "fact_qa",
        "task_label": "问答失败",
        "confidence": 0.0,
        "sources": [],
        "items": [],
        "evidence": [],
        "selection_reason": "",
        "warnings": [warning],
        "total_count": 0,
        "decision": {},
    }


def _ask_empty_payload(session_id: str) -> dict:
    return {
        "type": "result",
        "session_id": session_id,
        "answer": "问答暂时失败：未生成有效结果。",
        "speech": "",
        "mode": "fallback",
        "task_type": "fact_qa",
        "task_label": "问答失败",
        "confidence": 0.0,
        "sources": [],
        "items": [],
        "evidence": [],
        "selection_reason": "",
        "warnings": ["未生成有效结果。"],
        "total_count": 0,
        "decision": {},
    }


def _extract_result_items(result, kb) -> tuple[list[str], list[dict]]:
    """Pull item titles and full dicts from an AgentResult."""
    item_titles: list[str] = []
    items_full: list[dict] = []
    seen_item_ids: set[str] = set()
    for it in result.items or []:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        if title and title not in item_titles:
            item_titles.append(title)
        item_id = str(it.get("id") or "").strip()
        if item_id and item_id not in seen_item_ids:
            item_obj = kb.get(item_id)
            if item_obj is not None:
                items_full.append(item_to_dict(item_obj, include_content=True))
                seen_item_ids.add(item_id)
        elif not item_id and title:
            items_full.append(dict(it))
    return item_titles, items_full


def _ask_success_payload(result, session_id: str, kb) -> dict:
    """Persist turn and build the success response payload."""
    item_titles, items_full = _extract_result_items(result, kb)
    conv_store.add_turn(
        session_id=session_id,
        query=result.answer or "",
        answer=result.answer or "",
        item_titles=item_titles,
        items_full=items_full,
    )
    return {
        "type": "result",
        "session_id": session_id,
        "answer": result.answer,
        "speech": result.speech,
        "mode": result.mode,
        "task_type": result.task_type.value,
        "task_label": task_type_label(result.task_type),
        "confidence": result.confidence,
        "sources": result.sources,
        "items": result.items,
        "evidence": result.evidence,
        "selection_reason": result.selection_reason,
        "warnings": result.warnings,
        "total_count": result.total_count,
        "decision": result.decision,
    }


def main() -> None:
    create_app().run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
