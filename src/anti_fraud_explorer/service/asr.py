"""Server-side ASR: Volcengine speech-to-text."""

import base64
import json
import logging
import uuid
from typing import Any

from ..config import settings

LOGGER = logging.getLogger(__name__)


class VolcASRError(RuntimeError):
    """Raised when ASR recognition fails."""


def asr_available() -> bool:
    """True when Volcengine ASR credentials are configured."""
    return bool(
        settings.volc_asr_app_id and settings.volc_asr_access_token
    ) or bool(settings.volc_asr_api_key)


def recognize_speech(audio_bytes: bytes, format: str = "webm") -> str:
    """Recognize speech using Volcengine ASR flash API."""
    if not asr_available():
        raise VolcASRError("ASR is not configured")

    from urllib import request

    headers = _asr_headers()
    payload = _build_flash_payload(audio_bytes, format)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = request.Request(
        settings.volc_asr_endpoint,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.volc_asr_timeout) as response:
            response_body = response.read()
    except Exception as exc:
        raise VolcASRError("Volcengine ASR request failed") from exc

    try:
        data = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise VolcASRError("Volcengine ASR returned invalid JSON") from exc

    status_code = data.get("code") or data.get("status_code")
    if status_code and str(status_code) != "20000000":
        message = str(data.get("message") or data.get("msg") or "unknown error")
        raise VolcASRError(f"Volcengine ASR failed: {message}")

    result = data.get("result") or {}
    text = result.get("text") or ""
    if not text and "utterances" in result:
        text = "".join(u.get("text", "") for u in result["utterances"])

    return text.strip()


def _asr_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": settings.volc_asr_resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1",
    }
    if settings.volc_asr_api_key:
        headers["X-Api-Key"] = settings.volc_asr_api_key
    else:
        headers["X-Api-App-Key"] = settings.volc_asr_app_id
        headers["X-Api-Access-Key"] = settings.volc_asr_access_token
    return headers


def _build_flash_payload(audio_bytes: bytes, format: str) -> dict[str, Any]:
    return {
        "user": {"uid": "anti-fraud-web"},
        "audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": _normalize_format(format),
        },
        "request": {
            "model_name": "bigmodel",
            "enable_punc": True,
            "enable_itn": True,
            "enable_ddc": True,
        },
    }


def _normalize_format(fmt: str) -> str:
    fmt = fmt.lower().strip(".")
    mapping = {
        "webm": "webm",
        "ogg": "ogg",
        "mp3": "mp3",
        "mp4": "mp4",
        "wav": "wav",
        "m4a": "m4a",
    }
    return mapping.get(fmt, "webm")
