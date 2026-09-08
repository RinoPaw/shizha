"""Streaming Xunfei ASR transport for browser-captured 16 kHz PCM audio.

识诈只使用讯飞“中英识别大模型”这一条实时识别链路。该模型本身支持
普通话、英语以及 202 种中文方言免切换识别，因此不再并行探测方言模型
和多语种模型，也不再做三路结果仲裁。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
from collections.abc import Awaitable, Callable
from email.utils import formatdate
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

LOGGER = logging.getLogger(__name__)


class VoiceProviderError(RuntimeError):
    """Provider failure without secret-bearing diagnostic text."""


class _FinishRequest:
    """A queue barrier used to serialize the Xunfei status=2 packet."""

    def __init__(self) -> None:
        self.done: asyncio.Future[None] = asyncio.get_running_loop().create_future()


class XfyunStream:
    """One utterance over Xunfei's streaming IAT WebSocket protocol."""

    host = "iat.xf-yun.com"
    path = "/v1"
    chunk_size = 1280
    # 1280 bytes at 16 kHz, 16-bit, mono PCM represent exactly 40 ms.
    frame_interval = 0.04

    def __init__(
        self,
        *,
        app_id: str,
        api_key: str,
        api_secret: str,
        on_partial: Callable[[str], Awaitable[None]] | None = None,
        hotwords: str | list[str] | tuple[str, ...] | None = None,
        resource_id: str | None = None,
        host: str | None = None,
        path: str = "/v1",
        language: str = "zh_cn",
        accent: str = "mandarin",
        domain: str = "slm",
        language_hint: str = "",
        dynamic_correction: bool = True,
        eos: int = 1800,
    ) -> None:
        self.app_id = app_id.strip()
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.host = str(host or type(self).host).strip()
        self.path = str(path or "/v1").strip()
        self.language = str(language or "zh_cn").strip()
        self.accent = str(accent or "mandarin").strip()
        self.domain = str(domain or "slm").strip()
        self.language_hint = str(language_hint or "").strip()
        self.dynamic_correction = bool(dynamic_correction)
        self.eos = min(max(int(eos), 600), 10000)
        self.resource_id = (resource_id or "").strip()
        self._hotword_spec = self._format_hotwords(hotwords)
        self._socket: ClientConnection | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._sender: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._buffer = bytearray()
        self._send_queue: asyncio.Queue[bytes | _FinishRequest] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._pieces: dict[int, str] = {}
        self._candidate_pieces: dict[int, tuple[str, ...]] = {}
        self._language_pieces: dict[int, str] = {}
        self._sequence = 0
        self._first = True
        self._has_audio = False
        self._next_send_at: float | None = None
        self._send_failure: VoiceProviderError | None = None
        self._closed = False
        self._error = False
        self._on_partial = on_partial

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.api_key and self.api_secret)

    @staticmethod
    def _format_hotwords(hotwords: str | list[str] | tuple[str, ...] | None) -> str:
        """Return Xunfei's ``dhw`` value, bounded to 1024 UTF-8 bytes."""

        if hotwords is None:
            return ""
        values = [hotwords] if isinstance(hotwords, str) else hotwords
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "")
            for item in re.split(r"[|,，、;；\r\n\t]+", text):
                item = "".join(item.split())
                if not item or item in seen:
                    continue
                seen.add(item)
                cleaned.append(item)

        prefix = "utf-8;"
        remaining = 1024 - len(prefix.encode("utf-8"))
        selected: list[str] = []
        used = 0
        for item in cleaned:
            encoded = item.encode("utf-8")
            required = len(encoded) if not selected else len(encoded) + 1
            if required <= remaining - used:
                selected.append(item)
                used += required
                continue
            if not selected and remaining > 0:
                item = encoded[:remaining].decode("utf-8", errors="ignore")
                if item:
                    selected.append(item)
            break
        return prefix + "|".join(selected) if selected else ""

    def signed_url(self) -> str:
        date = formatdate(usegmt=True)
        origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        digest = hmac.new(
            self.api_secret.encode(), origin.encode(), digestmod=hashlib.sha256
        ).digest()
        signature = base64.b64encode(digest).decode()
        authorization = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        params = {
            "authorization": base64.b64encode(authorization.encode()).decode(),
            "date": date,
            "host": self.host,
        }
        return f"wss://{self.host}{self.path}?{urlencode(params)}"

    async def start(self) -> None:
        if not self.configured:
            raise VoiceProviderError("voice_provider_unconfigured")
        async with self._send_lock:
            if self._closed:
                raise VoiceProviderError("voice_stream_closed")
        try:
            socket = await connect(
                self.signed_url(), open_timeout=10, close_timeout=2, max_size=2**20
            )
        except Exception as exc:
            raise VoiceProviderError("voice_provider_unavailable") from exc
        should_close = False
        async with self._send_lock:
            if self._closed:
                should_close = True
            else:
                self._socket = socket
                self._next_send_at = None
                self._receiver = asyncio.create_task(self._receive())
                self._sender = asyncio.create_task(self._send_loop())
                self._enqueue_complete_frames()
        if should_close:
            await socket.close()
            raise VoiceProviderError("voice_stream_closed")
        await asyncio.sleep(0)

    async def send_audio(self, data: bytes) -> None:
        if not data:
            return
        async with self._send_lock:
            if self._closed:
                return
            self._has_audio = True
            self._buffer.extend(data)
            if self._socket is not None:
                self._enqueue_complete_frames()

    def _enqueue_complete_frames(self) -> None:
        while len(self._buffer) >= self.chunk_size:
            frame = bytes(self._buffer[: self.chunk_size])
            del self._buffer[: self.chunk_size]
            self._send_queue.put_nowait(frame)

    async def _send_loop(self) -> None:
        current: bytes | _FinishRequest | None = None
        try:
            while True:
                current = await self._send_queue.get()
                if isinstance(current, _FinishRequest):
                    if self._socket is None:
                        raise VoiceProviderError("voice_provider_disconnected")
                    if self._send_failure is not None:
                        raise self._send_failure
                    try:
                        await self._socket.send(json.dumps(self._last_packet()))
                    except Exception as exc:
                        raise VoiceProviderError("voice_provider_disconnected") from exc
                    if not current.done.done():
                        current.done.set_result(None)
                    current = None
                    continue

                now = asyncio.get_running_loop().time()
                while self._next_send_at is not None and self._next_send_at > now:
                    await asyncio.sleep(self._next_send_at - now)
                    now = asyncio.get_running_loop().time()
                await self._send_frame(current)
                self._next_send_at = asyncio.get_running_loop().time() + self.frame_interval
                current = None
        except asyncio.CancelledError:
            if isinstance(current, _FinishRequest) and not current.done.done():
                current.done.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all provider transport failures
            failure = exc if isinstance(exc, VoiceProviderError) else VoiceProviderError(
                "voice_provider_disconnected"
            )
            self._send_failure = failure
            self._error = True
            self._done.set()
            if isinstance(current, _FinishRequest) and not current.done.done():
                current.done.set_exception(failure)
            while True:
                try:
                    pending = self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(pending, _FinishRequest) and not pending.done.done():
                    pending.done.set_exception(failure)

    async def finish(self) -> str:
        try:
            async with self._send_lock:
                if self._socket is None:
                    raise VoiceProviderError("voice_stream_not_started")
                if self._send_failure is not None:
                    raise self._send_failure
                self._enqueue_complete_frames()
                if self._buffer:
                    self._send_queue.put_nowait(bytes(self._buffer))
                    self._buffer.clear()
                empty = not self._has_audio
                sender = self._sender
                if not empty:
                    if sender is None or sender.done():
                        raise VoiceProviderError("voice_provider_disconnected")
                    request = _FinishRequest()
                    self._send_queue.put_nowait(request)
            if empty:
                return ""
            await asyncio.shield(request.done)
            await asyncio.wait_for(self._done.wait(), timeout=8)
        except TimeoutError as exc:
            raise VoiceProviderError("voice_provider_timeout") from exc
        finally:
            await self.close()
        if self._error:
            raise VoiceProviderError("voice_provider_failed")
        return self.current_text()

    async def close(self) -> None:
        async with self._send_lock:
            self._closed = True
            socket, self._socket = self._socket, None
            sender, self._sender = self._sender, None
            receiver, self._receiver = self._receiver, None
            pending = []
            while True:
                try:
                    pending.append(self._send_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._buffer.clear()
        for item in pending:
            if isinstance(item, _FinishRequest) and not item.done.done():
                item.done.cancel()
        if sender is not None and sender is not asyncio.current_task():
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        if socket is not None:
            try:
                await socket.close()
            except Exception:
                LOGGER.info("asr.socket.close_failed", exc_info=True)
        if receiver is not None and receiver is not asyncio.current_task():
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def _send_frame(self, frame: bytes) -> None:
        if self._socket is None:
            return
        audio = base64.b64encode(frame).decode()
        packet = self._first_packet(audio) if self._first else self._middle_packet(audio)
        self._first = False
        try:
            await self._socket.send(json.dumps(packet))
        except Exception as exc:
            self._error = True
            self._done.set()
            raise VoiceProviderError("voice_provider_disconnected") from exc

    async def _receive(self) -> None:
        assert self._socket is not None
        try:
            async for raw in self._socket:
                message = json.loads(raw)
                header = message.get("header", {})
                if header.get("code", -1) != 0:
                    self._error = True
                    self._done.set()
                    return
                result = message.get("payload", {}).get("result") or {}
                encoded = result.get("text")
                decoded: dict[str, object] | None = None
                if encoded:
                    decoded_value = json.loads(base64.b64decode(encoded).decode())
                    if isinstance(decoded_value, dict):
                        decoded = decoded_value
                    if (
                        decoded is not None
                        and self.apply_result(decoded)
                        and self._on_partial is not None
                    ):
                        await self._on_partial(self.current_text())
                # Different IAT deployments place the terminal marker in
                # different layers: retain compatibility with all of them.
                if (
                    self._is_status_two(header.get("status"))
                    or self._is_status_two(result.get("status"))
                    or (
                        decoded is not None
                        and (
                            self._is_status_two(decoded.get("status"))
                            or self._is_last_result(decoded.get("ls"))
                        )
                    )
                ):
                    self._done.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all provider transport failures
            LOGGER.warning("asr.socket.receive_failed reason=%s", type(exc).__name__)
            self._error = True
            self._done.set()

    @staticmethod
    def extract_text(result: dict[str, object]) -> str:
        return XfyunStream.extract_candidates(result)[0]

    @staticmethod
    def extract_candidates(result: dict[str, object]) -> tuple[str, ...]:
        """Extract full-text hypotheses, preserving provider candidate order."""

        words: list[list[str]] = []
        segments = result.get("ws", [])
        if not isinstance(segments, list):
            return ("",)
        for segment in segments:
            raw_candidates = segment.get("cw", []) if isinstance(segment, dict) else []
            if not isinstance(raw_candidates, list):
                continue
            candidates = []
            for candidate in raw_candidates:
                if isinstance(candidate, dict):
                    word = str(candidate.get("w", ""))
                    if word:
                        candidates.append(word)
            if candidates:
                words.append(candidates)
        if not words:
            return ("",)
        count = max(len(items) for items in words)
        output: list[str] = []
        for index in range(count):
            output.append(
                "".join(items[index] if index < len(items) else items[0] for items in words)
            )
        return tuple(output)

    @staticmethod
    def _is_status_two(value: object) -> bool:
        return value == 2 or value == "2"

    @staticmethod
    def _is_last_result(value: object) -> bool:
        return value is True or value == "true" or value == 1 or value == "1"

    def apply_result(self, result: dict[str, object]) -> bool:
        candidates = self.extract_candidates(result)
        piece = candidates[0] if candidates else ""
        sequence = int(result.get("sn", max(self._pieces, default=-1) + 1))
        replacement = result.get("rg") if result.get("pgs") == "rpl" else None
        if isinstance(replacement, list) and len(replacement) == 2:
            start, end = int(replacement[0]), int(replacement[1])
            for key in range(start, end + 1):
                self._pieces.pop(key, None)
                self._candidate_pieces.pop(key, None)
                self._language_pieces.pop(key, None)
        if piece:
            self._pieces[sequence] = piece
        if candidates and any(candidates):
            self._candidate_pieces[sequence] = tuple(
                candidate
                for index, candidate in enumerate(candidates)
                if candidate and candidate not in candidates[:index]
            )
        source_language = self.extract_language(result)
        if source_language:
            self._language_pieces[sequence] = source_language
        return bool(piece or replacement)

    def current_text(self) -> str:
        return "".join(self._pieces[key] for key in sorted(self._pieces)).strip()

    @staticmethod
    def extract_language(result: dict[str, object]) -> str:
        """Return the dominant provider ``cw.lg`` source-language tag."""

        counts: dict[str, int] = {}
        segments = result.get("ws", [])
        if not isinstance(segments, list):
            return ""
        for segment in segments:
            candidates = segment.get("cw", []) if isinstance(segment, dict) else []
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                language = str(candidate.get("lg") or "").strip().casefold()
                if language:
                    counts[language] = counts.get(language, 0) + 1
                    break
        return max(counts, key=counts.get) if counts else ""

    @property
    def detected_language(self) -> str:
        counts: dict[str, int] = {}
        for language in self._language_pieces.values():
            counts[language] = counts.get(language, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    @property
    def alternative_texts(self) -> tuple[str, ...]:
        """Final, de-duplicated full-transcript hypotheses (top-1 first)."""

        if not self._candidate_pieces:
            text = self.current_text()
            return (text,) if text else ()
        keys = sorted(self._candidate_pieces)
        count = max(len(self._candidate_pieces[key]) for key in keys)
        texts: list[str] = []
        for index in range(count):
            parts = []
            for key in keys:
                candidates = self._candidate_pieces[key]
                parts.append(candidates[index] if index < len(candidates) else candidates[0])
            text = "".join(parts).strip()
            if text and text not in texts:
                texts.append(text)
        top1 = self.current_text()
        if top1 and top1 not in texts:
            texts.insert(0, top1)
        return tuple(texts)

    @property
    def candidates(self) -> tuple[str, ...]:
        return self.alternative_texts

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _audio(self, audio: str, status: int) -> dict[str, object]:
        return {
            "encoding": "raw",
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "seq": self._next_sequence(),
            "status": status,
            "audio": audio,
        }

    def _header(self, status: int) -> dict[str, object]:
        header: dict[str, object] = {"app_id": self.app_id, "status": status}
        if self.resource_id:
            header["res_id"] = self.resource_id
        return header

    def _first_packet(self, audio: str) -> dict[str, object]:
        iat: dict[str, object] = {
            "domain": self.domain,
            "language": self.language,
            "accent": self.accent,
            "eos": self.eos,
            "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
        }
        if self.dynamic_correction:
            iat["dwa"] = "wpgs"
        if self.language_hint:
            iat["ln"] = self.language_hint
        if self._hotword_spec:
            iat["dhw"] = self._hotword_spec
        return {
            "header": self._header(0),
            "parameter": {"iat": iat},
            "payload": {"audio": self._audio(audio, 0)},
        }

    def _middle_packet(self, audio: str) -> dict[str, object]:
        return {
            "header": self._header(1),
            "payload": {"audio": self._audio(audio, 1)},
        }

    def _last_packet(self) -> dict[str, object]:
        return {
            "header": self._header(2),
            "payload": {"audio": self._audio("", 2)},
        }


class AutoXfyunStream:
    """Compatibility facade backed by one Xunfei Chinese/English ASR stream.

    Older callers still pass the three-stream configuration fields.  They are
    intentionally accepted here so the browser WebSocket contract can stay
    unchanged while the runtime only opens the official Chinese/English model
    endpoint (``zh_cn`` + ``mandarin``).  That model also covers 202 Chinese
    dialects, so no multilingual/dialect arbitration is necessary.
    """

    CHINESE = "chinese"
    DIALECT = "dialect"  # compatibility-only constants for older callers
    MULTILINGUAL = "multilingual"
    LEGACY = "legacy"
    AUTO = "auto"

    def __init__(
        self,
        *,
        app_id: str,
        api_key: str,
        api_secret: str,
        multilingual_app_id: str = "",
        multilingual_api_key: str = "",
        multilingual_api_secret: str = "",
        host: str | None = None,
        multilingual_host: str | None = None,
        multilingual_language_hint: str = "",
        legacy_host: str | None = "iat.xf-yun.com",
        on_partial: Callable[[str], Awaitable[None]] | None = None,
        hotwords: str | list[str] | tuple[str, ...] | None = None,
        resource_id: str | None = None,
        mode: str = AUTO,
        preferred_mode: str = DIALECT,
    ) -> None:
        # The unused parameters stay in the signature temporarily so a rolling
        # deployment does not require an atomic api.py/config.py migration.
        del (
            multilingual_app_id,
            multilingual_api_key,
            multilingual_api_secret,
            host,
            multilingual_host,
            multilingual_language_hint,
            mode,
            preferred_mode,
        )
        self._stream = XfyunStream(
            app_id=app_id,
            api_key=api_key,
            api_secret=api_secret,
            host=legacy_host or "iat.xf-yun.com",
            language="zh_cn",
            accent="mandarin",
            domain="slm",
            dynamic_correction=True,
            on_partial=on_partial,
            hotwords=hotwords,
            resource_id=resource_id,
        )
        self._selected_mode = ""

    async def start(self) -> None:
        await self._stream.start()

    async def send_audio(self, data: bytes) -> None:
        await self._stream.send_audio(data)

    async def finish(self) -> str:
        result = await self._stream.finish()
        self._selected_mode = self.CHINESE
        return result

    async def close(self) -> None:
        await self._stream.close()

    @property
    def selected_mode(self) -> str:
        return self._selected_mode or self.CHINESE

    @property
    def detected_language(self) -> str:
        return self._stream.detected_language

    @property
    def candidates(self) -> tuple[str, ...]:
        return self._stream.candidates


__all__ = ["AutoXfyunStream", "VoiceProviderError", "XfyunStream"]
