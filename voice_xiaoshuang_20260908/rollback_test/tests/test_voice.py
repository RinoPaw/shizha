from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

import anti_fraud_explorer.voice as voice_module
from anti_fraud_explorer.voice import AutoXfyunStream, XfyunStream


def make_stream() -> XfyunStream:
    return XfyunStream(app_id="app", api_key="key", api_secret="secret")


def test_extract_text_uses_best_candidate_from_each_segment() -> None:
    result = {
        "ws": [
            {"cw": [{"w": "朱仙镇"}, {"w": "朱仙阵"}]},
            {"cw": [{"w": "木版年画"}]},
        ]
    }

    assert XfyunStream.extract_text(result) == "朱仙镇木版年画"


def test_extract_candidates_joins_nbest_words_in_provider_order() -> None:
    result = {
        "ws": [
            {"cw": [{"w": "朱仙镇"}, {"w": "朱仙阵"}]},
            {"cw": [{"w": "木版年画"}, {"w": "木板年画"}, {"w": "木版年华"}]},
        ]
    }

    assert XfyunStream.extract_candidates(result) == (
        "朱仙镇木版年画",
        "朱仙阵木板年画",
        "朱仙镇木版年华",
    )


def test_wpgs_replacement_discards_stale_top1_and_alternatives() -> None:
    stream = make_stream()
    assert stream.apply_result({"sn": 0, "ws": [{"cw": [{"w": "前文"}]}]})
    assert stream.apply_result(
        {"sn": 1, "ws": [{"cw": [{"w": "书秀"}, {"w": "旧候选"}]}]}
    )
    assert stream.alternative_texts == ("前文书秀", "前文旧候选")

    assert stream.apply_result(
        {
            "sn": 2,
            "pgs": "rpl",
            "rg": [1, 1],
            "ws": [{"cw": [{"w": "刷单返利"}, {"w": "新候选"}]}],
        }
    )
    assert stream.current_text() == "前文刷单返利"
    assert stream.candidates == ("前文刷单返利", "前文新候选")
    assert "旧候选" not in "|".join(stream.alternative_texts)


def test_chinese_model_packet_matches_official_contract() -> None:
    stream = XfyunStream(
        app_id="app",
        api_key="key",
        api_secret="secret",
        resource_id="  res-123 ",
        hotwords=[" 刷单返利 ", "刷单返利", "安全|账户", "", "甲\n乙"],
    )
    packet = stream._first_packet("")
    iat = packet["parameter"]["iat"]

    assert stream.host == "iat.xf-yun.com"
    assert packet["header"]["res_id"] == "res-123"
    assert stream._middle_packet("")["header"]["res_id"] == "res-123"
    assert stream._last_packet()["header"]["res_id"] == "res-123"
    assert iat["domain"] == "slm"
    assert iat["language"] == "zh_cn"
    assert iat["accent"] == "mandarin"
    assert "nbest" not in iat
    assert "wbest" not in iat
    assert iat["dhw"] == "utf-8;刷单返利|安全|账户|甲|乙"
    assert len(iat["dhw"].encode("utf-8")) <= 1024


def test_signed_url_uses_selected_host() -> None:
    stream = XfyunStream(
        app_id="app",
        api_key="key",
        api_secret="secret",
        host="iat.xf-yun.com",
    )

    parsed = urlparse(stream.signed_url())

    assert parsed.hostname == "iat.xf-yun.com"
    assert parse_qs(parsed.query)["host"] == ["iat.xf-yun.com"]


def test_provider_language_tracks_dominant_cw_lg_and_wpgs_replacement() -> None:
    result = {
        "ws": [
            {"cw": [{"w": "hello", "lg": "en"}]},
            {"cw": [{"w": "heritage", "lg": "en"}]},
            {"cw": [{"w": "遗产", "lg": "zh"}]},
        ]
    }
    assert XfyunStream.extract_language(result) == "en"

    stream = make_stream()
    stream.apply_result({"sn": 0, "ws": [{"cw": [{"w": "hello", "lg": "en"}]}]})
    stream.apply_result({"sn": 1, "ws": [{"cw": [{"w": "there", "lg": "en"}]}]})
    assert stream.detected_language == "en"

    stream.apply_result(
        {
            "sn": 2,
            "pgs": "rpl",
            "rg": [0, 1],
            "ws": [{"cw": [{"w": "你好", "lg": "zh"}]}],
        }
    )
    assert stream.detected_language == "zh"
    assert stream.current_text() == "你好"


def test_auto_stream_builds_only_one_chinese_english_model(monkeypatch) -> None:
    class FakeProviderStream:
        instances: list["FakeProviderStream"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.detected_language = ""
            self.candidates: tuple[str, ...] = ()
            self.__class__.instances.append(self)

        async def start(self) -> None:
            return

        async def send_audio(self, _data: bytes) -> None:
            return

        async def finish(self) -> str:
            return ""

        async def close(self) -> None:
            return

    monkeypatch.setattr(voice_module, "XfyunStream", FakeProviderStream)

    AutoXfyunStream(
        app_id="main-app",
        api_key="main-key",
        api_secret="main-secret",
        multilingual_app_id="unused-app",
        multilingual_api_key="unused-key",
        multilingual_api_secret="unused-secret",
        host="iat.cn-huabei-1.xf-yun.com",
        multilingual_host="some-multilingual-host",
        multilingual_language_hint="en|ja|ko",
        legacy_host="iat.xf-yun.com",
        hotwords=("苏绣",),
        resource_id="resource-1",
    )

    assert len(FakeProviderStream.instances) == 1
    kwargs = FakeProviderStream.instances[0].kwargs
    assert kwargs["app_id"] == "main-app"
    assert kwargs["api_key"] == "main-key"
    assert kwargs["api_secret"] == "main-secret"
    assert kwargs["host"] == "iat.xf-yun.com"
    assert kwargs["language"] == "zh_cn"
    assert kwargs["accent"] == "mandarin"
    assert kwargs["domain"] == "slm"
    assert kwargs["dynamic_correction"] is True
    assert kwargs["hotwords"] == ("苏绣",)
    assert kwargs["resource_id"] == "resource-1"


def test_auto_stream_delegates_chinese_english_and_dialect_results(monkeypatch) -> None:
    class FakeProviderStream:
        def __init__(self, **_kwargs: object) -> None:
            self.detected_language = "en"
            self.candidates = ("hello heritage", "hello")
            self.audio: list[bytes] = []
            self.started = False
            self.closed = False

        async def start(self) -> None:
            self.started = True

        async def send_audio(self, data: bytes) -> None:
            self.audio.append(data)

        async def finish(self) -> str:
            return "hello heritage"

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(voice_module, "XfyunStream", FakeProviderStream)

    async def scenario() -> None:
        stream = AutoXfyunStream(app_id="app", api_key="key", api_secret="secret")
        await stream.start()
        await stream.send_audio(b"pcm")

        assert await stream.finish() == "hello heritage"
        assert stream.selected_mode == AutoXfyunStream.CHINESE
        assert stream.detected_language == "en"
        assert stream.candidates == ("hello heritage", "hello")
        assert stream._stream.audio == [b"pcm"]

        await stream.close()
        assert stream._stream.closed

    asyncio.run(scenario())


def test_hotwords_are_bounded_to_1024_utf8_bytes() -> None:
    stream = XfyunStream(
        app_id="app",
        api_key="key",
        api_secret="secret",
        hotwords=["词" * 1000],
    )

    assert len(stream._hotword_spec.encode("utf-8")) <= 1024
    assert stream._hotword_spec.startswith("utf-8;")


def test_receive_accepts_terminal_status_from_result_or_decoded_ls() -> None:
    async def scenario() -> None:
        for message in (
            {"header": {"code": 0, "status": 0}, "payload": {"result": {"status": 2}}},
            {
                "header": {"code": 0, "status": 0},
                "payload": {
                    "result": {
                        "text": base64.b64encode(json.dumps({"ls": True}).encode()).decode()
                    }
                },
            },
        ):
            class FakeSocket:
                def __aiter__(self):
                    return self

                async def __anext__(self, _message=message):
                    if getattr(self, "sent", False):
                        raise StopAsyncIteration
                    self.sent = True
                    return json.dumps(_message)

            stream = make_stream()
            stream._socket = FakeSocket()
            await stream._receive()
            assert stream._done.is_set()

    asyncio.run(scenario())


def test_dynamic_correction_replaces_only_requested_range() -> None:
    stream = make_stream()
    assert stream.apply_result({"sn": 0, "ws": [{"cw": [{"w": "后还有"}]}]})
    assert stream.apply_result({"sn": 1, "ws": [{"cw": [{"w": "书秀"}]}]})
    assert stream.current_text() == "后还有书秀"
    assert stream.apply_result({"sn": 2, "pgs": "rpl", "rg": [1, 1], "ws": [{"cw": [{"w": "刷单返利"}]}]})
    assert stream.current_text() == "后还有刷单返利"


def test_packets_follow_streaming_status_contract() -> None:
    stream = make_stream()
    audio = base64.b64encode(bytes(1280)).decode()

    first = stream._first_packet(audio)
    middle = stream._middle_packet(audio)
    last = stream._last_packet()

    assert first["header"] == {"app_id": "app", "status": 0}
    assert first["parameter"]["iat"]["dwa"] == "wpgs"
    assert first["payload"]["audio"]["status"] == 0
    assert middle["payload"]["audio"]["status"] == 1
    assert last["payload"]["audio"]["status"] == 2
    assert last["payload"]["audio"]["audio"] == ""
    assert [
        first["payload"]["audio"]["seq"],
        middle["payload"]["audio"]["seq"],
        last["payload"]["audio"]["seq"],
    ] == [1, 2, 3]


def test_audio_captured_before_provider_connect_is_flushed_in_order(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.packets: list[str] = []

        async def send(self, packet: str) -> None:
            self.packets.append(packet)

        async def close(self) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    socket = FakeSocket()

    async def fake_connect(*_args, **_kwargs):
        return socket

    monkeypatch.setattr(voice_module, "connect", fake_connect)

    async def scenario() -> None:
        stream = make_stream()
        await stream.send_audio(bytes(stream.chunk_size))
        assert socket.packets == []
        await stream.start()
        assert len(socket.packets) == 1
        assert '"status": 0' in socket.packets[0]
        await stream.close()

    asyncio.run(scenario())


def test_handshake_cache_is_sent_on_a_40ms_paced_queue(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.packets: list[str] = []
            self.sent_at: list[float] = []

        async def send(self, packet: str) -> None:
            self.packets.append(packet)
            self.sent_at.append(asyncio.get_running_loop().time())

        async def close(self) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    socket = FakeSocket()

    async def fake_connect(*_args, **_kwargs):
        return socket

    monkeypatch.setattr(voice_module, "connect", fake_connect)

    async def scenario() -> None:
        stream = make_stream()
        await stream.send_audio(bytes(stream.chunk_size) * 3)
        await stream.start()
        while len(socket.packets) < 3:
            await asyncio.sleep(0.005)
        statuses = [json.loads(packet)["header"]["status"] for packet in socket.packets]
        assert statuses == [0, 1, 1]
        assert socket.sent_at[1] - socket.sent_at[0] >= stream.frame_interval * 0.8
        assert socket.sent_at[2] - socket.sent_at[1] >= stream.frame_interval * 0.8
        await stream.close()

    asyncio.run(scenario())


def test_finish_queues_status_two_after_all_audio(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self, stream: XfyunStream) -> None:
            self.stream = stream
            self.packets: list[str] = []

        async def send(self, packet: str) -> None:
            self.packets.append(packet)
            if json.loads(packet)["header"]["status"] == 2:
                self.stream._done.set()

        async def close(self) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    async def scenario() -> None:
        stream = make_stream()
        socket = FakeSocket(stream)

        async def fake_connect(*_args, **_kwargs):
            return socket

        monkeypatch.setattr(voice_module, "connect", fake_connect)
        await stream.send_audio(bytes(stream.chunk_size) * 2)
        await stream.start()
        assert await stream.finish() == ""
        statuses = [json.loads(packet)["header"]["status"] for packet in socket.packets]
        assert statuses == [0, 1, 2]

    asyncio.run(scenario())


def test_cancel_and_close_stop_sender_without_leaking_queued_frames(monkeypatch) -> None:
    class FakeSocket:
        async def send(self, _packet: str) -> None:
            return

        async def close(self) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    socket = FakeSocket()

    async def fake_connect(*_args, **_kwargs):
        return socket

    monkeypatch.setattr(voice_module, "connect", fake_connect)

    async def scenario() -> None:
        stream = make_stream()
        await stream.send_audio(bytes(stream.chunk_size) * 4)
        await stream.start()
        finishing = asyncio.create_task(stream.finish())
        await asyncio.sleep(0.005)
        finishing.cancel()
        await asyncio.gather(finishing, return_exceptions=True)
        await stream.close()
        assert stream._sender is None
        assert stream._receiver is None
        assert stream._send_queue.empty()

    asyncio.run(scenario())
