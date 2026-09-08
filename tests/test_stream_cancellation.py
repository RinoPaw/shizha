from __future__ import annotations

import asyncio

from anti_fraud_explorer.assistant import AssistantService
from anti_fraud_explorer.models import SearchResponse
from anti_fraud_explorer.sessions import SessionStore


class EmptySearch:
    def search(self, query: str, **kwargs: object) -> SearchResponse:
        return SearchResponse((), 0)


class FirstCallBlocks:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.first_closed = asyncio.Event()

    async def stream_chat(self, messages: object, **kwargs: object):
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                self.first_closed.set()
            return
        yield "第二个回答"


def test_superseding_turn_closes_blocked_provider_stream() -> None:
    async def scenario() -> None:
        llm = FirstCallBlocks()
        store = SessionStore()
        service = AssistantService(search=EmptySearch(), sessions=store, llm=llm)

        async def collect(question: str, turn_id: str):
            return [
                event
                async for event in service.stream_turn(
                    question,
                    session_id="session",
                    turn_id=turn_id,
                )
            ]

        first = asyncio.create_task(collect("第一个问题", "first"))
        await asyncio.wait_for(llm.first_started.wait(), timeout=1)
        second_events = await asyncio.wait_for(collect("第二个问题", "second"), timeout=1)
        first_events = await asyncio.wait_for(first, timeout=1)

        assert first_events[-1].type == "turn.cancelled"
        assert first_events[-1].payload["reason"] == "superseded"
        assert llm.first_closed.is_set()
        assert second_events[-1].type == "turn.completed"
        assert [turn.turn_id for turn in store.history("session")] == ["second"]

    asyncio.run(scenario())
