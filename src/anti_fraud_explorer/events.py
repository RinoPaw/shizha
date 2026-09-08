"""Event helpers for assistant transports."""

from __future__ import annotations

from typing import Any

from .models import AssistantEvent


class EventSequence:
    """Monotonic per-turn sequence generator."""

    def __init__(self, session_id: str, turn_id: str) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self._seq = 0

    def make(self, event_type: str, **data: Any) -> AssistantEvent:
        event = AssistantEvent(
            type=event_type,
            session_id=self.session_id,
            turn_id=self.turn_id,
            seq=self._seq,
            payload=data,
        )
        self._seq += 1
        return event

__all__ = ["AssistantEvent", "EventSequence"]
