"""Small, transport-independent models used by the new assistant core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ConversationTurn:
    """A completed turn kept in a session's bounded context."""

    turn_id: str
    question: str
    answer: str
    source_ids: tuple[str, ...] = ()
    locale: str = "zh-CN"
    created_at: datetime = field(default_factory=utc_now)

    def context_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "question": self.question,
            "answer": self.answer,
            "source_ids": list(self.source_ids),
            "locale": self.locale,
        }


@dataclass(frozen=True)
class AssistantEvent:
    """The transport-neutral assistant event envelope.

    Event-specific values always live under ``payload``.  Keeping the
    envelope stable is important for SSE consumers: a new event type can be
    added without changing the shape of every existing event.
    """

    type: str
    session_id: str
    turn_id: str
    seq: int
    timestamp: datetime = field(default_factory=utc_now)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class SearchResponse:
    items: tuple[Any, ...]
    total: int


# Public model compatibility: transport models stay here while the complete
# source-backed case record is implemented in dataset.py.
from .dataset import FraudCase

__all__ = ["AssistantEvent", "ConversationTurn", "FraudCase", "SearchResponse", "utc_now"]
