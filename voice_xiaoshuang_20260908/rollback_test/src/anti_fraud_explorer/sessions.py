"""Bounded, expiring in-memory sessions.

The store is intentionally transport agnostic.  Replacing it with Redis later
only requires implementing the same small interface.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock

from .models import ConversationTurn


@dataclass
class Session:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    active_turns: dict[str, asyncio.Event] = field(default_factory=dict)
    cancel_reasons: dict[str, str] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_seen = time.monotonic()


class SessionStore:
    """Thread-safe session store with TTL, LRU capacity and bounded history."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 1800,
        max_sessions: int = 1000,
        max_turns: int = 20,
    ) -> None:
        if ttl_seconds <= 0 or max_sessions <= 0 or max_turns <= 0:
            raise ValueError("session limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create_id(self) -> str:
        return uuid.uuid4().hex

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    def get_or_create(self, session_id: str | None = None) -> Session:
        with self._lock:
            self._purge_locked()
            session = self._get_or_create_locked(session_id)
            session.touch()
            self._evict_locked()
            return session

    def history(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.touch()
            # Copy while holding the lock so an append cannot race a slice or
            # expose the mutable internal list to a caller.
            return list(session.turns)

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self._purge_locked()
            session = self._get_or_create_locked(session_id)
            session.turns.append(turn)
            del session.turns[:-self.max_turns]
            session.touch()
            self._evict_locked()

    def begin_turn(self, session_id: str, turn_id: str | None = None) -> tuple[Session, str, asyncio.Event]:
        with self._lock:
            self._purge_locked()
            session = self._get_or_create_locked(session_id)
            turn_id = (turn_id or uuid.uuid4().hex).strip() or uuid.uuid4().hex
            # A session has one conversational foreground turn.  Setting all
            # prior events makes barge-in deterministic even when their
            # generators are suspended in retrieval or provider I/O.
            for old_turn_id, old_event in session.active_turns.items():
                old_event.set()
                session.cancel_reasons[old_turn_id] = "superseded"
            cancel_event = asyncio.Event()
            session.active_turns[turn_id] = cancel_event
            session.cancel_reasons.pop(turn_id, None)
            session.touch()
            self._evict_locked()
        return session, turn_id, cancel_event

    def finish_turn(
        self,
        session_id: str,
        turn_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # An old generator must not finish a replacement turn that
                # reused the same client-supplied turn_id.
                current = session.active_turns.get(turn_id)
                if cancel_event is None or current is cancel_event:
                    session.active_turns.pop(turn_id, None)
                    session.cancel_reasons.pop(turn_id, None)
                session.touch()
                self._evict_locked()

    def cancel_turn(self, session_id: str, turn_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            event = session.active_turns.get(turn_id) if session else None
            if event is None:
                return False
            event.set()
            session.cancel_reasons[turn_id] = "client_cancelled"
            session.touch()
            return True

    def cancel_reason(self, session_id: str, turn_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.cancel_reasons.get(turn_id)

    def size(self) -> int:
        with self._lock:
            self._purge_locked()
            self._evict_locked()
            return len(self._sessions)

    def _purge_locked(self) -> None:
        now = time.monotonic()
        expired = [
            key for key, session in self._sessions.items()
            if now - session.last_seen >= self.ttl_seconds and not session.active_turns
        ]
        for key in expired:
            self._sessions.pop(key, None)

    def _evict_locked(self) -> None:
        while len(self._sessions) > self.max_sessions:
            idle = [session for session in self._sessions.values() if not session.active_turns]
            # Active sessions are never evicted.  Capacity may temporarily be
            # exceeded while all sessions are serving a turn.
            if not idle:
                return
            oldest = min(idle, key=lambda session: session.last_seen)
            self._sessions.pop(oldest.session_id, None)

    def _get_or_create_locked(self, session_id: str | None) -> Session:
        requested = (session_id or "").strip()
        session = self._sessions.get(requested) if requested else None
        if session is None:
            requested = requested or self.create_id()
            session = Session(requested)
            self._sessions[requested] = session
        return session


__all__ = ["Session", "SessionStore"]
