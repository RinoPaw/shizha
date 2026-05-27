"""Model-calling functions for the anti-fraud AI."""

from ..config import settings
from ..domain.dataset import CaseItem
from ..prompts import QA_SYSTEM_PROMPT
from ..ai.context import build_context


def call_model_with_messages(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    """Send a chat completion via the unified http_client."""
    from ..service.http_client import chat_completion

    return chat_completion(messages, temperature, max_tokens)


def build_messages(question: str, sources: list[CaseItem]) -> list[dict[str, str]]:
    """Build QA chat messages (system + user) from question and sources."""
    context = build_context(sources, settings.ai_max_context_chars)
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
    ]


def call_chat_model(question: str, sources: list[CaseItem]) -> str:
    return call_model_with_messages(
        build_messages(question, sources),
        temperature=0.2,
        max_tokens=2000,
    )


def call_spoken_model(
    answer: str,
    question: str = "",
    sources: list[CaseItem] | None = None,
    max_chars: int = 1800,
) -> str:
    from ..ai.spoken import build_spoken_prompt

    return call_model_with_messages(
        build_spoken_prompt(answer, question=question, sources=sources or [], max_chars=max_chars),
        temperature=0.1,
        max_tokens=max(900, min(2400, max_chars + 300)),
    )


def describe_model_error(exc: Exception) -> str:
    msg = str(exc)
    if not msg:
        return type(exc).__name__
    return f"{type(exc).__name__}: {msg[:180]}"
