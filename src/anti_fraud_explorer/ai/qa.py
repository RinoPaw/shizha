"""Question answering over the anti-fraud dataset."""

import logging
import re
import textwrap
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..domain.dataset import CaseItem, KnowledgeBase, item_to_dict
from ..service.search import normalize_search_query, search_items
from ..text import normalize_text


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    answer: str
    mode: str
    sources: list[dict[str, Any]]
    speech: str = ""


def answer_question(
    kb: KnowledgeBase, question: str, category: str = "", include_speech: bool = True
) -> Answer:
    from ..ai.client import call_chat_model, describe_model_error
    from ..ai.spoken import build_spoken_answer

    question = normalize_text(question)
    category = normalize_text(category)
    if not question:
        answer = "请先输入问题。"
        return Answer(
            answer=answer, mode="empty", sources=[], speech=answer if include_speech else ""
        )

    sources = fact_question_sources(kb, question=question, category=category, limit=5)
    if not sources:
        answer = "没有在数据集中找到足够相关的资料。"
        return Answer(
            answer=answer, mode="no_context", sources=[], speech=answer if include_speech else ""
        )

    if settings.ai_api_key:
        try:
            answer = call_chat_model(question, sources)
            return Answer(
                answer=answer,
                mode="llm",
                sources=[source_payload(item) for item in sources],
                speech=build_spoken_answer(answer, question=question, sources=sources)
                if include_speech
                else "",
            )
        except Exception as exc:  # noqa: BLE001 - API failures should gracefully fall back.
            LOGGER.warning("Chat model unavailable: %s", describe_model_error(exc))
            fallback = build_local_answer(question, sources)
            fallback += "\n\n模型服务暂时不可用，已为你切换成本地依据式回答。"
            return Answer(
                answer=fallback,
                mode="fallback",
                sources=[source_payload(item) for item in sources],
                speech=build_spoken_answer(fallback, question=question, sources=sources)
                if include_speech
                else "",
            )

    answer = build_local_answer(question, sources)
    return Answer(
        answer=answer,
        mode="local",
        sources=[source_payload(item) for item in sources],
        speech=build_spoken_answer(answer, question=question, sources=sources)
        if include_speech
        else "",
    )


def fact_question_sources(
    kb: KnowledgeBase,
    question: str,
    category: str = "",
    limit: int = 5,
) -> list[CaseItem]:
    """Choose grounded sources for factual answers.

    Direct item questions like "刷单返利是什么" should stay anchored to the
    exact item rather than using semantic search to fill the source list
    with loosely related items.
    """
    search_query = normalize_search_query(question)
    direct_matches = direct_item_matches(kb, search_query, category=category)
    if direct_matches:
        return direct_matches[:limit]

    sources, _ = search_items(kb, query=search_query or question, category=category, limit=limit)
    return sources


def direct_item_matches(
    kb: KnowledgeBase,
    search_query: str,
    category: str = "",
) -> list[CaseItem]:
    search_query = normalize_text(search_query)
    category = normalize_text(category)
    if len(search_query) < 2:
        return []

    exact_title: list[CaseItem] = []
    exact_category: list[CaseItem] = []
    title_contains: list[CaseItem] = []
    category_contains: list[CaseItem] = []
    for item in kb.items:
        if category and item.ccl2023_category != category:
            continue
        title = normalize_text(item.title)
        item_category = normalize_text(item.ccl2023_category)

        if search_query == title:
            exact_title.append(item)
        elif title and (search_query in title or title in search_query):
            title_contains.append(item)
        elif item_category and search_query == item_category:
            exact_category.append(item)
        elif item_category and (search_query in item_category or item_category in search_query):
            category_contains.append(item)

    return _dedupe_items(exact_title) or _dedupe_items(
        title_contains + exact_category + category_contains
    )


def _dedupe_items(items: list[CaseItem]) -> list[CaseItem]:
    seen: set[str] = set()
    result: list[CaseItem] = []
    for item in items:
        key = "|".join(
            normalize_text(part)
            for part in (
                item.title,
                item.ccl2023_category,
                item.custom_subcategory,
            )
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_local_answer(question: str, sources: list[CaseItem]) -> str:
    from ..ai.context import item_context_text

    lead = f"根据数据集中与“{question}”最相关的资料，可以先这样理解："
    bullets = []
    for item in sources[:3]:
        text = item_context_text(item) or item.summary or item.content
        snippet = summarize_snippet(text)
        bullets.append(f"- {item.title}（{item.ccl2023_category}）：{snippet}")
    return "\n".join([lead, *bullets])


def summarize_snippet(text: str, max_chars: int = 180) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(案例标题|场景简述|关键手法|风险信号|防范建议)[:：]\\s*", "", text)
    if not text:
        return "暂无摘要。"
    sentences = [part.strip() for part in text.replace("；", "。").split("。") if part.strip()]
    snippet = "。".join(sentences[:2]) if sentences else text
    snippet = textwrap.shorten(snippet, width=max_chars, placeholder="...")
    return snippet.rstrip("。") + "。"


def source_payload(item: CaseItem) -> dict[str, Any]:
    data = item_to_dict(item)
    data["excerpt"] = summarize_snippet(item.content, max_chars=120)
    return data
