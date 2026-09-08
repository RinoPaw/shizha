"""Deterministic lexical/semantic retrieval for anti-fraud cases."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

from . import config
from .dataset import FraudCase, KnowledgeBase, normalize_text

LOGGER = logging.getLogger(__name__)
_PUNCTUATION = "？?！!。.，,、；;：:（）()[]【】{} \t\r\n"
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_FILLERS = (
    "请介绍", "介绍一下", "有哪些", "有那些", "有什么", "想了解", "值得了解",
    "我想知道", "讲讲", "说说", "搜索", "查找", "找一下", "推荐", "诈骗案例",
    "反诈案例", "案例资料", "案例", "资料", "一下", "的", "请问",
)


def normalize_search_query(query: str) -> str:
    text = normalize_text(query).casefold().strip(_PUNCTUATION)
    for filler in _FILLERS:
        text = text.replace(filler, " ")
    text = re.sub(r"(?:\d{1,2}|[一二两三四五六七八九十]+)\s*(?:个|项|种|类)", " ", text)
    return re.sub(r"\s+", " ", text).strip(_PUNCTUATION + " ")


def tokenize(query: str) -> list[str]:
    text = normalize_search_query(query)
    result: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if token not in result:
            result.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            result.extend(token[index:index + 2] for index in range(len(token) - 1))
    return list(dict.fromkeys(result))


@lru_cache(maxsize=8)
def _category_names(kb: KnowledgeBase) -> tuple[str, ...]:
    values = {normalize_text(item.ccl2023_category) for item in kb.items if item.ccl2023_category}
    values.update(normalize_text(row.name) for row in kb.categories if row.name)
    return tuple(sorted((value for value in values if value), key=lambda value: (-len(value), value)))


def _resolve_category(kb: KnowledgeBase, value: str) -> str:
    normalized = normalize_search_query(value)
    matches = [name for name in _category_names(kb) if name == normalized or name in normalized]
    return max(matches, key=len) if matches else ""


@lru_cache(maxsize=8192)
def _fields(item: FraudCase) -> dict[str, str]:
    join = lambda values: " ".join(normalize_text(v).casefold() for v in values if v)
    return {
        "title": normalize_text(item.title).casefold(),
        "category": normalize_text(item.ccl2023_category or item.custom_subcategory).casefold(),
        "subcategory": normalize_text(item.custom_subcategory).casefold(),
        "summary": normalize_text(item.summary).casefold(),
        "content": normalize_text(item.search_text).casefold(),
        "channel": join(item.entry_channels),
        "method": join(item.key_methods),
        "signal": normalize_text(item.risk_signals).casefold(),
        "victim": normalize_text(item.victim_group).casefold(),
        "platform": join(item.involved_platforms),
        "tag": join(item.tags),
    }


def _lexical_score(item: FraudCase, query: str, tokens: Sequence[str]) -> float:
    if not query:
        return 0.0
    fields = _fields(item)
    score = 0.0
    weights = {"title": 120.0, "category": 52.0, "subcategory": 42.0, "method": 32.0, "channel": 30.0, "victim": 28.0, "signal": 24.0, "summary": 18.0, "tag": 16.0, "content": 8.0}
    for name, weight in weights.items():
        value = fields[name]
        if query == value:
            score += weight
        elif query in value:
            score += weight * 0.62
    for token in tokens:
        if len(token) < 2 and len(tokens) > 1:
            continue
        for name, weight in weights.items():
            if token in fields[name]:
                score += weight * (0.28 if name == "content" else 0.42)
    return score


def _diversity_key(item: FraudCase) -> tuple[str, str, str]:
    return (
        normalize_text(item.custom_subcategory or item.title).casefold(),
        normalize_text(item.victim_group or "未知人群").casefold(),
        normalize_text(item.ccl2023_category or "未分类").casefold(),
    )


def _diversify(scored: Iterable[tuple[float, FraudCase]], prefix_size: int = 40) -> list[tuple[float, FraudCase]]:
    ordered = sorted(scored, key=lambda pair: (-pair[0], pair[1].title.casefold(), pair[1].case_id))
    prefix = ordered[:prefix_size]
    selected: list[tuple[float, FraudCase]] = []
    while prefix:
        if not selected:
            selected.append(prefix.pop(0))
            continue
        best = prefix[0][0]
        tolerance = max(4.0, abs(best) * 0.2)
        eligible = [pair for pair in prefix if pair[0] >= best - tolerance] or [prefix[0]]

        def preference(pair: tuple[float, FraudCase]) -> tuple[int, int, int, float, str]:
            key = _diversity_key(pair[1])
            counts = [sum(_diversity_key(item)[idx] == key[idx] for _, item in selected) for idx in range(3)]
            return (*counts, -pair[0], pair[1].title.casefold())

        picked = min(eligible, key=preference)
        prefix.remove(picked)
        selected.append(picked)
    return selected + ordered[prefix_size:]


def embedding_scores(kb: KnowledgeBase, query: str, candidates: Sequence[FraudCase], min_score: float = 0.0) -> dict[str, float]:
    from .embeddings import embedding_scores as score_embeddings
    return score_embeddings(kb, query, candidates, min_score=min_score)


def _rank(kb: KnowledgeBase, candidates: Sequence[FraudCase], query: str) -> list[FraudCase]:
    tokens = tokenize(query)
    lexical = {item.case_id: _lexical_score(item, query, tokens) for item in candidates}
    semantic_mode = False
    if config.SEARCH_USE_EMBEDDING and query and not any(score >= 8 for score in lexical.values()):
        try:
            semantic = embedding_scores(kb, query, candidates, min_score=config.EMBEDDING_MIN_SCORE)
        except Exception as exc:  # noqa: BLE001 - semantic failure must preserve lexical search
            LOGGER.info("retrieval.semantic.fallback reason=%s", type(exc).__name__)
            semantic = {}
        if semantic:
            semantic_mode = True
            maximum = max(semantic.values()) or 1.0
            lexical_max = max(lexical.values(), default=1.0) or 1.0
            for item in candidates:
                lexical[item.case_id] = 0.3 * lexical[item.case_id] / lexical_max + 0.7 * semantic.get(item.case_id, 0.0) / maximum
    threshold = 0.08 if semantic_mode else 8.0
    scored = [(lexical[item.case_id], item) for item in candidates if lexical[item.case_id] >= threshold]
    return [item for _score, item in _diversify(scored)]


def search_items(
    kb: KnowledgeBase,
    query: str = "",
    category: str = "",
    risk_level: str = "",
    entry_channel: str = "",
    victim_group: str = "",
    *,
    limit: int = 30,
    offset: int = 0,
    use_pinyin: bool = True,
) -> tuple[list[FraudCase], int]:
    del use_pinyin
    category = normalize_text(category)
    risk_level = normalize_text(risk_level)
    entry_channel = normalize_text(entry_channel)
    victim_group = normalize_text(victim_group)
    implicit = _resolve_category(kb, query) if not category else ""
    category = category or implicit
    candidates = [
        item for item in kb.items
        if (not category or normalize_text(item.ccl2023_category) == category or category in normalize_text(item.ccl2023_category))
        and (not risk_level or normalize_text(item.risk_level) == risk_level)
        and (not entry_channel or any(entry_channel.casefold() in value.casefold() for value in item.entry_channels))
        and (not victim_group or victim_group.casefold() in normalize_text(item.victim_group).casefold())
    ]
    combined = normalize_search_query(query)
    if implicit:
        combined = combined.replace(normalize_search_query(implicit), " ").strip()
    if combined:
        ranked = _rank(kb, candidates, combined)
    else:
        ranked = [item for _, item in _diversify((0.0, item) for item in candidates)]
    start, size = max(int(offset), 0), max(int(limit), 0)
    return ranked[start:start + size], len(ranked)


__all__ = ["embedding_scores", "normalize_search_query", "search_items", "tokenize"]
