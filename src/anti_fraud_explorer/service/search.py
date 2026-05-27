"""Small dependency-free lexical search for the normalized dataset."""

import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from ..config import settings
from ..domain.dataset import CaseItem, KnowledgeBase
from ..text import normalize_text


LOGGER = logging.getLogger(__name__)
HYBRID_LEXICAL_CANDIDATES = 80
HYBRID_SEMANTIC_CANDIDATES = 80
RRF_K = 60
LEXICAL_RANK_WEIGHT = 1.3
SEMANTIC_RANK_WEIGHT = 1.35
LEXICAL_MIN_SCORE = 10  # minimum score for an item to count as a lexical match
HYBRID_MIN_SCORE = 0.015  # RRF-based scores are small; this keeps weak tail matches out.

# Lexical scoring weights
TITLE_EXACT_SCORE = 100
TITLE_SUBSTRING_SCORE = 40
CATEGORY_SUBSTRING_SCORE = 30
SUMMARY_SUBSTRING_SCORE = 10
CONTENT_SUBSTRING_SCORE = 12
TOKEN_TITLE_SCORE = 12
TOKEN_CATEGORY_SCORE = 10
TOKEN_SUMMARY_SCORE = 3
TOKEN_SEARCH_TEXT_SCORE = 1

# Hybrid strong-match bonuses
STRONG_TITLE_EXACT_BONUS = 0.7
STRONG_TITLE_SUBSTRING_BONUS = 0.35
STRONG_CATEGORY_EXACT_BONUS = 0.4
STRONG_CATEGORY_SUBSTRING_BONUS = 0.15
STRONG_TOKEN_TITLE_EXACT_BONUS = 0.06
STRONG_TOKEN_TITLE_SUBSTRING_BONUS = 0.004
STRONG_TOKEN_CATEGORY_EXACT_BONUS = 0.10
STRONG_TOKEN_CATEGORY_SUBSTRING_BONUS = 0.005

# Pinyin fuzzy search constants
_PINYIN_MIN_QUERY_LEN = 2  # minimum query chars to try pinyin matching
_SEARCH_TRAILING_PUNCTUATION = "？?！!。.，,、 \t\r\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_items(
    kb: KnowledgeBase,
    query: str = "",
    category: str = "",
    risk_level: str = "",
    entry_channel: str = "",
    limit: int = 30,
    offset: int = 0,
) -> tuple[list[CaseItem], int]:
    """Search or browse cases.

    Empty query means browse mode:
    - apply explicit filters such as category/risk_level/entry_channel
    - return all matching items
    - do not run lexical ranking, pinyin fallback, or embedding retrieval

    Non-empty query means retrieval mode:
    - run lexical ranking
    - optionally run embedding fusion
    - apply pinyin fallback only when lexical title matching is weak
    """
    query = normalize_search_query(query)
    category = normalize_text(category)
    risk_level = normalize_text(risk_level)
    entry_channel = normalize_text(entry_channel)

    candidates: Iterable[CaseItem] = kb.items

    if category:
        candidates = (item for item in candidates if item.ccl2023_category == category)
    if risk_level:
        candidates = (item for item in candidates if item.risk_level == risk_level)
    if entry_channel:
        candidates = (item for item in candidates if entry_channel in item.entry_channels)

    candidates = list(candidates)

    # Browse/filter mode: clicking category/filter chips should not trigger
    # lexical ranking, pinyin fallback, or embedding retrieval.
    if not query:
        result = sorted(candidates, key=lambda item: (item.ccl2023_category, item.title))
        return result[offset : offset + limit], len(result)

    tokens = tokenize(query)

    # Keep lexical ranking separately. Even when hybrid is enabled, lexical score
    # is still used to decide whether pinyin fallback is needed.
    lexical_ranked = rank_lexical(candidates, query, tokens)
    ranked = lexical_ranked

    using_hybrid = False
    if settings.search_use_embedding:
        try:
            ranked = rank_hybrid(kb, candidates, query, tokens)
            using_hybrid = True
        except Exception:  # noqa: BLE001 - semantic retrieval should degrade to lexical search.
            ranked = lexical_ranked

    min_score = HYBRID_MIN_SCORE if using_hybrid else LEXICAL_MIN_SCORE
    result = [item for score, item in ranked if score >= min_score]

    # Pinyin is only a fallback. If lexical search already has a strong title
    # substring match, do not let pinyin results jump ahead.
    lexical_top_score = lexical_ranked[0][0] if lexical_ranked else 0
    if len(query) >= _PINYIN_MIN_QUERY_LEN and lexical_top_score < TITLE_SUBSTRING_SCORE:
        result = prepend_pinyin_matches(kb, result, query, candidates)

    return result[offset : offset + limit], len(result)


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------


def normalize_search_query(query: str) -> str:
    """Normalize a raw search string without trying to interpret user intent."""
    text = normalize_text(query).lower().strip(_SEARCH_TRAILING_PUNCTUATION)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip(_SEARCH_TRAILING_PUNCTUATION)


def tokenize(query: str) -> list[str]:
    if not query:
        return []
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
    if len(tokens) == 1:
        text = tokens[0]
        if len(text) > 2:
            tokens.extend(text[i : i + 2] for i in range(len(text) - 1))
    return list(dict.fromkeys(tokens))


# ---------------------------------------------------------------------------
# Lexical ranking
# ---------------------------------------------------------------------------


def rank_lexical(
    candidates: Iterable[CaseItem],
    lowered_query: str,
    tokens: list[str],
) -> list[tuple[float, CaseItem]]:
    ranked: list[tuple[float, CaseItem]] = []

    for item in candidates:
        score = score_item(item, lowered_query, tokens)
        if score > 0:
            ranked.append((score, item))

    ranked.sort(key=lambda pair: (-pair[0], pair[1].title))
    return ranked


def score_item(item: CaseItem, query: str, tokens: list[str]) -> float:
    title = item.title.lower()
    category_text = item.ccl2023_category.lower()
    summary = item.summary.lower()
    content = item.content.lower()
    search_text = build_search_text(item).lower()

    score = 0.0
    if query == title:
        score += TITLE_EXACT_SCORE
    if query and query in title:
        score += TITLE_SUBSTRING_SCORE
    if query and query in category_text:
        score += CATEGORY_SUBSTRING_SCORE
    if query and query in summary:
        score += SUMMARY_SUBSTRING_SCORE
    if query and query in content:
        score += CONTENT_SUBSTRING_SCORE

    for token in tokens:
        if token in title:
            score += TOKEN_TITLE_SCORE
        if token in category_text:
            score += TOKEN_CATEGORY_SCORE
        if token in summary:
            score += TOKEN_SUMMARY_SCORE
        if token in search_text:
            score += TOKEN_SEARCH_TEXT_SCORE

    return score


# ---------------------------------------------------------------------------
# Search document building
# ---------------------------------------------------------------------------


def build_search_text(item: CaseItem | dict[str, Any]) -> str:
    """Build the lightweight lexical-search text for an item."""
    parts = [
        _item_text_value(item, "title"),
        _item_text_value(item, "ccl2023_category"),
        _item_text_value(item, "custom_subcategory"),
    ]
    parts.extend(_item_text_list(item, "tags"))
    parts.extend(_item_text_list(item, "key_methods"))
    parts.extend(_item_text_list(item, "entry_channels"))
    parts.extend(_item_text_list(item, "impersonated_identity"))
    parts.extend(_item_text_list(item, "official_category"))
    victim_group = _item_text_value(item, "victim_group")
    if victim_group:
        parts.append(victim_group)
    return " ".join(part for part in parts if part)


def _item_text_value(item: CaseItem | dict[str, Any], key: str) -> str:
    value = item.get(key, "") if isinstance(item, dict) else getattr(item, key, "")
    return normalize_text(str(value or ""))


def _item_text_list(item: CaseItem | dict[str, Any], key: str) -> list[str]:
    value = item.get(key, ()) if isinstance(item, dict) else getattr(item, key, ())
    if not isinstance(value, (list, tuple)):
        return []
    return [part for raw in value if (part := normalize_text(str(raw)))]


# ---------------------------------------------------------------------------
# Pinyin fuzzy fallback
# ---------------------------------------------------------------------------


def prepend_pinyin_matches(
    kb: KnowledgeBase,
    ranked_items: list[CaseItem],
    query: str,
    candidates: list[CaseItem],
) -> list[CaseItem]:
    """Prepend deduplicated pinyin matches within the current candidate set."""
    candidate_ids = {item.id for item in candidates}
    pinyin_results = [item for item in search_items_pinyin(kb, query) if item.id in candidate_ids]
    if not pinyin_results:
        return ranked_items

    pinyin_ids = {item.id for item in pinyin_results}
    return pinyin_results + [item for item in ranked_items if item.id not in pinyin_ids]


def search_items_pinyin(
    kb: KnowledgeBase,
    query: str,
) -> list[CaseItem]:
    """Try pinyin matching as a fallback for homophones, pinyin input, or IME mistakes."""
    if not query or len(query) < _PINYIN_MIN_QUERY_LEN:
        return []

    index = _build_pinyin_index(_pinyin_index_rows(kb))
    if not index:
        return []

    try:
        from pypinyin import lazy_pinyin  # noqa: PLC0415 - optional dependency

        query_py = "".join(lazy_pinyin(query))
    except ImportError:
        return []

    matched_ids: list[str] = []

    if query_py in index:
        matched_ids.extend(index[query_py])

    for py, ids in index.items():
        if py == query_py:
            continue
        if query_py in py or _is_substantial_pinyin_part(py, query_py):
            matched_ids.extend(ids)

    seen: set[str] = set()
    result: list[CaseItem] = []
    for item_id in matched_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        item = kb.get(item_id)
        if item is not None:
            result.append(item)

    return result


@lru_cache(maxsize=1)
def _build_pinyin_index(rows: tuple[tuple[str, str, str], ...]) -> dict[str, list[str]]:
    """Build a title/category pinyin index for fuzzy Chinese retrieval fallback."""
    try:
        from pypinyin import lazy_pinyin  # noqa: PLC0415 - optional dependency

        index: dict[str, list[str]] = {}
        for item_id, title, category in rows:
            texts = [title]
            if category:
                texts.append(category)
            for text in texts:
                py = "".join(lazy_pinyin(text))
                py_compact = py.replace(" ", "")
                if py_compact:
                    index.setdefault(py_compact, []).append(item_id)
        LOGGER.info("Pinyin index built: %d entries", len(index))
        return index
    except ImportError:
        LOGGER.debug("pypinyin not installed, pinyin fuzzy search disabled")
        return {}


def _pinyin_index_rows(kb: KnowledgeBase) -> tuple[tuple[str, str, str], ...]:
    """Flatten the minimal item fields needed to build the cached pinyin index."""
    return tuple(
        (
            item.id,
            item.title,
            item.ccl2023_category or "",
        )
        for item in kb.items
    )


def _is_substantial_pinyin_part(candidate_py: str, query_py: str) -> bool:
    """Allow contained pinyin only when it covers a real chunk of the query."""
    if candidate_py not in query_py:
        return False
    if len(candidate_py) < 6:
        return False
    return len(candidate_py) >= len(query_py) * 0.45


# ---------------------------------------------------------------------------
# Optional hybrid semantic ranking
# ---------------------------------------------------------------------------


def rank_hybrid(
    kb: KnowledgeBase,
    candidates: list[CaseItem],
    lowered_query: str,
    tokens: list[str],
) -> list[tuple[float, CaseItem]]:
    from .embeddings import embedding_scores

    semantic_scores = embedding_scores(kb, lowered_query, candidates, min_score=0.0)
    lexical_ranked = rank_lexical(candidates, lowered_query, tokens)
    lexical_scores = {item.id: score for score, item in lexical_ranked}
    items_by_id = {item.id: item for item in candidates}
    candidate_ids: set[str] = set()
    rank_scores: dict[str, float] = {}

    add_rank_signal(
        rank_scores,
        lexical_ranked[:HYBRID_LEXICAL_CANDIDATES],
        LEXICAL_RANK_WEIGHT,
    )
    candidate_ids.update(item.id for _, item in lexical_ranked[:HYBRID_LEXICAL_CANDIDATES])

    semantic_ranked = [
        (score, item)
        for item_id, score in semantic_scores.items()
        if (item := items_by_id.get(item_id)) is not None
    ]
    semantic_ranked.sort(key=lambda pair: (-pair[0], pair[1].title))
    add_rank_signal(
        rank_scores,
        semantic_ranked[:HYBRID_SEMANTIC_CANDIDATES],
        SEMANTIC_RANK_WEIGHT,
    )
    candidate_ids.update(item.id for _, item in semantic_ranked[:HYBRID_SEMANTIC_CANDIDATES])

    for item in candidates:
        if strong_match_bonus(item, lowered_query, tokens) > 0:
            candidate_ids.add(item.id)

    ranked: list[tuple[float, CaseItem]] = []
    for item_id in candidate_ids:
        item = items_by_id[item_id]
        score = rank_scores.get(item_id, 0.0)
        score += strong_match_bonus(item, lowered_query, tokens)
        score += lexical_tiebreak(lexical_scores.get(item_id, 0.0))
        ranked.append((score, item))

    ranked.sort(key=lambda pair: (-pair[0], pair[1].title))
    return ranked


def add_rank_signal(
    rank_scores: dict[str, float],
    ranked: list[tuple[float, CaseItem]],
    weight: float,
) -> None:
    for rank, (_, item) in enumerate(ranked, start=1):
        rank_scores[item.id] = rank_scores.get(item.id, 0.0) + weight / (RRF_K + rank)


def strong_match_bonus(item: CaseItem, query: str, tokens: list[str]) -> float:
    if not query:
        return 0.0

    title = item.title.lower()
    category_text = item.ccl2023_category.lower()
    bonus = 0.0

    if query == title:
        bonus += STRONG_TITLE_EXACT_BONUS
    elif query in title:
        bonus += STRONG_TITLE_SUBSTRING_BONUS

    if query == category_text:
        bonus += STRONG_CATEGORY_EXACT_BONUS
    elif category_text and query in category_text:
        bonus += STRONG_CATEGORY_SUBSTRING_BONUS

    for token in tokens:
        if not token:
            continue
        if token == title:
            bonus += STRONG_TOKEN_TITLE_EXACT_BONUS
        elif token in title:
            bonus += STRONG_TOKEN_TITLE_SUBSTRING_BONUS
        if token == category_text:
            bonus += STRONG_TOKEN_CATEGORY_EXACT_BONUS
        elif category_text and token in category_text:
            bonus += STRONG_TOKEN_CATEGORY_SUBSTRING_BONUS

    return bonus


def lexical_tiebreak(score: float) -> float:
    if score <= 0:
        return 0.0
    return min(score, 100.0) / 10000.0
