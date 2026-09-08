"""High-confidence, conservative ASR correction for anti-fraud vocabulary."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from weakref import WeakKeyDictionary

try:
    from pypinyin import lazy_pinyin
except ImportError:  # ASR correction remains usable without optional pinyin support.
    def lazy_pinyin(value: str, errors: str = "default") -> list[str]:
        return list(value)


@dataclass(frozen=True)
class NormalizedSpan:
    start: int
    end: int
    raw: str
    canonical: str
    score: float
    reason: str


@dataclass(frozen=True)
class NormalizedTranscript:
    raw_text: str
    canonical_text: str
    spans: tuple[NormalizedSpan, ...]


@dataclass(frozen=True)
class _Entry:
    canonical: str
    category: str
    forms: tuple[str, ...]
    pinyin: tuple[str, ...]
    kind: str = "term"


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    entry: _Entry
    score: float
    reason: str


_CHINESE_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_COMMON_TERMS = (
    "验证码", "共享屏幕", "远程控制", "安全账户", "刷单返利", "杀猪盘", "冒充客服",
    "冒充公检法", "网络投资理财", "虚假购物", "贷款诈骗", "游戏交易", "诱导转账",
    "陌生链接", "陌生App", "止付", "冻结", "96110",
)


def _items_from_kb(kb: Any) -> Iterable[Any]:
    if kb is None:
        return ()
    return kb.get("items", ()) if isinstance(kb, dict) else getattr(kb, "items", ())


def _value(obj: Any, name: str, default: Any = "") -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _texts(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    try:
        return tuple(str(v).strip() for v in value if str(v).strip())
    except TypeError:
        return (str(value).strip(),) if str(value).strip() else ()


def _py(value: str) -> tuple[str, ...]:
    return tuple(lazy_pinyin(value, errors="default"))


def _char_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    positional = sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))
    return (positional + SequenceMatcher(None, left, right).ratio()) / 2


def _entry(canonical: str, category: str, forms: Iterable[str], kind: str) -> _Entry:
    clean = tuple(dict.fromkeys(form.strip() for form in forms if form and form.strip()))
    return _Entry(canonical.strip(), category.strip(), clean, _py(canonical), kind)


class _Index:
    def __init__(self, kb: Any) -> None:
        entries: dict[tuple[str, str], _Entry] = {}
        for item in _items_from_kb(kb):
            title = str(_value(item, "title", "") or "").strip()
            category = str(_value(item, "ccl2023_category", _value(item, "category", "")) or "").strip()
            subcategory = str(_value(item, "custom_subcategory", "") or "").strip()
            aliases = _texts(_value(item, "aliases", ())) + _texts(_value(item, "alternate_names", ()))
            if title:
                entries[(title, "title")] = _entry(title, category, (title, *aliases), "title")
            if category:
                entries[(category, "category")] = _entry(category, category, (category,), "category")
            for channel in _texts(_value(item, "entry_channels", ())):
                if len(channel) >= 2:
                    entries[(channel, "channel")] = _entry(channel, category, (channel,), "channel")
            if subcategory:
                entries[(subcategory, "subcategory")] = _entry(subcategory, category, (subcategory,), "subcategory")
        for term in _COMMON_TERMS:
            entries[(term, "common")] = _entry(term, "", (term,), "common")
        self.entries = tuple(entries.values())
        self.lengths = frozenset(len(form) for entry in self.entries for form in entry.forms if _CHINESE_RUN.fullmatch(form))
        self.max_length = max(self.lengths, default=0)
        self.by_py: dict[tuple[str, ...], list[_Entry]] = {}
        self.by_first: dict[tuple[int, str], list[_Entry]] = {}
        for entry in self.entries:
            for form in entry.forms:
                if not _CHINESE_RUN.fullmatch(form):
                    continue
                self.by_py.setdefault(_py(form), []).append(entry)
                self.by_first.setdefault((len(form), form[0]), []).append(entry)
        self.exact_by_length: dict[int, frozenset[str]] = {}
        exact: dict[int, set[str]] = {}
        for entry in self.entries:
            for form in entry.forms:
                if _CHINESE_RUN.fullmatch(form):
                    exact.setdefault(len(form), set()).add(form)
        self.exact_by_length = {length: frozenset(values) for length, values in exact.items()}

    def candidates(self, text: str) -> tuple[_Entry, ...]:
        if len(text) not in self.lengths:
            return ()
        found = {entry.canonical: entry for entry in self.by_py.get(_py(text), ())}
        if not found:
            found = {entry.canonical: entry for entry in self.by_first.get((len(text), text[0]), ())}
        return tuple(found.values())


_INDEX_CACHE: WeakKeyDictionary[Any, _Index] = WeakKeyDictionary()


def _get_index(kb: Any) -> _Index:
    if isinstance(kb, dict) or kb is None:
        return _Index(kb)
    try:
        index = _INDEX_CACHE.get(kb)
    except (TypeError, ValueError):
        index = None
    if index is None:
        index = _Index(kb)
        try:
            _INDEX_CACHE[kb] = index
        except (TypeError, ValueError):
            pass
    return index


def prepare_asr_normalization(kb: Any) -> None:
    _get_index(kb)


def _as_texts(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result = []
    for value in values or ():
        if isinstance(value, dict):
            value = value.get("text", value.get("title", ""))
        elif not isinstance(value, str):
            value = _value(value, "title", value)
        value = str(value or "").strip()
        if value:
            result.append(value)
    return tuple(result)


def normalize_asr_final(
    raw_text: str,
    *,
    kb: Any,
    category: str = "",
    recent_items: Iterable[Any] = (),
    asr_candidates: Iterable[Any] = (),
    language: str = "zh",
) -> NormalizedTranscript:
    text = str(raw_text or "")
    language_code = str(language or "").strip().casefold().replace("-", "_")
    if language_code not in {"zh", "cn", "zh_cn", "cn_cbm", "chinese"}:
        return NormalizedTranscript(text, text, ())
    index = _get_index(kb)
    if not text or not index.entries:
        return NormalizedTranscript(text, text, ())
    recent, nbest = _as_texts(recent_items), _as_texts(asr_candidates)
    exact_ranges = []
    for start in range(len(text)):
        for length, forms in index.exact_by_length.items():
            if text[start:start + length] in forms:
                exact_ranges.append((start, start + length))
    matches: list[_Match] = []
    category_clean = re.sub(r"\s+", "", str(category or "")).casefold()
    for run in _CHINESE_RUN.finditer(text):
        start, _end = run.span()
        run_text = run.group()
        for length in index.lengths:
            if length < 2 or length > len(run_text):
                continue
            for offset in range(len(run_text) - length + 1):
                span_start, span_end = start + offset, start + offset + length
                if any(span_start < right and span_end > left for left, right in exact_ranges):
                    continue
                raw = run_text[offset:offset + length]
                candidates = index.candidates(raw)
                ranked = []
                for entry in candidates:
                    best_form = max(entry.forms, key=lambda form: _char_similarity(raw, form) + SequenceMatcher(None, _py(raw), _py(form)).ratio())
                    char_score = _char_similarity(raw, best_form)
                    py_score = SequenceMatcher(None, _py(raw), _py(best_form)).ratio()
                    context = 0.0
                    reasons = []
                    if category_clean and category_clean == re.sub(r"\s+", "", entry.category).casefold():
                        context += 0.10; reasons.append("category")
                    if any(entry.canonical in value or any(form in value for form in entry.forms) for value in (*recent, *nbest)):
                        context += 0.12; reasons.append("asr-context")
                    base = 0.52 * py_score + 0.48 * char_score
                    ranked.append((min(1.0, base + context), entry, py_score, char_score, reasons))
                if not ranked:
                    continue
                ranked.sort(key=lambda row: row[0], reverse=True)
                best = ranked[0]
                second = ranked[1][0] if len(ranked) > 1 else 0.0
                if best[1].canonical == raw or best[0] < 0.88 or best[0] - second < 0.12 or (not best[4] and best[1].kind not in {"common", "category", "channel"}):
                    continue
                matches.append(_Match(span_start, span_end, best[1], best[0], "; ".join(best[4]) or "vocabulary"))
    matches.sort(key=lambda match: (match.start, -(match.end - match.start), -match.score))
    chosen: list[_Match] = []
    for match in matches:
        if any(match.start < old.end and match.end > old.start for old in chosen):
            continue
        chosen.append(match)
    chosen.sort(key=lambda match: match.start)
    if not chosen:
        return NormalizedTranscript(text, text, ())
    output, spans, cursor = [], [], 0
    for match in chosen:
        output.append(text[cursor:match.start])
        raw = text[match.start:match.end]
        output.append(match.entry.canonical)
        spans.append(NormalizedSpan(match.start, match.end, raw, match.entry.canonical, match.score, match.reason))
        cursor = match.end
    output.append(text[cursor:])
    return NormalizedTranscript(text, "".join(output), tuple(spans))


__all__ = ["NormalizedSpan", "NormalizedTranscript", "normalize_asr_final", "prepare_asr_normalization"]
