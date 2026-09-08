"""Automatic conversation-language detection and speech profile selection.

The public APIs accept locale *hints*, never provider voice names.  The text
itself remains authoritative so one browser session can naturally switch
languages without adding a settings control to the UI.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEFAULT_LOCALE = "zh-CN"


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    response_instruction: str
    tts_voice: str
    asr_language: str
    is_dialect: bool = False


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "zh-CN": LanguageProfile(
        "zh-CN",
        "简体中文（普通话）",
        "使用自然、清晰的简体中文普通话回答。",
        "zh-CN-XiaoshuangNeural",
        "zh",
    ),
    "yue-CN": LanguageProfile(
        "yue-CN",
        "粤语",
        "用户正在使用粤语；请用自然、易懂的书面粤语回答，案例名称保持资料中的标准写法。",
        "zh-HK-HiuMaanNeural",
        "zh",
        True,
    ),
    "zh-CN-sichuan": LanguageProfile(
        "zh-CN-sichuan",
        "四川话",
        "用户正在使用四川话；请用自然、克制、易懂的四川话口吻回答，不要用生造的谐音字，项目专名保持标准写法。",
        "zh-CN-XiaoshuangNeural",
        "zh",
        True,
    ),
    "zh-CN-henan": LanguageProfile(
        "zh-CN-henan",
        "河南话",
        "用户正在使用河南话；请用自然、克制、易懂的河南口吻回答，不要堆砌或生造方言字，项目专名保持标准写法。",
        "zh-CN-XiaoshuangNeural",
        "zh",
        True,
    ),
    "en-US": LanguageProfile(
        "en-US",
        "English",
        "Reply in natural English. Keep each fraud case’s canonical Chinese title when it is first mentioned.",
        "en-US-JennyNeural",
        "en",
    ),
    "ja-JP": LanguageProfile(
        "ja-JP",
        "日本語",
        "自然で分かりやすい日本語で答えてください。詐欺事例の固有名は、初出時に中国語の正式名称も残してください。",
        "ja-JP-NanamiNeural",
        "ja",
    ),
    "ko-KR": LanguageProfile(
        "ko-KR",
        "한국어",
        "자연스럽고 이해하기 쉬운 한국어로 답하세요. 사기 사례의 고유명은 처음 언급할 때 중국어 정식 명칭도 함께 남기세요.",
        "ko-KR-SunHiNeural",
        "ko",
    ),
}


_LOCALE_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-sg": "zh-CN",
    "cmn": "zh-CN",
    "yue": "yue-CN",
    "yue-cn": "yue-CN",
    "yue-hk": "yue-CN",
    "zh-hk": "yue-CN",
    "zh-cn-sichuan": "zh-CN-sichuan",
    "zh-sichuan": "zh-CN-sichuan",
    "sichuan": "zh-CN-sichuan",
    "zh-cn-henan": "zh-CN-henan",
    "zh-henan": "zh-CN-henan",
    "henan": "zh-CN-henan",
    "en": "en-US",
    "ja": "ja-JP",
    "jp": "ja-JP",
    "ko": "ko-KR",
    "kr": "ko-KR",
}

_PROVIDER_LANGUAGE_LOCALES = {
    "zh": "zh-CN",
    "cn": "zh-CN",
    "zh_cn": "zh-CN",
    "cn_cbm": "zh-CN",
    "yue": "yue-CN",
    "yue_cn": "yue-CN",
    "yue_hk": "yue-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "jp": "ja-JP",
    "ja_jp": "ja-JP",
    "ko": "ko-KR",
    "kr": "ko-KR",
    "ko_kr": "ko-KR",
}

_CANTONESE_MARKERS = (
    "唔",
    "冇",
    "喺",
    "佢",
    "咩",
    "嘅",
    "啲",
    "嚟",
    "乜",
    "噉",
    "咗",
    "睇下",
    "点解",
)
_SICHUAN_MARKERS = (
    "啥子",
    "咋个",
    "巴适",
    "要得",
    "莫得",
    "摆龙门阵",
    "雄起",
    "安逸得很",
    "瓜娃子",
)

_HENAN_STRONG_MARKERS = (
    "河南话",
    "豫语",
    "中不中",
    "弄啥嘞",
    "弄啥咧",
    "可得劲",
    "喷空",
    "木牛",
    "恁好",
)
_HENAN_WEAK_MARKERS = (
    "恁",
    "俺",
    "得劲",
    "咋恁",
    "啥嘞",
    "咋弄",
)

_LATIN_LANGUAGE_WORDS: dict[str, frozenset[str]] = {
    "en-US": frozenset(
        {
            "hello",
            "thanks",
            "english",
            "heritage",
            "history",
            "what",
            "which",
            "tell",
            "about",
            "the",
            "and",
            "is",
            "are",
        }
    ),
}

_SCRIPT_PATTERNS = (
    (re.compile(r"[\u3040-\u30ff]"), "ja-JP"),
    (re.compile(r"[\uac00-\ud7af]"), "ko-KR"),
)
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_LATIN_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def normalize_locale_hint(value: object) -> str | None:
    """Return a supported canonical locale for an untrusted BCP-47 hint."""

    raw = str(value or "").strip().replace("_", "-")
    if not raw or raw.casefold() == "auto":
        return None
    folded = raw.casefold()
    if folded in _LOCALE_ALIASES:
        return _LOCALE_ALIASES[folded]
    for code in LANGUAGE_PROFILES:
        if code.casefold() == folded:
            return code
    primary = folded.split("-", 1)[0]
    return _LOCALE_ALIASES.get(primary)


def _dialect_locale(text: str) -> str | None:
    cantonese_hits = sum(marker in text for marker in _CANTONESE_MARKERS)
    sichuan_hits = sum(marker in text for marker in _SICHUAN_MARKERS)
    henan_strong_hits = sum(marker in text for marker in _HENAN_STRONG_MARKERS)
    henan_weak_hits = sum(marker in text for marker in _HENAN_WEAK_MARKERS)
    henan_hits = (henan_strong_hits * 2) + (henan_weak_hits if henan_weak_hits >= 2 else 0)
    candidates = (
        (cantonese_hits, "yue-CN"),
        (sichuan_hits, "zh-CN-sichuan"),
        (henan_hits, "zh-CN-henan"),
    )
    score, locale = max(candidates, key=lambda item: item[0])
    return locale if score else None


def _latin_locale(text: str, hint: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = _LATIN_WORD.findall(normalized)
    scores = {
        locale: sum(word in vocabulary for word in words)
        for locale, vocabulary in _LATIN_LANGUAGE_WORDS.items()
    }
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score:
        tied = [locale for locale, score in scores.items() if score == best_score]
        if hint in tied:
            return hint
        return best
    if hint in {"en-US", "ja-JP", "ko-KR"}:
        return hint
    return "en-US"


def detect_locale(text: object, *, hint: object = None) -> str:
    """Detect a response/TTS locale, using the browser locale only as a tie-breaker."""

    content = str(text or "").strip()
    normalized_hint = normalize_locale_hint(hint)
    if not content:
        return normalized_hint or DEFAULT_LOCALE

    dialect = _dialect_locale(content)
    if dialect:
        return dialect
    for pattern, locale in _SCRIPT_PATTERNS:
        if pattern.search(content):
            return locale

    han_count = len(_HAN.findall(content))
    latin_count = len(_LATIN.findall(content))
    if latin_count and (not han_count or latin_count >= han_count * 2):
        return _latin_locale(content, normalized_hint)
    if han_count:
        # Han-only text is ambiguous between Chinese and Japanese.  Preserve
        # text authority for explicit dialect markers above, then let the
        # browser/provider hint break this otherwise undecidable tie.
        if normalized_hint in {"yue-CN", "zh-CN-sichuan", "zh-CN-henan", "ja-JP"}:
            return normalized_hint
        return "zh-CN"
    if latin_count:
        return _latin_locale(content, normalized_hint)
    return normalized_hint or DEFAULT_LOCALE


def get_language_profile(locale: object = None, *, text: object = "") -> LanguageProfile:
    code = (
        detect_locale(text, hint=locale)
        if text
        else (normalize_locale_hint(locale) or DEFAULT_LOCALE)
    )
    return LANGUAGE_PROFILES.get(code, LANGUAGE_PROFILES[DEFAULT_LOCALE])


def locale_from_provider_language(value: object) -> str | None:
    tag = str(value or "").strip().casefold().replace("-", "_")
    return _PROVIDER_LANGUAGE_LOCALES.get(tag)


def is_chinese_locale(locale: object) -> bool:
    return get_language_profile(locale).asr_language == "zh"


__all__ = [
    "DEFAULT_LOCALE",
    "LANGUAGE_PROFILES",
    "LanguageProfile",
    "detect_locale",
    "get_language_profile",
    "is_chinese_locale",
    "locale_from_provider_language",
    "normalize_locale_hint",
]
