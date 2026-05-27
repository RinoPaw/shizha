"""Shared text normalization helpers for Python services."""

import re


EMOJI_RE = re.compile("[\U0001f1e6-\U0001f1ff\U0001f300-\U0001faff\u2600-\u27bf\u200d\ufe0f]+")


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def normalize_text(value: str) -> str:
    return collapse_whitespace(value)


def strip_emoji(value: str) -> str:
    return collapse_whitespace(EMOJI_RE.sub(" ", str(value or "")))
