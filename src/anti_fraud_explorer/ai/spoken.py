"""Spoken text pipeline — converts display answers into readable speech text."""

import logging
import re

from ..domain.dataset import CaseItem
from ..prompts import SPOKEN_SYSTEM_PROMPT
from ..text import normalize_text, strip_emoji

LOGGER = logging.getLogger(__name__)


def build_spoken_prompt(
    answer: str,
    question: str = "",
    sources: list[CaseItem] | None = None,
    max_chars: int = 1800,
) -> list[dict[str, str]]:
    source_titles = "、".join(item.title for item in (sources or [])) or "无"
    return [
        {"role": "system", "content": SPOKEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question or '未提供'}\n"
                f"相关资料标题：{source_titles}\n"
                f"展示版回答：\n{answer}\n\n"
                f"如果它无需修改，请直接输出原文；如果需要修改，请输出润色后的文本。"
                f"最终文本应尽量接近展示版回答的内容和长度；只有超过 {max_chars} 字时才适度压缩。"
            ),
        },
    ]


def build_spoken_answer(
    answer: str,
    question: str = "",
    sources: list[CaseItem] | None = None,
    max_chars: int = 1800,
) -> str:
    from ..ai.client import call_spoken_model

    try:
        spoken = call_spoken_model(answer, question=question, sources=sources, max_chars=max_chars)
        return _clean_spoken_text(spoken, max_chars=max_chars)
    except Exception:
        LOGGER.warning("Spoken rewrite failed, returning raw answer")
        return _clean_spoken_text(answer, max_chars=max_chars)


def _clean_spoken_text(text: str, max_chars: int = 1800) -> str:
    """Strip Markdown, symbols, and LLM prefix cruft from spoken model output."""
    text = _remove_symbols(str(text or ""))
    text = re.sub(r"^\s*(?:无需修改|需要修改|润色后|播报稿|最终播报文本)\s*[:：]\s*", "", text)

    # Markdown
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)、]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[>#*_~|]+", " ", text)

    text = normalize_text(text)
    text = re.sub(r"[，、；：]\s*([。！？])", r"\1", text)
    text = re.sub(r"。{2,}", "。", text).strip(" 。")
    if not text:
        return ""

    # Truncate at sentence boundary: rfind returns -1 when no punctuation found,
    # so we fall through to max_chars (the whole text).
    boundary = max(text.rfind(mark, 0, max_chars) for mark in "。！？")
    if boundary < max_chars // 2:
        boundary = max_chars
    text = text[: boundary + 1].rstrip("，、；： ")
    if text and text[-1] not in "。！？":
        text += "。"
    return text


def _remove_symbols(text: str) -> str:
    return strip_emoji(text)
