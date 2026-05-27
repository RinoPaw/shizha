"""Formatting helpers for LLM contexts, candidate summaries, and keyword extraction."""

from typing import Any

from ..prompts import FRAUD_LABEL_MAP
from ..service.item_cards import title_with_family
from ..text import normalize_text


def format_context_item_for_llm(item: Any) -> str:
    """Format a single context item for the LLM prompt."""
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title") or "").strip()
    item_id = str(item.get("id") or "").strip()
    if not title and not item_id:
        return ""

    meta = " | ".join(
        part
        for part in [
            str(item.get("ccl2023_category") or "").strip(),
            str(item.get("custom_subcategory") or "").strip(),
            str(item.get("risk_level") or "").strip(),
            str(item.get("victim_group") or "").strip(),
        ]
        if part
    )
    label = f"- [{item_id}] {title}" if item_id else f"- {title}"
    lines = [f"{label} | {meta}" if meta else label]

    for key in ("summary", "source_name", "prevention_advice", "content"):
        value = normalize_text(item.get(key) or "")
        if value:
            label_name = FRAUD_LABEL_MAP.get(key, key)
            lines.append(f"  {label_name}：{value[:240]}")

    forms = item.get("entry_channels")
    if isinstance(forms, list) and forms:
        label_name = FRAUD_LABEL_MAP.get("entry_channels", "入口渠道")
        lines.append(f"  {label_name}：{'、'.join(str(form) for form in forms[:6])}")

    return "\n".join(lines)


def items_to_llm_context(items: list[Any], total: int) -> str:
    """Format search results as compact context for the answer LLM."""
    lines = [f"从资料库中检索到 {total} 条相关反诈案例，以下是其中最相关的：\n"]
    for i, item in enumerate(items[:30], 1):
        forms = "、".join(item.entry_channels) if item.entry_channels else ""
        lines.append(
            f"{i}. [{item.id}] {title_with_family(item)}\n"
            f"   类别：{item.ccl2023_category} | 风险等级：{item.risk_level}\n"
            f"   简介：{item.summary[:200]}"
        )
        if forms:
            lines.append(f"   入口渠道：{forms}")
        content_snippet = item.content[:300].replace("\n", " ")
        if content_snippet:
            lines.append(f"   正文：{content_snippet}")
        lines.append("")
    return "\n".join(lines)


def items_to_title_context(items: list[Any], total: int) -> str:
    """Format broad first-round candidates as title-only planning context."""
    lines = [f"第 1 轮候选标题共 {total} 项，以下为标题和基础元数据：\n"]
    for i, item in enumerate(items[:100], 1):  # INITIAL_TITLE_CONTEXT_LIMIT
        forms = "、".join(item.entry_channels[:4]) if item.entry_channels else ""
        meta = " | ".join(
            part
            for part in [item.ccl2023_category, item.risk_level, item.custom_subcategory]
            if part
        )
        extra = "；".join(part for part in [f"入口渠道：{forms}" if forms else ""] if part)
        suffix = f" | {extra}" if extra else ""
        lines.append(f"{i}. [{item.id}] {title_with_family(item)} | {meta}{suffix}")
    return "\n".join(lines)


def context_title_keywords(items: list[Any]) -> list[str]:
    """Extract keywords from context items to expand retrieval."""
    keywords: list[str] = []
    suffixes = (
        "诈骗",
        "刷单",
        "返利",
        "冒充",
        "贷款",
        "游戏",
        "理财",
        "养老",
        "客服",
        "公检法",
        "征信",
        "虚假",
        "投资",
    )
    for item in items:
        texts = [
            getattr(item, "ccl2023_category", ""),
            getattr(item, "custom_subcategory", ""),
            getattr(item, "title", ""),
        ]
        for text in texts:
            text = normalize_text(text)
            if not text:
                continue
            for suffix in suffixes:
                if suffix in text and suffix not in keywords:
                    keywords.append(suffix)
            if len(text) <= 4 and text not in keywords:
                keywords.append(text)
        if len(keywords) >= 6:
            break
    return keywords[:6]


def candidate_summaries_for_llm(items: list[Any], limit: int) -> str:
    """Build item summaries for LLM recommendation selection."""
    lines = []
    for item in items:
        forms = "、".join(item.entry_channels) if item.entry_channels else "无"
        lines.append(
            f"[{item.id}] {title_with_family(item)} | "
            f"{item.ccl2023_category} | {item.risk_level} | "
            f"入口渠道：{forms} | "
            f"{item.summary[:80]}"
        )
    return "\n".join(lines)
