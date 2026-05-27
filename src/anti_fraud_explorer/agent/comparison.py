"""Comparison task handler for the anti-fraud agent."""

import re
from typing import Any

from .models import AgentResult, TaskType
from ..domain.dataset import KnowledgeBase
from ..service.item_cards import enriched_item_card, source_payload, title_with_family
from ..service.retriever import _PROVINCE_PATTERN, _SHORT_PROVINCE_MAP
from ..text import normalize_text


_COMPARISON_TARGET_TRAILING_RE = re.compile(
    r"(?:有什么区别|有什么不同|有何区别|有何不同|的区别|的差异|哪个更受欢迎|哪个更适合|哪个更|哪个好|的比较|的对比|对比一下|比较一下)$"
)


def handle_comparison(kb: KnowledgeBase, analysis) -> AgentResult:
    """Handle a multi-entity structured comparison without fabricating matches."""
    from ..service.search import search_items

    # Resolve target entities — try explicit entities first, fall back to splitting
    targets: list[str] = []
    if analysis.entities:
        targets = [_clean_comparison_target(e) for e in analysis.entities]
    else:
        # Fallback: split rewritten query on common separators
        parts = re.split(r"\s+", analysis.rewritten_query)
        targets = [_clean_comparison_target(p) for p in parts if len(p) >= 2]
    targets = [target for target in targets if target]

    if len(targets) < 2:
        # Not enough entities to compare — fall through to LLM
        from ..ai import Answer, answer_question

        answer: Answer = answer_question(
            kb,
            question=analysis.rewritten_query or analysis.original_query,
        )
        return AgentResult(
            task_type=TaskType.COMPARISON,
            answer=answer.answer,
            speech=answer.speech,
            sources=answer.sources,
            mode=answer.mode,
            confidence=0.7,
        )

    # Search each target entity in the KB
    resolved: list[tuple[str, Any]] = []
    unmatched: list[str] = []
    used_item_ids: set[str] = set()

    for t in targets:
        match = _resolve_comparison_target(kb, t, used_item_ids)
        if match:
            display_name, item = match
            used_item_ids.add(item.id)
            resolved.append((display_name, item))
        else:
            unmatched.append(t)

    if len(resolved) < 2:
        suggestion_query = _comparison_suggestion_query(targets)
        suggestions: list[Any] = []
        if suggestion_query and not _has_explicit_region_targets(targets):
            suggestions, _ = search_items(kb, query=suggestion_query, limit=4)
        suggestion_cards = [enriched_item_card(item) for item in suggestions]
        suggestion_sources = [source_payload(item) for item in suggestions]
        missing = unmatched or targets
        missing_text = "、".join(missing)
        answer_lines = [
            f"资料库中暂未找到可直接对应「{missing_text}」的反诈案例，因此当前不能做依据式对比。",
        ]
        if suggestion_cards:
            topic_text = f"与“{suggestion_query}”相关" if suggestion_query else "当前最接近"
            answer_lines.extend(
                [
                    "",
                    f"资料库里 {topic_text} 的案例有：",
                    "",
                ]
            )
            for index, item in enumerate(suggestions, 1):
                desc = " · ".join(part for part in [item.ccl2023_category, item.risk_level] if part)
                answer_lines.append(
                    f"{index}. {title_with_family(item)}" + (f"（{desc}）" if desc else "")
                )
            answer_lines.extend(
                [
                    "",
                    "你可以继续追问这些已收录案例之间的区别，或改问资料库中实际存在的地区化案例。",
                ]
            )
        elif _has_explicit_region_targets(targets):
            answer_lines.extend(
                [
                    "",
                    "这类问题带有明确地域约束，系统不会用其他省份的同类案例替代，以免把参考资料误当成对比对象。",
                ]
            )

        speech = f"资料库中暂时没有可直接对应{missing_text}的条目，所以现在不能做依据式对比。" + (
            f"当前最接近的案例主要有：{'、'.join(title_with_family(item) for item in suggestions)}。"
            if suggestions
            else ""
        )
        return AgentResult(
            task_type=TaskType.COMPARISON,
            answer="\n".join(answer_lines),
            speech=speech,
            items=suggestion_cards,
            sources=suggestion_sources,
            mode="local",
            confidence=0.45,
            warnings=[f"未在资料库中找到可直接对应的比较项：{missing_text}"],
        )

    # Build comparison answer
    lines: list[str] = []
    lines.append(f"## {' vs '.join(name for name, _ in resolved)} 对比\n")

    # ── Table header ──
    col_width = 18
    header = (
        f"| {'维度':<{col_width - 4}}"
        + "".join(f" | {name[: col_width - 2]:<{col_width - 2}}" for name, _ in resolved)
        + " |"
    )
    sep = (
        "|" + "-" * (col_width - 1) + "|" + "|".join("-" * (col_width - 1) for _ in resolved) + "|"
    )
    lines.append(header)
    lines.append(sep)

    def _row(label: str, *values: str) -> str:
        return (
            f"| {label:<{col_width - 4}}"
            + "".join(f" | {v[: col_width - 2]:<{col_width - 2}}" for v in values)
            + " |"
        )

    # Category row
    lines.append(_row("类别", *(item.ccl2023_category for _, item in resolved)))

    lines.append(_row("细分类", *(item.custom_subcategory or "—" for _, item in resolved)))
    lines.append(_row("风险等级", *(item.risk_level or "—" for _, item in resolved)))

    lines.append(
        _row(
            "入口渠道",
            *(
                "、".join(item.entry_channels) if item.entry_channels else "\u2014"
                for _, item in resolved
            ),
        )
    )

    # ── Narrative sections ──
    lines.append("")
    for entity_name, item in resolved:
        lines.append(f"### {entity_name}")
        if item.key_methods:
            lines.append(f"**诈骗手法：**{'；'.join(item.key_methods)[:200]}")
        if item.source_name:
            lines.append(f"**来源：**{item.source_name[:200]}")
        if item.prevention_advice:
            lines.append(f"**防范建议：**{item.prevention_advice[:200]}")
        if not (item.key_methods or item.source_name or item.prevention_advice):
            lines.append(f"{item.summary[:300]}")
        lines.append("")

    # Comparison summary
    lines.append("### 对比小结")
    summary_parts: list[str] = []

    # Level comparison
    levels = [item.risk_level or "" for _, item in resolved]
    unique_levels = list(dict.fromkeys(levels))
    if len(unique_levels) > 1:
        summary_parts.append(
            f"风险等级上，{'、'.join(f'{name}为{lv}' for (name, _), lv in zip(resolved, levels))}"
        )
    else:
        summary_parts.append(f"两项风险等级均为{unique_levels[0]}")

    cats = [item.ccl2023_category for _, item in resolved]
    unique_cats = list(dict.fromkeys(cats))
    if len(unique_cats) > 1:
        summary_parts.append(f"分属{'和'.join(unique_cats)}不同类别")
    else:
        summary_parts.append(f"同属{unique_cats[0]}类别")

    lines.append("；".join(summary_parts) + "。")

    # Build evidence
    evidence: list[dict[str, Any]] = []
    for entity_name, item in resolved:
        evidence.append(
            {
                "type": "source",
                "claim": f"对比项：{entity_name}",
                "basis": f"search query={entity_name!r}",
                "item_id": item.id,
            }
        )

    sources = [source_payload(item) for _, item in resolved]
    items = [enriched_item_card(item) for _, item in resolved]

    warnings: list[str] = []
    if unmatched:
        warnings.append(f"未在资料库中找到：{'、'.join(unmatched)}")

    return AgentResult(
        task_type=TaskType.COMPARISON,
        answer="\n".join(lines),
        items=items,
        sources=sources,
        evidence=evidence,
        mode="local",
        confidence=0.85 if not unmatched else 0.6,
        warnings=warnings,
    )


def _clean_comparison_target(value: str) -> str:
    cleaned = normalize_text(value)
    cleaned = _COMPARISON_TARGET_TRAILING_RE.sub("", cleaned)
    cleaned = cleaned.strip("，,。！？?、；：:~～ ")
    return cleaned


def _comparison_target_parts(target: str) -> tuple[str, str]:
    cleaned = _clean_comparison_target(target)
    province = ""
    match = _PROVINCE_PATTERN.search(cleaned)
    if match:
        province = match.group(1)
    else:
        for short, full in sorted(
            _SHORT_PROVINCE_MAP.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if short in cleaned:
                province = full
                cleaned = cleaned.replace(short, "", 1)
                break

    if province:
        cleaned = cleaned.replace(province, "")
        for short, full in _SHORT_PROVINCE_MAP.items():
            if full == province:
                cleaned = cleaned.replace(short, "")
                break

    core = cleaned.strip(" 的")
    return province, core or _clean_comparison_target(target)


def _resolve_comparison_target(kb: KnowledgeBase, target: str, used_item_ids: set[str]):
    from ..service.search import search_items

    cleaned = _clean_comparison_target(target)
    province, core = _comparison_target_parts(cleaned)

    candidate_queries = [cleaned]
    if core and core != cleaned:
        candidate_queries.append(core)

    candidates: list[Any] = []
    seen_ids: set[str] = set()
    for query in candidate_queries:
        if not query:
            continue
        result, _ = search_items(kb, query=query, limit=8)
        for item in result:
            if item.id in used_item_ids or item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            candidates.append(item)

    best = None
    best_score = 0
    for item in candidates:
        names = [
            item.title,
            item.ccl2023_category,
            item.custom_subcategory,
        ]
        score = 0
        if cleaned in names:
            score += 120
        if core in names:
            score += 100
        if cleaned and cleaned in item.title:
            score += 90
        if core and core in item.title:
            score += 70
        if core and core in item.summary:
            score += 30

        if score > best_score:
            best = (title_with_family(item), item)
            best_score = score

    if best_score < 60:
        return None
    return best


def _comparison_suggestion_query(targets: list[str]) -> str:
    cores = []
    for target in targets:
        _, core = _comparison_target_parts(target)
        if core:
            cores.append(core)
    unique = list(dict.fromkeys(core for core in cores if len(core) >= 2))
    if len(unique) == 1:
        return unique[0]
    for candidate in sorted(unique, key=len, reverse=True):
        if all(candidate in core for core in unique):
            return candidate
    return unique[0] if unique else ""


def _has_explicit_region_targets(targets: list[str]) -> bool:
    return any(_comparison_target_parts(target)[0] for target in targets)
