"""Context building and text extraction for the anti-fraud AI — v3 schema."""

from ..domain.dataset import CaseItem
from ..text import normalize_text


def build_context(sources: list[CaseItem], max_chars: int) -> str:
    chunks = []
    remaining = max_chars
    for index, item in enumerate(sources, start=1):
        text = item_context_text(item)
        chunk = f"[{index}] 标题：{item.title}\n类别：{item.ccl2023_category}\n资料：{text}"
        if len(chunk) > remaining:
            chunk = chunk[: max(0, remaining - 20)] + "..."
        chunks.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            break
    return "\n\n".join(chunks)


def item_context_text(item: CaseItem) -> str:
    """Build structured context from the LLM-normalized fields directly."""
    parts = []
    for label, value in [
        ("场景简述", item.summary),
        ("性质判断", item.nature_judgment),
        ("判断理由", item.judgment_reason),
        ("入口渠道", "；".join(item.entry_channels)),
        ("冒充身份", "；".join(item.impersonated_identity)),
        ("关键手法", "；".join(item.key_methods)),
        ("目标资产", "；".join(item.target_assets)),
        ("诈骗阶段", "；".join(item.fraud_stage)),
        ("风险信号", item.risk_signals),
        ("防范建议", item.prevention_advice),
    ]:
        if value:
            parts.append(f"{label}：{normalize_text(value)}")
    if parts:
        return "\n".join(parts)
    return normalize_text(item.summary)


def clean_knowledge_text(text: str) -> str:
    return normalize_text(text)


def extract_structured_field(text: str, label: str) -> str:
    """Kept for backward compat — search text for a label: marker.
    In v3 schema, use item fields directly instead."""
    text = normalize_text(text)
    marker = f"{label}:"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    # Find next label boundary
    labels = [
        "场景简述",
        "性质判断",
        "判断理由",
        "入口渠道",
        "冒充身份",
        "关键手法",
        "目标资产",
        "诈骗阶段",
        "风险信号",
        "防范建议",
        "来源名称",
        "采集日期",
    ]
    end = len(text)
    for next_label in labels:
        if next_label == label:
            continue
        for next_marker in (f", {next_label}:", f"，{next_label}:"):
            position = text.find(next_marker, start)
            if position >= 0:
                end = min(end, position)
    return text[start:end].strip(" ，,")
