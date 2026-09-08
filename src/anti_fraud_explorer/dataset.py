"""Load and normalize the local anti-fraud case catalogue."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import DATASET_PATH


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(normalize_text(str(v)) for v in value if normalize_text(str(v)))
    return (normalize_text(str(value)),) if normalize_text(str(value)) else ()


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    item_count: int


@dataclass(frozen=True)
class FraudCase:
    """Complete schema-v4 case record plus derived retrieval text."""

    case_id: str
    title: str
    summary: str = ""
    nature_judgment: str = ""
    judgment_reason: str = ""
    entry_channels: tuple[str, ...] = ()
    impersonated_identity: tuple[str, ...] = ()
    false_belief: tuple[str, ...] = ()
    key_methods: tuple[str, ...] = ()
    target_assets: tuple[str, ...] = ()
    fraud_stage: tuple[str, ...] = ()
    risk_signals: str = ""
    risk_level: str = ""
    loss_occurred: str = ""
    loss_type: tuple[str, ...] = ()
    prevention_advice: str = ""
    source_name: str = ""
    source_type: str = ""
    collection_date: str = ""
    is_desensitized: str = ""
    official_category: tuple[str, ...] = ()
    ccl2023_category: str = ""
    custom_subcategory: str = ""
    tags: tuple[str, ...] = ()
    involved_platforms: tuple[str, ...] = ()
    victim_group: str = ""
    emergency_plan_id: str = ""
    law_basis_ids: tuple[str, ...] = ()
    source_links: tuple[str, ...] = ()
    publish_date: str = ""
    remark: str = ""
    content: str = field(init=False)
    search_text: str = field(init=False)

    def __post_init__(self) -> None:
        parts = (self.summary, self.judgment_reason, self.risk_signals, self.prevention_advice)
        content = normalize_text(" ".join(part for part in parts if part))
        search_parts = (
            self.title, self.ccl2023_category, self.custom_subcategory,
            " ".join(self.tags), " ".join(self.entry_channels),
            " ".join(self.key_methods), self.victim_group, content,
        )
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "search_text", normalize_text(" ".join(search_parts)))

    @property
    def id(self) -> str:
        return self.case_id

    @property
    def category(self) -> str:
        return self.ccl2023_category or self.custom_subcategory


def _record(item: dict[str, Any]) -> FraudCase:
    text_fields = {
        "case_id": item.get("case_id", item.get("id", "")),
        "title": item.get("title", ""),
        "summary": item.get("summary", ""),
        "nature_judgment": item.get("nature_judgment", ""),
        "judgment_reason": item.get("judgment_reason", ""),
        "risk_signals": item.get("risk_signals", ""),
        "risk_level": item.get("risk_level", ""),
        "loss_occurred": item.get("loss_occurred", ""),
        "prevention_advice": item.get("prevention_advice", ""),
        "source_name": item.get("source_name", ""),
        "source_type": item.get("source_type", ""),
        "collection_date": item.get("collection_date", ""),
        "is_desensitized": item.get("is_desensitized", ""),
        "ccl2023_category": item.get("ccl2023_category", ""),
        "custom_subcategory": item.get("custom_subcategory", ""),
        "victim_group": item.get("victim_group", ""),
        "emergency_plan_id": item.get("emergency_plan_id", ""),
        "publish_date": item.get("publish_date", ""),
        "remark": item.get("remark", ""),
    }
    tuple_fields = (
        "entry_channels", "impersonated_identity", "false_belief", "key_methods",
        "target_assets", "fraud_stage", "loss_type", "official_category", "tags",
        "involved_platforms", "law_basis_ids", "source_links",
    )
    values = {key: normalize_text(str(value or "")) for key, value in text_fields.items()}
    values.update({key: _as_tuple(item.get(key)) for key in tuple_fields})
    return FraudCase(**values)


class KnowledgeBase:
    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]):
        if isinstance(payload, list):
            payload = {"items": payload}
        self.schema_version = int(payload.get("schema_version", 1))
        self.generated_at = str(payload.get("generated_at", ""))
        self.source = payload.get("source", {}) or {}
        raw_items = payload.get("items", payload.get("case_items", []))
        self.items = [_record(item) for item in raw_items if isinstance(item, dict)]
        raw_categories = payload.get("categories", []) or []
        if raw_categories:
            self.categories = [
                Category(int(row.get("id", index + 1)), str(row.get("name", "")), int(row.get("item_count", 0)))
                for index, row in enumerate(raw_categories)
            ]
        else:
            names = sorted({item.ccl2023_category for item in self.items if item.ccl2023_category})
            self.categories = [Category(i, name, sum(item.ccl2023_category == name for item in self.items)) for i, name in enumerate(names, 1)]
        self._by_id = {item.case_id: item for item in self.items}

    def get(self, item_id: str) -> FraudCase | None:
        return self._by_id.get(str(item_id))


def load_dataset(path: Path = DATASET_PATH) -> KnowledgeBase:
    with path.open("r", encoding="utf-8") as handle:
        return KnowledgeBase(json.load(handle))


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return load_dataset()


def item_to_dict(item: FraudCase, *, include_content: bool = False, include_enrichment: bool = False) -> dict[str, Any]:
    """Serialize the complete source schema; enrichment is a no-op API flag."""
    data: dict[str, Any] = {
        "case_id": item.case_id,
        "id": item.case_id,
        "title": item.title,
        "summary": item.summary,
        "nature_judgment": item.nature_judgment,
        "judgment_reason": item.judgment_reason,
        "entry_channels": list(item.entry_channels),
        "impersonated_identity": list(item.impersonated_identity),
        "false_belief": list(item.false_belief),
        "key_methods": list(item.key_methods),
        "target_assets": list(item.target_assets),
        "fraud_stage": list(item.fraud_stage),
        "risk_signals": item.risk_signals,
        "risk_level": item.risk_level,
        "loss_occurred": item.loss_occurred,
        "loss_type": list(item.loss_type),
        "prevention_advice": item.prevention_advice,
        "source_name": item.source_name,
        "source_type": item.source_type,
        "collection_date": item.collection_date,
        "is_desensitized": item.is_desensitized,
        "official_category": list(item.official_category),
        "ccl2023_category": item.ccl2023_category,
        "category": item.ccl2023_category or item.custom_subcategory,
        "custom_subcategory": item.custom_subcategory,
        "tags": list(item.tags),
        "involved_platforms": list(item.involved_platforms),
        "victim_group": item.victim_group,
        "emergency_plan_id": item.emergency_plan_id,
        "law_basis_ids": list(item.law_basis_ids),
        "source_links": list(item.source_links),
        "publish_date": item.publish_date,
        "remark": item.remark,
    }
    if include_content:
        data["content"] = item.content
    return data


__all__ = ["Category", "FraudCase", "KnowledgeBase", "get_knowledge_base", "item_to_dict", "load_dataset", "normalize_text"]
