"""Dataset loading and normalized in-memory access for the current case schema."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import settings
from ..text import normalize_text


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    item_count: int


@dataclass(frozen=True)
class CaseItem:
    """Anti-fraud case item matching the normalized `case_items.json` schema."""

    id: str
    title: str
    summary: str
    nature_judgment: str
    judgment_reason: str
    entry_channels: tuple[str, ...]
    impersonated_identity: tuple[str, ...]
    false_belief: tuple[str, ...]
    key_methods: tuple[str, ...]
    target_assets: tuple[str, ...]
    fraud_stage: tuple[str, ...]
    risk_signals: str
    risk_level: str
    loss_occurred: str
    loss_type: tuple[str, ...]
    prevention_advice: str
    source_name: str
    source_type: str
    collection_date: str
    is_desensitized: str
    official_category: tuple[str, ...]
    ccl2023_category: str
    custom_subcategory: str
    tags: tuple[str, ...]
    involved_platforms: tuple[str, ...]
    victim_group: str
    emergency_plan_id: str
    law_basis_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    publish_date: str
    remark: str

    @property
    def content(self) -> str:
        """Canonical text payload used by embedding, retrieval, and answer context."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        if self.judgment_reason:
            parts.append(self.judgment_reason)
        if self.risk_signals:
            parts.append(self.risk_signals)
        if self.prevention_advice:
            parts.append(self.prevention_advice)
        return "\n".join(parts)


class KnowledgeBase:
    def __init__(self, payload: dict[str, Any]):
        def text_field(item: dict[str, Any], key: str) -> str:
            return normalize_text(str(item.get(key) or ""))

        def multivalue_field(item: dict[str, Any], key: str) -> tuple[str, ...]:
            value = item.get(key) or []
            if not isinstance(value, list):
                raise TypeError(f"{key} must be a list in case_items.json")
            return tuple(part for raw in value if (part := normalize_text(str(raw))))

        self.schema_version = payload.get("schema_version", 1)
        self.generated_at = payload.get("generated_at", "")
        self.source = payload.get("source", {})
        self.categories = [
            Category(
                id=int(category["id"]),
                name=str(category["name"]),
                item_count=int(category.get("item_count", 0)),
            )
            for category in payload.get("categories", [])
        ]
        self.items = [
            CaseItem(
                id=normalize_text(str(item.get("case_id") or item.get("id") or "")),
                title=text_field(item, "title"),
                summary=text_field(item, "summary"),
                nature_judgment=text_field(item, "nature_judgment"),
                judgment_reason=text_field(item, "judgment_reason"),
                entry_channels=multivalue_field(item, "entry_channels"),
                impersonated_identity=multivalue_field(item, "impersonated_identity"),
                false_belief=multivalue_field(item, "false_belief"),
                key_methods=multivalue_field(item, "key_methods"),
                target_assets=multivalue_field(item, "target_assets"),
                fraud_stage=multivalue_field(item, "fraud_stage"),
                risk_signals=text_field(item, "risk_signals"),
                risk_level=text_field(item, "risk_level"),
                loss_occurred=text_field(item, "loss_occurred"),
                loss_type=multivalue_field(item, "loss_type"),
                prevention_advice=text_field(item, "prevention_advice"),
                source_name=text_field(item, "source_name"),
                source_type=text_field(item, "source_type"),
                collection_date=text_field(item, "collection_date"),
                is_desensitized=text_field(item, "is_desensitized"),
                official_category=multivalue_field(item, "official_category"),
                ccl2023_category=text_field(item, "ccl2023_category"),
                custom_subcategory=text_field(item, "custom_subcategory"),
                tags=multivalue_field(item, "tags"),
                involved_platforms=multivalue_field(item, "involved_platforms"),
                victim_group=text_field(item, "victim_group"),
                emergency_plan_id=text_field(item, "emergency_plan_id"),
                law_basis_ids=multivalue_field(item, "law_basis_ids"),
                source_links=multivalue_field(item, "source_links"),
                publish_date=text_field(item, "publish_date"),
                remark=text_field(item, "remark"),
            )
            for item in payload.get("items", [])
        ]
        self._by_id = {item.id: item for item in self.items}

    def get(self, item_id: str) -> CaseItem | None:
        return self._by_id.get(item_id)

    def category_names(self) -> list[str]:
        return [category.name for category in self.categories]


def load_dataset(path: Path | None = None) -> KnowledgeBase:
    if path is None:
        path = settings.dataset_path
    with path.open("r", encoding="utf-8") as f:
        return KnowledgeBase(json.load(f))


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return load_dataset()


def item_to_dict(item: CaseItem, include_content: bool = False) -> dict[str, Any]:
    data = {
        "id": item.id,
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
