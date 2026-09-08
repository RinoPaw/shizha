"""Validate and normalize structured anti-fraud case data.

The runtime dataset is deterministic: this script never asks an LLM to invent
missing facts. It accepts a JSON object containing ``items`` (or
``records``/``list``), keeps the complete anti-fraud schema, and rebuilds the
category summary used by the application.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

CORE_FIELDS = (
    "case_id",
    "title",
    "summary",
    "nature_judgment",
    "judgment_reason",
    "entry_channels",
    "impersonated_identity",
    "false_belief",
    "key_methods",
    "target_assets",
    "fraud_stage",
    "risk_signals",
    "risk_level",
    "loss_occurred",
    "loss_type",
    "prevention_advice",
    "source_name",
    "source_type",
    "collection_date",
    "is_desensitized",
    "official_category",
    "ccl2023_category",
    "custom_subcategory",
    "tags",
    "involved_platforms",
    "victim_group",
    "emergency_plan_id",
    "law_basis_ids",
    "source_links",
    "publish_date",
    "remark",
)

MULTI_VALUE_FIELDS = {
    "entry_channels",
    "impersonated_identity",
    "false_belief",
    "key_methods",
    "target_assets",
    "fraud_stage",
    "loss_type",
    "official_category",
    "tags",
    "involved_platforms",
    "law_basis_ids",
    "source_links",
}

REQUIRED_TEXT_FIELDS = {
    "case_id",
    "title",
    "summary",
    "judgment_reason",
    "risk_signals",
    "risk_level",
    "prevention_advice",
    "source_name",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).replace("；", ";").split(";")
    result: list[str] = []
    for raw in raw_values:
        normalized = _text(raw)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def normalize_case(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in CORE_FIELDS:
        value = record.get(field)
        normalized[field] = _values(value) if field in MULTI_VALUE_FIELDS else _text(value)

    if not normalized["case_id"]:
        normalized["case_id"] = f"FZ-{index:03d}"

    missing = sorted(field for field in REQUIRED_TEXT_FIELDS if not normalized[field])
    if missing:
        raise ValueError(
            f"{normalized['case_id'] or f'row {index}'} missing required fields: "
            + ", ".join(missing)
        )
    return normalized


def build_dataset(source: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        records = payload
        source_meta: Any = {"name": source.name}
    elif isinstance(payload, dict):
        records = payload.get("items") or payload.get("records") or payload.get("list") or []
        source_meta = payload.get("source") or {"name": source.name}
    else:
        raise TypeError("source must be a JSON object or array")

    if not isinstance(records, list):
        raise TypeError("items/records/list must be an array")

    items = [normalize_case(record, index=index) for index, record in enumerate(records, 1)]
    ids = [item["case_id"] for item in items]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate case_id values: " + ", ".join(duplicates))

    category_counts = Counter(
        item["ccl2023_category"]
        or (item["official_category"][0] if item["official_category"] else "未分类")
        for item in items
    )
    categories = [
        {"id": index, "name": name, "item_count": count}
        for index, (name, count) in enumerate(
            sorted(category_counts.items(), key=lambda entry: (-entry[1], entry[0])),
            1,
        )
    ]
    return {
        "schema_version": 4,
        "generated_at": "",
        "source": source_meta,
        "categories": categories,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate structured anti-fraud case data")
    parser.add_argument("source", type=Path, help="input JSON containing items/records/list")
    parser.add_argument("output", type=Path, help="normalized output JSON")
    args = parser.parse_args()

    result = build_dataset(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(result['items'])} cases to {args.output.name}")


if __name__ == "__main__":
    main()
