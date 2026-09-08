from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_dataset import CORE_FIELDS, build_dataset


def _record(case_id: str = "FZ-T01") -> dict[str, object]:
    record: dict[str, object] = {field: "" for field in CORE_FIELDS}
    for field in (
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
    ):
        record[field] = []
    record.update(
        {
            "case_id": case_id,
            "title": "冒充客服退款诈骗",
            "summary": "对方冒充客服，以退款为由诱导受害人共享屏幕。",
            "nature_judgment": "确认诈骗",
            "judgment_reason": "正规客服不会要求共享屏幕或提供验证码。",
            "entry_channels": ["电话"],
            "key_methods": ["屏幕共享", "索要验证码"],
            "risk_signals": "要求共享屏幕并索要验证码",
            "risk_level": "极高",
            "prevention_advice": "立即停止共享，联系平台官方客服核验。",
            "source_name": "公安机关反诈通报",
            "source_type": "官方",
            "ccl2023_category": "冒充电商物流客服类",
        }
    )
    return record


def test_builder_keeps_complete_schema_and_rebuilds_categories(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"source": {"name": "fixture"}, "items": [_record()]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_dataset(source)

    assert result["schema_version"] == 4
    assert tuple(result["items"][0]) == CORE_FIELDS
    assert result["categories"] == [
        {"id": 1, "name": "冒充电商物流客服类", "item_count": 1}
    ]
    assert result["items"][0]["key_methods"] == ["屏幕共享", "索要验证码"]


def test_builder_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"items": [_record(), _record()]}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        build_dataset(source)


def test_production_dataset_retains_all_86_cases_and_31_fields() -> None:
    payload = json.loads(
        (ROOT / "data" / "processed" / "case_items.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 4
    assert len(payload["items"]) == 86
    assert len({item["case_id"] for item in payload["items"]}) == 86
    assert all(set(item) == set(CORE_FIELDS) for item in payload["items"])
    assert sum(category["item_count"] for category in payload["categories"]) == 86
    assert {category["name"] for category in payload["categories"]} == {
        "虚假购物、服务类",
        "虚假网络投资理财类",
        "冒充电商物流客服类",
        "刷单返利类",
        "冒充领导、熟人类",
        "冒充公检法及政府机关类",
        "网络游戏产品虚假交易类",
        "贷款、代办信用卡类",
    }
