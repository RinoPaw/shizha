"""Smoke tests for dataset loading."""

import json

import pytest
from anti_fraud_explorer.config import settings
from anti_fraud_explorer.domain.dataset import (
    CaseItem,
    load_dataset,
    get_knowledge_base,
    KnowledgeBase,
    item_to_dict,
    normalize_text,
)
from anti_fraud_explorer.service.search import build_search_text


pytestmark = pytest.mark.skipif(
    not settings.dataset_path.exists(),
    reason="Dataset file not found — run scripts/process_raw_data.py first",
)


def test_normalize_text_whitespace():
    assert normalize_text("  你好  世界  ") == "你好 世界"


def test_normalize_text_nbsp():
    assert normalize_text("hello\u00a0world") == "hello world"


def test_load_dataset_returns_knowledge_base():
    kb = load_dataset()
    assert isinstance(kb, KnowledgeBase)
    assert len(kb.items) > 0


def test_knowledge_base_categories():
    kb = load_dataset()
    assert len(kb.categories) > 0
    for cat in kb.categories:
        assert cat.name
        assert cat.item_count > 0


def test_dataset_items_have_required_fields():
    kb = load_dataset()
    item = kb.items[0]
    assert item.id
    assert item.title
    assert item.ccl2023_category
    assert item.content


def test_dataset_multivalue_fields_are_tuples():
    kb = load_dataset()
    item = kb.items[0]
    assert isinstance(item.entry_channels, tuple)
    assert isinstance(item.impersonated_identity, tuple)
    assert isinstance(item.key_methods, tuple)


def test_item_to_dict_emits_lists_for_multivalue_fields():
    item = CaseItem(
        id="FZ-TEST-001",
        title="测试案例",
        summary="摘要",
        nature_judgment="确认诈骗",
        judgment_reason="理由",
        entry_channels=("电话", "短信"),
        impersonated_identity=("冒充客服", "冒充平台官方"),
        false_belief=("误以为对方是官方",),
        key_methods=("钓鱼链接",),
        target_assets=("钱款",),
        fraud_stage=("接触引流", "诱导操作"),
        risk_signals="陌生来电；可疑链接",
        risk_level="高",
        loss_occurred="是",
        loss_type=("金钱损失",),
        prevention_advice="不要点链接",
        source_name="测试来源",
        source_type="官方",
        collection_date="2026-05-20",
        is_desensitized="是",
        official_category=("其他/无法对应",),
        ccl2023_category="虚假购物、服务类",
        custom_subcategory="测试细分类",
        tags=("测试", "反诈"),
        involved_platforms=("短信",),
        victim_group="普通用户",
        emergency_plan_id="",
        law_basis_ids=("FG-001",),
        source_links=("https://example.com",),
        publish_date="",
        remark="",
    )
    data = item_to_dict(item)
    assert data["entry_channels"] == ["电话", "短信"]
    assert data["impersonated_identity"] == ["冒充客服", "冒充平台官方"]
    assert data["fraud_stage"] == ["接触引流", "诱导操作"]


def test_build_search_text_comes_from_search_layer():
    item = CaseItem(
        id="FZ-TEST-002",
        title="测试搜索案例",
        summary="摘要",
        nature_judgment="确认诈骗",
        judgment_reason="理由",
        entry_channels=("电话", "短信"),
        impersonated_identity=("冒充客服",),
        false_belief=("误以为对方是官方",),
        key_methods=("钓鱼链接",),
        target_assets=("钱款",),
        fraud_stage=("接触引流",),
        risk_signals="风险信号",
        risk_level="高",
        loss_occurred="是",
        loss_type=("金钱损失",),
        prevention_advice="不要点击",
        source_name="测试来源",
        source_type="官方",
        collection_date="2026-05-20",
        is_desensitized="是",
        official_category=("其他/无法对应",),
        ccl2023_category="虚假购物、服务类",
        custom_subcategory="测试细分类",
        tags=("测试", "反诈"),
        involved_platforms=("短信",),
        victim_group="普通用户",
        emergency_plan_id="",
        law_basis_ids=(),
        source_links=(),
        publish_date="",
        remark="",
    )
    search_text = build_search_text(item)
    assert "测试搜索案例" in search_text
    assert "钓鱼链接" in search_text
    assert "冒充客服" in search_text


def test_raw_dataset_uses_lists_for_multivalue_fields():
    payload = json.loads(settings.dataset_path.read_text(encoding="utf-8"))
    item = payload["items"][0]
    assert isinstance(item["entry_channels"], list)
    assert isinstance(item["impersonated_identity"], list)
    assert isinstance(item["key_methods"], list)


def test_get_knowledge_base_caches():
    kb1 = get_knowledge_base()
    kb2 = get_knowledge_base()
    assert kb1 is kb2
