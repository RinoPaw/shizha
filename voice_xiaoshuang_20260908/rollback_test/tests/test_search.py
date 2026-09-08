from __future__ import annotations

from anti_fraud_explorer import search
from anti_fraud_explorer.dataset import KnowledgeBase


def _case(
    case_id: str,
    title: str,
    category: str,
    *,
    risk: str,
    channels: tuple[str, ...],
    methods: tuple[str, ...],
    victim_group: str = "",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "title": title,
        "summary": f"{title}的案例摘要",
        "nature_judgment": "确认诈骗",
        "judgment_reason": "存在明确诱导操作",
        "entry_channels": list(channels),
        "impersonated_identity": [],
        "false_belief": [],
        "key_methods": list(methods),
        "target_assets": ["钱款"],
        "fraud_stage": ["诱导操作"],
        "risk_signals": "要求立即操作并转账",
        "risk_level": risk,
        "loss_occurred": "是",
        "loss_type": ["金钱损失"],
        "prevention_advice": "停止操作并通过官方渠道核验",
        "source_name": "公安机关反诈通报",
        "source_type": "官方",
        "collection_date": "2026-01-01",
        "is_desensitized": "是",
        "official_category": [],
        "ccl2023_category": category,
        "custom_subcategory": "",
        "tags": [],
        "involved_platforms": [],
        "victim_group": victim_group,
        "emergency_plan_id": "",
        "law_basis_ids": [],
        "source_links": [],
        "publish_date": "",
        "remark": "",
    }


def make_kb() -> KnowledgeBase:
    items = [
        _case(
            "FZ-001",
            "冒充客服退款并诱导共享屏幕",
            "冒充电商物流客服类",
            risk="极高",
            channels=("电话",),
            methods=("屏幕共享", "索要验证码"),
            victim_group="网购用户",
        ),
        _case(
            "FZ-002",
            "刷单返利后要求垫资",
            "刷单返利类",
            risk="高",
            channels=("微信",),
            methods=("小额返利", "虚假订单/任务"),
            victim_group="学生",
        ),
        _case(
            "FZ-003",
            "冒充公检法要求转入安全账户",
            "冒充公检法及政府机关类",
            risk="极高",
            channels=("电话",),
            methods=("安全账户", "诱导转账"),
            victim_group="老人",
        ),
        _case(
            "FZ-004",
            "虚假投资平台承诺稳赚不赔",
            "虚假网络投资理财类",
            risk="高",
            channels=("App",),
            methods=("虚假投资平台",),
            victim_group="投资者",
        ),
        _case(
            "FZ-005",
            "游戏装备脱离平台交易",
            "网络游戏产品虚假交易类",
            risk="中",
            channels=("游戏平台私信",),
            methods=("虚假交易",),
            victim_group="游戏玩家",
        ),
    ]
    return KnowledgeBase({"items": items})


def test_empty_query_filters_and_pagination_are_stable() -> None:
    items, total = search.search_items(make_kb(), risk_level="极高", limit=1, offset=0)
    assert total == 2
    assert len(items) == 1
    assert items[0].case_id in {"FZ-001", "FZ-003"}

    first, total = search.search_items(make_kb(), limit=2, offset=0, use_pinyin=False)
    second, _ = search.search_items(make_kb(), limit=2, offset=2, use_pinyin=False)
    assert total == 5
    assert len(first) == len(second) == 2
    assert {item.case_id for item in first}.isdisjoint(item.case_id for item in second)


def test_query_searches_title_methods_signals_and_advice() -> None:
    kb = make_kb()
    items, total = search.search_items(kb, query="共享屏幕验证码", use_pinyin=False)
    assert total >= 1
    assert items[0].case_id == "FZ-001"

    items, total = search.search_items(kb, query="稳赚不赔", use_pinyin=False)
    assert total == 1
    assert items[0].case_id == "FZ-004"


def test_explicit_filters_cover_fraud_dimensions() -> None:
    kb = make_kb()
    items, total = search.search_items(kb, category="刷单返利类", use_pinyin=False)
    assert total == 1 and items[0].case_id == "FZ-002"

    items, total = search.search_items(kb, entry_channel="电话", use_pinyin=False)
    assert total == 2
    assert {item.case_id for item in items} == {"FZ-001", "FZ-003"}

    items, total = search.search_items(kb, victim_group="游戏玩家", use_pinyin=False)
    assert total == 1 and items[0].case_id == "FZ-005"


def test_broad_catalogue_query_is_not_artificially_reduced_to_four() -> None:
    items, total = search.search_items(
        make_kb(),
        query="有哪些诈骗案例值得了解？",
        limit=30,
        use_pinyin=False,
    )
    assert total == 5
    assert len(items) == 5
    assert len({item.category for item in items}) == 5


def test_embedding_scores_can_hybrid_rerank(monkeypatch) -> None:
    kb = make_kb()
    monkeypatch.setattr(search.config, "SEARCH_USE_EMBEDDING", True)

    def fake_embedding_scores(_kb, _query, candidates, min_score=0.0):
        scores = {item.case_id: (0.99 if item.case_id == "FZ-003" else 0.2) for item in candidates}
        return {item.case_id: score for item in candidates if (score := scores[item.case_id]) >= min_score}

    monkeypatch.setattr("anti_fraud_explorer.embeddings.embedding_scores", fake_embedding_scores)
    items, total = search.search_items(kb, query="电话转账", use_pinyin=False)
    assert total >= 2
    assert items[0].case_id == "FZ-003"
