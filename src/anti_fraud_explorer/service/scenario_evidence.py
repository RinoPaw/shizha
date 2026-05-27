"""Scenario matching with hard evidence separated from soft inferred labels."""

from typing import Any

from ..text import normalize_text


SOFT_SCENARIO_SCORE = 2
HARD_SCENARIO_THRESHOLD = 4

SCENARIO_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "校园宣讲": ("校园宣讲", "校园", "学校", "学生", "班会", "宿舍", "求职", "游戏交易"),
    "社区宣传": ("社区宣传", "社区", "居民", "街道", "讲座", "日常提醒", "网购"),
    "老年防骗": ("老年防骗", "老人", "养老", "保健品", "补贴", "投资养老"),
    "企业培训": ("企业培训", "财务", "领导", "对公", "转账", "公司账户"),
    "新媒体提醒": ("新媒体提醒", "短视频", "推文", "海报", "社交平台", "客服", "刷单"),
    "以案说法": ("以案说法", "案例复盘", "拆解", "风险信号", "预警"),
}

SCENARIO_STRONG_TERMS: dict[str, tuple[str, ...]] = {
    "校园宣讲": ("校园宣讲", "学校", "学生", "班会", "校园贷"),
    "社区宣传": ("社区宣传", "社区", "居民", "讲座", "网购"),
    "老年防骗": ("老年防骗", "老人", "养老", "保健品"),
    "企业培训": ("企业培训", "财务", "领导", "对公", "公司"),
    "新媒体提醒": ("新媒体提醒", "短视频", "海报", "社交平台", "刷单"),
    "以案说法": ("以案说法", "复盘", "拆解", "风险信号"),
}


def scenario_match_score(item: Any, scenario: str) -> int:
    """Return scenario support score.

    Hard evidence comes from display forms and source text.  Soft category-inferred
    labels are allowed as weak tie-breakers only and cannot pass the threshold alone.
    """
    scenario = normalize_text(scenario)
    if not scenario:
        return 0

    terms = SCENARIO_EVIDENCE_TERMS.get(scenario, (scenario,))
    display_text = normalize_text(" ".join(getattr(item, "entry_channels", ()) or ()))
    source_text = normalize_text(
        " ".join(
            str(part or "")
            for part in [
                getattr(item, "title", ""),
                getattr(item, "ccl2023_category", ""),
                getattr(item, "custom_subcategory", ""),
                getattr(item, "summary", ""),
                getattr(item, "content", "")[:800],
            ]
        )
    )

    score = 0
    if scenario and scenario in display_text:
        score += 10
    display_hits = sum(1 for term in terms if term and term in display_text)
    source_hits = sum(1 for term in terms if term and term in source_text)
    strong_source_hits = sum(
        1 for term in SCENARIO_STRONG_TERMS.get(scenario, ()) if term and term in source_text
    )
    score += min(display_hits * 6, 12)
    if strong_source_hits:
        score += min(strong_source_hits * 5, 10)
    elif source_hits >= 2:
        score += min(source_hits * 3, 6)
    return score


def scenario_is_hard_match(item: Any, scenario: str) -> bool:
    return scenario_match_score(item, scenario) >= HARD_SCENARIO_THRESHOLD
