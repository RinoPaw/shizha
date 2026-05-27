"""Handler functions for each TaskType (comparison, study task, etc.)."""

import json
import re
from dataclasses import replace
from typing import Any

from ..config import settings
from .models import TaskType, AgentResult
from ..domain.dataset import KnowledgeBase
from ..service.item_cards import enriched_item_card, source_payload, title_with_family
from ..service.scenario_evidence import scenario_match_score
from ..prompts import DEFAULT_TRANSFORM_TYPE, TRANSFORM_MAX_TOKENS, TRANSFORM_PROMPTS
from ..text import normalize_text
from .formatting import candidate_summaries_for_llm
from .rendering import render_template, build_transform_local

# Forward declaration for type hint
Agent = Any  # Will be resolved at runtime from agent.py


def handle_comparison(kb: KnowledgeBase, analysis) -> AgentResult:
    """COMPARISON: multi-entity structured comparison, no LLM."""
    from .comparison import handle_comparison as _comparison_impl

    return _comparison_impl(kb, analysis)


def handle_study_task(agent: Agent, analysis) -> AgentResult:
    """STUDY_TASK: curriculum/teaching plan generation."""
    from ..service.search import search_items

    target_item = None
    if analysis.entities:
        entity = analysis.entities[0]
        result, _ = search_items(agent.kb, query=entity, limit=1)
        if result:
            target_item = result[0]

    if target_item is None:
        rec_result = agent._handle_recommend(analysis)  # agent passed in
        if rec_result.items:
            target_item = agent.kb.get(rec_result.items[0]["id"])

    if target_item is None:
        from ..ai import Answer, answer_question

        answer: Answer = answer_question(
            agent.kb,
            question=analysis.rewritten_query or analysis.original_query,
            include_speech=False,
        )
        return AgentResult(
            task_type=TaskType.STUDY_TASK,
            answer=answer.answer,
            speech=answer.speech,
            sources=answer.sources,
            mode=answer.mode,
            confidence=0.5,
            warnings=["未找到可用的反诈案例，已退回通用问答"],
        )

    audience = analysis.audience or "中小学生"
    time_budget = analysis.time_budget or "45分钟"
    scenario = analysis.scenario or "校园宣讲"

    audience_label: str
    if audience in ("儿童", "小学生"):
        audience_label = "小学中高年级"
    elif audience in ("青少年", "中学生"):
        audience_label = "初中生"
    elif audience == "大学生":
        audience_label = "大学生"
    elif audience == "家庭":
        audience_label = "亲子家庭"
    else:
        audience_label = "中小学生"

    title = title_with_family(target_item)
    category = target_item.ccl2023_category
    summary = target_item.summary[:200]

    features = "；".join(target_item.key_methods)[:200] or summary
    history = target_item.source_name[:200] or ""
    display = "、".join(target_item.entry_channels) if target_item.entry_channels else "展板 + 讲解"
    prevention_advice = target_item.prevention_advice[:200] or ""

    answer = render_template(
        "study_task.md.j2",
        title=title,
        audience_label=audience_label,
        time_budget=time_budget,
        scenario=scenario,
        category=category,
        display=display,
        features=features,
        history=history,
        cultural_value=prevention_advice,
    ).strip()

    sources = [source_payload(target_item)]
    items = [enriched_item_card(target_item)]
    evidence = [
        {
            "type": "source",
            "claim": "教案主体",
            "basis": f"entity={analysis.entities[0] if analysis.entities else '推荐'}",
            "item_id": target_item.id,
        }
    ]

    return AgentResult(
        task_type=TaskType.STUDY_TASK,
        answer=answer,
        items=items,
        sources=sources,
        evidence=evidence,
        mode="local",
        confidence=0.8,
        warnings=["教案由模板辅助整理，建议教师根据实际学情调整教学环节和时间分配。"],
    )


def handle_content_transform(agent: Agent, analysis) -> AgentResult:
    """CONTENT_TRANSFORM: rewrite / spoken script / creative brief."""
    from ..service.search import search_items
    from ..ai import Answer, answer_question

    target_item = None
    if analysis.entities:
        entity = analysis.entities[0]
        result, _ = search_items(agent.kb, query=entity, limit=1)
        if result:
            target_item = result[0]

    if target_item is None:
        answer: Answer = answer_question(
            agent.kb,
            question=analysis.original_query,
            include_speech=False,
        )
        return AgentResult(
            task_type=TaskType.CONTENT_TRANSFORM,
            answer=answer.answer,
            speech=answer.speech,
            sources=answer.sources,
            mode=answer.mode,
            confidence=0.5,
            warnings=["未识别到具体反诈案例，已退回通用问答"],
        )

    transform_type = analysis.transform_type
    if not transform_type:
        q = analysis.original_query
        if re.search(r"讲解词|讲解稿|口播稿|解说词", q):
            transform_type = "讲解词"
        elif re.search(r"年轻化|朋友圈|口语化|轻松", q):
            transform_type = "年轻化"
        elif re.search(r"文创|设计.*产品|纹样|包装|IP|联名|周边|创意", q):
            transform_type = "文创文案"
        else:
            transform_type = "改写"

    context_lines = [
        f"标题：{target_item.title}",
        f"类别：{target_item.ccl2023_category}",
    ]
    if target_item.custom_subcategory:
        context_lines.append(f"细分类：{target_item.custom_subcategory}")
    if target_item.risk_level:
        context_lines.append(f"风险等级：{target_item.risk_level}")
    if target_item.key_methods:
        context_lines.append(f"关键手法：{'；'.join(target_item.key_methods)}")
    if target_item.source_name:
        context_lines.append(f"来源：{target_item.source_name}")
    if target_item.prevention_advice:
        context_lines.append(f"防范建议：{target_item.prevention_advice}")
    context_lines.append(f"简介：{target_item.summary}")
    context_lines.append(f"正文片段：{target_item.content[:800]}")
    context = "\n".join(context_lines)

    if settings.ai_api_key:
        try:
            answer_text = _call_transform_model(
                transform_type=transform_type,
                context=context,
                query=analysis.original_query,
            )
            return AgentResult(
                task_type=TaskType.CONTENT_TRANSFORM,
                answer=answer_text,
                sources=[source_payload(target_item)],
                items=[enriched_item_card(target_item)],
                mode="llm",
                confidence=0.8,
            )
        except Exception:
            pass  # fall through to local

    local_answer = build_transform_local(transform_type, target_item)
    return AgentResult(
        task_type=TaskType.CONTENT_TRANSFORM,
        answer=local_answer,
        items=[enriched_item_card(target_item)],
        sources=[source_payload(target_item)],
        mode="local",
        confidence=0.5,
        warnings=["模型接口不可用，已切回本地模板。如需更丰富的内容，请配置 API Key。"],
    )


def handle_browse(agent: Agent, analysis) -> AgentResult:
    """BROWSE_QUERY: structured filters + local listing, no LLM."""
    from ..service.search import search_items

    province = analysis.metadata_filters.get("province", "")
    level = analysis.metadata_filters.get("level", "")
    category = analysis.metadata_filters.get("category", "")
    limit = analysis.retrieval_count

    result, total = search_items(
        agent.kb,
        query=analysis.rewritten_query,
        category=category,
        risk_level=level,
        limit=limit,
    )

    items = [enriched_item_card(item) for item in result]
    filter_desc = _describe_filters(category, province, level)
    header = (
        f"找到 {total} 条{filter_desc}相关案例：\n"
        if total
        else f"未找到匹配的{filter_desc}相关案例。"
    )
    lines = [header]
    for i, item in enumerate(result, 1):
        level_str = f" | {item.risk_level}" if item.risk_level else ""
        subtype_str = f" | {item.custom_subcategory}" if item.custom_subcategory else ""
        lines.append(
            f"{i}. {title_with_family(item)} -- {item.ccl2023_category}{level_str}{subtype_str}"
        )

    evidence = []
    for item in result:
        evidence.append(
            {
                "type": "source",
                "claim": "筛选命中",
                "basis": f"province={province}, category={category}, risk_level={level}",
                "item_id": item.id,
            }
        )

    warnings = []
    if province:
        warnings.append(f"当前资料库未提供地区字段，已忽略地域筛选：{province}。")
    if not total:
        warnings.append(f"未找到{filter_desc}相关案例")
    elif total > limit:
        warnings.append(
            f"共匹配 {total} 项，当前仅展示前 {limit} 项。可通过筛选条件缩小范围或调整展示数量。"
        )

    return AgentResult(
        task_type=TaskType.BROWSE_QUERY,
        answer="\n".join(lines),
        items=items,
        sources=[source_payload(item) for item in result],
        evidence=evidence,
        mode="local",
        confidence=0.95,
        total_count=total,
        warnings=warnings,
    )


def handle_recommend(agent: Agent, analysis) -> AgentResult:
    """RECOMMENDATION: LLM selects best items from candidate pool."""
    from ..service.search import search_items
    from ..service.http_client import chat_completion

    scenario = analysis.scenario
    audience = analysis.audience
    limit = min(analysis.retrieval_count or 5, 10)

    query = analysis.rewritten_query or " ".join(analysis.entities)
    candidates, _ = search_items(agent.kb, query=query, limit=30)

    seen_ids = set()
    unique = []
    for item in candidates:
        if item.id not in seen_ids:
            seen_ids.add(item.id)
            unique.append(item)

    if not unique:
        return AgentResult(
            task_type=TaskType.RECOMMENDATION,
            answer="未找到相关案例，请尝试调整问题。",
            mode="local",
            confidence=0.3,
            warnings=["检索候选池为空"],
        )

    candidate_text = candidate_summaries_for_llm(unique[:20], limit)

    scene_desc = f"场景：{scenario}" if scenario else ""
    audience_desc = f"受众：{audience}" if audience else ""
    try:
        if not settings.ai_api_key:
            raise RuntimeError("AI_API_KEY is not configured")
        response = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个反诈案例推荐助手。用户要求推荐案例，你要从候选池中选出最合适的案例。\n"
                        "选择原则：\n"
                        "1. 优先选择与用户目标人群或宣传场景匹配的案例\n"
                        "2. 优先选择风险等级更高、提醒价值更强的案例\n"
                        "3. 优先选择手法清晰、风险信号明确的案例\n"
                        "4. 避免重复选择高度相似的案例\n"
                        f"请选出恰好 {limit} 个案例，输出 JSON："
                        '{"selected": ["item_id_1", "item_id_2", ...], "reason": "选择理由"}\n'
                        "只输出 JSON，不要其他内容。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户需求：{analysis.original_query}\n"
                        f"{scene_desc}{audience_desc}\n\n"
                        f"候选案例（共 {len(unique)} 个）：\n{candidate_text}\n\n"
                        f"请从中选出最好的 {limit} 个。"
                    ),
                },
            ],
            temperature=0.3,
        )
        selected = _parse_llm_selection(response, limit)
    except Exception:
        unique.sort(
            key=lambda x: (
                5
                if x.risk_level == "极高"
                else 3
                if x.risk_level == "高"
                else 1
                if x.risk_level == "中"
                else 0
            ),
            reverse=True,
        )
        selected = [item.id for item in unique[:limit]]

    top = [agent.kb.get(item_id) for item_id in selected if agent.kb.get(item_id)]
    if len(top) < limit:
        used = set(selected)
        for item in unique:
            if item.id not in used:
                top.append(item)
                used.add(item.id)
                if len(top) >= limit:
                    break

    parts = ["## 场景推荐", ""]
    parts.append(
        f"下面推荐 {len(top)} 个更适合「{scenario or '通用'}」的反诈案例。"
        f"我优先看它们是否便于{audience or '目标人群'}理解、是否具备清晰的风险信号，"
        "以及资料里能否支撑具体的提醒重点。"
    )
    parts.append("")
    for i, item in enumerate(top, 1):
        title = title_with_family(item)
        display = "、".join(item.entry_channels) if item.entry_channels else "案例讲解、风险提示"
        feature_text = "；".join(item.key_methods) or item.summary
        feature_text = _short_text(feature_text, 150)
        summary = _short_text(item.summary, 120)
        boundary = _recommendation_boundary(item, scenario)
        parts.append(f"### {i}. {title}")
        parts.append(
            f"{title}属于{item.ccl2023_category}"
            f"{f'，级别为{item.risk_level}' if item.risk_level else ''}。"
            f"它适合放在「{scenario or '通用'}」里，是因为资料中明确呈现了“{display}”等风险触点，"
            f"听众可以先通过讲解了解案情背景，再围绕诈骗入口、话术和转账节点进行讨论。"
            f"{feature_text or summary}"
            f"{boundary}"
        )
        parts.append("")
    parts.append(
        "总体上，这类推荐更适合做成“先看案例、再拆话术、最后练判断”的活动结构："
        "先用案情片段建立代入感，再由讲解员补充风险信号和处置建议，互动环节只写资料能够支撑的部分。"
    )

    selection_reason = f"共推荐 {len(top)} 个案例，排序依据：模型智能选择"
    cards = []
    for item in top:
        card = enriched_item_card(item)
        card["reason_tags"] = _item_reason_tags(item, scenario)
        cards.append(card)

    evidence = []
    for item in top:
        evidence.append(
            {
                "type": "inferred",
                "claim": "推荐排序",
                "basis": f"scenario={scenario}, risk_level={item.risk_level}",
                "item_id": item.id,
            }
        )

    return AgentResult(
        task_type=TaskType.RECOMMENDATION,
        answer="\n".join(parts),
        items=cards,
        evidence=evidence,
        selection_reason=selection_reason,
        mode="local",
        confidence=0.7,
        warnings=[] if top else [f"未找到适合「{scenario or '通用'}」的案例，建议放宽条件"],
    )


def handle_lecture(agent: Agent, analysis) -> AgentResult:
    """LECTURE_PLAN: recommendation sub-pipeline + lecture template."""
    rec_analysis = analysis
    if analysis.item_count == 1:
        rec_analysis = replace(analysis, retrieval_count=max(analysis.retrieval_count, 5))

    rec = handle_recommend(agent, rec_analysis)
    if analysis.item_count == 1 and rec.items:
        selected_index, selected_reason = _select_exhibition_core_item(
            rec.items,
            scene=analysis.scenario or "反诈宣传",
            audience=analysis.audience or "公众",
            time_budget=analysis.time_budget or "待定",
        )
        rec.items = [rec.items[selected_index]]
        if rec.evidence:
            rec.evidence = [rec.evidence[min(selected_index, len(rec.evidence) - 1)]]
        rec.selection_reason = (
            selected_reason
            or f"先筛出 {max(analysis.retrieval_count, 5)} 个候选，再确定 1 个核心案例"
        )

    rec.task_type = TaskType.LECTURE_PLAN

    scene = analysis.scenario or "反诈宣传"
    audience = analysis.audience or "公众"
    time_budget = analysis.time_budget or "待定"

    template_items = []
    for item_data in rec.items:
        display_str = "、".join(item_data.get("entry_channels", ["展板"]))
        item_title = item_data["title"]
        category_name = item_data.get("ccl2023_category") or ""
        display_title = (
            f"{item_title}（{category_name}）"
            if category_name and category_name not in item_title
            else item_title
        )
        template_items.append(
            {
                "display_title": display_title,
                "display_str": display_str,
                "summary": item_data["summary"],
            }
        )

    rec.answer = render_template(
        "lecture_plan.md.j2",
        scene=scene,
        audience=audience,
        time_budget=time_budget,
        items=template_items,
    ).strip()
    rec.warnings.append("展示方案由模板辅助整理，互动环节与物料建议待人工补充。")
    return rec


# ---------------------------------------------------------------------------
# Helpers used by handlers
# ---------------------------------------------------------------------------


def _describe_filters(category: str, province: str, level: str) -> str:
    parts = [p for p in [province, level, category] if p]
    return "".join(parts) or ""


def _short_text(text: str, limit: int) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if len(text) <= limit:
        return text if text.endswith(("。", "！", "？")) else f"{text}。"
    clipped = text[:limit].rstrip("，、；;：: ")
    return f"{clipped}。"


def _recommendation_boundary(item, scene: str) -> str:
    forms = "、".join(item.entry_channels) if item.entry_channels else ""
    scene_text = normalize_text(scene)
    if "亲子" in scene_text:
        if any(key in forms for key in ("电话", "短信", "微信", "App", "二维码", "不明链接")):
            return "活动设计上可以保留家庭情景问答，但要避免公开真实隐私信息或让未成年人接触真实转账流程。"
        return "需要注意的是，资料更支持案例讲解和轻量讨论，不宜擅自扩展成复杂演练。"
    if "社区" in scene_text:
        return "落地时建议控制时长，把互动设计成风险信号问答或处置流程复盘，方便不同年龄居民参与。"
    if "校园" in scene_text:
        return "用于校园时可配合展板、讲解卡和判断题，互动部分应以安全、低门槛、易组织为准。"
    if "企业" in scene_text:
        return "用于企业培训时要突出核验流程和转账审批边界，不宜把资料外的处置规定写成确定结论。"
    return "具体活动可以从资料明确写到的诈骗入口、风险信号和处置建议出发，避免加入没有依据的细节。"


def _item_reason_tags(item, scenario: str = "") -> list[str]:
    tags = []
    lvl = item.risk_level
    if lvl == "极高":
        tags.append("极高风险")
    elif lvl == "高":
        tags.append("高风险")
    elif lvl == "中":
        tags.append("中风险")
    if item.entry_channels:
        forms = item.entry_channels[:3]
        tags.append(f"📐 {'·'.join(forms)}")
    if scenario and scenario_match_score(item, scenario) >= 4:
        tags.append(f"🎯 {scenario}")
    return tags


def _parse_llm_selection(response: str, limit: int) -> list[str]:
    text = response.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(text[start : end + 1])
        selected = payload.get("selected", [])
        if isinstance(selected, list):
            return [str(s) for s in selected[:limit] if s]
    return []


def _select_exhibition_core_item(candidate_items, scene, audience, time_budget):
    if not candidate_items:
        return 0, ""
    if len(candidate_items) == 1:
        return 0, "用户要求 1 个核心案例，当前候选仅 1 个。"
    if not settings.ai_api_key:
        return 0, f"先筛出 {len(candidate_items)} 个候选，再按当前排序取第 1 个核心案例。"

    from ..service.http_client import chat_completion

    candidate_lines = []
    for index, item in enumerate(candidate_items, 1):
        meta = " · ".join(
            part
            for part in [
                str(item.get("ccl2023_category") or ""),
                str(item.get("custom_subcategory") or ""),
                str(item.get("risk_level") or ""),
            ]
            if part
        )
        entry_channels = "、".join(item.get("entry_channels") or [])
        summary = str(item.get("summary") or "").strip()
        candidate_lines.append(
            f"{index}. {item.get('title', '')}\n"
            f"   信息：{meta or '无'}\n"
            f"   入口渠道：{entry_channels or '未标注'}\n"
            f"   简介：{summary[:120]}"
        )

    system_prompt = (
        "你是反诈宣传策划顾问。现在有 5 个候选案例，需要为一次宣传活动只选出 1 个最适合作为核心案例。"
        "请综合场景、受众、时间预算、风险等级、提醒价值和互动潜力判断。"
        "只输出 JSON，不要输出解释文字，不要输出 Markdown。\n"
        '{"selected_index": 1, "reason": "一句中文理由"}'
    )
    user_prompt = (
        f"场景：{scene}\n"
        f"受众：{audience}\n"
        f"时间预算：{time_budget}\n\n"
        "候选案例：\n" + "\n\n".join(candidate_lines)
    )
    try:
        raw = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        match = re.search(r"\{[\s\S]*\}", raw or "")
        if not match:
            raise ValueError("no json")
        data = json.loads(match.group(0))
        selected_index = int(data.get("selected_index", 1)) - 1
        if not 0 <= selected_index < len(candidate_items):
            raise ValueError("index out of range")
        reason = str(data.get("reason") or "").strip()
        if reason:
            reason = f"先筛出 {len(candidate_items)} 个候选，再由模型选定 1 个核心案例：{reason}"
        else:
            reason = f"先筛出 {len(candidate_items)} 个候选，再由模型选定 1 个核心案例。"
        return selected_index, reason
    except Exception:
        return 0, f"先筛出 {len(candidate_items)} 个候选，再按当前排序取第 1 个核心案例。"


def _call_transform_model(transform_type: str, context: str, query: str) -> str:
    from ..service.http_client import chat_completion

    system_prompt = TRANSFORM_PROMPTS.get(transform_type, TRANSFORM_PROMPTS[DEFAULT_TRANSFORM_TYPE])
    max_tokens = TRANSFORM_MAX_TOKENS.get(
        transform_type, TRANSFORM_MAX_TOKENS[DEFAULT_TRANSFORM_TYPE]
    )
    return chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"任务要求：{query}\n\n反诈案例资料：\n{context}"},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
