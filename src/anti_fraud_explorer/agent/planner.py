"""Model-based intent planner for the anti-fraud agent."""

import json
from typing import Any

from ..config import settings
from .models import AgentDecision, TaskType, task_type_from_str
from ..domain.dataset import KnowledgeBase
from ..text import normalize_text


def call_agent_planner_model(
    query: str,
    kb: KnowledgeBase,
    category: str = "",
    context: dict | None = None,
) -> AgentDecision:
    if not settings.ai_agent_planner:
        raise RuntimeError("AI_AGENT_PLANNER is disabled")
    if not settings.ai_api_key:
        raise RuntimeError("AI_API_KEY is not configured")
    from ..service.http_client import chat_completion

    content = chat_completion(
        build_agent_planner_messages(query, kb, category, context),
        temperature=0.3,
        extra_options=agent_planner_extra_options(),
    )
    plan = json.loads(extract_json_object(content))
    return decision_from_planner_payload(plan, query, kb, category)


def build_agent_planner_messages(
    query: str,
    kb: KnowledgeBase,
    category: str = "",
    context: dict | None = None,
) -> list[dict[str, str]]:
    categories = "、".join(category.name for category in kb.categories[:12])
    user_content = (
        f"用户问题：{query}\n"
        f"显式分类筛选：{category or '无'}\n"
        f"资料库概况：{len(kb.items)} 个反诈案例，{len(kb.categories)} 类，类别包括：{categories}\n"
    )
    if context and isinstance(context, dict):
        # Multi-turn conversation history (new format)
        history = context.get("history")
        if isinstance(history, list) and history:
            user_content += "\\n对话历史（最近几轮）：\\n"
            for idx, turn in enumerate(history[-5:], 1):
                q = str(turn.get("q") or "").strip()
                a = str(turn.get("a") or "").strip()
                if q:
                    user_content += f"第{idx}轮 - 用户：{q}\\n"
                if a:
                    user_content += f"第{idx}轮 - 助手：{a[:200]}\\n"
            user_content += "请根据对话历史理解本轮问题中的指代（如「它」「这个」「刚才说的」），"
            user_content += "将其解析为具体项目名，并输出合理的 search_queries。\\n"
        else:
            # Legacy single-turn context
            prev_question = str(context.get("question") or "").strip()
            prev_items = _context_item_titles(context.get("items"))
            prev_answer = str(context.get("answer") or "").strip()
            if prev_question or prev_items:
                user_content += "\\n上一轮对话上下文：\\n"
                if prev_question:
                    user_content += f"上一轮用户问题：{prev_question}\\n"
                if prev_items:
                    user_content += f"上一轮涉及项目：{'、'.join(prev_items)}\\n"
                user_content += "请根据上下文理解本轮问题中的指代（如「它」「这个」「这份」），"
                user_content += "将其解析为具体项目名，并在 reason 中说明你的理解。\\n"
                if prev_answer:
                    prev_answer_short = prev_answer[:300]
                    user_content += f"上一轮回答概要（仅供参考，请勿在 direct_answer 中复述）：{prev_answer_short}\\n"
    user_content += (
        "\n请输出 JSON："
        '{"task_type":"...", "confidence":0.0, "needs_retrieval":true, '
        '"needs_llm":true, "reason":"...", "direct_answer":"", '
        '"mode":"local", "warnings":[], "search_queries":[]}'
        "\n\nsearch_queries 是你认为检索资料库的最佳关键词列表。"
        "\n规则："
        "\n- 如果问题涉及具体项目名（含上下文指代如「它」），列出项目名作为检索词"
        '\n- 如果问题是浏览/推荐类，列出类别名或特征词（如["刷单返利","学生"]）'
        "\n- comparison 任务列出待对比的各项目名"
        "\n- chitchat / direct_answer 留空数组"
        "\n请确保 search_queries 中的每个词都是可以直接用来检索资料库的。"
    )
    return [
        {
            "role": "system",
            "content": (
                "你是识诈智能体的内部规划器，用户不可见，不直接编造资料。"
                "你的任务只是在后台决定下一步行动，不是扮演最终回答者。"
                "可选任务类型：chitchat, fact_qa, browse_query, comparison, recommendation, "
                "lecture_plan, study_task, content_transform。"
                "可选动作：direct_answer（身份/能力/寒暄/越界说明）、retrieval_tool（查资料库）、"
                "rule_handler（筛选/对比/推荐/策划/教案）、llm_generation（基于检索资料生成）。"
                "任务边界：content_transform 用于把一个或多个反诈案例改写成口播稿、提醒文案、"
                "海报文案、短视频脚本、年轻化版本等成稿内容；"
                "study_task 用于班会、课程、学习单、课堂活动、互动提问等教学设计；"
                "lecture_plan 用于社区宣传角、校园宣传栏、反诈讲座流程、专题宣传方案等展示方案。"
                "如果用户要求为单个案例整理口播稿或提醒稿，优先选择 content_transform，"
                "不要因为出现「讲解」就归为 study_task，也不要因为用于宣传场景就误判。"
                "如果有对话历史上下文，你必须理解其中的指代关系，"
                "例如「它」指代上一轮的项目、要把「它和同类对比」理解为「项目A和同类对比」。"
                "多轮对话时，优先从历史中找到用户真正关心的项目，再生成 search_queries。"
                "请根据用户最终意图自主选择最合适的任务类型和动作。"
                "direct_answer 必须使用用户可见角色「识诈」的口吻回答。"
                "不要在 direct_answer 中提到内部规划器、后台、决策层、路由、JSON、工具选择等实现细节。"
                "只输出 JSON，不要输出 markdown，不要解释。"
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _context_item_titles(items) -> list[str]:
    if not isinstance(items, list):
        return []
    titles: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= 5:
            break
    return titles


def agent_planner_extra_options() -> dict[str, Any]:
    opts: dict[str, Any] = {"response_format": {"type": "json_object"}}
    model = settings.ai_model.lower()
    if any(name in model for name in ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")):
        opts["thinking"] = {"type": "disabled"}
    return opts


def extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Planner did not return JSON: {text[:120]}")
    return text[start : end + 1]


def decision_from_planner_payload(
    payload: dict[str, Any],
    query: str,
    kb: KnowledgeBase,
    category: str = "",
) -> AgentDecision:
    task_type = task_type_from_str(str(payload.get("task_type") or "fact_qa"))
    confidence = clamp_float(payload.get("confidence"), default=0.7)
    needs_retrieval = bool(payload.get("needs_retrieval", task_type is not TaskType.CHITCHAT))
    needs_llm = bool(
        payload.get("needs_llm", task_type in {TaskType.FACT_QA, TaskType.CONTENT_TRANSFORM})
    )
    reason = normalize_text(payload.get("reason") or "模型 planner 已选择下一步行动。")
    direct_answer = normalize_text(payload.get("direct_answer") or "")
    mode = str(payload.get("mode") or "local")
    warnings = payload.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    if task_type is TaskType.CHITCHAT:
        needs_retrieval = False
        needs_llm = False
        mode = "local"
    elif not needs_retrieval and not needs_llm:
        if not warnings:
            warnings = ["模型 planner 判断问题不需要资料库或生成模型处理。"]

    return AgentDecision(
        task_type=task_type,
        confidence=confidence,
        needs_retrieval=needs_retrieval,
        needs_llm=needs_llm,
        reason=reason,
        direct_answer=direct_answer,
        mode=mode,
        warnings=[str(item) for item in warnings],
        planner="model",
        search_queries=_safe_str_list(payload.get("search_queries")),
    )


def _safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clamp_float(value: Any, default: float = 0.7) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
