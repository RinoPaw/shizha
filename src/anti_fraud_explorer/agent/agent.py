"""Top-level agent: intent classification -> query analysis -> dispatch."""

import json
import logging
import re
from dataclasses import replace
from typing import Any

from ..config import settings
from ..domain.dataset import KnowledgeBase
from ..service.item_cards import enriched_item_card, source_payload, title_with_family
from ..service.search import (
    LEXICAL_MIN_SCORE,
    normalize_search_query,
    rank_lexical,
    search_items,
    tokenize,
)
from ..service.retriever import QueryAnalyzer
from ..service.scenario_evidence import scenario_is_hard_match, scenario_match_score
from ..text import normalize_text
from ..prompts import SUBSEQUENT_TURN_SYSTEM_PROMPT

from .models import (
    AgentDecision,
    AgentResult,
    TaskType,
    task_type_from_str,
)
from .router import IntentRouter
from .formatting import (
    context_title_keywords,
    format_context_item_for_llm,
    items_to_llm_context,
    items_to_title_context,
)

LOGGER = logging.getLogger(__name__)

MAX_SEARCH_ROUNDS_PER_TURN = 2
INITIAL_TITLE_CANDIDATE_LIMIT = 100
INITIAL_TITLE_CONTEXT_LIMIT = 100
DETAIL_SEARCH_LIMIT_PER_QUERY = 8


def _normalize_answer_text(value: Any) -> str:
    """Clean line endings and whitespace while preserving Markdown line breaks."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


class Agent:
    """Top-level agent: intent classification -> query analysis -> dispatch."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.router = IntentRouter()
        self.query_analyzer = QueryAnalyzer(kb)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def dispatch(
        self,
        query: str,
        category: str = "",
        include_speech: bool = True,
        context: dict | None = None,
    ) -> AgentResult:
        result = None
        speech_text = ""
        for event in self.dispatch_stream(
            query, category, include_speech=include_speech, context=context
        ):
            if isinstance(event, AgentResult):
                result = event
            elif isinstance(event, dict) and event.get("type") == "speech":
                speech_text = str(event.get("text") or "")
        if result and speech_text and not result.speech:
            result = replace(result, speech=speech_text)
        return result

    def dispatch_stream(
        self,
        query: str,
        category: str = "",
        include_speech: bool = True,
        context: dict | None = None,
    ):
        query = query.strip()
        if not query:
            yield AgentResult(
                task_type=TaskType.FACT_QA,
                answer="请先输入问题。",
                speech="请先输入问题。" if include_speech else "",
                mode="empty",
            )
            return

        has_legacy_context = bool(
            context and (context.get("question") or context.get("items") or context.get("answer"))
        )
        is_first = not context or (context.get("turn_count", 0) == 0 and not has_legacy_context)
        if is_first:
            yield from self._dispatch_subsequent_turn(query, category, include_speech, context=None)
            return

        yield from self._dispatch_subsequent_turn(query, category, include_speech, context)

    # ------------------------------------------------------------------ #
    # Core dispatch loop
    # ------------------------------------------------------------------ #

    def _dispatch_subsequent_turn(
        self,
        query: str,
        category: str,
        include_speech: bool,
        context: dict | None,
    ):
        from ..ai import describe_model_error

        context = context or {}
        yield self._progress_event("search", "检索资料", "按原问题筛选候选标题。")
        title_candidates, initial_total_count, initial_note = self._search_initial_candidates(
            query, category, context
        )
        search_rounds_used = 1
        used_queries: list[str] = [query]
        detailed_items: list[Any] = []
        collected_items = self._merge_items(title_candidates, detailed_items)
        total_count = initial_total_count
        retrieval_note = initial_note
        warnings: list[str] = []

        if not settings.ai_api_key:
            yield self._progress_event(
                "generate", "整理结论", "未配置模型 Key，使用本地案例资料直接回答。"
            )
            result, decision = self._subsequent_fallback_result(
                query=query,
                context=context,
                collected_items=collected_items,
                used_queries=used_queries,
                total_count=total_count,
                warnings=warnings,
                reason="未配置 AI_API_KEY，服务器使用本地检索资料回答。",
                mode="local_context",
                planner="local_no_key",
            )
            yield from self._stream_completed_result(result, decision, include_speech, query=query)
            return

        while True:
            if search_rounds_used >= MAX_SEARCH_ROUNDS_PER_TURN:
                yield self._progress_event("generate", "思考回答", "资料已齐，正在组织回答。")
            else:
                yield self._progress_event(
                    "classify", "理解问题", "结合上下文和候选标题，判断是否需要精查。"
                )

            try:
                payload = self._call_subsequent_turn_model(
                    query=query,
                    context=context,
                    title_candidates=title_candidates,
                    detailed_items=detailed_items,
                    search_rounds_used=search_rounds_used,
                    retrieval_note=retrieval_note,
                )
            except Exception as exc:
                warning = describe_model_error(exc)
                LOGGER.warning("Subsequent-turn LLM decision unavailable: %s", warning)
                warnings.append(warning)
                result, decision = self._subsequent_fallback_result(
                    query=query,
                    context=context,
                    collected_items=collected_items,
                    used_queries=used_queries,
                    total_count=total_count,
                    warnings=warnings,
                )
                yield from self._stream_completed_result(
                    result, decision, include_speech, query=query
                )
                return

            action = self._payload_action(payload)
            answer = _normalize_answer_text(payload.get("answer") or "")
            search_queries = self._payload_str_list(payload.get("search_queries"))

            if action == "answer" and answer:
                yield self._progress_event("generate", "思考回答", "资料已齐，正在组织回答。")
                result, decision = self._subsequent_answer_result(
                    payload=payload,
                    answer=answer,
                    context=context,
                    collected_items=collected_items,
                    used_queries=used_queries,
                    total_count=total_count,
                    warnings=warnings,
                )
                yield from self._stream_completed_result(
                    result, decision, include_speech, query=query
                )
                return

            if search_rounds_used >= MAX_SEARCH_ROUNDS_PER_TURN:
                warnings.append("搜索预算已用尽，已基于现有上下文兜底回答。")
                result, decision = self._subsequent_fallback_result(
                    query=query,
                    context=context,
                    collected_items=collected_items,
                    used_queries=used_queries,
                    total_count=total_count,
                    warnings=warnings,
                )
                yield from self._stream_completed_result(
                    result, decision, include_speech, query=query
                )
                return

            if not search_queries:
                search_queries = self._fallback_queries_from_context(context)

            if not search_queries:
                warnings.append("模型未给出答案或检索词，且上下文中没有可兜底检索的案例。")
                result, decision = self._subsequent_fallback_result(
                    query=query,
                    context=context,
                    collected_items=collected_items,
                    used_queries=used_queries,
                    total_count=total_count,
                    warnings=warnings,
                )
                yield from self._stream_completed_result(
                    result, decision, include_speech, query=query
                )
                return

            search_rounds_used += 1
            used_queries.extend(q for q in search_queries if q not in used_queries)
            yield self._progress_event(
                "search", "检索资料", f"精查资料：{'、'.join(search_queries[:4])}"
            )
            new_items, total = self._search_subsequent_items(search_queries, category)
            total_count += total
            if total == 0:
                warnings.append(f"检索词未命中资料库：{'、'.join(search_queries)}")
            detailed_items = self._merge_items(detailed_items, new_items)
            collected_items = self._merge_items(title_candidates, detailed_items)
            retrieval_note = (
                f"服务器已根据模型查询词补充详情检索：{'、'.join(search_queries)}。"
                if new_items
                else f"服务器根据模型查询词没有查询到高相关结果：{'、'.join(search_queries)}。"
            )

    # ------------------------------------------------------------------ #
    # Model call helpers
    # ------------------------------------------------------------------ #

    def _call_subsequent_turn_model(
        self,
        query: str,
        context: dict,
        title_candidates: list[Any],
        detailed_items: list[Any],
        search_rounds_used: int,
        retrieval_note: str = "",
    ) -> dict[str, Any]:
        from ..service.http_client import chat_completion
        from .planner import agent_planner_extra_options, extract_json_object

        raw = chat_completion(
            self._build_subsequent_turn_messages(
                query=query,
                context=context,
                title_candidates=title_candidates,
                detailed_items=detailed_items,
                search_rounds_used=search_rounds_used,
                retrieval_note=retrieval_note,
            ),
            temperature=0.2,
            max_tokens=2200,
            extra_options=agent_planner_extra_options(),
        )
        payload = json.loads(extract_json_object(raw))
        if not isinstance(payload, dict):
            raise ValueError("Subsequent-turn model did not return a JSON object")
        return payload

    def _build_subsequent_turn_messages(
        self,
        query: str,
        context: dict,
        title_candidates: list[Any],
        detailed_items: list[Any],
        search_rounds_used: int,
        retrieval_note: str = "",
    ) -> list[dict[str, str]]:
        remaining = max(MAX_SEARCH_ROUNDS_PER_TURN - search_rounds_used, 0)
        history_text = self._format_history_for_llm(context)
        title_text = (
            items_to_title_context(
                title_candidates[:INITIAL_TITLE_CONTEXT_LIMIT], len(title_candidates)
            )
            if title_candidates
            else "无"
        )
        detail_text = (
            items_to_llm_context(detailed_items[:30], len(detailed_items))
            if detailed_items
            else "无"
        )

        user_prompt = (
            f"搜索状态：已连续搜索 {search_rounds_used} 轮，剩余 {remaining} 轮。\n\n"
            f"对话历史（最多最近五轮）：\n{history_text}\n\n"
            f"检索说明：{retrieval_note or '服务器尚未提供额外检索说明。'}\n\n"
            f"第1轮标题候选（仅含标题和基础元数据，不等同于完整事实依据）：\n{title_text}\n\n"
            f"第2轮详情资料（模型查询后由服务器补充）：\n{detail_text}\n\n"
            f"当前问题：{query}"
        )
        return [
            {"role": "system", "content": SUBSEQUENT_TURN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _format_history_for_llm(self, context: dict) -> str:
        history = context.get("history")
        if not isinstance(history, list) or not history:
            q = normalize_text(context.get("question") or "")
            a = normalize_text(context.get("answer") or "")
            items = context.get("items") or []
            lines = []
            if q:
                lines.append(f"上一轮问：{q}")
            if a:
                lines.append(f"上一轮答：{a[:300]}")
            if isinstance(items, list) and items:
                title_text = "、".join(
                    normalize_text(item.get("title") if isinstance(item, dict) else str(item))
                    for item in items[:8]
                )
                if title_text:
                    lines.append(f"上一轮涉及案例：{title_text}")
            return "\n".join(lines) if lines else "无"

        blocks: list[str] = []
        for idx, turn in enumerate(history[-5:], 1):
            if not isinstance(turn, dict):
                continue
            q = normalize_text(turn.get("q") or "")
            a = normalize_text(turn.get("a") or "")
            lines = [f"第{idx}轮"]
            if q:
                lines.append(f"问：{q}")
            if a:
                lines.append(f"答：{a[:300]}")

            items_full = turn.get("items_full") or []
            if isinstance(items_full, list) and items_full:
                lines.append("涉及案例：")
                for item in items_full[:8]:
                    item_text = format_context_item_for_llm(item)
                    if item_text:
                        lines.append(item_text)
            else:
                titles = turn.get("items") or []
                if isinstance(titles, list) and titles:
                    title_text = "、".join(str(title) for title in titles[:8] if str(title).strip())
                    if title_text:
                        lines.append(f"涉及案例：{title_text}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks) if blocks else "无"

    # ------------------------------------------------------------------ #
    # Search / merge / fallback helpers
    # ------------------------------------------------------------------ #

    def _search_initial_candidates(
        self,
        query: str,
        category: str,
        context: dict | None = None,
    ) -> tuple[list[Any], int, str]:
        analysis = self.query_analyzer.analyze(query, context=context)
        search_query = normalize_search_query(analysis.rewritten_query or query)
        lowered_query = search_query or normalize_text(query).lower()
        context_items = self._context_items(context or {})
        contextual_items = self._contextual_initial_candidates(context_items, category)
        if not lowered_query:
            if contextual_items:
                return (
                    contextual_items,
                    len(contextual_items),
                    self._contextual_candidate_note(contextual_items),
                )
            return [], 0, "服务器根据原问题没有查询到候选标题；你可以直接回答或组织关键词重新查询。"

        candidates = [
            item for item in self.kb.items if not category or item.ccl2023_category == category
        ]
        structured_items = self._structured_initial_candidates(
            analysis, limit=INITIAL_TITLE_CANDIDATE_LIMIT
        )
        ranked = rank_lexical(candidates, lowered_query, tokenize(search_query))
        scenario = analysis.scenario
        lexical_items = [
            item
            for score, item in ranked
            if score >= LEXICAL_MIN_SCORE
            and (not scenario or scenario_is_hard_match(item, scenario))
        ][:INITIAL_TITLE_CANDIDATE_LIMIT]

        title_candidates = self._merge_items(
            contextual_items,
            self._merge_items(structured_items, lexical_items),
        )[:INITIAL_TITLE_CANDIDATE_LIMIT]
        if not title_candidates:
            return (
                [],
                0,
                (
                    "服务器根据原问题没有查询到候选标题；"
                    "如果历史上下文不足，你可以发送 search_queries 重新组织关键词查询。"
                ),
            )

        note_prefix = (
            "服务器已附带历史案例相关候选，是否采用由你根据对话判断；" if contextual_items else ""
        )
        note = (
            note_prefix
            + f"服务器已完成第 1 轮标题候选检索，提供 {len(title_candidates)} 个候选案例的标题和基础元数据；"
            "这些候选用于判断下一步，不包含完整事实依据。"
        )
        return title_candidates, len(title_candidates), note

    def _contextual_initial_candidates(self, context_items: list[Any], category: str) -> list[Any]:
        if not context_items:
            return []
        categories = {item.ccl2023_category for item in context_items if item.ccl2023_category}
        if category:
            categories.add(category)
        title_keywords = context_title_keywords(context_items)
        context_ids = {item.id for item in context_items}
        forms = {form for item in context_items for form in item.entry_channels}

        scored: list[tuple[int, str, Any]] = []
        for item in self.kb.items:
            score = 0
            if item.id in context_ids:
                score += 12
            if categories and item.ccl2023_category in categories:
                score += 6
            if any(
                keyword
                and (
                    keyword in item.title
                    or keyword in item.ccl2023_category
                    or keyword in item.custom_subcategory
                )
                for keyword in title_keywords
            ):
                score += 10
            if forms and any(form in forms for form in item.entry_channels):
                score += 2
            if item.risk_level == "极高":
                score += 3
            elif item.risk_level == "高":
                score += 2
            if score <= 0:
                continue
            scored.append((score, item.title, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:INITIAL_TITLE_CANDIDATE_LIMIT]]

    def _contextual_candidate_note(self, items: list[Any]) -> str:
        return (
            f"服务器根据历史案例附带 {len(items)} 个相关候选标题；"
            "是否承接上一轮由你根据对话历史判断。如果需要新增案例详情，请在 search_queries 中给出案例标题。"
        )

    def _structured_initial_candidates(self, analysis, limit: int) -> list[Any]:
        query = analysis.original_query
        scenario = analysis.scenario
        wants_recommendation = bool(
            re.search(r"推荐|适合|哪些|有哪些|找|筛选|展示|宣传|宣讲|班会|活动|互动|亲子", query)
        )
        if not wants_recommendation or not scenario:
            return []

        scored: list[tuple[int, str, Any]] = []
        for item in self.kb.items:
            score = 0
            if scenario:
                scenario_score = scenario_match_score(item, scenario)
                if scenario_score < 4:
                    continue
                score += scenario_score
            if re.search(r"展示|宣传|宣讲|班会", query) and item.entry_channels:
                score += 4
            if re.search(r"活动|互动", query) and item.entry_channels:
                score += 4
            if item.risk_level == "极高":
                score += 4
            elif item.risk_level == "高":
                score += 3
            elif item.risk_level == "中":
                score += 1
            if item.entry_channels:
                score += min(len(item.entry_channels), 3)
            if score <= 0:
                continue
            scored.append((score, item.title, item))

        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:limit]]

    def _search_subsequent_items(self, queries: list[str], category: str) -> tuple[list[Any], int]:
        items: list[Any] = []
        seen: set[str] = set()
        total = 0
        for search_query in queries[:6]:
            exact_items = self._exact_items_for_query(search_query, category)
            if exact_items:
                result = exact_items
                result_total = len(exact_items)
            else:
                result, result_total = search_items(
                    self.kb,
                    query=search_query,
                    category=category,
                    limit=DETAIL_SEARCH_LIMIT_PER_QUERY,
                )
            total += result_total
            for item in result:
                if item.id in seen:
                    continue
                seen.add(item.id)
                items.append(item)
        return items, total

    def _exact_items_for_query(self, query: str, category: str) -> list[Any]:
        ref = normalize_text(query)
        if not ref:
            return []
        matches: list[Any] = []
        seen: set[str] = set()
        for item in self.kb.items:
            if category and item.ccl2023_category != category:
                continue
            if ref in (item.id, item.title, title_with_family(item), item.ccl2023_category):
                if item.id not in seen:
                    seen.add(item.id)
                    matches.append(item)
        return matches

    def _merge_items(self, existing: list[Any], incoming: list[Any]) -> list[Any]:
        merged = list(existing)
        seen = {item.id for item in merged if hasattr(item, "id")}
        for item in incoming:
            item_id = getattr(item, "id", "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item)
        return merged

    # ------------------------------------------------------------------ #
    # Answer / fallback result builders
    # ------------------------------------------------------------------ #

    def _subsequent_answer_result(
        self,
        payload: dict[str, Any],
        answer: str,
        context: dict,
        collected_items: list[Any],
        used_queries: list[str],
        total_count: int,
        warnings: list[str],
    ) -> tuple[AgentResult, AgentDecision]:
        from .planner import clamp_float

        task_type = task_type_from_str(str(payload.get("task_type") or TaskType.FACT_QA.value))
        confidence = clamp_float(payload.get("confidence"), default=0.78)
        reason = normalize_text(payload.get("reason") or "模型根据最近五轮上下文完成回答。")
        display_pool = self._merge_items(self._context_items(context), collected_items)
        raw_display_items = payload.get("display_items")
        if isinstance(raw_display_items, list):
            display_refs = self._payload_str_list(raw_display_items)
            display_items = self._select_display_items(
                display_refs, display_pool, fallback_items=[]
            )
        else:
            display_items = self._select_display_items(
                [], display_pool, fallback_items=collected_items
            )
        cards = [enriched_item_card(item) for item in display_items[:8]]
        sources = [source_payload(item) for item in display_items[:5]]

        decision = AgentDecision(
            task_type=task_type,
            confidence=confidence,
            needs_retrieval=bool(used_queries),
            needs_llm=True,
            reason=reason,
            mode="llm_context",
            warnings=list(warnings),
            planner="llm_decision",
            search_queries=list(used_queries),
        )
        result = AgentResult(
            task_type=task_type,
            answer=answer,
            items=cards,
            sources=sources,
            mode="llm_context",
            confidence=confidence,
            warnings=list(warnings),
            total_count=len(display_items),
        )
        return result, decision

    def _select_display_items(
        self, refs: list[str], pool: list[Any], fallback_items: list[Any]
    ) -> list[Any]:
        selected: list[Any] = []
        seen: set[str] = set()

        def add(item: Any) -> None:
            item_id = getattr(item, "id", "")
            if item_id and item_id not in seen:
                seen.add(item_id)
                selected.append(item)

        for ref in refs:
            for item in pool:
                if self._item_matches_ref(item, ref):
                    add(item)
                    break
        if not selected:
            for item in fallback_items[:8]:
                add(item)
        return selected

    def _item_matches_ref(self, item: Any, ref: str) -> bool:
        ref = normalize_text(ref)
        if not ref:
            return False
        return ref in (item.id, item.title, title_with_family(item), item.ccl2023_category)

    def _subsequent_fallback_result(
        self,
        query: str,
        context: dict,
        collected_items: list[Any],
        used_queries: list[str],
        total_count: int,
        warnings: list[str],
        reason: str = "后续轮模型未能给出可用 answer，服务器使用上下文兜底。",
        mode: str = "llm_context_fallback",
        planner: str = "llm_decision",
    ) -> tuple[AgentResult, AgentDecision]:
        display_items = collected_items[:5] or self._context_items(context)[:5]
        task_type = _fallback_task_type(query)
        if display_items:
            lead = (
                "我先按本地案例库推荐几条："
                if task_type is TaskType.RECOMMENDATION
                else "我先基于当前已有资料回答："
            )
            lines = [lead]
            for item in display_items[:3]:
                meta = " | ".join(
                    part
                    for part in [item.ccl2023_category, item.risk_level, item.custom_subcategory]
                    if part
                )
                lines.append(f"- **{title_with_family(item)}**：{meta}")
                if item.summary:
                    lines.append(f"  {item.summary[:140]}")
            if used_queries:
                lines.append(f"\n已尝试检索：{'、'.join(used_queries)}。")
            answer = "\n".join(lines)
        else:
            answer = "这轮我没有拿到足够可靠的资料来回答。可以换成更具体的案例名、骗局类型或地区再问一次。"

        decision = AgentDecision(
            task_type=task_type,
            confidence=0.4,
            needs_retrieval=bool(used_queries),
            needs_llm=False,
            reason=reason,
            mode=mode,
            warnings=list(warnings),
            planner=planner,
            search_queries=list(used_queries),
        )
        result = AgentResult(
            task_type=task_type,
            answer=answer,
            items=[enriched_item_card(item) for item in display_items],
            sources=[source_payload(item) for item in display_items[:5]],
            mode=mode,
            confidence=0.4,
            warnings=list(warnings),
            total_count=len(display_items),
        )
        return result, decision

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def _payload_action(self, payload: dict[str, Any]) -> str:
        action = str(payload.get("action") or "").strip().lower()
        if action in {"answer", "search"}:
            return action
        if self._payload_str_list(payload.get("search_queries")):
            return "search"
        return "answer"

    def _payload_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = normalize_text(str(item))
            if text and text not in result:
                result.append(text)
        return result

    def _fallback_queries_from_context(self, context: dict) -> list[str]:
        queries: list[str] = []
        for item in self._context_item_payloads(context):
            title = normalize_text(item.get("title") or "")
            if title and title not in queries:
                queries.append(title)
            category = normalize_text(item.get("ccl2023_category") or "")
            if category and category not in queries:
                queries.append(category)
            if len(queries) >= 4:
                break
        if queries:
            return queries
        for item in context.get("items") or []:
            if not isinstance(item, dict):
                continue
            title = normalize_text(item.get("title") or "")
            if title and title not in queries:
                queries.append(title)
            if len(queries) >= 4:
                break
        return queries

    def _context_item_payloads(self, context: dict) -> list[dict]:
        payloads: list[dict] = []
        seen: set[str] = set()
        for item in context.get("items_full") or []:
            if isinstance(item, dict):
                key = str(item.get("id") or item.get("title") or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    payloads.append(item)
        history = context.get("history")
        if isinstance(history, list):
            for turn in history[-5:]:
                if not isinstance(turn, dict):
                    continue
                for item in turn.get("items_full") or []:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("id") or item.get("title") or "").strip()
                    if key and key not in seen:
                        seen.add(key)
                        payloads.append(item)
        return payloads

    def _context_items(self, context: dict) -> list[Any]:
        items: list[Any] = []
        seen: set[str] = set()
        for payload in self._context_item_payloads(context):
            item = self._resolve_context_item(payload)
            if item is None or item.id in seen:
                continue
            seen.add(item.id)
            items.append(item)
        return items

    def _resolve_context_item(self, payload: dict) -> Any | None:
        item_id = str(payload.get("id") or "").strip()
        if item_id:
            item = self.kb.get(item_id)
            if item is not None:
                return item
        title = normalize_text(payload.get("title") or "")
        if not title:
            return None
        for item in self.kb.items:
            if item.title == title or title_with_family(item) == title:
                return item
        return None

    # ------------------------------------------------------------------ #
    # Speech (delegated to ai.spoken)
    # ------------------------------------------------------------------ #

    def _ensure_speech(self, result: AgentResult, query: str = "") -> AgentResult:
        if result.speech:
            return result
        from ..ai.spoken import build_spoken_answer

        speech = build_spoken_answer(
            result.answer,
            question=query,
            sources=self._speech_source_items(result),
        )
        return replace(result, speech=speech)

    def _speech_source_items(self, result: AgentResult) -> list[Any]:
        source_items: list[Any] = []
        seen: set[str] = set()
        for payload in [*result.sources, *result.items]:
            item_id = payload.get("id") if isinstance(payload, dict) else ""
            if not item_id or item_id in seen:
                continue
            item = self.kb.get(item_id)
            if item is None:
                continue
            seen.add(item_id)
            source_items.append(item)
        return source_items

    def _stream_completed_result(
        self,
        result: AgentResult,
        decision: AgentDecision,
        include_speech: bool,
        query: str = "",
    ):
        if not include_speech or result.speech:
            yield with_agent_decision(result, decision, include_speech)
            return
        yield with_agent_decision(replace(result, speech=""), decision, include_speech)
        if result.answer:
            result = self._ensure_speech(result, query=query)
            yield {"type": "speech", "text": result.speech}

    def _progress_event(self, step: str, title: str, detail: str) -> dict[str, str]:
        return {"type": "progress", "step": step, "title": title, "detail": detail}


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #


def _fallback_task_type(query: str) -> TaskType:
    text = normalize_text(query)
    if re.search(r"推荐|适合|筛选|找.*案例", text):
        return TaskType.RECOMMENDATION
    if re.search(r"比较|对比|区别|不同", text):
        return TaskType.COMPARISON
    if re.search(r"班会|课堂|任务单|教学|宣教任务", text):
        return TaskType.STUDY_TASK
    if re.search(r"策划|宣传角|宣传栏|方案|流程", text):
        return TaskType.LECTURE_PLAN
    if re.search(r"改写|改成|口播|文案|海报|短视频|提醒稿", text):
        return TaskType.CONTENT_TRANSFORM
    return TaskType.FACT_QA


def with_agent_decision(
    result: AgentResult,
    decision: AgentDecision,
    include_speech: bool,
) -> AgentResult:
    result = replace(result, decision=decision.to_payload())
    if not include_speech:
        result = replace(result, speech="")
    return result


def normalize_query_with_pinyin_anchor(kb: KnowledgeBase, query: str, category: str = "") -> str:
    query = normalize_text(query)
    if not query:
        return query
    from ..service.search import search_items_pinyin

    for item in search_items_pinyin(kb, query):
        if category and item.ccl2023_category != category:
            continue
        corrected = replace_homophone_span(query, item.title)
        if corrected != query:
            return corrected
    return query


def replace_homophone_span(text: str, canonical: str) -> str:
    if canonical in text:
        return text
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return text
    canonical_py = "".join(lazy_pinyin(canonical))
    if not canonical_py:
        return text
    for start in range(len(text)):
        for end in range(start + 2, len(text) + 1):
            span = text[start:end]
            if "".join(lazy_pinyin(span)) == canonical_py:
                return f"{text[:start]}{canonical}{text[end:]}"
    return text
