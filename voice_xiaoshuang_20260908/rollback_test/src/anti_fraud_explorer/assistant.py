"""The single answer pipeline shared by text and future voice transports."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from functools import lru_cache

from .config import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_FIRST_TOKEN_MAX_ATTEMPTS,
    AI_FIRST_TOKEN_TIMEOUT,
    AI_MAX_CONTEXT_CHARS,
    AI_MODEL,
    AI_TIMEOUT,
)
from .dataset import FraudCase, KnowledgeBase, get_knowledge_base, item_to_dict, normalize_text
from .events import EventSequence
from .language import DEFAULT_LOCALE, detect_locale, get_language_profile
from .models import AssistantEvent, ConversationTurn, SearchResponse
from .providers.llm import LLMProvider, OpenAICompatibleLLM
from .search import normalize_search_query, search_items, tokenize
from .sessions import SessionStore

MAX_QUESTION_CHARS = 4000
GREETING_QUESTIONS = frozenset({"你好", "您好", "嗨", "哈喽", "在吗"})
MULTILINGUAL_GREETINGS = {
    "zh-CN-henan": frozenset({"恁好"}),
    "en-US": frozenset({"hello", "hi", "hey", "good morning", "good afternoon"}),
    "ja-JP": frozenset({"こんにちは", "もしもし", "おはよう", "こんばんは"}),
    "ko-KR": frozenset({"안녕하세요", "안녕", "여보세요"}),
}
SHORT_REPLY_MODES = {
    "嗯": "continuation",
    "嗯嗯": "continuation",
    "好": "continuation",
    "好的": "continuation",
    "行": "continuation",
    "可以": "continuation",
    "继续": "continuation",
    "接着说": "continuation",
    "然后呢": "continuation",
    "中": "continuation",
    "等一下": "pause",
    "等等": "pause",
    "先等一下": "pause",
    "停一下": "pause",
}
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
ITEM_COUNT_WORDS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
CATALOGUE_BROWSE_ACTIONS = ("推荐", "有哪些", "哪些", "列举", "浏览", "查找", "搜索", "找几个", "找一些")
CATALOGUE_BROWSE_OBJECTS = ("案例", "资料", "类别", "诈骗", "风险", "手法")
FRAUD_DOMAIN_TERMS = ("诈骗", "反诈", "风险", "转账", "验证码", "共享屏幕", "被骗", "诈骗案例")
MULTILINGUAL_FRAUD_TERMS = {
    "en-US": ("fraud", "scam", "phishing", "transfer", "verification code"),
    "ja-JP": ("詐欺", "フィッシング", "送金", "認証コード"),
    "ko-KR": ("사기", "피싱", "송금", "인증번호"),
}
MULTILINGUAL_SEARCH_GLOSSES = {
    "en-US": (
        (("refund scam", "fake customer service"), "冒充客服"),
        (("investment scam", "fake investment"), "虚假投资理财"),
        (("task scam", "rebate scam"), "刷单返利"),
        (("impersonation scam", "police scam"), "冒充公检法"),
        (("phishing",), "网络钓鱼"),
    ),
    "ja-JP": (
        (("還付金詐欺", "偽カスタマーサービス"), "冒充客服"),
        (("投資詐欺",), "虚假投资理财"),
        (("フィッシング",), "网络钓鱼"),
    ),
    "ko-KR": (
        (("환불 사기", "가짜 고객센터"), "冒充客服"),
        (("투자 사기",), "虚假投资理财"),
        (("피싱",), "网络钓鱼"),
    ),
}


class SearchService:
    """One stable search entry point for APIs, chat and voice."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.knowledge_base = knowledge_base or get_knowledge_base()

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        risk_level: str = "",
        entry_channel: str = "",
        victim_group: str = "",
        limit: int = 30,
        offset: int = 0,
    ) -> SearchResponse:
        safe_limit = min(max(int(limit), 1), 100)
        safe_offset = max(int(offset), 0)
        items, total = search_items(
            self.knowledge_base,
            query=normalize_text(query),
            category=normalize_text(category),
            risk_level=normalize_text(risk_level),
            entry_channel=normalize_text(entry_channel),
            victim_group=normalize_text(victim_group),
            limit=safe_limit,
            offset=safe_offset,
            use_pinyin=True,
        )
        return SearchResponse(items=tuple(items), total=total)


class AssistantService:
    def __init__(
        self,
        *,
        search: SearchService | None = None,
        sessions: SessionStore | None = None,
        llm: LLMProvider | None = None,
        max_candidates: int = 12,
    ) -> None:
        self.search = search or SearchService()
        self.sessions = sessions or SessionStore()
        self.llm = llm or OpenAICompatibleLLM(
            api_key=AI_API_KEY,
            base_url=AI_BASE_URL,
            model=AI_MODEL,
            timeout=AI_TIMEOUT,
        )
        self.max_candidates = max(1, min(max_candidates, 20))

    async def aclose(self) -> None:
        """Release the shared provider transport during application shutdown."""

        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()

    async def stream_turn(
        self,
        question: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        category: str = "",
        locale_hint: str = "",
    ) -> AsyncIterator[AssistantEvent]:
        question = normalize_text(str(question or ""))
        locale = detect_locale(question, hint=locale_hint)
        if not question:
            session = self.sessions.get_or_create(session_id)
            turn = turn_id or uuid.uuid4().hex
            yield EventSequence(session.session_id, turn).make(
                "turn.failed",
                code="empty_question",
            )
            return
        if len(question) > MAX_QUESTION_CHARS:
            session = self.sessions.get_or_create(session_id)
            turn = turn_id or uuid.uuid4().hex
            yield EventSequence(session.session_id, turn).make(
                "turn.failed",
                code="question_too_long",
                max_chars=MAX_QUESTION_CHARS,
            )
            return

        session = self.sessions.get_or_create(session_id)
        _, turn_id, cancel_event = self.sessions.begin_turn(session.session_id, turn_id)
        sequence = EventSequence(session.session_id, turn_id)
        history = self.sessions.history(session.session_id)
        short_reply_mode = _short_reply_mode(question)
        retrieval_basis = (
            "conversation_reply"
            if short_reply_mode
            else _retrieval_basis(
                self.search,
                question,
                category,
                locale=locale,
            )
        )
        retrieval_query = _localized_search_query(question, locale, retrieval_basis)
        answer_parts: list[str] = []
        candidates: tuple[FraudCase, ...] = ()
        try:
            yield sequence.make("turn.started", question=question, locale=locale)
            if cancel_event.is_set():
                yield sequence.make(
                    "turn.cancelled", reason=self._cancel_reason(session.session_id, turn_id)
                )
                return

            if _is_greeting(question, locale):
                answer = _localized_copy(locale, "greeting")
                self.sessions.append(
                    session.session_id,
                    ConversationTurn(
                        turn_id=turn_id,
                        question=question,
                        answer=answer,
                        source_ids=(),
                        locale=locale,
                    ),
                )
                yield sequence.make("response.text.delta", delta=answer, locale=locale)
                yield sequence.make("response.sources", sources=[])
                yield sequence.make(
                    "turn.completed",
                    answer=answer,
                    confidence=1.0,
                    suggested_questions=_localized_greeting_suggestions(locale),
                    locale=locale,
                )
                return

            urgent = _urgent_intervention(question)
            if urgent:
                answer = urgent
                self.sessions.append(
                    session.session_id,
                    ConversationTurn(turn_id=turn_id, question=question, answer=answer, source_ids=()),
                )
                yield sequence.make("response.text.delta", delta=answer)
                yield sequence.make("response.sources", sources=[])
                yield sequence.make("turn.completed", answer=answer, confidence=1.0, suggested_questions=["我还没有转账，下一步怎么办？", "如何保留证据？"])
                return

            yield sequence.make("retrieval.started")
            candidate_limit = (
                self.max_candidates
                if retrieval_basis == "multilingual_catalogue"
                else self._candidate_limit(question, category)
            )
            result = (
                SearchResponse(items=(), total=0)
                if retrieval_basis in {"none", "conversation_reply"}
                else await asyncio.to_thread(
                    self.search.search,
                    retrieval_query,
                    category=category,
                    limit=candidate_limit,
                )
            )
            if (
                not short_reply_mode
                and result.total > candidate_limit
                and (
                    retrieval_basis == "multilingual_catalogue"
                    or _is_scope_browse(self.search, question, category)
                )
            ):
                offset = _exploration_offset(
                    session.session_id,
                    turn_id,
                    retrieval_query,
                    result.total,
                    candidate_limit,
                )
                result = await asyncio.to_thread(
                    self.search.search,
                    retrieval_query,
                    category=category,
                    limit=candidate_limit,
                    offset=offset,
                )
                LOGGER.info(
                    "[trace=%s turn=%s] retrieval.window offset=%s total=%s",
                    session.session_id,
                    turn_id,
                    offset,
                    result.total,
                )
            candidates = tuple(result.items)
            LOGGER.info(
                "[trace=%s turn=%s] retrieval.completed basis=%s history_turns=%s total=%s candidates=%s candidate_titles=%s",
                session.session_id,
                turn_id,
                retrieval_basis,
                len(history),
                result.total,
                len(candidates),
                "|".join(item.title for item in candidates) or "-",
            )
            yield sequence.make(
                "retrieval.completed",
                total=result.total,
                source_count=len(candidates),
            )
            if cancel_event.is_set():
                yield sequence.make(
                    "turn.cancelled", reason=self._cancel_reason(session.session_id, turn_id)
                )
                return

            # Avoid even constructing a network client when no key is
            # configured.  The local retrieval answer remains fully usable.
            configured = getattr(self.llm, "api_key", object())
            api_key_configured = not (
                configured is None or (isinstance(configured, str) and not configured.strip())
            )
            if api_key_configured:
                messages = self._messages(
                    question,
                    candidates,
                    history,
                    short_reply_mode=short_reply_mode,
                    retrieval_basis=retrieval_basis,
                    locale=locale,
                )
                try:
                    llm_started = time.perf_counter()
                    first_delta_at = None
                    delta_index = 0
                    text_chars = 0
                    next_progress_chars = 100
                    LOGGER.info(
                        "[trace=%s turn=%s] llm.request.start attempts=%s first_token_timeout=%.3fs candidates=%s history_turns=%s retrieval_basis=%s",
                        session.session_id,
                        turn_id,
                        AI_FIRST_TOKEN_MAX_ATTEMPTS,
                        AI_FIRST_TOKEN_TIMEOUT,
                        len(candidates),
                        len(history),
                        retrieval_basis,
                    )

                    def start_provider_stream():
                        return self.llm.stream_chat(
                            messages,
                            temperature=0.2,
                            max_tokens=700,
                        )

                    async for delta in _stream_with_first_token_retry(
                        start_provider_stream,
                        cancel_event,
                        timeout=AI_FIRST_TOKEN_TIMEOUT,
                        max_attempts=AI_FIRST_TOKEN_MAX_ATTEMPTS,
                        log_context=f"trace={session.session_id} turn={turn_id}",
                    ):
                        if not delta:
                            continue
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                            LOGGER.info(
                                "[trace=%s turn=%s] llm.first_text_delta +%.3fs",
                                session.session_id,
                                turn_id,
                                first_delta_at - llm_started,
                            )
                        answer_parts.append(delta)
                        delta_index += 1
                        text_chars += len(delta)
                        if text_chars >= next_progress_chars:
                            LOGGER.info(
                                "[trace=%s turn=%s] llm.text.progress chunks=%s chars=%s",
                                session.session_id,
                                turn_id,
                                delta_index,
                                text_chars,
                            )
                            next_progress_chars += 100
                        yield sequence.make("response.text.delta", delta=delta, locale=locale)
                    if cancel_event.is_set():
                        yield sequence.make(
                            "turn.cancelled",
                            reason=self._cancel_reason(session.session_id, turn_id),
                        )
                        return
                    LOGGER.info(
                        "[trace=%s turn=%s] llm.stream.complete +%.3fs chunks=%s chars=%s",
                        session.session_id,
                        turn_id,
                        time.perf_counter() - llm_started,
                        delta_index,
                        text_chars,
                    )
                except asyncio.CancelledError:
                    yield sequence.make("turn.cancelled", reason="transport_closed")
                    return
                except _LLMFirstTokenTimeout as exc:
                    LOGGER.error(
                        "[trace=%s turn=%s] llm.failed code=llm_first_token_timeout attempts=%s",
                        session.session_id,
                        turn_id,
                        exc.attempts,
                    )
                    yield sequence.make(
                        "turn.failed",
                        code="llm_first_token_timeout",
                        attempts=exc.attempts,
                    )
                    return
                except _LLMEmptyStream as exc:
                    LOGGER.error(
                        "[trace=%s turn=%s] llm.failed code=llm_empty_stream attempts=%s",
                        session.session_id,
                        turn_id,
                        exc.attempts,
                    )
                    yield sequence.make(
                        "turn.failed",
                        code="llm_empty_stream",
                        attempts=exc.attempts,
                    )
                    return
                except Exception:  # provider details stay server-side
                    LOGGER.exception(
                        "[trace=%s turn=%s] llm.failed code=llm_unavailable",
                        session.session_id,
                        turn_id,
                    )
                    yield sequence.make("turn.failed", code="llm_unavailable")
                    return

            if cancel_event.is_set():
                yield sequence.make(
                    "turn.cancelled", reason=self._cancel_reason(session.session_id, turn_id)
                )
                return

            answer = "".join(answer_parts).strip() or _fallback_answer(
                question,
                candidates,
                history=history,
                locale=locale,
            )
            if not answer_parts:
                yield sequence.make("response.text.delta", delta=answer, locale=locale)
            used_sources = _used_sources(answer, candidates)
            confidence = _confidence(used_sources, answer)
            source_ids = tuple(item.id for item in used_sources)
            self.sessions.append(
                session.session_id,
                ConversationTurn(
                    turn_id=turn_id,
                    question=question,
                    answer=answer,
                    source_ids=source_ids,
                    locale=locale,
                ),
            )
            source_payload = [item_to_dict(item) for item in used_sources]
            yield sequence.make(
                "response.sources",
                sources=source_payload,
            )
            LOGGER.info(
                "[trace=%s turn=%s] text.complete chars=%s sources=%s",
                session.session_id,
                turn_id,
                len(answer),
                len(source_payload),
            )
            yield sequence.make(
                "turn.completed",
                answer=answer,
                confidence=confidence,
                suggested_questions=_suggestions(used_sources, locale=locale),
                locale=locale,
            )
        finally:
            self.sessions.finish_turn(session.session_id, turn_id, cancel_event)

    def _cancel_reason(self, session_id: str, turn_id: str) -> str:
        return self.sessions.cancel_reason(session_id, turn_id) or "client_cancelled"

    def _candidate_limit(self, question: str, category: str = "") -> int:
        """Use a wider evidence set for comparison/recommendation questions.

        ``max_candidates`` is an upper bound, not a promise to stuff every turn
        with the same number of records. Focused questions need a small context;
        broad questions need enough distinct cases for the model to compare.
        """
        normalized = normalize_search_query(question).lower()
        knowledge_base = getattr(self.search, "knowledge_base", None)
        source_items = getattr(knowledge_base, "items", ())
        titles = {normalize_search_query(item.title).lower() for item in source_items if item.title}
        exact_titles = [title for title in titles if title and title in normalized]
        categories = {
            normalize_search_query(item.ccl2023_category).lower()
            for item in source_items
            if item.ccl2023_category
        }
        category_match = any(category and category in normalized for category in categories)
        focused_broad_markers = (
            "哪些",
            "有哪些",
            "推荐",
            "值得了解",
            "想了解",
            "各类",
            "比较",
            "分别",
            "适合",
        )
        broad_markers = (*focused_broad_markers, "了解")
        requested = _requested_item_count(question)
        if exact_titles and not category_match and requested is None:
            return 1
        if requested is not None:
            return min(self.max_candidates, max(requested + 2, requested))
        if category.strip() or any(marker in question for marker in broad_markers):
            return self.max_candidates
        # For an unqualified question, let query complexity determine the
        # evidence width instead of using another global magic number.
        return min(self.max_candidates, max(2, len(tokenize(question)) + 1))

    def _messages(
        self,
        question: str,
        candidates: Sequence[FraudCase],
        history: Sequence[ConversationTurn],
        *,
        short_reply_mode: str | None = None,
        retrieval_basis: str | None = None,
        locale: str = DEFAULT_LOCALE,
    ) -> list[dict[str, str]]:
        short_reply_mode = short_reply_mode or _short_reply_mode(question)
        retrieval_basis = retrieval_basis or ("retrieval" if candidates else "none")
        language_profile = get_language_profile(locale)
        system = (
            f"本轮已自动识别用户语言为{language_profile.name}。"
            f"{language_profile.response_instruction}"
            "你是识诈，称呼可以是熊猫警官；你是反诈知识助手，不是公安机关工作人员，也不要自称警方。"
            "说话自然、冷静、具体，不过度戏剧化，不制造恐慌。输出是会被朗读的台词，不写动作、旁白或舞台说明。"
            "事实只能来自本轮提供的候选案例；没有候选或资料不足时坦白说明，不编造号码、金额、时间线或法律结论。"
            "系统检索到的资料不是用户提及，不能把候选内容说成用户刚才说过。"
            "回答优先给可执行判断：指出类别、风险信号、常见手法和防范建议；涉及正在转账、提供验证码、共享屏幕、下载陌生App或远程控制时，"
            "先明确要求立即停止操作、挂断或退出共享屏幕，不提供验证码和密码，联系银行止付/冻结，并按紧急程度拨打110或96110。"
            "若已经转账，提醒尽快联系银行和110/96110并保存聊天、账号、订单、转账凭证；不要承诺一定追回。"
            "宽泛问题可从候选中按相关性和差异选取若干案例，不要固定输出四条，不要堆砌字段。"
            "保持简洁自然，用户只是寒暄时友好回应并询问他遇到的情况。正文使用清晰 Markdown，不输出 JSON、XML 或内部流程标签。"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for turn in history[-5:]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer[:500]})
        if retrieval_basis == "none":
            messages.append({
                "role": "system",
                "content": (
                    "本轮没有识别到明确的案例、类别、渠道或反诈问题，因此系统没有提供资料候选。"
                    "这不是用户说的话。不要猜测案例名，也不要主动补出‘刚才提到’的案例；"
                    "若用户原话含混，就自然请他重说或补充想聊的对象。"
                ),
            })
        else:
            context = _candidate_context(candidates, AI_MAX_CONTEXT_CHARS)
            messages.append({
                "role": "system",
                "content": (
                    "以下内容是系统为本轮自动检索的参考资料，不是用户说的话，也不是用户点名的案例。"
                    "只能用它核对事实，不能据此声称‘你刚才提到/讲到/问到’。\n\n"
                    f"{context}"
                ),
            })
        if short_reply_mode == "continuation":
            messages.append({
                "role": "system",
                "content": "用户本轮只是简短回应，未重新点名案例；请沿着最近一条 assistant 回答自然接续，不要把那条回答的内容归到用户身上。",
            })
        elif short_reply_mode == "pause":
            messages.append(
                {
                    "role": "system",
                    "content": "用户本轮是在请求暂缓。简短回应并停住，不要展开新项目，也不要把上一条 assistant 回答说成用户讲过。",
                }
            )
        messages.append({
            "role": "system",
            "content": "下一条 user 消息是用户本轮逐字原话；不要把历史 assistant 内容或检索资料拼接进这条用户消息。",
        })
        messages.append({"role": "user", "content": question})
        return messages


def _candidate_context(items: Sequence[FraudCase], max_chars: int) -> str:
    if not items:
        return "未检索到匹配资料。涉及具体事实时请说明资料库暂无对应条目。"
    blocks: list[str] = []
    used = 0
    for item in items:
        payload = item_to_dict(item, include_content=True)
        block = "\n".join(
            part
            for part in (
                f"[{payload['id']}] {payload['title']}",
                f"类别：{payload.get('ccl2023_category') or payload.get('category', '')}；风险等级：{payload.get('risk_level', '')}；受害人群：{payload.get('victim_group', '')}",
                f"入口渠道：{'、'.join(payload.get('entry_channels', []))}；关键手法：{'、'.join(payload.get('key_methods', []))}",
                f"风险信号：{str(payload.get('risk_signals') or '')[:300]}",
                f"防范建议：{str(payload.get('prevention_advice') or '')[:360]}",
                f"来源：{payload.get('source_name', '')}",
                f"事实：{str(payload.get('content') or '')[:560]}",
            ) if part.strip()
        )
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


class _LLMFirstTokenTimeout(RuntimeError):
    def __init__(self, attempts: int) -> None:
        super().__init__("llm_first_token_timeout")
        self.attempts = attempts


class _LLMEmptyStream(RuntimeError):
    def __init__(self, attempts: int) -> None:
        super().__init__("llm_empty_stream")
        self.attempts = attempts


def _urgent_intervention(question: str) -> str:
    """Return a deterministic stop-and-protect response for active incidents."""
    text = normalize_text(question).casefold()
    direct_transfer = any(
        term in text
        for term in ("正在转账", "已经转账", "准备转账", "要我转账", "让我转账", "转账中")
    )
    sensitive_action = any(
        term in text for term in ("验证码", "共享屏幕", "远程控制", "屏幕共享", "陌生app")
    )
    active_cue = any(
        term in text
        for term in (
            "对方", "让我", "要我", "叫我", "正在", "准备", "已经", "刚刚",
            "提供", "发给", "输入", "下载",
        )
    )
    active = direct_transfer or (sensitive_action and active_cue)
    if not active:
        return ""
    actions = []
    if any(term in text for term in ("验证码", "共享屏幕", "远程控制", "屏幕共享")):
        actions.append("立即挂断或退出共享屏幕，不要提供验证码、密码，也不要按对方指示操作")
    if "转账" in text:
        actions.append("立刻停止转账；如果已经转出，马上联系银行尝试止付/冻结")
    actions.append("保存聊天、来电、账号和转账凭证，尽快拨打110或96110核实求助")
    return "这类情况先按紧急风险处理：" + "；".join(actions) + "。不要继续和对方周旋，也不要相信所谓‘安全账户’。"


async def _close_iterator(iterator: object | None) -> None:
    if iterator is None:
        return
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def _stream_with_first_token_retry(
    factory: Callable[[], AsyncIterator[str]],
    cancel_event: asyncio.Event,
    *,
    timeout: float,
    max_attempts: int,
    log_context: str,
) -> AsyncIterator[str]:
    """Retry only pre-first-token failures, always closing the old stream first."""

    attempts = min(max(int(max_attempts), 1), 2)
    first_token_timeout = max(float(timeout), 0.001)
    for attempt in range(1, attempts + 1):
        iterator: AsyncIterator[str] | None = None
        emitted = False
        started = time.perf_counter()
        LOGGER.info(
            "[%s] llm.attempt.start attempt=%s/%s first_token_timeout=%.3fs",
            log_context,
            attempt,
            attempts,
            first_token_timeout,
        )
        try:
            iterator = factory().__aiter__()
            while not cancel_event.is_set():
                remaining = None
                if not emitted:
                    remaining = first_token_timeout - (time.perf_counter() - started)
                    if remaining <= 0:
                        LOGGER.warning(
                            "[%s] llm.first-token-timeout attempt=%s/%s timeout=%.3fs",
                            log_context,
                            attempt,
                            attempts,
                            first_token_timeout,
                        )
                        raise _LLMFirstTokenTimeout(attempt)

                next_task = asyncio.create_task(anext(iterator))
                cancelled = asyncio.create_task(cancel_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        {next_task, cancelled},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        next_task.cancel()
                        await asyncio.gather(next_task, return_exceptions=True)
                        cancelled.cancel()
                        await asyncio.gather(cancelled, return_exceptions=True)
                        LOGGER.warning(
                            "[%s] llm.first-token-timeout attempt=%s/%s timeout=%.3fs",
                            log_context,
                            attempt,
                            attempts,
                            first_token_timeout,
                        )
                        raise _LLMFirstTokenTimeout(attempt)
                    if cancelled in done and cancel_event.is_set():
                        next_task.cancel()
                        await asyncio.gather(next_task, return_exceptions=True)
                        return
                    cancelled.cancel()
                    await asyncio.gather(cancelled, return_exceptions=True)
                    try:
                        delta = next_task.result()
                    except StopAsyncIteration:
                        if not emitted:
                            raise _LLMEmptyStream(attempt)
                        return
                finally:
                    for task in (next_task, cancelled):
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)

                if not delta:
                    continue
                emitted = True
                yield delta
            return
        except (_LLMFirstTokenTimeout, _LLMEmptyStream) as exc:
            if emitted:
                raise
            if attempt >= attempts:
                raise
            LOGGER.info(
                "[%s] llm.retry attempt=%s/%s reason=%s",
                log_context,
                attempt,
                attempts,
                "first-token-timeout" if isinstance(exc, _LLMFirstTokenTimeout) else "empty-stream",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if emitted or attempt >= attempts:
                raise
            LOGGER.info(
                "[%s] llm.retry attempt=%s/%s reason=provider-failure",
                log_context,
                attempt,
                attempts,
            )
        finally:
            # This runs before the next attempt starts, so a timed-out HTTP
            # response cannot remain alive while its retry is sent.
            await _close_iterator(iterator)


_FALLBACK_COPY = {
    "zh-CN": {
        "greeting": "你好，我是熊猫警官。把可疑消息、电话或操作告诉我，我们一起判断风险。",
        "pause": "好，你慢慢来。我先停在这里。",
        "continuation": "好，我们就接着刚才的内容看。你想先听哪一处？",
        "no_results": "资料库暂时没有找到与这个问题直接对应的项目。你可以换一个项目名称、地区或类别再试试。",
        "list_intro": "如果想先抓住这一类非遗的不同气质，我会带你从{names}看起。",
        "list_close": "你对哪一项更有感觉？我可以接着带你往它的历史和现场里走。",
        "missing_detail": "它的详细资料还在整理中。",
        "single_missing": "资料库中暂未提供该项目的详细简介。",
    },
    "yue-CN": {
        "greeting": "你好呀。想由边一项非遗开始了解？",
        "pause": "好呀，你慢慢嚟，我先停喺度。",
        "continuation": "好，我哋接住头先嘅内容讲。你想先听边一部分？",
        "no_results": "资料库暂时搵唔到同呢个问题直接对应嘅项目。你可以换个项目名、地区或者类别再试。",
        "list_intro": "想先睇清呢类非遗各自嘅味道，我会由{names}讲起。",
        "list_close": "你对边一项最有感觉？我可以继续讲佢嘅历史同现场。",
        "missing_detail": "呢一项嘅详细资料仲整理紧。",
        "single_missing": "资料库暂时未有呢个项目嘅详细简介。",
    },
    "zh-CN-sichuan": {
        "greeting": "你好。想从哪一项非遗开始摆起？",
        "pause": "要得，你慢慢来，我先停到这儿。",
        "continuation": "要得，我们接到刚才的内容摆。你想先听哪一截？",
        "no_results": "资料库暂时没找到直接对应的项目。你可以换个项目名、地区或者类别再试一下。",
        "list_intro": "想先看出这一类非遗各自的味道，我带你从{names}摆起。",
        "list_close": "你对哪一项更感兴趣？我可以接到讲它的历史和现场。",
        "missing_detail": "这项的详细资料还在整理中。",
        "single_missing": "资料库暂时还没有这个项目的详细简介。",
    },
    "zh-CN-henan": {
        "greeting": "恁好。想从哪一项非遗开始聊？",
        "pause": "中，恁慢慢来，我先停到这儿。",
        "continuation": "中，咱接着刚才的内容聊。恁想先听哪一段？",
        "no_results": "资料库暂时没找着直接对应的项目。恁可以换个项目名、地区或者类别再试试。",
        "list_intro": "想先看看这一类非遗各有啥味道，咱可以从{names}聊起。",
        "list_close": "恁对哪一项更感兴趣？我可以接着聊它的历史和现在。",
        "missing_detail": "这一项的详细资料还在整理中。",
        "single_missing": "资料库暂时还没有这个项目的详细介绍。",
    },
    "en-US": {
        "greeting": "Hello. I’m your anti-fraud assistant. Tell me about the suspicious message, call, or action, and we’ll assess the risk together.",
        "pause": "Of course. Take your time; I'll pause here.",
        "continuation": "Let's continue from where we left off. Which part would you like to hear first?",
        "no_results": "I couldn't find a project in the collection that directly matches that question. Try a project name, region, or category.",
        "list_intro": "To see the different character of this heritage category, I would begin with {names}.",
        "list_close": "Which one draws you in most? I can take you further into its history and living practice.",
        "missing_detail": "Its detailed record is still being prepared.",
        "single_missing": "The collection does not yet include a detailed introduction for this project.",
    },
    "ja-JP": {
        "greeting": "こんにちは。どの中国無形文化遺産から見てみましょうか。",
        "pause": "はい、ゆっくりどうぞ。ここでいったん止めます。",
        "continuation": "では、先ほどの続きから見ていきましょう。まずどこを聞きたいですか。",
        "no_results": "この質問に直接対応する項目は資料庫で見つかりませんでした。項目名、地域、または分類を変えてお試しください。",
        "list_intro": "この分野の違いをつかむなら、まず{names}から見ていきましょう。",
        "list_close": "どの項目が一番気になりましたか。歴史や現在の姿をさらにご案内できます。",
        "missing_detail": "詳しい資料は現在整理中です。",
        "single_missing": "この項目の詳しい紹介は、まだ資料庫に収録されていません。",
    },
    "ko-KR": {
        "greeting": "안녕하세요. 어떤 중국 무형문화유산부터 살펴볼까요?",
        "pause": "네, 천천히 하세요. 여기서 잠시 멈출게요.",
        "continuation": "그럼 앞의 이야기에서 이어가겠습니다. 어느 부분부터 듣고 싶으세요?",
        "no_results": "이 질문과 직접 맞는 항목을 자료에서 찾지 못했습니다. 종목명, 지역 또는 분류를 바꿔 다시 시도해 보세요.",
        "list_intro": "이 분야의 서로 다른 매력을 보려면 {names}부터 살펴보는 것이 좋습니다.",
        "list_close": "어느 항목이 가장 마음에 닿나요? 역사와 현장의 모습까지 이어서 안내해 드릴게요.",
        "missing_detail": "상세 자료는 현재 정리 중입니다.",
        "single_missing": "이 항목의 상세 소개는 아직 자료에 수록되지 않았습니다.",
    },
    "fr-FR": {
        "greeting": "Bonjour. Quel patrimoine culturel immatériel chinois souhaitez-vous découvrir ?",
        "pause": "Bien sûr. Prenez votre temps ; je m'arrête ici un instant.",
        "continuation": "Reprenons là où nous en étions. Quelle partie souhaitez-vous entendre d'abord ?",
        "no_results": "Je n'ai pas trouvé de projet correspondant directement à cette question. Essayez un nom de projet, une région ou une catégorie.",
        "list_intro": "Pour saisir les différentes facettes de ce patrimoine, je commencerais par {names}.",
        "list_close": "Lequel vous attire le plus ? Je peux poursuivre avec son histoire et sa pratique vivante.",
        "missing_detail": "Sa notice détaillée est encore en préparation.",
        "single_missing": "La collection ne contient pas encore de présentation détaillée de ce projet.",
    },
    "es-ES": {
        "greeting": "Hola. ¿Qué patrimonio cultural inmaterial de China te gustaría descubrir?",
        "pause": "Claro. Tómate tu tiempo; haré una pausa aquí.",
        "continuation": "Sigamos desde donde lo dejamos. ¿Qué parte te gustaría escuchar primero?",
        "no_results": "No encontré un proyecto que coincida directamente con la pregunta. Prueba con el nombre de un proyecto, una región o una categoría.",
        "list_intro": "Para apreciar los distintos matices de este patrimonio, empezaría por {names}.",
        "list_close": "¿Cuál te atrae más? Puedo continuar con su historia y su práctica viva.",
        "missing_detail": "Su ficha detallada todavía está en preparación.",
        "single_missing": "La colección aún no incluye una introducción detallada de este proyecto.",
    },
    "de-DE": {
        "greeting": "Hallo. Welches immaterielle Kulturerbe Chinas möchten Sie entdecken?",
        "pause": "Gern. Nehmen Sie sich Zeit; ich pausiere hier.",
        "continuation": "Machen wir dort weiter, wo wir aufgehört haben. Welchen Teil möchten Sie zuerst hören?",
        "no_results": "Ich habe kein Projekt gefunden, das direkt zu dieser Frage passt. Versuchen Sie einen Projektnamen, eine Region oder eine Kategorie.",
        "list_intro": "Um die unterschiedlichen Facetten dieses Kulturerbes zu erfassen, würde ich mit {names} beginnen.",
        "list_close": "Welches spricht Sie am meisten an? Ich kann seine Geschichte und heutige Praxis weiter erläutern.",
        "missing_detail": "Der ausführliche Eintrag wird noch vorbereitet.",
        "single_missing": "Die Sammlung enthält noch keine ausführliche Einführung zu diesem Projekt.",
    },
    "ru-RU": {
        "greeting": "Здравствуйте. С каким объектом нематериального наследия Китая вы хотели бы познакомиться?",
        "pause": "Конечно. Не спешите — я пока остановлюсь здесь.",
        "continuation": "Продолжим с того места, где остановились. О чём рассказать сначала?",
        "no_results": "Я не нашёл в коллекции проект, напрямую соответствующий вопросу. Попробуйте указать название, регион или категорию.",
        "list_intro": "Чтобы увидеть разные грани этого наследия, я бы начал с {names}.",
        "list_close": "Что заинтересовало вас больше всего? Я могу продолжить рассказ об истории и живой практике.",
        "missing_detail": "Подробная запись ещё готовится.",
        "single_missing": "В коллекции пока нет подробного описания этого проекта.",
    },
    "ar-SA": {
        "greeting": "مرحباً. أي عنصر من التراث الثقافي غير المادي في الصين تود استكشافه؟",
        "pause": "بكل تأكيد. خذ وقتك؛ سأتوقف هنا قليلاً.",
        "continuation": "لنواصل من حيث توقفنا. أي جزء تود سماعه أولاً؟",
        "no_results": "لم أجد في المجموعة مشروعاً يطابق هذا السؤال مباشرة. جرّب اسم مشروع أو منطقة أو فئة.",
        "list_intro": "لرؤية الجوانب المختلفة لهذا التراث، سأبدأ بـ {names}.",
        "list_close": "أيها جذبك أكثر؟ يمكنني متابعة الحديث عن تاريخه وممارسته الحية.",
        "missing_detail": "لا يزال السجل التفصيلي قيد الإعداد.",
        "single_missing": "لا تتضمن المجموعة بعد مقدمة تفصيلية لهذا المشروع.",
    },
    "hi-IN": {
        "greeting": "नमस्ते। आप चीन की किस अमूर्त सांस्कृतिक विरासत के बारे में जानना चाहेंगे?",
        "pause": "ज़रूर। आराम से समय लें; मैं यहीं रुकता हूँ।",
        "continuation": "जहाँ रुके थे वहीं से आगे बढ़ते हैं। आप पहले कौन-सा भाग सुनना चाहेंगे?",
        "no_results": "संग्रह में इस प्रश्न से सीधे मेल खाने वाली परियोजना नहीं मिली। किसी परियोजना का नाम, क्षेत्र या श्रेणी आज़माएँ।",
        "list_intro": "इस विरासत के अलग-अलग रूप समझने के लिए मैं {names} से शुरुआत करूँगा।",
        "list_close": "इनमें से किसने आपको सबसे अधिक आकर्षित किया? मैं उसके इतिहास और जीवित परंपरा पर आगे बता सकता हूँ।",
        "missing_detail": "इसका विस्तृत अभिलेख अभी तैयार किया जा रहा है।",
        "single_missing": "संग्रह में अभी इस परियोजना का विस्तृत परिचय उपलब्ध नहीं है।",
    },
    "th-TH": {
        "greeting": "สวัสดี คุณอยากสำรวจมรดกทางวัฒนธรรมที่จับต้องไม่ได้ของจีนรายการใด?",
        "pause": "ได้เลย ค่อย ๆ ใช้เวลานะ ฉันจะหยุดไว้ตรงนี้ก่อน",
        "continuation": "มาต่อจากที่เราค้างไว้ คุณอยากฟังส่วนไหนก่อน?",
        "no_results": "ฉันไม่พบรายการในคลังที่ตรงกับคำถามนี้โดยตรง ลองใช้ชื่อโครงการ ภูมิภาค หรือหมวดหมู่",
        "list_intro": "หากต้องการเห็นลักษณะที่หลากหลายของมรดกประเภทนี้ ฉันขอเริ่มจาก {names}",
        "list_close": "รายการไหนดึงดูดคุณมากที่สุด? ฉันเล่าต่อถึงประวัติและการสืบทอดในปัจจุบันได้",
        "missing_detail": "ข้อมูลฉบับละเอียดยังอยู่ระหว่างการจัดทำ",
        "single_missing": "คลังยังไม่มีคำแนะนำฉบับละเอียดสำหรับโครงการนี้",
    },
}


def _copy_catalog(locale: str) -> dict[str, str]:
    return _FALLBACK_COPY.get(locale, _FALLBACK_COPY["en-US"])


def _localized_copy(locale: str, key: str, **values: str) -> str:
    return _copy_catalog(locale)[key].format(**values)


def _localized_greeting_suggestions(locale: str) -> list[str]:
    suggestions = {
        "zh-CN": ("按诈骗类别浏览案例", "按风险等级筛选"),
        "zh-CN-sichuan": ("按诈骗类别浏览案例", "按风险等级筛选"),
        "zh-CN-henan": ("按诈骗类别找案例", "按风险等级筛选"),
        "yue-CN": ("按诈骗类别搵案例", "按风险等级筛选"),
        "en-US": ("Browse cases by scam type", "Filter by risk level"),
        "ja-JP": ("詐欺類型から事例を探す", "危険度で絞り込む"),
        "ko-KR": ("사기 유형별 사례 찾기", "위험 수준으로 필터링"),
        "fr-FR": ("Explorer le patrimoine par région", "Parcourir la collection par catégorie"),
        "es-ES": ("Explorar el patrimonio por región", "Ver la colección por categoría"),
        "de-DE": ("Kulturerbe nach Region erkunden", "Sammlung nach Kategorie durchsuchen"),
        "ru-RU": ("Искать наследие по регионам", "Просматривать коллекцию по категориям"),
        "ar-SA": ("استكشاف التراث حسب المنطقة", "تصفح المجموعة حسب الفئة"),
        "hi-IN": ("क्षेत्र के अनुसार विरासत खोजें", "श्रेणी के अनुसार संग्रह देखें"),
        "th-TH": ("สำรวจมรดกตามภูมิภาค", "ดูคลังตามหมวดหมู่"),
    }
    return list(suggestions.get(locale, suggestions["en-US"]))


def _is_greeting(question: str, locale: str) -> bool:
    compact = question.casefold().strip("。！!？?～~，,、；;：: .")
    if compact in GREETING_QUESTIONS:
        return True
    return compact in MULTILINGUAL_GREETINGS.get(locale, ())


def _fallback_answer(
    question: str,
    items: Sequence[FraudCase],
    *,
    history: Sequence[ConversationTurn] = (),
    locale: str = DEFAULT_LOCALE,
) -> str:
    mode = _short_reply_mode(question)
    if mode == "pause":
        return _localized_copy(locale, "pause")
    if mode == "continuation" and history:
        return _localized_copy(locale, "continuation")
    if not items:
        return "我暂时没有找到与这个问题直接对应的案例。你可以补充诈骗类别、联系渠道、风险等级或受害人群。"
    if any(marker in question for marker in ("有哪些", "推荐", "值得", "几个", "案例")) and len(items) > 1:
        requested = _requested_item_count(question)
        selected: list[FraudCase] = []
        remaining_chars = 760
        for item in items:
            summary = normalize_text(item.summary or item.content)
            cost = max(80, min(len(summary), 180))
            if selected and requested is None and remaining_chars < cost:
                break
            selected.append(item)
            remaining_chars -= cost
            if requested is not None and len(selected) >= requested:
                break
            if len(selected) >= 6:
                break
        separator = (
            "、"
            if locale in {"zh-CN", "yue-CN", "zh-CN-sichuan", "zh-CN-henan"}
            else ", "
        )
        names = separator.join(item.title for item in selected)
        passages = []
        for item in selected:
            summary = normalize_text(item.summary or item.content)
            summary = summary[:180]
            passages.append(f"**{item.title}**。{summary or '该案例的详细资料还在整理中。'}")
        return (
            f"如果想先把这个风险类型看清楚，我会从{names}这些案例说起。\n\n"
            + "\n\n".join(passages)
            + "\n\n你想先看哪一个案例的风险信号和防范做法？"
        )
    item = items[0]
    summary = normalize_text(item.summary or item.content)[:500]
    return f"### {item.title}\n\n{summary or '案例库中暂未提供该案例的详细简介。'}"


def _requested_item_count(question: str) -> int | None:
    match = re.search(r"(\d{1,2}|[一二两三四五六七八九十])\s*(?:个|项|种|类)", question)
    if not match:
        match = re.search(
            r"\b(\d{1,2})\s*(?:projects?|items?|types?|traditions?|examples?)\b",
            question,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else ITEM_COUNT_WORDS[token]


@lru_cache(maxsize=8)
def _catalogue_anchors(
    knowledge_base: KnowledgeBase,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    item_names: set[str] = set()
    categories: set[str] = set()
    channels: set[str] = set()
    for item in knowledge_base.items:
        for value in (item.title, item.custom_subcategory, *item.tags, *item.key_methods):
            normalized = normalize_text(value).casefold()
            if len(normalized) >= 2:
                item_names.add(normalized)
        normalized_category = normalize_text(item.ccl2023_category).casefold()
        if len(normalized_category) >= 2:
            categories.add(normalized_category)
        for value in item.entry_channels:
            normalized_channel = normalize_text(value).casefold()
            if len(normalized_channel) >= 2:
                channels.add(normalized_channel)
    return frozenset(item_names), frozenset(categories), frozenset(channels)


def _translated_search_anchor(question: str, locale: str) -> str:
    folded = normalize_text(question).casefold()
    for phrases, canonical in MULTILINGUAL_SEARCH_GLOSSES.get(locale, ()):
        if any(phrase.casefold() in folded for phrase in phrases):
            return canonical
    return ""


def _localized_search_query(question: str, locale: str, retrieval_basis: str) -> str:
    canonical = _translated_search_anchor(question, locale)
    if canonical:
        return canonical
    if retrieval_basis == "multilingual_catalogue":
        # An empty search query intentionally opens a bounded, rotating
        # catalogue window.  The original user wording remains untouched in
        # history and in the LLM's user message.
        return ""
    return question


def _retrieval_basis(
    search: SearchService,
    question: str,
    category: str = "",
    *,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Return the explicit user signal that authorizes knowledge retrieval.

    Full-text search intentionally accepts weak content matches for the case
    browser. Conversation grounding is stricter: a candidate may enter the LLM
    only when the user actually named an item, scope or catalogue action.
    """
    if normalize_text(category):
        return "ui_category"
    text = normalize_text(question).casefold()
    if not text:
        return "none"
    knowledge_base = getattr(search, "knowledge_base", None)
    if knowledge_base is not None:
        item_names, categories, channels = _catalogue_anchors(knowledge_base)
        if any(name in text for name in item_names):
            return "item_name"
        if any(name in text for name in categories):
            return "category"
        if any(name in text for name in channels):
            return "channel"
    if any(term in text for term in FRAUD_DOMAIN_TERMS):
        return "fraud_domain"
    if any(term.casefold() in text for term in MULTILINGUAL_FRAUD_TERMS.get(locale, ())):
        return "multilingual_catalogue"
    if (
        any(action in text for action in CATALOGUE_BROWSE_ACTIONS)
        and any(target in text for target in CATALOGUE_BROWSE_OBJECTS)
    ):
        return "catalogue_browse"
    return "none"


def _is_scope_browse(search: SearchService, question: str, category: str) -> bool:
    """Identify broad catalogue browsing where rotating a result window is useful."""
    if not category and not any(
        marker in question for marker in ("哪些", "有哪些", "推荐", "值得", "想了解", "各类")
    ):
        return False
    residual = normalize_search_query(question)
    knowledge_base = getattr(search, "knowledge_base", None)
    category_names = [category]
    category_names.extend(
        category_item.name for category_item in getattr(knowledge_base, "categories", ())
    )
    for name in sorted(set(category_names), key=len, reverse=True):
        normalized = normalize_search_query(name)
        if normalized:
            residual = residual.replace(normalized, " ")
    return not normalize_text(residual).strip()


def _exploration_offset(
    session_id: str,
    turn_id: str,
    question: str,
    total: int,
    limit: int,
) -> int:
    max_start = max(0, total - limit)
    if max_start == 0:
        return 0
    digest = hashlib.blake2s(
        f"{session_id}\0{turn_id}\0{normalize_search_query(question)}".encode(),
        digest_size=8,
    ).digest()
    return 1 + int.from_bytes(digest, "big") % max_start


def _short_reply_mode(question: str) -> str | None:
    compact = normalize_text(question).lower().strip("。！？!?，,、；;：: ")
    multilingual = {
        "ok": "continuation",
        "okay": "continuation",
        "yes": "continuation",
        "continue": "continuation",
        "go on": "continuation",
        "wait": "pause",
        "wait a moment": "pause",
        "hold on": "pause",
        "stop": "pause",
        "はい": "continuation",
        "続けて": "continuation",
        "ちょっと待って": "pause",
        "네": "continuation",
        "계속": "continuation",
        "잠깐만": "pause",
        "oui": "continuation",
        "continuez": "continuation",
        "attendez": "pause",
        "sí": "continuation",
        "continúa": "continuation",
        "espera": "pause",
        "ja": "continuation",
        "weiter": "continuation",
        "warten": "pause",
        "да": "continuation",
        "продолжайте": "continuation",
        "подождите": "pause",
        "نعم": "continuation",
        "تابع": "continuation",
        "انتظر": "pause",
        "हाँ": "continuation",
        "जारी रखें": "continuation",
        "रुकिए": "pause",
        "ใช่": "continuation",
        "ต่อเลย": "continuation",
        "รอก่อน": "pause",
    }
    return SHORT_REPLY_MODES.get(compact) or multilingual.get(compact)


def _confidence(items: Sequence[FraudCase], answer: str) -> float:
    if not items:
        return 0.2
    return 0.85 if answer else 0.4


def _used_sources(answer: str, candidates: Sequence[FraudCase]) -> tuple[FraudCase, ...]:
    """Keep citations tied to cases the final answer actually names."""
    if not candidates:
        return ()
    text = normalize_text(answer).casefold()
    alias_counts: dict[str, int] = {}
    for item in candidates:
        aliases = {
            normalize_text(value).casefold()
            for value in (item.custom_subcategory, *item.tags)
            if len(normalize_text(value)) >= 3
        }
        for alias in aliases:
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
    matches: list[tuple[int, int, FraudCase]] = []
    for index, item in enumerate(candidates):
        title = normalize_text(item.title).casefold()
        names = {title} if len(title) >= 2 else set()
        names.update(
            alias
            for alias in {
                normalize_text(value).casefold()
                for value in (item.custom_subcategory, *item.tags)
                if len(normalize_text(value)) >= 3
            }
            if alias_counts.get(alias) == 1
        )
        positions = [text.find(name) for name in names if name in text]
        if positions:
            matches.append((min(positions), index, item))
    if not matches:
        return (candidates[0],)
    matches.sort(key=lambda match: (match[0], match[1]))
    return tuple(match[2] for match in matches)


def _suggestions(
    items: Sequence[FraudCase],
    *,
    locale: str = DEFAULT_LOCALE,
) -> list[str]:
    if not items:
        return _localized_greeting_suggestions(locale) + [_localized_safety_question(locale)]
    suggestions: list[str] = []
    seen: set[str] = set()
    item_templates = {
        "zh-CN": "{title}有哪些风险信号？",
        "zh-CN-sichuan": "{title}有哪些风险信号？",
        "zh-CN-henan": "{title}有啥风险信号？",
        "yue-CN": "{title}有咩风险信号？",
        "en-US": "What are the warning signs in {title}?",
        "ja-JP": "{title}の危険信号は何ですか？",
        "ko-KR": "{title}의 위험 신호는 무엇인가요?",
    }
    template = item_templates.get(locale, item_templates["en-US"])
    for item in items:
        title = normalize_text(item.title)
        if not title or title in seen:
            continue
        seen.add(title)
        suggestions.append(template.format(title=title))
        if len(suggestions) == 3:
            break
    browse = {
        "zh-CN": ("按渠道继续筛选", "按类别继续浏览"),
        "zh-CN-sichuan": ("按渠道继续筛选", "按类别继续浏览"),
        "zh-CN-henan": ("按渠道接着筛选", "按类别接着看"),
        "yue-CN": ("按渠道继续筛选", "按类别继续睇"),
        "en-US": ("Filter by contact channel", "Browse by scam type"),
        "ja-JP": ("接触経路で絞り込む", "詐欺類型で見る"),
        "ko-KR": ("접촉 경로로 필터링", "사기 유형별 보기"),
    }
    suggestions.extend(browse.get(locale, browse["en-US"]))
    return suggestions[:3]


def _localized_safety_question(locale: str) -> str:
    questions = {
        "zh-CN": "遇到转账要求该怎么办？",
        "zh-CN-sichuan": "遇到转账要求该怎么办？",
        "zh-CN-henan": "碰到转账要求咋办？",
        "yue-CN": "遇到转账要求点算？",
        "en-US": "What should I do if someone asks me to transfer money?",
        "ja-JP": "送金を求められたらどうすればよいですか？",
        "ko-KR": "송금을 요구받으면 어떻게 해야 하나요?",
    }
    return questions.get(locale, questions["en-US"])


__all__ = ["AssistantService", "SearchService"]
