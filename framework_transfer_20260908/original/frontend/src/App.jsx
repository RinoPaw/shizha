import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  ArrowUp,
  CaretRight,
  SpinnerGap,
  Stop,
  X,
} from "@phosphor-icons/react";
import { REALTIME_VOICE_STATUS, useVoiceConversation } from "./hooks/useVoiceConversation.js";
import DigitalHuman from "./components/DigitalHuman.jsx";
import { composerAction, effectiveVoiceStatus } from "./lib/conversationUiState.js";
import { createConversationSessionId } from "./lib/conversationSession.js";
import {
  mergePageItems,
  nextPageState,
  pageLimitForState,
  PROJECT_PAGE_SIZE,
} from "./lib/pagination.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const VOICE_COPY = {
  [REALTIME_VOICE_STATUS.IDLE]: { label: "实时对话", detail: "等待开启", owner: "system" },
  [REALTIME_VOICE_STATUS.CONNECTING]: { label: "连接中", detail: "请允许使用麦克风", owner: "assistant" },
  [REALTIME_VOICE_STATUS.LISTENING]: { label: "等待语音", detail: "麦克风已就绪", owner: "user" },
  [REALTIME_VOICE_STATUS.USER_SPEAKING]: { label: "聆听中", detail: "继续说", owner: "user" },
  [REALTIME_VOICE_STATUS.TRANSCRIBING]: { label: "识别中", detail: "请稍候", owner: "user" },
  [REALTIME_VOICE_STATUS.THINKING]: { label: "生成回答中", detail: "查找资料中", owner: "assistant" },
  [REALTIME_VOICE_STATUS.RESPONDING]: { label: "播报中", detail: "可随时打断", owner: "assistant" },
  [REALTIME_VOICE_STATUS.ERROR]: { label: "语音不可用", detail: "请重试", owner: "assistant" },
};

function getSessionSeed() {
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }
  return Math.floor(Math.random() * 0x1_0000_0000);
}

function rotateAndLimit(values, seed, limit = 3) {
  const unique = [...new Set((Array.isArray(values) ? values : []).map((value) => String(value || "").trim()).filter(Boolean))];
  if (unique.length <= 1) return unique.slice(0, limit);
  const offset = Math.abs(Number(seed) || 0) % unique.length;
  const rotated = [...unique.slice(offset), ...unique.slice(0, offset)];
  return rotated.slice(0, limit);
}

const initialState = {
  sessionId: null,
  turnId: null,
  phase: "idle",
  sources: [],
  error: "",
  announcement: "",
  sequence: 0,
  messages: [],
};

function updateLastAssistant(messages, patch) {
  const last = messages.at(-1);
  if (last?.role !== "assistant") return messages;
  return [...messages.slice(0, -1), { ...last, ...patch(last) }];
}

function upsertCurrentAssistant(messages, sequence, patch) {
  const last = messages.at(-1);
  if (last?.role === "assistant") return updateLastAssistant(messages, patch);
  const assistant = {
    id: `a-${sequence}`,
    role: "assistant",
    content: "",
    status: "streaming",
    sources: [],
    suggestions: [],
  };
  return [...messages, { ...assistant, ...patch(assistant) }];
}

function appendTurn(state, text, phase) {
  const sequence = state.sequence + 1;
  return {
    ...state,
    sequence,
    phase,
    sources: [],
    error: "",
    announcement: "",
    turnId: null,
    messages: [...state.messages, { id: `u-${sequence}`, role: "user", content: text, status: "complete" }],
  };
}

function updateRealtimeUserPartial(state, text) {
  const content = String(text || "").trim();
  if (!content) return state;
  const last = state.messages.at(-1);
  if (last?.role === "user" && last.status === "transcribing") {
    return {
      ...state,
      phase: "realtime",
      messages: [...state.messages.slice(0, -1), { ...last, content }],
    };
  }
  const sequence = state.sequence + 1;
  return {
    ...state,
    sequence,
    phase: "realtime",
    error: "",
    messages: [...state.messages, { id: `u-${sequence}`, role: "user", content, status: "transcribing" }],
  };
}

function completeRealtimeUser(state, text) {
  const content = String(text || "").trim();
  const last = state.messages.at(-1);
  if (!content) return state;
  if (last?.role !== "user" || last.status !== "transcribing") return appendTurn(state, content, "realtime");
  return {
    ...state,
    phase: "realtime",
    sources: [],
    error: "",
    messages: [...state.messages.slice(0, -1), { ...last, content, status: "complete" }],
  };
}

function normalizeSuggestions(values) {
  return Array.isArray(values)
    ? [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))]
    : [];
}

function reducer(state, action) {
  if (action.type === "ask") return appendTurn(state, action.text, "retrieving");
  if (action.type === "realtime.user.partial") return updateRealtimeUserPartial(state, action.text);
  if (action.type === "realtime.user") return completeRealtimeUser(state, action.text);

  if (action.type === "event") {
    const { event } = action;
    const payload = event.payload || {};
    const next = {
      ...state,
      sessionId: event.session_id || state.sessionId,
      turnId: event.turn_id || state.turnId,
    };

    if (event.type === "retrieval.started") return { ...next, phase: "retrieving" };
    if (event.type === "retrieval.completed") return { ...next, phase: "composing" };

    if (event.type === "response.text.delta") {
      next.phase = "streaming";
      next.messages = upsertCurrentAssistant(state.messages, state.sequence, (message) => ({
        content: `${message.content}${payload.delta || ""}`,
        status: "streaming",
        sources: message.sources?.length ? message.sources : state.sources,
      }));
    }

    if (event.type === "response.sources") {
      const sources = Array.isArray(payload.sources) ? payload.sources : [];
      next.sources = sources;
      next.messages = updateLastAssistant(next.messages, () => ({ sources }));
    }

    if (event.type === "turn.completed") {
      const answer = String(payload.answer || "");
      const suggestions = normalizeSuggestions(payload.suggested_questions);
      return {
        ...next,
        phase: "complete",
        announcement: answer,
        messages: upsertCurrentAssistant(next.messages, state.sequence, (message) => ({
          content: answer || message.content,
          status: "complete",
          suggestions,
          sources: message.sources?.length ? message.sources : next.sources,
        })),
      };
    }

    if (event.type === "turn.failed") {
      return {
        ...next,
        phase: "error",
        error: payload.code === "question_too_long" ? `问题过长（最多 ${payload.max_chars} 字）` : "暂时无法回答",
        messages: updateLastAssistant(next.messages, () => ({ status: "error" })),
      };
    }

    if (event.type === "turn.cancelled") {
      return {
        ...next,
        phase: "cancelled",
        messages: updateLastAssistant(next.messages, () => ({ status: "cancelled" })),
      };
    }

    return next;
  }

  if (action.type === "realtime.answer.delta") {
    return {
      ...state,
      phase: "realtime",
      messages: upsertCurrentAssistant(state.messages, state.sequence, (message) => ({
        content: `${message.content}${action.text}`,
        status: "realtime",
        sources: message.sources?.length ? message.sources : state.sources,
      })),
    };
  }

  if (action.type === "realtime.answer.done") {
    const answer = String(action.text || "");
    const last = state.messages.at(-1);
    const previous = last?.role === "assistant" ? last.content || "" : "";
    const completed = answer || previous;
    return {
      ...state,
      phase: "complete",
      announcement: answer,
      messages: upsertCurrentAssistant(state.messages, state.sequence, (message) => ({
        content: completed || message.content,
        status: "complete",
        sources: message.sources?.length ? message.sources : state.sources,
      })),
    };
  }

  if (action.type === "realtime.sources") {
    const sources = Array.isArray(action.sources) ? action.sources : [];
    return {
      ...state,
      sources,
      messages: updateLastAssistant(state.messages, () => ({ sources })),
    };
  }

  if (action.type === "realtime.interrupted") return { ...state, phase: "realtime", error: "" };
  if (action.type === "error") return { ...state, phase: "error", error: action.message || "连接暂时不可用" };
  if (action.type === "clear.error") return { ...state, error: "", phase: state.phase === "error" ? "idle" : state.phase };
  if (action.type === "cancel") {
    return {
      ...state,
      phase: "cancelled",
      error: "",
      messages: updateLastAssistant(state.messages, () => ({ status: "cancelled" })),
    };
  }

  return state;
}

const apiUrl = (path) => `${API_BASE}${path}`;
const asList = (value) => (Array.isArray(value) ? value : value ? [value] : [])
  .map((entry) => String(entry || "").trim())
  .filter(Boolean);
const formatValues = (value, fallback = "未提供") => asList(value).join("、") || fallback;
const formatDetailContent = (value) => Array.isArray(value) ? value.map((entry) => `- ${entry}`).join("\n") : String(value || "");
function friendlyVoiceError(error, available) {
  if (!available) return "实时语音暂未就绪。";
  const message = String(error?.message || "");
  if (/notallowed|permission|denied/i.test(message)) return "请允许浏览器使用麦克风。";
  if (/notfound|device|microphone/i.test(message)) return "没有找到可用的麦克风。";
  if (/生成等待过久|first.token.timeout/i.test(message)) return "回答生成超时，请再试一次。";
  if (/回答服务暂时不可用/i.test(message)) return "回答服务暂时不可用，请再试一次。";
  return "实时语音连接失败，请重试。";
}

function Markdown({ text, status, className = "" }) {
  if (!text) {
    const copy = status === "cancelled" ? "已停止回答" : status === "error" ? "没有完成回答" : "正在回答…";
    return <span className="muted-copy">{copy}</span>;
  }
  return (
    <div className={`markdown ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{String(text)}</ReactMarkdown>
    </div>
  );
}

function isCompactMessage(message) {
  const text = String(message?.content || "").trim();
  if (!text || message?.status !== "complete" || text.includes("\n") || text.length > 72) return false;
  return !/[#*_>`|\[\]]/.test(text);
}

function CitationCards({ sources, onOpen }) {
  if (!sources?.length) return null;
  return (
    <div className="citation-group" aria-label={`本轮引用 ${sources.length} 项资料`}>
      <div className="citation-heading">资料引用 · {sources.length}</div>
      <div className="citation-cards">
        {sources.map((item) => (
          <button key={item.id} type="button" className="citation-card" onClick={() => onOpen(item)}>
            <span className="citation-title">{item.title || "未命名案例"}</span>
            <span className="citation-meta">{item.category || "类别未注明"} · 风险{item.risk_level || "未注明"}</span>
            <CaretRight aria-hidden="true" />
          </button>
        ))}
      </div>
    </div>
  );
}

function PromptChips({ prompts, onAsk, className = "", rotationSeed = 0 }) {
  const visiblePrompts = useMemo(() => rotateAndLimit(prompts, rotationSeed), [prompts, rotationSeed]);
  if (!visiblePrompts.length) return null;
  return (
    <div className={`prompt-chips ${className}`}>
      {visiblePrompts.map((prompt) => (
        <button key={prompt} type="button" onClick={() => void onAsk(prompt)}>{prompt}</button>
      ))}
    </div>
  );
}

function VoiceControl({ available, connected, status, onStart, onStop }) {
  const { label, detail } = VOICE_COPY[status] || VOICE_COPY[REALTIME_VOICE_STATUS.IDLE];
  return (
    <button
      type="button"
      className={`voice-orb ${connected ? "active" : ""}`}
      onClick={connected ? onStop : onStart}
      disabled={!available || status === REALTIME_VOICE_STATUS.CONNECTING}
      aria-label={connected ? `结束实时对话，${detail}` : "开启连续实时对话"}
      title={connected ? `${label} · 点击结束` : (available ? "开启连续实时对话" : "实时语音暂未就绪")}
    >
      {status === REALTIME_VOICE_STATUS.CONNECTING ? <SpinnerGap className="spin" /> : (
        <span className="voice-bars" aria-hidden="true"><i /><i /><i /><i /><i /></span>
      )}
      <span className="sr-only">{label}</span>
    </button>
  );
}

function VoiceSpectrum({ values }) {
  return (
    <div className="voice-spectrum" aria-hidden="true">
      {(values?.length ? values : Array(24).fill(0)).map((value, index) => (
        <i key={index} style={{ "--level": Math.max(0.08, Number(value) || 0) }} />
      ))}
    </div>
  );
}

function VoiceStatusRow({ status, connected, error, hasUserPartial, hasAssistantBubble }) {
  const copy = VOICE_COPY[status];
  // Voice state is scoped to an active realtime session. The hook can finish
  // a socket transition with a stale status, but a closed session must never
  // leak a voice prompt into ordinary text chat.
  const disconnectedActionableStatus = status === REALTIME_VOICE_STATUS.CONNECTING;
  if ((!connected && !disconnectedActionableStatus) || !copy || status === REALTIME_VOICE_STATUS.IDLE) return null;
  if (status === REALTIME_VOICE_STATUS.ERROR) {
    return <div className="error-banner voice-error" role="alert">{error || copy.label}</div>;
  }

  const hasVisibleBubble = copy.owner === "user" ? hasUserPartial : hasAssistantBubble;
  if (hasVisibleBubble) return null;

  return (
    <article className={`message-row ${copy.owner} thinking-row voice-status-row`} aria-live="polite">
      {copy.owner === "assistant" && <span className="chat-avatar">识</span>}
      <div className="message-content">
        {copy.owner === "assistant" && <span className="message-author">识诈</span>}
        <div className="thinking-status">
          <span>{copy.label}</span>
          <i /><i /><i />
        </div>
      </div>
    </article>
  );
}

function App() {
  const [pageSessionId] = useState(createConversationSessionId);
  const [state, dispatch] = useReducer(reducer, { ...initialState, sessionId: pageSessionId });
  const [searchDraft, setSearchDraft] = useState("");
  const [questionDraft, setQuestionDraft] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState({ category: "", risk_level: "" });
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [hasMoreItems, setHasMoreItems] = useState(false);
  const [categories, setCategories] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchError, setSearchError] = useState("");
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [promptSeed] = useState(() => getSessionSeed());

  const abortRef = useRef(null);
  const turnRef = useRef(null);
  const requestRef = useRef(0);
  const searchAbort = useRef(null);
  const searchSeq = useRef(0);
  const detailAbort = useRef(null);
  const sourceListRef = useRef(null);
  const itemPageRef = useRef({ key: "", nextOffset: 0, startOffset: 0, wrapped: false, hasMore: true, loading: false, requestId: 0 });
  const chatEnd = useRef(null);
  const sessionRef = useRef(pageSessionId);
  const realtimeRef = useRef(null);

  const cancelActive = useCallback(async () => {
    abortRef.current?.abort();
    if (sessionRef.current && turnRef.current) {
      try {
        await fetch(apiUrl(`/api/chat/${encodeURIComponent(sessionRef.current)}/turn/${encodeURIComponent(turnRef.current)}/cancel`), { method: "POST" });
      } catch {
        // The local stream is already stopped.
      }
    }
    turnRef.current = null;
  }, []);

  // Every new turn enters through this gate. It invalidates the old client
  // stream and stops the browser scheduler before any new text is submitted.
  // Keeping this at the boundary prevents queued/prefetched TTS from
  // surviving a later user question.
  const prepareSubmission = useCallback(() => {
    requestRef.current += 1;
    realtimeRef.current?.stopSpeaking?.();
    return cancelActive();
  }, [cancelActive]);

  const cancelQuestion = useCallback(async () => {
    await prepareSubmission();
    dispatch({ type: "cancel" });
  }, [prepareSubmission]);

  const ask = useCallback(async (raw) => {
    const text = String(raw || "").trim();
    if (!text) return;
    const cancelPromise = prepareSubmission();
    const requestId = requestRef.current;
    const isCurrent = () => requestRef.current === requestId;
    await cancelPromise;
    if (!isCurrent()) return;

    const controller = new AbortController();
    abortRef.current = controller;
    dispatch({ type: "ask", text });
    console.info(`[识诈][trace=${requestId}] llm.request.client.start`);
    let buffer = "";
    let eventName = "message";
    let dataLines = [];
    let firstTextLogged = false;

    const consume = (block) => {
      if (!isCurrent()) return;
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length) {
        try {
          const event = JSON.parse(dataLines.join("\n"));
          const type = event.type || eventName;
          dispatch({ type: "event", event: { ...event, type } });
          if (type === "response.text.delta") {
            if (!firstTextLogged) {
              firstTextLogged = true;
              console.info(`[识诈][trace=${requestId}] llm.first_text_delta`);
            }
            realtimeRef.current?.appendSpeechDelta?.(event.payload?.delta || "");
          }
          if (type === "turn.completed") {
            console.info(`[识诈][trace=${requestId}] text.complete`);
            realtimeRef.current?.finishSpeechStream?.(event.payload?.answer || "");
          }
          if (["turn.failed", "turn.cancelled"].includes(type)) realtimeRef.current?.stopSpeaking?.();
          if (event.turn_id) turnRef.current = event.turn_id;
        } catch {
          // Ignore malformed heartbeat frames.
        }
      }
      eventName = "message";
      dataLines = [];
    };

    try {
      const response = await fetch(apiUrl("/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ question: text, session_id: sessionRef.current, category: filters.category }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error("request_failed");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || "";
        blocks.forEach(consume);
        if (done) {
          if (buffer.trim()) consume(buffer);
          break;
        }
      }
    } catch (error) {
      if (error?.name !== "AbortError" && isCurrent()) {
        dispatch({ type: "error", message: "回答服务暂时不可用" });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [filters.category, prepareSubmission]);

  const recognitionContext = useMemo(() => ({
    category: filters.category,
    risk_level: filters.risk_level,
    visibleItems: items
      .filter((item) => !filters.category || item.category === filters.category)
      .slice(0, 8),
    selectedItem: selected,
    sessionId: sessionRef.current,
  }), [filters.category, filters.risk_level, items, selected]);

  const realtime = useVoiceConversation({
    websocketPath: apiUrl("/api/voice"),
    recognitionContext,
    onUserPartial: (text, event) => dispatch({
      type: "realtime.user.partial",
      text,
      utteranceId: event?.utterance_id,
      revision: event?.revision,
    }),
    onUserTranscript: (text, event) => dispatch({
      type: "realtime.user",
      text,
      utteranceId: event?.utterance_id,
      revision: event?.revision,
    }),
    onAssistantTranscript: (text, event) => dispatch({ type: event?.done ? "realtime.answer.done" : "realtime.answer.delta", text }),
    onBargeIn: () => {
      dispatch({ type: "realtime.interrupted" });
      void cancelActive();
    },
    onSources: (sources) => dispatch({ type: "realtime.sources", sources }),
    onError: (error) => dispatch({ type: "error", message: friendlyVoiceError(error, Boolean(meta?.capabilities?.realtime_voice)) }),
  });

  useEffect(() => {
    realtimeRef.current = realtime;
  }, [realtime]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetch(apiUrl("/api/meta"), { signal: controller.signal }),
      fetch(apiUrl("/api/categories"), { signal: controller.signal }),
    ]).then(async ([metaResponse, categoriesResponse]) => {
      if (metaResponse.ok) setMeta(await metaResponse.json());
      if (categoriesResponse.ok) setCategories(await categoriesResponse.json());
    }).catch((error) => {
      if (error?.name !== "AbortError") setMeta({ risk_levels: [], capabilities: { realtime_voice: false } });
    });
    return () => controller.abort();
  }, []);

  const fetchItemPage = useCallback(async ({ reset = false } = {}) => {
    const initialBrowse = !searchQuery && !filters.category && !filters.risk_level;
    if (initialBrowse && meta === null) return;
    const key = JSON.stringify([searchQuery, filters.category, filters.risk_level]);
    const page = itemPageRef.current;
    if (!reset && (page.loading || (!page.hasMore && !page.error) || page.key !== key)) return;

    const requestId = reset ? ++searchSeq.current : page.requestId;
    if (reset) {
      searchAbort.current?.abort();
      const itemCount = Math.max(0, Number(meta?.item_count) || 0);
      const startOffset = initialBrowse && itemCount > PROJECT_PAGE_SIZE
        ? promptSeed % (itemCount - PROJECT_PAGE_SIZE + 1)
        : 0;
      itemPageRef.current = { key, nextOffset: startOffset, startOffset, wrapped: false, hasMore: true, error: false, loading: true, requestId };
      setItems([]);
      setTotal(0);
      setHasMoreItems(true);
      setSearchError("");
    } else {
      itemPageRef.current = { ...page, error: false, loading: true };
    }
    const controller = new AbortController();
    searchAbort.current = controller;
    setLoading(true);
    const currentPage = itemPageRef.current;
    const offset = currentPage.nextOffset;
    const pageLimit = pageLimitForState(currentPage);
    const query = new URLSearchParams({
      q: searchQuery,
      category: filters.category,
      risk_level: filters.risk_level,
      limit: String(pageLimit),
      offset: String(offset),
    });

    try {
      const response = await fetch(apiUrl(`/api/items?${query}`), { signal: controller.signal });
      if (!response.ok) throw new Error("search_failed");
      const data = await response.json();
      const current = itemPageRef.current;
      if (requestId !== searchSeq.current || current.key !== key) return;
      const incoming = Array.isArray(data.items) ? data.items : [];
      const pageState = nextPageState({
        offset: Number(data.offset ?? offset),
        limit: Number(data.limit ?? PROJECT_PAGE_SIZE),
        total: Number(data.total || 0),
        received: incoming.length,
        startOffset: current.startOffset,
        wrapped: current.wrapped,
      });
      setItems((previous) => reset ? mergePageItems([], incoming) : mergePageItems(previous, incoming));
      setTotal(Number(data.total || 0));
      setHasMoreItems(pageState.hasMore);
      itemPageRef.current = { ...current, nextOffset: pageState.nextOffset, wrapped: pageState.wrapped, hasMore: pageState.hasMore, error: false, loading: false };
    } catch (error) {
      if (error?.name !== "AbortError" && requestId === searchSeq.current && itemPageRef.current.key === key) {
        // Keep already loaded pages visible; the footer offers a retry.
        setSearchError("加载失败");
        itemPageRef.current = { ...itemPageRef.current, loading: false, error: true };
        setHasMoreItems(false);
      }
    } finally {
      if (requestId === searchSeq.current && itemPageRef.current.key === key) {
        itemPageRef.current = { ...itemPageRef.current, loading: false };
        setLoading(false);
      }
    }
  }, [filters, meta, promptSeed, searchQuery]);

  const searchItems = useCallback(() => fetchItemPage({ reset: true }), [fetchItemPage]);
  const loadMoreItems = useCallback(() => fetchItemPage(), [fetchItemPage]);

  useEffect(() => { void searchItems(); }, [searchItems]);
  useEffect(() => {
    const list = sourceListRef.current;
    if (!list || loading || !hasMoreItems || items.length === 0) return;
    if (list.scrollHeight <= list.clientHeight + 8) void loadMoreItems();
  }, [hasMoreItems, items.length, loadMoreItems, loading]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "auto", block: "end" }); }, [state.messages]);
  useEffect(() => () => {
    abortRef.current?.abort();
    searchAbort.current?.abort();
    detailAbort.current?.abort();
    realtime.stop();
  }, [realtime.stop]);

  const openItem = useCallback(async (item) => {
    if (!item?.id) return;
    setRightOpen(true);
    setSelected(item);
    setDetailLoading(true);
    setDetailError("");
    detailAbort.current?.abort();
    const controller = new AbortController();
    detailAbort.current = controller;
    try {
      const response = await fetch(apiUrl(`/api/items/${encodeURIComponent(item.id)}`), { signal: controller.signal });
      if (!response.ok) throw new Error("detail_failed");
      setSelected(await response.json());
    } catch (error) {
      if (error?.name !== "AbortError") setDetailError("暂时无法读取案例详情");
    } finally {
      if (detailAbort.current === controller) setDetailLoading(false);
    }
  }, []);

  const available = Boolean(meta?.capabilities?.realtime_voice);
  const connected = Boolean(realtime.isConnected);
  const voiceStatus = effectiveVoiceStatus(connected, realtime.status, REALTIME_VOICE_STATUS.IDLE);
  const riskLevels = Array.isArray(meta?.risk_levels)
    ? meta.risk_levels
    : (Array.isArray(meta?.levels) ? meta.levels : []);
  const showRiskFilter = riskLevels.length > 1;
  const results = items;
  const shownTotal = total;
  const personaMode = realtime.isPlaying ? "speaking" : "idle";
  const welcomePrompts = useMemo(() => {
    const visibleItems = items.filter((item) => item?.title);
    const category = categories.length ? categories[promptSeed % categories.length] : null;
    const item = visibleItems.length ? visibleItems[(promptSeed * 7) % visibleItems.length] : null;
    const riskCase = visibleItems.length ? visibleItems[(promptSeed * 11 + 1) % visibleItems.length] : null;
    return [
      category ? `有哪些${category.name}诈骗案例值得警惕？` : "按类别浏览反诈案例",
      item ? `${item.title}有哪些风险信号？` : "查找一个反诈案例",
      riskCase ? `遇到${riskCase.title}该如何防范？` : "如何识别常见诈骗套路？",
    ];
  }, [categories, items, promptSeed]);

  const startVoice = async () => {
    if (!available) {
      dispatch({ type: "error", message: friendlyVoiceError(null, false) });
      return;
    }
    const interruptedTextTurn = ["retrieving", "composing", "streaming"].includes(state.phase);
    await prepareSubmission();
    if (interruptedTextTurn) dispatch({ type: "cancel" });
    dispatch({ type: "clear.error" });
    await realtime.start();
  };

  const stopVoice = () => {
    requestRef.current += 1;
    realtime.stop();
    void cancelActive();
    dispatch({ type: "clear.error" });
  };

  const submitQuestion = (raw) => {
    const text = String(raw || "").trim();
    if (!text || voiceStatus === REALTIME_VOICE_STATUS.CONNECTING) return false;
    if (connected) {
      // Stop the previous response before sending the new realtime turn. The
      // voice hook also sends the server barge-in frame as part of sendText;
      // this local gate makes the UI/audio transition immediate.
      void prepareSubmission();
      if (!realtime.sendText(text)) return false;
      dispatch({ type: "realtime.user", text });
      return true;
    }
    void ask(text);
    return true;
  };

  const submit = (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const text = String(form.elements.question.value || "").trim();
    if (!submitQuestion(text)) return;
    setQuestionDraft("");
  };

  const answerInProgress = ["retrieving", "composing", "streaming"].includes(state.phase);
  // Browser TTS can outlive the text turn and the realtime socket may already
  // be disconnected. Use the speech pipeline itself, not voiceStatus, so the
  // ordinary composer keeps showing Stop during request/prefetch/playback.
  const speechInProgress = Boolean(realtime.isSpeechPending || realtime.isPlaying);
  const composerMode = composerAction({
    connected,
    draft: questionDraft,
    answerInProgress,
    speechInProgress,
  });
  const stopComposer = () => {
    if (composerMode === "stop") void cancelQuestion();
  };

  const handleSourceScroll = useCallback((event) => {
    const list = event.currentTarget;
    if (loading || !hasMoreItems) return;
    if (list.scrollHeight - list.scrollTop - list.clientHeight < 180) void loadMoreItems();
  }, [hasMoreItems, loadMoreItems, loading]);

  const applyFilters = (next) => {
    // A filter change submits the whole form the user can currently see.
    // Keeping an older committed keyword while the field shows another value
    // makes a valid category look empty (for example an invisible “皮影戏”
    // combined with “传统体育、游艺与杂技”).
    setSelected(null);
    setSearchQuery(searchDraft.trim());
    setFilters(next);
  };

  const updateSearchDraft = (value) => {
    setSearchDraft(value);
    // Clearing the visible field is itself an unambiguous search action; do
    // not leave an invisible committed query affecting later filters.
    if (!value.trim()) setSearchQuery("");
  };

  const waitingForFirstToken = !connected && ["retrieving", "composing"].includes(state.phase)
    && state.messages.at(-1)?.role === "user";
  const latestMessage = state.messages.at(-1);
  const hasUserPartial = latestMessage?.role === "user"
    && latestMessage.status === "transcribing"
    && Boolean(String(latestMessage.content || "").trim());
  const hasAssistantBubble = latestMessage?.role === "assistant"
    && Boolean(String(latestMessage.content || "").trim());
  const showConversationEmpty = state.messages.length === 0
    && voiceStatus === REALTIME_VOICE_STATUS.IDLE
    && !state.error;
  const showVoiceStatusOnly = state.messages.length === 0 && !showConversationEmpty;

  return (
    <main className={`app-shell voice-${voiceStatus}`}>
      <div className="mobile-toggle-row">
        <button type="button" aria-expanded={leftOpen} onClick={() => setLeftOpen((value) => !value)}>熊猫警官</button>
        <button type="button" aria-expanded={rightOpen} onClick={() => setRightOpen((value) => !value)}>反诈案例</button>
      </div>

      <div className="workspace">
        <aside className={`left-panel ${leftOpen ? "is-open" : ""}`}>
          <button className="panel-dismiss" type="button" onClick={() => setLeftOpen(false)} aria-label="关闭熊猫警官面板"><X /></button>
          <DigitalHuman mode={personaMode} />
        </aside>

        <section className="center-panel">
          <div className="conversation-toolbar">
            <span className="conversation-title">与识诈对话</span>
          </div>

          <div className="chat-timeline">
            <div className={`message-stack ${showConversationEmpty ? "is-empty" : ""} ${showVoiceStatusOnly ? "is-status-only" : ""}`}>
              {showConversationEmpty && (
                <section className="conversation-empty" aria-label="开始对话">
                  <h1>想先了解哪类诈骗风险？</h1>
                  <PromptChips prompts={welcomePrompts} rotationSeed={promptSeed} onAsk={submitQuestion} className="starter-row" />
                </section>
              )}
              {state.messages.map((message) => {
                const compact = isCompactMessage(message);
                return (
                  <article key={message.id} className={`message-row ${message.role}`}>
                    {message.role === "assistant" && <span className="chat-avatar">识</span>}
                    <div className="message-content">
                      {message.role === "assistant" && <span className="message-author">识诈</span>}
                      <div className={`message-bubble ${compact ? "compact" : ""}`}><Markdown text={message.content} status={message.status} /></div>
                      <CitationCards sources={message.sources} onOpen={openItem} />
                      {message.role === "assistant" && <PromptChips prompts={message.suggestions} rotationSeed={promptSeed + Number(String(message.id).replace(/\D/g, ""))} onAsk={submitQuestion} />}
                    </div>
                  </article>
                );
              })}
              {waitingForFirstToken && (
                <article className="message-row assistant thinking-row" aria-live="polite">
                  <span className="chat-avatar">识</span>
                  <div className="message-content">
                    <span className="message-author">识诈</span>
                    <div className="thinking-status">
                      <span>{state.phase === "retrieving" ? "查找资料中" : "生成回答中"}</span>
                      <i /><i /><i />
                    </div>
                  </div>
                </article>
              )}
              <VoiceStatusRow
                status={voiceStatus}
                connected={connected}
                error={state.error || friendlyVoiceError(realtime.error, available)}
                hasUserPartial={hasUserPartial}
                hasAssistantBubble={hasAssistantBubble}
              />
              {state.error && voiceStatus !== REALTIME_VOICE_STATUS.ERROR && <div className="error-banner" role="alert">{state.error}</div>}
              <div ref={chatEnd} />
            </div>
          </div>

          <p className="sr-only" role="status" aria-live="polite">{state.announcement}</p>
          <div className="composer-card">
            <div className="composer-row">
              <form onSubmit={submit} className="composer-form">
                {connected
                  ? <VoiceSpectrum values={realtime.spectrum} />
                  : (
                    <textarea
                      name="question"
                      rows="1"
                      maxLength="500"
                      value={questionDraft}
                      onChange={(event) => setQuestionDraft(event.target.value)}
                      placeholder="向识诈提问"
                      aria-label="输入问题"
                    />
                  )}
                {!connected && (
                  <button
                    className={`send-button ${composerMode === "stop" ? "is-stop" : ""}`}
                    type={composerMode === "stop" ? "button" : "submit"}
                    onClick={composerMode === "stop" ? stopComposer : undefined}
                    aria-label={composerMode === "stop" ? "停止" : "发送"}
                  >
                    {composerMode === "stop" ? <Stop /> : <ArrowUp />}
                  </button>
                )}
              </form>
              <VoiceControl
                available={available}
                connected={connected}
                status={voiceStatus}
                onStart={startVoice}
                onStop={stopVoice}
              />
            </div>
          </div>
        </section>

        <aside className={`right-panel ${rightOpen ? "is-open" : ""}`}>
          <button className="panel-dismiss" type="button" onClick={() => setRightOpen(false)} aria-label="关闭反诈案例"><X /></button>
          {selected ? (
            <ItemDetail item={selected} loading={detailLoading} error={detailError} onBack={() => setSelected(null)} />
          ) : (
            <>
              <div className="right-tools">
                <div className="right-heading">
                  <h2>反诈案例</h2>
                  <span>{riskLevels.length === 1 ? `风险${riskLevels[0]}` : "案例库"}</span>
                </div>
                <form className="search-form" onSubmit={(event) => {
                  event.preventDefault();
                  setSearchQuery(searchDraft.trim());
                  if (searchQuery === searchDraft.trim()) void searchItems();
                }}>
                  <input value={searchDraft} onChange={(event) => updateSearchDraft(event.target.value)} placeholder="搜索案例、套路或关键词" aria-label="搜索反诈案例" />
                  <button type="submit">搜索</button>
                </form>
                <div className={`filter-row ${showRiskFilter ? "" : "single-filter"}`}>
                  {showRiskFilter && (
                    <label className="filter-select">
                      <span className="sr-only">风险等级</span>
                      <select value={filters.risk_level} onChange={(event) => applyFilters({ ...filters, risk_level: event.target.value })}>
                        <option value="">全部风险等级</option>
                        {riskLevels.map((riskLevel) => <option key={riskLevel} value={riskLevel}>{riskLevel}</option>)}
                      </select>
                    </label>
                  )}
                  <label className="filter-select">
                    <span className="sr-only">诈骗类别</span>
                    <select value={filters.category} onChange={(event) => applyFilters({ ...filters, category: event.target.value })}>
                      <option value="">全部类别</option>
                      {categories.map((category) => <option key={category.id || category.name} value={category.name}>{category.name}</option>)}
                    </select>
                  </label>
                </div>
                <div className="result-head">
                  <span>{loading && results.length === 0 ? "检索中" : `共 ${shownTotal} 个案例`}</span>
                  <span>{loading && results.length === 0 ? "载入中" : `已载入 ${results.length} 个`}</span>
                </div>
              </div>
              <div className="source-list-frame">
                <div ref={sourceListRef} className="source-list" aria-busy={loading} onScroll={handleSourceScroll}>
                  {results.map((item) => <SourceItem key={item.id} item={item} onOpen={openItem} />)}
                  {!loading && !searchError && results.length === 0 && <div className="empty-state">没有匹配案例</div>}
                  {loading && results.length > 0 && <div className="source-list-status" aria-live="polite">继续加载中</div>}
                  {!loading && searchError && (
                    <button type="button" className="source-list-retry" onClick={() => { setSearchError(""); void loadMoreItems(); }}>
                      加载失败，点击重试
                    </button>
                  )}
                  {!loading && !searchError && !hasMoreItems && results.length > 0 && (
                    <div className="source-list-status">已显示全部 {shownTotal} 个案例</div>
                  )}
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}

function SourceItem({ item, onOpen }) {
  return (
    <button type="button" className="source-item" onClick={() => onOpen(item)}>
      <div className="source-title"><h3>{item?.title || "未命名案例"}</h3><CaretRight /></div>
      <div className="source-meta">{item?.category || "类别未注明"} · 风险{item?.risk_level || "未注明"}</div>
      <p>{item?.risk_signals || item?.summary || "案例暂未提供风险信号。"}</p>
      <div className="source-meta">入口：{formatValues(item?.entry_channels)}</div>
    </button>
  );
}

function ItemDetail({ item, loading, error, onBack }) {
  const sections = [
    ["风险信号", item?.risk_signals],
    ["诈骗手法", item?.key_methods],
    ["防范建议", item?.prevention_advice],
    ["案例说明", item?.content || item?.summary],
  ].filter(([, value]) => value);

  return (
    <div className="item-detail">
      <div className="detail-top"><button type="button" onClick={onBack}><ArrowLeft />返回案例</button></div>
      <div className="detail-scroll">
        <span className="detail-kicker">风险{item?.risk_level || "未注明"}</span>
        <h2>{item?.title || "未命名案例"}</h2>
        <div className="detail-tags"><span>{item?.category || "类别未注明"}</span><span>入口：{formatValues(item?.entry_channels)}</span></div>
        {item?.source_name && <p className="detail-source">来源：{item.source_name}</p>}
        {loading && <div className="detail-loading"><SpinnerGap className="spin" />正在读取完整案例</div>}
        {error && <div className="search-error" role="alert">{error}</div>}
        {!loading && sections.map(([title, content]) => <section key={title}><h3>{title}</h3><Markdown text={formatDetailContent(content)} className="detail-markdown" /></section>)}
      </div>
    </div>
  );
}

export { App };
