const MAX_SEGMENTS = 2;
const MAX_AUDIO_RETRIES = 2;
const RETRY_DELAYS_MS = [300, 800];

function now() {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

function noop() {}

function callSafely(callback, value) {
  try {
    callback(value);
  } catch {
    // Observability must never break audio scheduling.
  }
}

function withRetryMarker(url, retry) {
  const separator = String(url).includes("?") ? "&" : "?";
  return `${url}${separator}tts_retry=${retry}`;
}

function guessSpeechLocale(text) {
  const value = String(text || "");
  if (/[\u3040-\u30ff]/u.test(value)) return "ja-JP";
  if (/[\uac00-\ud7af]/u.test(value)) return "ko-KR";
  if (/[\u3400-\u9fff]/u.test(value)) return "zh-CN";
  return "en-US";
}

function browserFallbackSpeak(text, { onStart = noop, onEnd = noop, onError = noop } = {}) {
  const synthesis = globalThis?.speechSynthesis;
  const Utterance = globalThis?.SpeechSynthesisUtterance;
  if (!synthesis || typeof synthesis.speak !== "function" || typeof Utterance !== "function") return null;

  let utterance;
  try {
    utterance = new Utterance(String(text || ""));
    utterance.lang = guessSpeechLocale(text);
    utterance.onstart = () => onStart();
    utterance.onend = () => onEnd();
    utterance.onerror = (event) => onError(event || new Error("speech_synthesis_failed"));
    synthesis.speak(utterance);
  } catch (error) {
    onError(error);
    return null;
  }

  return () => {
    utterance.onstart = null;
    utterance.onend = null;
    utterance.onerror = null;
    try {
      synthesis.cancel();
    } catch {
      // Browser speech synthesis may already have stopped itself.
    }
  };
}

/**
 * Schedules the two browser TTS segments produced by TtsTextPlan.
 *
 * Segment zero is played as soon as it is enqueued. Segment one is created and
 * preloaded as soon as it is enqueued (normally when the complete text arrives)
 * and is only started after segment zero ends. Every callback is tied to the
 * generation that created it, so a stopped response cannot resurrect audio.
 *
 * Transient media failures are retried twice. If the network TTS stream still
 * fails, only that segment falls back to the browser's local speech synthesis;
 * a narration failure must never tear down the realtime voice conversation.
 */
export class TtsScheduler {
  constructor({
    createAudio = (url) => new Audio(url),
    onEvent = noop,
    onPlayingChange = noop,
    onTerminal = noop,
    fallbackSpeak = browserFallbackSpeak,
    scheduleRetry = (callback, delay) => setTimeout(callback, delay),
    cancelRetry = (timer) => clearTimeout(timer),
  } = {}) {
    this.createAudio = createAudio;
    this.onEvent = onEvent;
    this.onPlayingChange = onPlayingChange;
    this.onTerminal = onTerminal;
    this.fallbackSpeak = fallbackSpeak;
    this.scheduleRetry = scheduleRetry;
    this.cancelRetry = cancelRetry;
    this.generation = 0;
    this.activeGeneration = 0;
    this.entries = new Map();
    this.nextSegment = 0;
    this.completed = false;
    this.current = null;
    this.playing = false;
    this.terminal = false;
  }

  get isPlaying() {
    return this.playing;
  }

  get segmentCount() {
    return this.entries.size;
  }

  get hasSegments() {
    return this.entries.size > 0;
  }

  begin() {
    this.stop();
    this.generation += 1;
    this.activeGeneration = this.generation;
    this.entries = new Map();
    this.nextSegment = 0;
    this.completed = false;
    this.current = null;
    this.terminal = false;
    return this.activeGeneration;
  }

  enqueue(text, { url, reason = "text_complete" } = {}) {
    const content = String(text || "").trim();
    if (!content || this.completed || this.terminal || this.entries.size >= MAX_SEGMENTS) {
      return false;
    }

    const segment = this.nextSegment;
    this.nextSegment += 1;
    if (segment >= MAX_SEGMENTS || !url) return false;

    const generation = this.activeGeneration;
    const startedAt = now();
    let audio;
    try {
      audio = this.createAudio(url, segment, 0);
      if (!audio) throw new Error("tts_audio_unavailable");
      audio.preload = "auto";
    } catch (error) {
      this.fail(generation, error);
      return false;
    }

    const entry = {
      audio,
      url,
      content,
      generation,
      segment,
      reason,
      startedAt,
      started: false,
      ended: false,
      firstChunk: false,
      handlers: null,
      retryCount: 0,
      retryTimer: null,
      useFallback: false,
      fallbackCancel: null,
    };
    this.entries.set(segment, entry);
    this.bind(entry);
    this.emit({ type: "request.start", segment, reason, elapsedMs: 0 });

    if (segment === 0) {
      this.start(entry);
    } else {
      // Calling load() explicitly makes the second request begin immediately;
      // it must not wait for the first audio element's ended event.
      try {
        audio.load?.();
      } catch (error) {
        this.handleAudioError(entry, error);
      }
      if (!this.current && this.entries.get(0)?.ended) this.start(entry);
    }
    return true;
  }

  complete() {
    if (this.terminal) return false;
    this.completed = true;
    if (!this.entries.size) {
      this.terminal = true;
      this.setPlaying(false);
      callSafely(this.onTerminal, { generation: this.activeGeneration, failed: false });
      return true;
    }
    this.maybeFinish();
    return true;
  }

  stop() {
    return this.confirmStop();
  }

  /** Confirm cancellation and release every current/prefetched resource. */
  confirmStop() {
    this.generation += 1;
    this.activeGeneration = this.generation;
    for (const entry of this.entries.values()) this.release(entry);
    this.entries.clear();
    this.current = null;
    this.completed = false;
    this.terminal = true;
    this.setPlaying(false);
    return true;
  }

  dispose() {
    this.stop();
  }

  isCurrent(entry) {
    return entry.generation === this.activeGeneration && this.entries.get(entry.segment) === entry;
  }

  bind(entry) {
    const { audio } = entry;
    const handlers = {
      loadeddata: () => {
        if (!this.isCurrent(entry) || entry.firstChunk) return;
        entry.firstChunk = true;
        this.emit({ type: "first_audio_chunk", segment: entry.segment, elapsedMs: now() - entry.startedAt });
      },
      playing: () => {
        if (!this.isCurrent(entry)) return;
        this.current = entry;
        this.setPlaying(true);
        this.emit({ type: "playing", segment: entry.segment, elapsedMs: now() - entry.startedAt });
      },
      ended: () => this.end(entry),
      error: (event) => this.handleAudioError(entry, event || new Error("tts_audio_error")),
    };
    entry.handlers = handlers;
    Object.entries(handlers).forEach(([type, handler]) => audio.addEventListener?.(type, handler));
  }

  start(entry) {
    if (!this.isCurrent(entry) || entry.started || this.current) return false;
    if (entry.useFallback) return this.startFallback(entry);

    entry.started = true;
    this.current = entry;
    let result;
    try {
      result = entry.audio.play?.();
    } catch (error) {
      this.handleAudioError(entry, error);
      return false;
    }
    if (result?.catch) result.catch((error) => this.handleAudioError(entry, error));
    return true;
  }

  startFallback(entry) {
    if (!this.isCurrent(entry) || entry.started || this.current) return false;
    entry.started = true;
    this.current = entry;
    this.emit({ type: "fallback.start", segment: entry.segment, elapsedMs: now() - entry.startedAt });

    const onStart = () => {
      if (!this.isCurrent(entry) || this.current !== entry) return;
      this.setPlaying(true);
      this.emit({ type: "fallback.playing", segment: entry.segment, elapsedMs: now() - entry.startedAt });
    };
    const onEnd = () => {
      if (!this.isCurrent(entry)) return;
      this.end(entry);
    };
    const onError = () => {
      if (!this.isCurrent(entry)) return;
      this.emit({ type: "fallback.failed", segment: entry.segment, elapsedMs: now() - entry.startedAt });
      this.end(entry);
    };

    const cancel = callFallback(this.fallbackSpeak, entry.content, { onStart, onEnd, onError });
    if (!cancel) {
      this.emit({ type: "fallback.unavailable", segment: entry.segment, elapsedMs: now() - entry.startedAt });
      this.end(entry);
      return false;
    }
    entry.fallbackCancel = typeof cancel === "function" ? cancel : null;
    return true;
  }

  end(entry) {
    if (!this.isCurrent(entry) || entry.ended) return;
    entry.ended = true;
    const wasCurrent = this.current === entry;
    if (wasCurrent) this.current = null;
    if (wasCurrent) this.setPlaying(false);
    entry.fallbackCancel = null;
    this.emit({ type: "segment.complete", segment: entry.segment, elapsedMs: now() - entry.startedAt });

    const next = this.entries.get(entry.segment + 1);
    if (next && !next.started) this.start(next);
    this.maybeFinish();
  }

  maybeFinish() {
    if (this.terminal || !this.completed || this.current || !this.entries.size) return;
    const allEnded = [...this.entries.values()].every((entry) => entry.ended);
    if (!allEnded) return;
    this.terminal = true;
    this.setPlaying(false);
    callSafely(this.onTerminal, { generation: this.activeGeneration, failed: false });
  }

  handleAudioError(entry, reason) {
    if (!this.isCurrent(entry) || this.terminal || entry.ended || entry.useFallback) return;
    if (!entry.firstChunk && entry.retryCount < MAX_AUDIO_RETRIES) {
      this.retry(entry, reason);
      return;
    }
    this.prepareFallback(entry, reason);
  }

  retry(entry, reason) {
    if (!this.isCurrent(entry) || entry.retryTimer !== null) return false;
    entry.retryCount += 1;
    const retry = entry.retryCount;
    const delay = RETRY_DELAYS_MS[Math.min(retry - 1, RETRY_DELAYS_MS.length - 1)] || 0;
    const wasCurrent = this.current === entry;
    if (wasCurrent) this.current = null;
    if (wasCurrent) this.setPlaying(false);
    entry.started = false;
    entry.firstChunk = false;
    this.emit({
      type: "request.retry",
      segment: entry.segment,
      retry,
      delayMs: delay,
      elapsedMs: now() - entry.startedAt,
      reason,
    });

    entry.retryTimer = this.scheduleRetry(() => {
      entry.retryTimer = null;
      if (!this.isCurrent(entry) || this.terminal || entry.ended) return;
      try {
        entry.audio.src = withRetryMarker(entry.url, retry);
        entry.audio.preload = "auto";
        entry.audio.load?.();
      } catch (error) {
        this.handleAudioError(entry, error);
        return;
      }

      const previous = this.entries.get(entry.segment - 1);
      if (entry.segment === 0 || previous?.ended) this.start(entry);
    }, delay);
    return true;
  }

  prepareFallback(entry, reason) {
    if (!this.isCurrent(entry) || entry.ended) return false;
    if (entry.retryTimer !== null) {
      this.cancelRetry(entry.retryTimer);
      entry.retryTimer = null;
    }
    const wasCurrent = this.current === entry;
    if (wasCurrent) this.current = null;
    if (wasCurrent) this.setPlaying(false);
    entry.started = false;
    entry.useFallback = true;
    try {
      entry.audio.pause?.();
      entry.audio.removeAttribute?.("src");
      entry.audio.src = "";
      entry.audio.load?.();
    } catch {
      // The network media element is no longer required once fallback begins.
    }
    this.emit({
      type: "request.degraded",
      segment: entry.segment,
      retries: entry.retryCount,
      elapsedMs: now() - entry.startedAt,
      reason,
    });

    const previous = this.entries.get(entry.segment - 1);
    if (entry.segment === 0 || previous?.ended) this.start(entry);
    return true;
  }

  fail(generation, reason) {
    if (generation !== this.activeGeneration || this.terminal) return;
    this.terminal = true;
    for (const entry of this.entries.values()) this.release(entry);
    this.entries.clear();
    this.current = null;
    this.setPlaying(false);
    callSafely(this.onTerminal, { generation, failed: true, reason });
  }

  release(entry) {
    if (entry.retryTimer !== null) {
      this.cancelRetry(entry.retryTimer);
      entry.retryTimer = null;
    }
    if (entry.fallbackCancel) {
      try {
        entry.fallbackCancel();
      } catch {
        // The local speech synthesizer may already have completed.
      }
      entry.fallbackCancel = null;
    }

    const { audio, handlers } = entry;
    if (handlers) {
      Object.entries(handlers).forEach(([type, handler]) => audio.removeEventListener?.(type, handler));
    }
    try {
      audio.pause?.();
    } catch {
      // A stale media element is already being torn down.
    }
    try {
      audio.removeAttribute?.("src");
      audio.src = "";
      audio.load?.();
    } catch {
      // Test doubles and partially initialized media elements may not expose all APIs.
    }
  }

  setPlaying(value) {
    const next = Boolean(value);
    if (this.playing === next) return;
    this.playing = next;
    callSafely(this.onPlayingChange, next);
  }

  emit(event) {
    callSafely(this.onEvent, { ...event, generation: this.activeGeneration });
  }
}

function callFallback(fallbackSpeak, text, callbacks) {
  try {
    return fallbackSpeak?.(text, callbacks) || null;
  } catch (error) {
    callbacks.onError?.(error);
    return null;
  }
}

export default TtsScheduler;
