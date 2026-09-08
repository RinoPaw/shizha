const MAX_SEGMENTS = 2;

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

/**
 * Schedules the two browser TTS segments produced by TtsTextPlan.
 *
 * Segment zero is played as soon as it is enqueued. Segment one is created and
 * preloaded as soon as it is enqueued (normally when the complete text arrives)
 * and is only started after segment zero ends. Every callback is tied to the
 * generation that created it, so a stopped response cannot resurrect audio.
 */
export class TtsScheduler {
  constructor({
    createAudio = (url) => new Audio(url),
    onEvent = noop,
    onPlayingChange = noop,
    onTerminal = noop,
  } = {}) {
    this.createAudio = createAudio;
    this.onEvent = onEvent;
    this.onPlayingChange = onPlayingChange;
    this.onTerminal = onTerminal;
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
      audio = this.createAudio(url, segment);
      if (!audio) throw new Error("tts_audio_unavailable");
      audio.preload = "auto";
    } catch (error) {
      this.fail(generation, error);
      return false;
    }

    const entry = {
      audio,
      content,
      generation,
      segment,
      reason,
      startedAt,
      started: false,
      ended: false,
      firstChunk: false,
      handlers: null,
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
        this.fail(generation, error);
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
      error: (event) => this.fail(entry.generation, event || new Error("tts_audio_error")),
    };
    entry.handlers = handlers;
    Object.entries(handlers).forEach(([type, handler]) => audio.addEventListener?.(type, handler));
  }

  start(entry) {
    if (!this.isCurrent(entry) || entry.started || this.current) return false;
    entry.started = true;
    this.current = entry;
    let result;
    try {
      result = entry.audio.play?.();
    } catch (error) {
      this.fail(entry.generation, error);
      return false;
    }
    if (result?.catch) result.catch((error) => this.fail(entry.generation, error));
    return true;
  }

  end(entry) {
    if (!this.isCurrent(entry) || entry.ended) return;
    entry.ended = true;
    const wasCurrent = this.current === entry;
    if (wasCurrent) this.current = null;
    if (wasCurrent) this.setPlaying(false);
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

export default TtsScheduler;
