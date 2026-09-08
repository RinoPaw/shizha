import { useCallback, useEffect, useRef, useState } from "react";
import {
  normalizeVoiceText,
  normalizeUtteranceId,
  normalizeVoiceId,
  REALTIME_VOICE_STATUS,
  shouldAcceptServerTurn,
  shouldAcceptServerVoiceStatus,
  shouldAcceptUtteranceEvent,
} from "./voiceState.js";
import { TtsTextPlan } from "../lib/ttsTextPlan.js";
import { TtsScheduler } from "../lib/ttsScheduler.js";
import { acceptPartialRevision, reconcilePartialText, startsNewTranscript } from "../lib/voiceTranscript.js";
import {
  BARGE_IN_PHASE,
  beginBargeInCandidate as makeBargeInCandidate,
  confirmBargeIn,
  createBargeInState,
  shouldConfirmBargeInText,
} from "./bargeInState.js";
import { advanceVADGate, createVADGateState, getVADOnsetPolicy } from "../lib/vadOnsetGate.js";
import { encodePcm, StatefulPcmResampler } from "../lib/pcmResampler.js";
import { appendPreRoll, createPreRollState, drainPreRoll, PRE_ROLL_MS } from "../lib/pcmPreRoll.js";
import { DEFAULT_LOCALE, getPreferredLocales, normalizeLocaleHint } from "../lib/locale.js";

const TARGET_RATE = 16000;
const NORMAL_VAD_POLICY = getVADOnsetPolicy(false);
const SILENCE_MS = 680;
const MAX_UTTERANCE_MS = 20000;

function websocketUrl(path) {
  const url = new URL(path, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export function resolveSpeechLocale(value, fallback = DEFAULT_LOCALE) {
  return normalizeLocaleHint(value) || normalizeLocaleHint(fallback) || DEFAULT_LOCALE;
}

export function assistantEventLocale(message, fallback = DEFAULT_LOCALE) {
  return resolveSpeechLocale(
    message?.locale
      || message?.response_locale
      || message?.payload?.locale
      || message?.payload?.response_locale,
    fallback,
  );
}

export function buildTtsUrl({
  websocketPath,
  text,
  traceId,
  segment,
  reason,
  locale,
}) {
  const ttsPath = String(websocketPath || "/api/voice").replace(/\/voice$/, "/tts");
  const speechLocale = resolveSpeechLocale(locale);
  return `${ttsPath}?text=${encodeURIComponent(String(text || ""))}`
    + `&trace_id=${encodeURIComponent(String(traceId || ""))}`
    + `&segment=${encodeURIComponent(String(segment ?? 0))}`
    + `&reason=${encodeURIComponent(String(reason || ""))}`
    + `&locale=${encodeURIComponent(speechLocale)}`;
}

export function compactRecognitionContext(context) {
  const value = context && typeof context === "object" ? context : {};
  const values = [];
  for (const key of ["selectedTitle", "selectedItem", "selected"]) {
    if (value[key]) values.push(value[key]);
  }
  for (const key of ["titles", "visibleTitles", "visibleItems", "items"]) {
    const entries = Array.isArray(value[key]) ? value[key] : [];
    values.push(...entries);
  }
  const titles = [];
  const seen = new Set();
  for (const entry of values) {
    const title = String(entry && typeof entry === "object" ? entry.title || "" : entry || "").trim();
    if (!title || seen.has(title)) continue;
    seen.add(title);
    titles.push(title.slice(0, 200));
    if (titles.length >= 8) break;
  }
  const rawPreferredLocales = Array.isArray(value.preferredLocales)
    ? value.preferredLocales
    : Array.isArray(value.preferred_locales)
      ? value.preferred_locales
      : [];
  const requestedLocale = normalizeLocaleHint(value.localeHint || value.locale_hint || value.locale);
  const preferredLocales = getPreferredLocales({
    languages: [...(requestedLocale ? [requestedLocale] : []), ...rawPreferredLocales],
  });
  const localeHint = requestedLocale || preferredLocales[0] || DEFAULT_LOCALE;

  return {
    category: String(value.category || "").trim().slice(0, 200),
    titles,
    selected_title: String(value.selectedTitle || value.selectedItem?.title || value.selected?.title || "").trim().slice(0, 200),
    session_id: String(value.sessionId || "").trim().slice(0, 128),
    locale_hint: localeHint,
    preferred_locales: preferredLocales,
  };
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(1, samples.length));
}

function rememberIgnoredTurn(ignoredTurns, turnId) {
  if (!turnId) return;
  ignoredTurns.add(turnId);
  if (ignoredTurns.size > 32) ignoredTurns.delete(ignoredTurns.values().next().value);
}

function acceptAssistantTurn(message, activeTurn, ignoredTurns) {
  const turnId = normalizeVoiceId(message?.turn_id);
  if (!turnId || ignoredTurns.has(turnId)) return false;
  if (activeTurn.current && activeTurn.current !== turnId) return false;
  activeTurn.current = turnId;
  return true;
}

export function useBrowserDuplexVoice({
  websocketPath = "/api/voice",
  recognitionContext = null,
  onUserPartial,
  onUserTranscript,
  onAssistantTranscript,
  onBargeIn,
  onSources,
  onError,
} = {}) {
  const [status, setStatus] = useState(REALTIME_VOICE_STATUS.IDLE);
  const [error, setError] = useState(null);
  const [isMuted, setMuted] = useState(false);
  const [connected, setConnected] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSpeechPending, setSpeechPending] = useState(false);
  const [spectrum, setSpectrum] = useState(() => Array(24).fill(0));
  const socketRef = useRef(null);
  const recognitionContextRef = useRef(recognitionContext);
  recognitionContextRef.current = recognitionContext;
  const recognitionContextKey = JSON.stringify(compactRecognitionContext(recognitionContext));
  const streamRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const transportGenerationRef = useRef(0);
  const statusRef = useRef(status);
  const utteranceRef = useRef(false);
  const utteranceStartedAtRef = useRef(0);
  const lastVoiceRef = useRef(0);
  const aboveRef = useRef(createVADGateState());
  const preRollRef = useRef(createPreRollState());
  const resamplerRef = useRef(null);
  const mutedRef = useRef(false);
  const speechActiveRef = useRef(false);
  const inputBlockedUntilRef = useRef(0);
  const spectrumUpdatedAtRef = useRef(0);
  const speechPipelineRef = useRef(false);
  const assistantPendingRef = useRef(false);
  const speechTraceRef = useRef("");
  const speechLocaleRef = useRef(DEFAULT_LOCALE);
  const speechTextPlanRef = useRef(new TtsTextPlan(DEFAULT_LOCALE));
  const ttsSchedulerRef = useRef(null);
  const bargeInRef = useRef(createBargeInState());
  const assistantTurnRef = useRef("");
  const ignoredAssistantTurnsRef = useRef(new Set());
  const latestUtteranceIdRef = useRef(0);
  const activeUtteranceIdRef = useRef(0);
  const nextUtteranceIdRef = useRef(0);
  const partialRevealRef = useRef({
    utteranceId: 0,
    revision: 0,
    finalized: false,
    visible: "",
    target: "",
    timer: null,
    event: null,
  });
  const callbacks = useRef({ onUserPartial, onUserTranscript, onAssistantTranscript, onBargeIn, onSources, onError });
  callbacks.current = { onUserPartial, onUserTranscript, onAssistantTranscript, onBargeIn, onSources, onError };

  const clearPartialReveal = useCallback((resetText = false) => {
    const reveal = partialRevealRef.current;
    if (reveal.timer !== null) {
      window.clearInterval(reveal.timer);
      reveal.timer = null;
    }
    if (resetText) {
      reveal.utteranceId = 0;
      reveal.revision = 0;
      reveal.finalized = false;
      reveal.visible = "";
      reveal.target = "";
      reveal.event = null;
    }
  }, []);

  const publishUserPartial = useCallback((message) => {
    const reveal = partialRevealRef.current;
    const accepted = acceptPartialRevision(reveal, message);
    if (!accepted.accepted) return false;
    if (startsNewTranscript(reveal, message)) {
      // A committed turn owns a new bubble. Multiple VAD utterances can still
      // belong to one spoken turn, so keep their already revealed prefix.
      clearPartialReveal(false);
      reveal.visible = "";
      reveal.target = "";
    }
    const target = String(message?.text || "");
    const reconciled = reconcilePartialText(reveal.visible, target);
    reveal.utteranceId = accepted.state.utteranceId;
    reveal.revision = accepted.state.revision;
    reveal.finalized = false;
    reveal.visible = reconciled.visible;
    reveal.target = reconciled.target;
    reveal.event = message;
    callbacks.current.onUserPartial?.(reveal.visible, message);

    if (reveal.timer === null && reveal.visible.length < reveal.target.length) {
      reveal.timer = window.setInterval(() => {
        const current = partialRevealRef.current;
        if (current.visible.length >= current.target.length) {
          window.clearInterval(current.timer);
          current.timer = null;
          return;
        }
        current.visible += current.target[current.visible.length];
        callbacks.current.onUserPartial?.(current.visible, current.event);
      }, 26);
    }
    return true;
  }, [clearPartialReveal]);

  const publishUserTranscript = useCallback((message, transcript) => {
    const reveal = partialRevealRef.current;
    const id = Number(message?.utterance_id);
    if (Number.isSafeInteger(id) && id > 0 && id < reveal.utteranceId) return false;
    clearPartialReveal(false);
    reveal.utteranceId = Number.isSafeInteger(id) && id > 0 ? id : reveal.utteranceId;
    reveal.revision = Number(message?.revision) || reveal.revision;
    reveal.finalized = true;
    reveal.visible = String(transcript || "");
    reveal.target = reveal.visible;
    reveal.event = message;
    callbacks.current.onUserTranscript?.(reveal.visible, message);
    return true;
  }, [clearPartialReveal]);

  const setStatusValue = useCallback((value) => {
    // A delayed server `listening` event must not erase the visible
    // transcribing/thinking state of the current turn. The transport may send
    // that event after a newer utterance has already been finalized.
    if (value === REALTIME_VOICE_STATUS.LISTENING
      && (utteranceRef.current || speechPipelineRef.current || assistantPendingRef.current)) return;
    statusRef.current = value;
    setStatus(value);
  }, []);

  const acceptUtteranceMessage = useCallback((message, { required = true } = {}) => {
    const id = normalizeUtteranceId(message?.utterance_id);
    if (!id) return !required;
    if (!shouldAcceptUtteranceEvent(
      id,
      latestUtteranceIdRef.current,
      activeUtteranceIdRef.current,
    )) return false;
    latestUtteranceIdRef.current = Math.max(latestUtteranceIdRef.current, id);
    nextUtteranceIdRef.current = Math.max(nextUtteranceIdRef.current, id);
    return true;
  }, []);

  const reportError = useCallback((value) => {
    const next = value instanceof Error ? value : new Error(String(value || "voice_error"));
    setError(next);
    setStatusValue(REALTIME_VOICE_STATUS.ERROR);
    callbacks.current.onError?.(next);
  }, [setStatusValue]);

  if (!ttsSchedulerRef.current) {
    ttsSchedulerRef.current = new TtsScheduler({
      onEvent: (event) => {
        const trace = speechTraceRef.current || "-";
        const elapsed = Number(event.elapsedMs || 0) / 1000;
        if (event.type === "request.start") {
          console.info(`[识诈][trace=${trace}] tts.request.start segment=${event.segment} reason=${event.reason}`);
        } else if (event.type === "first_audio_chunk") {
          console.info(`[识诈][trace=${trace}] tts.first_audio_chunk segment=${event.segment} +${elapsed.toFixed(3)}s`);
        } else if (event.type === "playing") {
          console.info(`[识诈][trace=${trace}] tts.playing segment=${event.segment} +${elapsed.toFixed(3)}s`);
          setStatusValue(REALTIME_VOICE_STATUS.RESPONDING);
        } else if (event.type === "segment.complete") {
          console.info(`[识诈][trace=${trace}] tts.sentence.complete segment=${event.segment} +${elapsed.toFixed(3)}s`);
        }
      },
      onPlayingChange: (playing) => {
        speechActiveRef.current = playing;
        setIsPlaying(playing);
      },
      onTerminal: ({ failed }) => {
        speechPipelineRef.current = false;
        assistantPendingRef.current = false;
        speechActiveRef.current = false;
        // A candidate cannot outlive the answer it was watching. It never
        // owns TTS, so terminal cleanup only drops the local marker.
        bargeInRef.current = createBargeInState();
        setSpeechPending(false);
        inputBlockedUntilRef.current = performance.now() + 450;
        setStatusValue(REALTIME_VOICE_STATUS.LISTENING);
        if (failed) reportError("speech_output_failed");
      },
    });
  }

  const send = useCallback((payload) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendRecognitionContext = useCallback(() => {
    return send({ type: "context", ...compactRecognitionContext(recognitionContextRef.current) });
  }, [send]);

  const clearBargeInCandidate = useCallback(() => {
    bargeInRef.current = createBargeInState();
  }, []);

  const stopSpeech = useCallback((bargeIn = false, notifyServer = true) => {
    clearBargeInCandidate();
    rememberIgnoredTurn(ignoredAssistantTurnsRef.current, assistantTurnRef.current);
    assistantTurnRef.current = "";
    ttsSchedulerRef.current?.stop();
    speechPipelineRef.current = false;
    assistantPendingRef.current = false;
    speechActiveRef.current = false;
    setSpeechPending(false);
    inputBlockedUntilRef.current = bargeIn ? 0 : performance.now() + 450;
    speechTextPlanRef.current.reset();
    if (bargeIn) {
      if (notifyServer) send({ type: "barge_in" });
      callbacks.current.onBargeIn?.();
    }
  }, [clearBargeInCandidate, send]);

  const confirmBargeInFromAsr = useCallback(() => {
    const confirmed = confirmBargeIn(bargeInRef.current);
    if (!confirmed.confirmed) return false;
    clearBargeInCandidate();
    // Energy onset is only a candidate. ASR text is the evidence that lets us
    // cancel the answer and tell the server about a real interruption.
    stopSpeech(true, true);
    setStatusValue(REALTIME_VOICE_STATUS.USER_SPEAKING);
    return true;
  }, [clearBargeInCandidate, setStatusValue, stopSpeech]);

  const beginBargeInCandidate = useCallback((utteranceId, startedAt) => {
    if (bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE) return false;
    if (!speechPipelineRef.current && !assistantPendingRef.current && !speechActiveRef.current) return false;
    bargeInRef.current = makeBargeInCandidate(
      bargeInRef.current,
      utteranceId,
      startedAt,
    );
    return true;
  }, []);

  const beginSpeechStream = useCallback((locale = "") => {
    const generation = ttsSchedulerRef.current?.begin();
    speechPipelineRef.current = true;
    assistantPendingRef.current = true;
    setSpeechPending(true);
    speechTraceRef.current = globalThis.crypto?.randomUUID?.() || `tts-${Date.now()}`;
    // Locale is owned by the answer generation, just like the TTS scheduler
    // generation.  Snapshot it once so a later browser/context update cannot
    // make the prefetched remainder speak with another voice.
    speechLocaleRef.current = resolveSpeechLocale(
      locale,
      compactRecognitionContext(recognitionContextRef.current).locale_hint,
    );
    speechTextPlanRef.current.reset(speechLocaleRef.current);
    speechActiveRef.current = false;
    aboveRef.current = createVADGateState();
    preRollRef.current = createPreRollState();
    setStatusValue(REALTIME_VOICE_STATUS.THINKING);
    return generation;
  }, [setStatusValue]);

  const enqueueSpeechSegment = useCallback((segment, reason) => {
    const scheduler = ttsSchedulerRef.current;
    if (!scheduler) return false;
    const spokenSegment = String(segment || "").replace(/[#*_`>-]/g, " ").trim();
    const segmentNumber = scheduler.segmentCount;
    const url = buildTtsUrl({
      websocketPath,
      text: spokenSegment,
      traceId: speechTraceRef.current,
      segment: segmentNumber,
      reason,
      locale: speechLocaleRef.current,
    });
    return scheduler.enqueue(spokenSegment, { url, reason });
  }, [websocketPath]);

  const appendSpeechDelta = useCallback((text, locale = "") => {
    const delta = String(text || "");
    if (!delta) return false;
    if (!speechPipelineRef.current) beginSpeechStream(locale);
    const firstSegment = speechTextPlanRef.current.append(delta);
    if (firstSegment) enqueueSpeechSegment(firstSegment, "first_sentence");
    return true;
  }, [beginSpeechStream, enqueueSpeechSegment]);

  const finishSpeechStream = useCallback((fallbackText = "", locale = "") => {
    const scheduler = ttsSchedulerRef.current;
    if (!speechPipelineRef.current) beginSpeechStream(locale);
    const remainder = speechTextPlanRef.current.finish();
    if (remainder) enqueueSpeechSegment(remainder, "text_complete");
    else if (fallbackText && !scheduler?.hasSegments) enqueueSpeechSegment(fallbackText, "text_complete");
    scheduler?.complete();
    return true;
  }, [beginSpeechStream, enqueueSpeechSegment]);

  const speak = useCallback((text, locale = "") => {
    const content = String(text || "").trim();
    if (!content) return false;
    beginSpeechStream(locale);
    appendSpeechDelta(content, locale);
    return finishSpeechStream("", locale);
  }, [appendSpeechDelta, beginSpeechStream, finishSpeechStream]);

  const processAudio = useCallback((samples, inputRate) => {
    if (mutedRef.current) return;
    const now = performance.now();
    if (now < inputBlockedUntilRef.current) {
      aboveRef.current = createVADGateState();
      preRollRef.current = createPreRollState();
      resamplerRef.current?.reset();
      return;
    }
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (!resamplerRef.current || resamplerRef.current.inputRate !== inputRate) {
      resamplerRef.current = new StatefulPcmResampler(inputRate, TARGET_RATE);
    }
    const pcm = encodePcm(samples, inputRate, resamplerRef.current);
    preRollRef.current = appendPreRoll(
      preRollRef.current,
      pcm,
      (samples.length / inputRate) * 1000,
      PRE_ROLL_MS,
    );
    const level = rms(samples);
    const isPlaybackActive = speechActiveRef.current;
    // VAD ownership follows the actual speech pipeline, never the label
    // currently visible in the UI.
    const isAgentBusy = speechPipelineRef.current
      || assistantPendingRef.current
      || isPlaybackActive;
    // Only the first onset needs the stricter echo-aware threshold. Once an
    // utterance owns the microphone, ordinary VAD must control its lifetime;
    // otherwise TTS residue keeps refreshing lastVoice and the ASR packet is
    // never finalized.
    const shouldInterrupt = !utteranceRef.current && isAgentBusy;
    // The capture stream already has browser AEC/noise suppression enabled.
    // Do not learn a second "echo floor" from the mixed microphone signal:
    // while narration is loud that floor rises with the assistant's own voice
    // and can make a real user's sustained speech mathematically unreachable.
    // Playback instead uses one slightly higher, duration-gated threshold.
    const playbackOnset = !utteranceRef.current && isPlaybackActive;
    const onsetPolicy = getVADOnsetPolicy(playbackOnset);
    const threshold = onsetPolicy.threshold;
    aboveRef.current = advanceVADGate(aboveRef.current, {
      isAboveThreshold: level >= threshold,
      samplesLength: samples.length,
      inputRate,
      thresholdMs: onsetPolicy.thresholdMs,
    });

    if (now - spectrumUpdatedAtRef.current >= 50) {
      spectrumUpdatedAtRef.current = now;
      const showInput = !playbackOnset || level >= threshold;
      const bars = Array.from({ length: 24 }, (_, index) => {
        if (!showInput) return 0;
        const start = Math.floor((index / 24) * samples.length);
        const end = Math.max(start + 1, Math.floor(((index + 1) / 24) * samples.length));
        let energy = 0;
        for (let cursor = start; cursor < end; cursor += 1) energy += samples[cursor] * samples[cursor];
        return Math.min(1, Math.sqrt(energy / Math.max(1, end - start)) / 0.12);
      });
      setSpectrum(bars);
    }

    if (!utteranceRef.current && aboveRef.current.onset) {
      const utteranceId = Math.max(
        nextUtteranceIdRef.current,
        latestUtteranceIdRef.current,
      ) + 1;
      nextUtteranceIdRef.current = utteranceId;
      activeUtteranceIdRef.current = utteranceId;
      socket.send(JSON.stringify({
        type: "utterance.start",
        interrupt: shouldInterrupt,
        level: Number(level.toFixed(4)),
        threshold: Number(threshold.toFixed(4)),
      }));
      utteranceRef.current = true;
      utteranceStartedAtRef.current = now;
      lastVoiceRef.current = now;
      if (shouldInterrupt) {
        // Energy-only onset is an internal candidate. Keep the current
        // thinking/responding state and TTS alive until ASR confirms speech.
        beginBargeInCandidate(utteranceId, now);
      } else {
        // Ordinary recording owns the visible user state immediately; only
        // possible barge-in is hidden until ASR provides evidence.
        setStatusValue(REALTIME_VOICE_STATUS.USER_SPEAKING);
      }
      // The duration gate has already isolated a sustained onset. Send the
      // retained window so short commands keep their first syllable; trimming
      // this back to a few AudioWorklet frames makes “等一下” arrive as only
      // its tail.
      drainPreRoll(preRollRef.current).chunks.forEach((frame) => socket.send(frame));
      preRollRef.current = createPreRollState();
      return;
    }
    if (!utteranceRef.current) return;
    if (pcm.byteLength > 0) socket.send(pcm);
    // Continue the utterance with ordinary VAD so a quieter second half is not
    // cut off by the stricter echo-aware onset threshold.
    if (level >= NORMAL_VAD_POLICY.threshold) lastVoiceRef.current = now;
    const utteranceTimedOut = now - utteranceStartedAtRef.current >= MAX_UTTERANCE_MS;
    if (utteranceTimedOut || now - lastVoiceRef.current > SILENCE_MS) {
      socket.send(JSON.stringify({ type: "utterance.end" }));
      utteranceRef.current = false;
      utteranceStartedAtRef.current = 0;
      aboveRef.current = createVADGateState();
      preRollRef.current = createPreRollState();
      if (bargeInRef.current.phase !== BARGE_IN_PHASE.TENTATIVE) {
        setStatusValue(REALTIME_VOICE_STATUS.TRANSCRIBING);
      }
    }
  }, [beginBargeInCandidate, setStatusValue]);

  const cleanup = useCallback(() => {
    transportGenerationRef.current += 1;
    clearPartialReveal(true);
    stopSpeech(false);
    utteranceRef.current = false;
    utteranceStartedAtRef.current = 0;
    latestUtteranceIdRef.current = 0;
    activeUtteranceIdRef.current = 0;
    nextUtteranceIdRef.current = 0;
    clearBargeInCandidate();
    preRollRef.current = createPreRollState();
    resamplerRef.current?.reset();
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current = null;
    if (contextRef.current && contextRef.current.state !== "closed") void contextRef.current.close();
    contextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "client_stop");
    setConnected(false);
  }, [clearBargeInCandidate, clearPartialReveal, stopSpeech]);

  const start = useCallback(async () => {
    if (connected || statusRef.current === REALTIME_VOICE_STATUS.CONNECTING) return;
    const generation = transportGenerationRef.current + 1;
    transportGenerationRef.current = generation;
    clearPartialReveal(true);
    latestUtteranceIdRef.current = 0;
    activeUtteranceIdRef.current = 0;
    nextUtteranceIdRef.current = 0;
    preRollRef.current = createPreRollState();
    resamplerRef.current = null;
    setError(null);
    setStatusValue(REALTIME_VOICE_STATUS.CONNECTING);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (transportGenerationRef.current !== generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const socket = new WebSocket(websocketUrl(websocketPath));
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      await new Promise((resolve, reject) => {
        const timer = window.setTimeout(() => reject(new Error("voice_socket_timeout")), 10000);
        socket.addEventListener("open", () => { window.clearTimeout(timer); resolve(); }, { once: true });
        socket.addEventListener("error", () => { window.clearTimeout(timer); reject(new Error("voice_socket_failed")); }, { once: true });
      });
      if (transportGenerationRef.current !== generation) {
        socket.close(1000, "stale_voice_start");
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      // Recognition context must arrive before the microphone can open a VAD
      // utterance; the effect below only handles later filter/selection
      // changes while this socket remains connected.
      sendRecognitionContext();
      socket.onmessage = (event) => {
        if (socketRef.current !== socket) return;
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === "status") {
          const map = {
            listening: REALTIME_VOICE_STATUS.LISTENING,
            user_speaking: REALTIME_VOICE_STATUS.USER_SPEAKING,
            transcribing: REALTIME_VOICE_STATUS.TRANSCRIBING,
            thinking: REALTIME_VOICE_STATUS.THINKING,
          };
          if (map[message.status]) {
            const nextStatus = map[message.status];
            if (nextStatus !== REALTIME_VOICE_STATUS.THINKING
              && !acceptUtteranceMessage(message, {
                required: nextStatus === REALTIME_VOICE_STATUS.USER_SPEAKING
                  || nextStatus === REALTIME_VOICE_STATUS.TRANSCRIBING,
              })) return;
            if (bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE
              && [REALTIME_VOICE_STATUS.USER_SPEAKING, REALTIME_VOICE_STATUS.TRANSCRIBING].includes(nextStatus)) return;
            if (nextStatus === REALTIME_VOICE_STATUS.THINKING
              && !shouldAcceptServerTurn(
                message.turn_id,
                assistantTurnRef.current,
                assistantPendingRef.current,
                ignoredAssistantTurnsRef.current,
              )) return;
            if (nextStatus === REALTIME_VOICE_STATUS.THINKING) {
              assistantTurnRef.current = normalizeVoiceId(message.turn_id);
            }
            if (!shouldAcceptServerVoiceStatus(nextStatus, statusRef.current, {
              utteranceActive: utteranceRef.current,
              speechPipeline: speechPipelineRef.current,
              assistantPending: assistantPendingRef.current,
              speechActive: speechActiveRef.current,
            })) return;
            setStatusValue(nextStatus);
          }
        } else if (message.type === "user.transcript") {
          if (!acceptUtteranceMessage(message)) return;
          activeUtteranceIdRef.current = 0;
          const transcript = normalizeVoiceText(message.text);
          // The server contract rejects an empty final ASR result. Keep this
          // guard at the transport boundary as well, so a malformed/provider
          // event cannot create a thinking turn with no user bubble.
          if (!transcript) {
            const wasCandidate = bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE;
            if (wasCandidate) clearBargeInCandidate();
            if (!wasCandidate) {
              setStatusValue(REALTIME_VOICE_STATUS.LISTENING);
            }
            return;
          }
          // A final ASR result is definitive proof of user speech. Stop every
          // local audio segment even when the provider emitted no partials.
          if (bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE) confirmBargeInFromAsr();
          else if (speechPipelineRef.current || speechActiveRef.current) stopSpeech(true, false);
          else {
            rememberIgnoredTurn(ignoredAssistantTurnsRef.current, assistantTurnRef.current);
            assistantTurnRef.current = "";
          }
          assistantPendingRef.current = true;
          setStatusValue(REALTIME_VOICE_STATUS.THINKING);
          publishUserTranscript(message, transcript);
        }
        else if (message.type === "user.partial") {
          if (!acceptUtteranceMessage(message)) return;
          if (bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE) {
            if (!shouldConfirmBargeInText(message.text)) return;
            confirmBargeInFromAsr();
          }
          publishUserPartial(message);
        }
        else if (message.type === "assistant.delta") {
          if (!acceptAssistantTurn(message, assistantTurnRef, ignoredAssistantTurnsRef.current)) return;
          const locale = assistantEventLocale(
            message,
            compactRecognitionContext(recognitionContextRef.current).locale_hint,
          );
          callbacks.current.onAssistantTranscript?.(message.text || "", { done: false, locale });
          appendSpeechDelta(message.text || "", locale);
        }
        else if (message.type === "assistant.done") {
          if (!acceptAssistantTurn(message, assistantTurnRef, ignoredAssistantTurnsRef.current)) return;
          const locale = assistantEventLocale(
            message,
            compactRecognitionContext(recognitionContextRef.current).locale_hint,
          );
          callbacks.current.onAssistantTranscript?.(message.text || "", { done: true, locale });
          assistantPendingRef.current = false;
          finishSpeechStream(message.text || "", locale);
          const turnId = normalizeVoiceId(message.turn_id);
          if (turnId) {
            rememberIgnoredTurn(ignoredAssistantTurnsRef.current, turnId);
            if (assistantTurnRef.current === turnId) assistantTurnRef.current = "";
          }
        } else if (message.type === "utterance.rejected") {
          if (!acceptUtteranceMessage(message)) return;
          const wasCandidate = bargeInRef.current.phase === BARGE_IN_PHASE.TENTATIVE;
          if (wasCandidate) clearBargeInCandidate();
          if (!wasCandidate) {
            setStatusValue(REALTIME_VOICE_STATUS.LISTENING);
          }
          activeUtteranceIdRef.current = 0;
        } else if (message.type === "sources") {
          if (!acceptAssistantTurn(message, assistantTurnRef, ignoredAssistantTurnsRef.current)) return;
          callbacks.current.onSources?.(message.items || []);
        }
        else if (message.type === "error") {
          const turnId = normalizeVoiceId(message.turn_id);
          if (turnId && !shouldAcceptServerTurn(
            turnId,
            assistantTurnRef.current,
            assistantPendingRef.current,
            ignoredAssistantTurnsRef.current,
          )) return;
          stopSpeech(false, false);
          reportError(message.message || "voice_error");
        }
      };
      socket.onclose = (event) => {
        // A socket can close after a newer connection has already replaced
        // it. Only the current transport is allowed to tear down media and
        // TTS resources or change the visible state.
        const isCurrentSocket = socketRef.current === socket;
        if (isCurrentSocket) cleanup();
        if (isCurrentSocket && event.code !== 1000 && statusRef.current !== REALTIME_VOICE_STATUS.IDLE) {
          reportError("voice_socket_closed");
        }
      };
      const context = new AudioContext({ latencyHint: "interactive" });
      contextRef.current = context;
      await context.audioWorklet.addModule("/audio-capture-worklet.js");
      if (transportGenerationRef.current !== generation) {
        await context.close();
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      const source = context.createMediaStreamSource(stream);
      const processor = new AudioWorkletNode(context, "shizha-pcm-capture");
      const silent = context.createGain();
      silent.gain.value = 0;
      processor.port.onmessage = (event) => processAudio(event.data, context.sampleRate);
      source.connect(processor).connect(silent).connect(context.destination);
      sourceRef.current = source;
      processorRef.current = processor;
      setConnected(true);
      setStatusValue(REALTIME_VOICE_STATUS.LISTENING);
    } catch (startError) {
      if (transportGenerationRef.current !== generation) return;
      cleanup();
      reportError(startError);
    }
  }, [acceptAssistantTurn, acceptUtteranceMessage, appendSpeechDelta, cleanup, clearBargeInCandidate, clearPartialReveal, connected, confirmBargeInFromAsr, finishSpeechStream, processAudio, publishUserPartial, publishUserTranscript, reportError, sendRecognitionContext, setStatusValue, stopSpeech, websocketPath]);

  const stop = useCallback(() => {
    cleanup();
    setError(null);
    setStatusValue(REALTIME_VOICE_STATUS.IDLE);
  }, [cleanup, setStatusValue]);

  const toggleMute = useCallback(() => {
    const next = !mutedRef.current;
    mutedRef.current = next;
    setMuted(next);
    if (next) {
      // Muted capture is intentionally omitted from the stream. Start a new
      // phase when capture resumes instead of bridging over that gap.
      resamplerRef.current?.reset();
      preRollRef.current = createPreRollState();
    }
    streamRef.current?.getAudioTracks().forEach((track) => { track.enabled = !next; });
    return next;
  }, []);

  useEffect(() => {
    if (connected) sendRecognitionContext();
  }, [connected, recognitionContextKey, sendRecognitionContext]);

  const sendText = useCallback((value) => {
    const text = String(value || "").trim();
    if (!text) return false;
    stopSpeech(true);
    const sent = send({ type: "text", text });
    if (sent) {
      assistantPendingRef.current = true;
      setStatusValue(REALTIME_VOICE_STATUS.THINKING);
    }
    return sent;
  }, [send, setStatusValue, stopSpeech]);

  const cancelResponse = useCallback(() => {
    stopSpeech(true);
    setStatusValue(REALTIME_VOICE_STATUS.LISTENING);
    return true;
  }, [setStatusValue, stopSpeech]);

  useEffect(() => () => {
    clearPartialReveal(true);
    cleanup();
  }, [clearPartialReveal, cleanup]);

  return {
    status, error, isMuted, isConnected: connected, isPlaying, isSpeechPending, spectrum,
    start, stop, toggleMute, mute: toggleMute, sendText, cancelResponse,
    speakText: speak, stopSpeaking: () => stopSpeech(false), beginSpeechStream, appendSpeechDelta, finishSpeechStream,
    sendToolResult: () => false,
  };
}

export default useBrowserDuplexVoice;
