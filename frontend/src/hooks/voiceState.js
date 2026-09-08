export const REALTIME_VOICE_STATUS = Object.freeze({
  IDLE: "idle",
  CONNECTING: "connecting",
  LISTENING: "listening",
  USER_SPEAKING: "user_speaking",
  TRANSCRIBING: "transcribing",
  THINKING: "thinking",
  RESPONDING: "responding",
  ERROR: "error",
});

export function normalizeVoiceText(value) {
  const text = String(value ?? "").trim();
  return text || "";
}

export function normalizeVoiceId(value) {
  const id = String(value ?? "").trim();
  return id || "";
}

export function normalizeUtteranceId(value) {
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : 0;
}

export function shouldAcceptUtteranceEvent(utteranceId, latestId = 0, activeId = 0) {
  const id = normalizeUtteranceId(utteranceId);
  if (!id) return false;
  return id >= latestId && (!activeId || id >= activeId);
}

export function shouldAcceptServerTurn(
  turnId,
  currentTurnId,
  assistantPending,
  ignoredTurns = new Set(),
) {
  const id = normalizeVoiceId(turnId);
  if (!id || ignoredTurns.has(id) || !assistantPending) return false;
  return !currentTurnId || currentTurnId === id;
}

/**
 * Server status messages are advisory: the browser owns the local VAD/TTS
 * phases and must not let a delayed status from an older phase overwrite them.
 */
export function shouldAcceptServerVoiceStatus(
  nextStatus,
  currentStatus,
  {
    utteranceActive = false,
    speechPipeline = false,
    assistantPending = false,
    speechActive = false,
  } = {},
) {
  // A stale listening event must not clear a live utterance or answer.
  if (nextStatus === REALTIME_VOICE_STATUS.LISTENING
    && (utteranceActive || speechPipeline || assistantPending)) return false;

  // Once the user has started a new utterance, an older answer's thinking
  // event is no longer authoritative.
  if (nextStatus === REALTIME_VOICE_STATUS.THINKING && utteranceActive) return false;

  // A transcribing/user-speaking event arriving after a final transcript is
  // stale; the current turn is already waiting for the assistant.
  if (assistantPending && !utteranceActive && [
    REALTIME_VOICE_STATUS.USER_SPEAKING,
    REALTIME_VOICE_STATUS.TRANSCRIBING,
  ].includes(nextStatus)) return false;

  // TTS has already started (or the UI has entered responding); a delayed
  // thinking event cannot move the visible state backwards.
  if (nextStatus === REALTIME_VOICE_STATUS.THINKING
    && (speechActive || currentStatus === REALTIME_VOICE_STATUS.RESPONDING)) return false;

  return true;
}
