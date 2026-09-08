/**
 * Two-phase barge-in state.
 *
 * An audio onset is only a candidate: it may be a cup, a chair, or speaker
 * echo.  It becomes a real interruption only after ASR supplies non-empty
 * text.  Keeping this as pure data makes the ordering rules testable without
 * constructing an AudioContext or a WebSocket.
 */
export const BARGE_IN_PHASE = Object.freeze({
  IDLE: "idle",
  TENTATIVE: "tentative",
});

export function createBargeInState() {
  return { phase: BARGE_IN_PHASE.IDLE, utteranceId: 0, startedAt: 0 };
}

export function beginBargeInCandidate(state, utteranceId, startedAt) {
  if (state?.phase === BARGE_IN_PHASE.TENTATIVE) return state;
  return {
    phase: BARGE_IN_PHASE.TENTATIVE,
    utteranceId: Number(utteranceId) || 0,
    startedAt: Number(startedAt) || 0,
  };
}

export function confirmBargeIn(state) {
  if (state?.phase !== BARGE_IN_PHASE.TENTATIVE) return { state, confirmed: false };
  return {
    state: createBargeInState(),
    confirmed: true,
  };
}

export function shouldConfirmBargeInText(text) {
  // ASR providers can emit punctuation for a click or clink. A confirmed
  // interruption must contain at least one spoken letter/number; one-character
  // commands such as “停” remain valid.
  return /[\p{L}\p{N}]/u.test(String(text ?? ""));
}
