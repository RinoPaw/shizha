import { REALTIME_VOICE_STATUS } from "../hooks/voiceState.js";

/**
 * UI state derived from the transport state. A closed realtime session owns
 * no visible voice state, even if a late socket callback left a stale status
 * in React for one render.
 */
export function effectiveVoiceStatus(connected, status, idleStatus) {
  if (connected) return status;
  // CONNECTING is the only realtime phase that legitimately precedes an open
  // session. Once the transport is closed, every other voice phase belongs to
  // no conversation; connection failures are rendered by the ordinary error
  // banner instead of masquerading as a voice message.
  if (status === REALTIME_VOICE_STATUS.CONNECTING) return status;
  return idleStatus;
}

/**
 * The ordinary composer has one action button. Text always wins; with no
 * text, an active answer/audio pipeline turns that same button into Stop.
 * Realtime mode has no action button inside the composer at all.
 */
export function composerAction({ connected, draft = "", answerInProgress = false, speechInProgress = false }) {
  if (connected) return "voice";
  if (String(draft).trim()) return "send";
  if (answerInProgress || speechInProgress) return "stop";
  return "send";
}
