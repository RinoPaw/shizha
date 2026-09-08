import test from "node:test";
import assert from "node:assert/strict";

import {
  normalizeUtteranceId,
  normalizeVoiceId,
  normalizeVoiceText,
  REALTIME_VOICE_STATUS,
  shouldAcceptServerTurn,
  shouldAcceptServerVoiceStatus,
  shouldAcceptUtteranceEvent,
} from "../src/hooks/voiceState.js";
import { composerAction, effectiveVoiceStatus } from "../src/lib/conversationUiState.js";
import {
  acceptPartialRevision,
  reconcilePartialText,
  startsNewTranscript,
} from "../src/lib/voiceTranscript.js";
import {
  BARGE_IN_PHASE,
  beginBargeInCandidate,
  confirmBargeIn,
  createBargeInState,
  shouldConfirmBargeInText,
} from "../src/hooks/bargeInState.js";

test("a disconnected socket cannot expose a stale realtime status", () => {
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.LISTENING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.IDLE);
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.USER_SPEAKING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.IDLE);
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.THINKING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.IDLE);
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.RESPONDING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.IDLE);
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.CONNECTING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.CONNECTING);
  assert.equal(effectiveVoiceStatus(false, REALTIME_VOICE_STATUS.ERROR, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.IDLE);
  assert.equal(effectiveVoiceStatus(true, REALTIME_VOICE_STATUS.LISTENING, REALTIME_VOICE_STATUS.IDLE), REALTIME_VOICE_STATUS.LISTENING);
});

test("the ordinary composer shares send and stop, while realtime has no button", () => {
  assert.equal(composerAction({ connected: true, draft: "hello", answerInProgress: true }), "voice");
  assert.equal(composerAction({ connected: false, draft: "new question", answerInProgress: true }), "send");
  assert.equal(composerAction({ connected: false, draft: "", answerInProgress: true }), "stop");
  assert.equal(composerAction({ connected: false, draft: "", speechInProgress: true }), "stop");
  assert.equal(composerAction({ connected: false, draft: "", speechInProgress: true, answerInProgress: false }), "stop");
  assert.equal(composerAction({ connected: false, draft: "", answerInProgress: false, speechInProgress: false }), "send");
});

test("empty final ASR text is not a user turn", () => {
  assert.equal(normalizeVoiceText("  \n\t"), "");
  assert.equal(normalizeVoiceText("  你好  "), "你好");
});

test("utterance IDs reject an older final after a newer VAD turn", () => {
  assert.equal(normalizeUtteranceId("2"), 2);
  assert.equal(normalizeUtteranceId("bad"), 0);
  assert.equal(shouldAcceptUtteranceEvent(1, 2, 2), false);
  assert.equal(shouldAcceptUtteranceEvent(2, 1, 2), true);
});

test("turn IDs reject an old thinking status", () => {
  const ignored = new Set(["old-turn"]);
  assert.equal(normalizeVoiceId(" old-turn "), "old-turn");
  assert.equal(shouldAcceptServerTurn("old-turn", "", true, ignored), false);
  assert.equal(shouldAcceptServerTurn("new-turn", "current-turn", true), false);
  assert.equal(shouldAcceptServerTurn("current-turn", "current-turn", true), true);
  assert.equal(shouldAcceptServerTurn("current-turn", "", false), false);
});

test("delayed listening cannot erase a pending answer", () => {
  assert.equal(shouldAcceptServerVoiceStatus(
    REALTIME_VOICE_STATUS.LISTENING,
    REALTIME_VOICE_STATUS.THINKING,
    { assistantPending: true },
  ), false);
});

test("delayed thinking cannot cover an active user utterance", () => {
  assert.equal(shouldAcceptServerVoiceStatus(
    REALTIME_VOICE_STATUS.THINKING,
    REALTIME_VOICE_STATUS.USER_SPEAKING,
    { utteranceActive: true },
  ), false);
});

test("delayed transcribing cannot move a finalized turn backwards", () => {
  assert.equal(shouldAcceptServerVoiceStatus(
    REALTIME_VOICE_STATUS.TRANSCRIBING,
    REALTIME_VOICE_STATUS.THINKING,
    { assistantPending: true },
  ), false);
});

test("thinking is accepted for a newly submitted turn", () => {
  assert.equal(shouldAcceptServerVoiceStatus(
    REALTIME_VOICE_STATUS.THINKING,
    REALTIME_VOICE_STATUS.LISTENING,
    { assistantPending: true },
  ), true);
});

test("server recognition statuses do not depend on obsolete tentative barge state", () => {
  for (const nextStatus of [
    REALTIME_VOICE_STATUS.USER_SPEAKING,
    REALTIME_VOICE_STATUS.TRANSCRIBING,
  ]) {
    assert.equal(shouldAcceptServerVoiceStatus(
      nextStatus,
      REALTIME_VOICE_STATUS.RESPONDING,
      {},
    ), true);
  }
});

test("streamed ASR accepts only newer revisions for the active utterance", () => {
  let state = { utteranceId: 4, revision: 2, finalized: false };
  let result = acceptPartialRevision(state, { utterance_id: 4, revision: 1, text: "你" });
  assert.equal(result.accepted, false);
  result = acceptPartialRevision(state, { utterance_id: 4, revision: 3, text: "你好" });
  assert.equal(result.accepted, true);
  state = result.state;
  result = acceptPartialRevision(state, { utterance_id: 3, revision: 99, text: "旧" });
  assert.equal(result.accepted, false);
});

test("ASR replacement keeps the confirmed common prefix and reveals only provider text", () => {
  assert.deepEqual(reconcilePartialText("你好", "你好呀"), { visible: "你好", target: "你好呀" });
  assert.deepEqual(reconcilePartialText("你好啊", "你好呀"), { visible: "你好", target: "你好呀" });
});

test("a newer utterance can resume after a finalized previous one", () => {
  const result = acceptPartialRevision(
    { utteranceId: 1, revision: 7, finalized: true },
    { utterance_id: 2, revision: 1, text: "下一句" },
  );
  assert.equal(result.accepted, true);
});

test("only a finalized turn resets the reveal when VAD advances", () => {
  assert.equal(startsNewTranscript(
    { utteranceId: 1, revision: 2, finalized: false },
    { utterance_id: 2, revision: 3, text: "前半句 后半句" },
  ), false);
  assert.equal(startsNewTranscript(
    { utteranceId: 1, revision: 2, finalized: true },
    { utterance_id: 2, revision: 1, text: "下一轮" },
  ), true);
});

test("barge-in onset is tentative until ASR has non-empty text", () => {
  const candidate = beginBargeInCandidate(createBargeInState(), 8, 100);
  assert.equal(candidate.phase, BARGE_IN_PHASE.TENTATIVE);
  assert.equal(shouldConfirmBargeInText("   "), false);
  assert.equal(shouldConfirmBargeInText("……！"), false);
  assert.equal(shouldConfirmBargeInText("碰"), true);
  assert.equal(confirmBargeIn(candidate).confirmed, true);
});

test("energy candidate remains tentative until ASR confirms text", () => {
  const candidate = beginBargeInCandidate(
    createBargeInState(),
    12,
    100,
  );
  assert.equal(candidate.phase, BARGE_IN_PHASE.TENTATIVE);
  assert.equal(shouldConfirmBargeInText("   "), false);
  assert.equal(shouldConfirmBargeInText("……"), false);
  assert.equal(shouldConfirmBargeInText("桦"), true);
});
