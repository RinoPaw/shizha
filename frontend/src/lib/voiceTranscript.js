/** Pure ordering/correction rules for streamed ASR hypotheses. */

function idOf(value) {
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : 0;
}

export function acceptPartialRevision(state, event) {
  const id = idOf(event?.utterance_id);
  const revision = Number(event?.revision);
  const nextRevision = Number.isSafeInteger(revision) && revision >= 0 ? revision : 0;
  if (!id) return { accepted: false, state };
  if (state?.finalized && id <= (state.utteranceId || 0)) return { accepted: false, state };
  if (id < (state.utteranceId || 0)) return { accepted: false, state };
  if (id === (state.utteranceId || 0) && nextRevision <= (state.revision || 0)) {
    return { accepted: false, state };
  }
  return {
    accepted: true,
    state: {
      utteranceId: id,
      revision: nextRevision,
      finalized: false,
    },
  };
}

export function startsNewTranscript(state, event) {
  const nextId = idOf(event?.utterance_id);
  const currentId = idOf(state?.utteranceId);
  return Boolean(state?.finalized && nextId > currentId);
}

/**
 * Keep the already confirmed prefix visible while a provider replacement is
 * being applied. The caller can reveal `target` one character at a time.
 */
export function reconcilePartialText(visible, target) {
  const current = String(visible || "");
  const next = String(target || "");
  if (next.startsWith(current)) return { visible: current, target: next };
  let prefixLength = 0;
  while (prefixLength < current.length
    && prefixLength < next.length
    && current[prefixLength] === next[prefixLength]) prefixLength += 1;
  return { visible: current.slice(0, prefixLength), target: next };
}
