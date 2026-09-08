export const PRE_ROLL_MS = 300;

export function createPreRollState() {
  return { chunks: [], durationMs: 0 };
}

function normalizeBuffer(buffer) {
  if (buffer instanceof ArrayBuffer) return buffer;
  if (ArrayBuffer.isView(buffer)) {
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  }
  return new ArrayBuffer(0);
}

/**
 * Add a PCM chunk while retaining the most recent `maxDurationMs` of input.
 * Chunks are timestamped by captured input duration, rather than by the
 * number of AudioWorklet callbacks, so this remains correct for any callback
 * size and for both 48 kHz and 44.1 kHz contexts.
 */
export function appendPreRoll(
  state,
  buffer,
  durationMs,
  maxDurationMs = PRE_ROLL_MS,
) {
  const chunkBuffer = normalizeBuffer(buffer);
  const chunkDuration = Math.max(0, Number(durationMs) || 0);
  const maxDuration = Math.max(0, Number(maxDurationMs) || 0);
  if (chunkBuffer.byteLength === 0 || chunkDuration === 0 || maxDuration === 0) {
    return state;
  }

  const chunks = [...(state?.chunks || []), {
    buffer: chunkBuffer,
    durationMs: chunkDuration,
  }];
  let totalDuration = (Number(state?.durationMs) || 0) + chunkDuration;

  while (totalDuration > maxDuration && chunks.length > 0) {
    const oldest = chunks[0];
    const excess = totalDuration - maxDuration;
    if (oldest.durationMs <= excess || oldest.buffer.byteLength < 2) {
      chunks.shift();
      totalDuration -= oldest.durationMs;
      continue;
    }

    // Keep the newest part of an oversized oldest chunk. PCM16 is used on
    // the wire, so trim whole samples and account for the exact duration
    // represented by the retained bytes.
    const frameCount = Math.floor(oldest.buffer.byteLength / 2);
    const framesToDrop = Math.min(
      frameCount,
      Math.max(1, Math.ceil(frameCount * excess / oldest.durationMs)),
    );
    const byteOffset = framesToDrop * 2;
    const retained = oldest.buffer.slice(byteOffset);
    const droppedDuration = oldest.durationMs * framesToDrop / frameCount;
    chunks[0] = {
      buffer: retained,
      durationMs: oldest.durationMs - droppedDuration,
    };
    totalDuration -= droppedDuration;
  }

  return { chunks, durationMs: Math.max(0, totalDuration) };
}

export function drainPreRoll(state) {
  const chunks = (state?.chunks || []).map(({ buffer }) => buffer);
  return { chunks, state: createPreRollState() };
}
