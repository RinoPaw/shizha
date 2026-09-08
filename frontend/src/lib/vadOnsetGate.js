/**
 * Duration-based VAD onset gate.
 *
 * AudioWorklet callback sizes are not a speech duration. The gate therefore
 * accumulates samples / inputRate instead of counting callbacks.
 */
const NORMAL_POLICY = Object.freeze({ threshold: 0.018, thresholdMs: 60 });
const PLAYBACK_POLICY = Object.freeze({ threshold: 0.022, thresholdMs: 160 });

export function getVADOnsetPolicy(playbackActive) {
  return playbackActive ? PLAYBACK_POLICY : NORMAL_POLICY;
}

export function createVADGateState() {
  return { aboveMs: 0 };
}

export function advanceVADGate(
  state,
  { isAboveThreshold, samplesLength, inputRate, thresholdMs },
) {
  const previous = Number(state?.aboveMs) || 0;
  const required = Math.max(0, Number(thresholdMs) || 0);
  const durationMs = Number(samplesLength) > 0 && Number(inputRate) > 0
    ? (Number(samplesLength) / Number(inputRate)) * 1000
    : 0;

  if (!isAboveThreshold || durationMs <= 0) {
    return { aboveMs: 0, onset: false };
  }

  const aboveMs = previous + durationMs;
  return { aboveMs, onset: previous < required && aboveMs >= required };
}
