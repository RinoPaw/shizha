import test from "node:test";
import assert from "node:assert/strict";

import {
  advanceVADGate,
  createVADGateState,
  getVADOnsetPolicy,
} from "../src/lib/vadOnsetGate.js";

test("accumulates voiced duration from samples and input rate", () => {
  let state = createVADGateState();
  let result = advanceVADGate(state, {
    isAboveThreshold: true, samplesLength: 128, inputRate: 48000, thresholdMs: 60,
  });
  assert.equal(result.onset, false);
  assert.equal(result.aboveMs, 128 / 48000 * 1000);

  state = result;
  for (let index = 0; index < 22; index += 1) {
    result = advanceVADGate(state, {
      isAboveThreshold: true, samplesLength: 128, inputRate: 48000, thresholdMs: 60,
    });
    state = result;
  }
  assert.equal(result.onset, true);
  assert.ok(result.aboveMs >= 60);
});

test("a transient below threshold resets the gate immediately", () => {
  const state = advanceVADGate(createVADGateState(), {
    isAboveThreshold: true, samplesLength: 4096, inputRate: 48000, thresholdMs: 100,
  });
  assert.ok(state.aboveMs > 0);
  const reset = advanceVADGate(state, {
    isAboveThreshold: false, samplesLength: 128, inputRate: 48000, thresholdMs: 100,
  });
  assert.equal(reset.aboveMs, 0);
  assert.equal(reset.onset, false);
});

test("playback uses a reachable fixed threshold and rejects short impacts", () => {
  assert.deepEqual(getVADOnsetPolicy(false), { threshold: 0.018, thresholdMs: 60 });
  assert.deepEqual(getVADOnsetPolicy(true), { threshold: 0.022, thresholdMs: 160 });

  const frame = { isAboveThreshold: true, samplesLength: 480, inputRate: 48000 };
  let normal = createVADGateState();
  let playback = createVADGateState();
  let normalOnset = false;
  let playbackOnset = false;
  for (let index = 0; index < 12; index += 1) {
    normal = advanceVADGate(normal, { ...frame, thresholdMs: 60 });
    playback = advanceVADGate(playback, { ...frame, thresholdMs: 160 });
    normalOnset ||= normal.onset;
    playbackOnset ||= playback.onset;
  }
  assert.equal(normalOnset, true);
  assert.equal(playbackOnset, false);

  for (let index = 0; index < 4; index += 1) {
    playback = advanceVADGate(playback, { ...frame, thresholdMs: 160 });
    playbackOnset ||= playback.onset;
  }
  assert.equal(playbackOnset, true);
});
