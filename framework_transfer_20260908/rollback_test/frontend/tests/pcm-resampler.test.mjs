import test from "node:test";
import assert from "node:assert/strict";

import { StatefulPcmResampler, encodePcm } from "../src/lib/pcmResampler.js";
import {
  appendPreRoll,
  createPreRollState,
  PRE_ROLL_MS,
} from "../src/lib/pcmPreRoll.js";

function joinPcm16(buffers) {
  const values = buffers.map((buffer) => new Int16Array(buffer));
  const joined = new Int16Array(values.reduce((total, value) => total + value.length, 0));
  let offset = 0;
  for (const value of values) {
    joined.set(value, offset);
    offset += value.length;
  }
  return joined;
}

test("retains resampling phase across 48 kHz callback boundaries", () => {
  const input = Float32Array.from({ length: 24 }, (_, index) => index / 24);
  const whole = new StatefulPcmResampler(48000);
  const split = new StatefulPcmResampler(48000);
  const expected = new Int16Array(whole.process(input));
  const actual = joinPcm16([
    split.process(input.slice(0, 5)),
    split.process(input.slice(5, 13)),
    split.process(input.slice(13)),
  ]);
  assert.deepEqual(actual, expected);
});

test("retains fractional phase for 44.1 kHz input", () => {
  const input = Float32Array.from({ length: 441 }, (_, index) => Math.sin(index / 13));
  const whole = new StatefulPcmResampler(44100);
  const split = new StatefulPcmResampler(44100);
  const expected = new Int16Array(whole.process(input));
  const outputs = [];
  for (let offset = 0; offset < input.length; offset += 37) {
    outputs.push(split.process(input.slice(offset, Math.min(input.length, offset + 37))));
  }
  assert.deepEqual(joinPcm16(outputs), expected);
});

test("encodePcm uses the supplied streaming state", () => {
  const state = new StatefulPcmResampler(48000);
  const first = encodePcm(new Float32Array([0, 0.25, 0.5, 0.75]), 48000, state);
  const second = encodePcm(new Float32Array([1, 0.5, 0, -0.5]), 48000, state);
  assert.ok(first.byteLength > 0);
  assert.ok(second.byteLength > 0);
  assert.notEqual(state.position, 0);
});

test("pre-roll is bounded by milliseconds rather than callback count", () => {
  let state = createPreRollState();
  const chunk = new Int16Array(80).buffer;
  for (let index = 0; index < 8; index += 1) {
    state = appendPreRoll(state, chunk, 50);
  }
  assert.ok(state.durationMs <= PRE_ROLL_MS);
  assert.ok(state.durationMs >= 250);
  assert.equal(state.chunks.length, 6);
});

test("the 300 ms pre-roll covers the 160 ms playback onset gate", () => {
  let state = createPreRollState();
  const chunk = new Int16Array(160).buffer;
  for (let index = 0; index < 6; index += 1) state = appendPreRoll(state, chunk, 50);
  assert.equal(state.durationMs, 300);
  assert.ok(state.durationMs - 160 >= 80);
  assert.ok(state.durationMs - 160 <= 150);
});
