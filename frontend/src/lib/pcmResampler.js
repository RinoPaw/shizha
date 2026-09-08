const DEFAULT_TARGET_RATE = 16000;

function clampSample(value) {
  return Math.max(-1, Math.min(1, Number(value) || 0));
}

function toPcm16(samples) {
  const pcm = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const value = clampSample(samples[index]);
    pcm[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return pcm.buffer;
}

/**
 * Streaming linear resampler for microphone PCM.
 *
 * The fractional source position and the final source sample are retained
 * between calls. This is important for AudioWorklet callbacks: a callback
 * boundary is not an audio boundary, so restarting the interpolation phase
 * for every block introduces a discontinuity and changes the sample count.
 */
export class StatefulPcmResampler {
  constructor(inputRate, targetRate = DEFAULT_TARGET_RATE) {
    this.inputRate = 0;
    this.targetRate = 0;
    this.step = 0;
    this.buffer = new Float32Array(0);
    this.position = 0;
    this.configure(inputRate, targetRate);
  }

  configure(inputRate, targetRate = DEFAULT_TARGET_RATE) {
    const nextInputRate = Number(inputRate);
    const nextTargetRate = Number(targetRate);
    if (!(nextInputRate > 0) || !(nextTargetRate > 0)) {
      throw new RangeError("Audio sample rates must be positive");
    }
    if (this.inputRate !== nextInputRate || this.targetRate !== nextTargetRate) {
      this.inputRate = nextInputRate;
      this.targetRate = nextTargetRate;
      this.step = nextInputRate / nextTargetRate;
      this.reset();
    }
    return this;
  }

  reset() {
    this.buffer = new Float32Array(0);
    this.position = 0;
    return this;
  }

  process(samples) {
    if (!samples || samples.length === 0) return new ArrayBuffer(0);
    const incoming = samples instanceof Float32Array
      ? samples
      : Float32Array.from(samples);
    const combined = new Float32Array(this.buffer.length + incoming.length);
    combined.set(this.buffer);
    combined.set(incoming, this.buffer.length);
    this.buffer = combined;

    const output = [];
    // Linear interpolation needs the following source sample. Keep that
    // final sample in `buffer` when compacting below.
    while (this.position + 1 < this.buffer.length) {
      const lower = Math.floor(this.position);
      const fraction = this.position - lower;
      const sample = this.buffer[lower] * (1 - fraction)
        + this.buffer[lower + 1] * fraction;
      output.push(sample);
      this.position += this.step;
    }

    // Do not compact past the final retained sample. `position` can be ahead
    // of it by up to one step while waiting for the next callback's samples.
    const consumed = Math.min(
      Math.floor(this.position),
      Math.max(0, this.buffer.length - 1),
    );
    if (consumed > 0) {
      this.buffer = this.buffer.slice(consumed);
      this.position -= consumed;
    }
    return toPcm16(output);
  }

  /**
   * Emit the final interpolated sample when a capture stream is ending.
   * Normal microphone operation does not need to call this because silence
   * callbacks continue to provide the look-ahead sample.
   */
  flush() {
    if (this.buffer.length === 0 || this.position >= this.buffer.length) {
      return new ArrayBuffer(0);
    }
    const lower = Math.floor(this.position);
    const fraction = this.position - lower;
    const next = this.buffer[Math.min(lower + 1, this.buffer.length - 1)];
    const sample = this.buffer[lower] * (1 - fraction) + next * fraction;
    this.reset();
    return toPcm16([sample]);
  }
}

/**
 * Encode one capture block with a caller-owned streaming resampler. Keeping
 * the state explicit prevents a call site from accidentally reintroducing a
 * per-callback phase reset.
 */
export function encodePcm(samples, inputRate, resampler) {
  if (!(resampler instanceof StatefulPcmResampler)) {
    throw new TypeError("encodePcm requires a StatefulPcmResampler");
  }
  resampler.configure(inputRate, DEFAULT_TARGET_RATE);
  return resampler.process(samples);
}

export { DEFAULT_TARGET_RATE };
