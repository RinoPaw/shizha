const SENTENCE_BOUNDARY = /(?<=[。！？!?；;\n])/u;

export class TtsTextPlan {
  constructor() { this.reset(); }

  reset() {
    this.buffer = "";
    this.remainder = "";
    this.firstCommitted = false;
    this.finalized = false;
  }

  append(text) {
    if (this.finalized) return null;
    this.buffer += String(text || "");
    const parts = this.buffer.split(SENTENCE_BOUNDARY);
    this.buffer = parts.pop() || "";
    const sentences = parts.map((part) => part.trim()).filter(Boolean);
    let first = null;
    if (!this.firstCommitted && sentences.length) {
      first = sentences.shift();
      this.firstCommitted = true;
    }
    if (sentences.length) this.remainder += sentences.join("");
    return first;
  }

  finish() {
    if (this.finalized) return null;
    this.finalized = true;
    const remainder = `${this.remainder}${this.buffer}`.trim();
    this.buffer = "";
    this.remainder = "";
    return remainder || null;
  }
}
