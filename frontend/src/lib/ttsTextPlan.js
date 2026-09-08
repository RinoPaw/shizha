const HARD_BOUNDARIES = new Set(["。", "！", "!", "？", "?", "；", ";", "\n"]);
const CLOSING_PUNCTUATION = new Set(["\"", "'", "”", "’", ")", "]", "】", "》"]);
const COMMON_ABBREVIATION = /(?:^|\s)(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc)\.$/iu;
const INITIALISM = /(?:^|\s)(?:[a-z]\.){2,}$/iu;

function isPeriodBoundary(text, index) {
  const previous = text[index - 1] || "";
  const next = text[index + 1] || "";
  if (/\d/u.test(previous) && /\d/u.test(next)) return false;
  if (next && !/\s/u.test(next) && !CLOSING_PUNCTUATION.has(next)) return false;
  const prefix = text.slice(Math.max(0, index - 16), index + 1);
  return !COMMON_ABBREVIATION.test(prefix) && !INITIALISM.test(prefix);
}

function splitCompletedSentences(text) {
  const sentences = [];
  let start = 0;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const boundary = HARD_BOUNDARIES.has(character)
      || (character === "." && isPeriodBoundary(text, index));
    if (!boundary) continue;
    let end = index + 1;
    while (end < text.length && CLOSING_PUNCTUATION.has(text[end])) end += 1;
    const sentence = text.slice(start, end).trim();
    if (sentence) sentences.push(sentence);
    start = end;
    index = end - 1;
  }
  return { sentences, remainder: text.slice(start) };
}

export class TtsTextPlan {
  constructor(locale = "zh-CN") { this.reset(locale); }

  reset(locale = this.locale || "zh-CN") {
    this.locale = String(locale || "zh-CN");
    this.buffer = "";
    this.remainder = "";
    this.firstCommitted = false;
    this.finalized = false;
  }

  append(text) {
    if (this.finalized) return null;
    this.buffer += String(text || "");
    const split = splitCompletedSentences(this.buffer);
    this.buffer = split.remainder;
    const sentences = split.sentences;
    let first = null;
    if (!this.firstCommitted && sentences.length) {
      first = sentences.shift();
      this.firstCommitted = true;
    }
    if (sentences.length) {
      const separator = /^(?:zh|yue|ja|ko)(?:-|$)/iu.test(this.locale) ? "" : " ";
      const joined = sentences.join(separator);
      this.remainder += this.remainder && separator ? `${separator}${joined}` : joined;
    }
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

export { splitCompletedSentences };
