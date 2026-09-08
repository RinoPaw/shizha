export function createConversationSessionId(cryptoLike = globalThis.crypto) {
  if (typeof cryptoLike?.randomUUID === "function") {
    return cryptoLike.randomUUID();
  }

  if (typeof cryptoLike?.getRandomValues === "function") {
    const words = cryptoLike.getRandomValues(new Uint32Array(4));
    return Array.from(words, (word) => word.toString(16).padStart(8, "0")).join("");
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}
