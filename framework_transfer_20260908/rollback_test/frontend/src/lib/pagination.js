/**
 * Small, UI-independent helpers for offset/limit project browsing.
 * The API owns ordering; the client only appends pages and removes repeated ids.
 */
export const PROJECT_PAGE_SIZE = 30;

export function mergePageItems(existing, incoming) {
  const result = Array.isArray(existing) ? [...existing] : [];
  const seen = new Set(result.map((item) => item?.id).filter(Boolean));
  for (const item of Array.isArray(incoming) ? incoming : []) {
    const id = item?.id;
    if (!id || seen.has(id)) continue;
    seen.add(id);
    result.push(item);
  }
  return result;
}

export function pageLimitForState(
  { nextOffset = 0, startOffset = 0, wrapped = false },
  pageSize = PROJECT_PAGE_SIZE,
) {
  const safeSize = Math.max(1, Number(pageSize) || PROJECT_PAGE_SIZE);
  const offset = Math.max(0, Number(nextOffset) || 0);
  const boundary = Math.max(0, Number(startOffset) || 0);
  if (!wrapped || offset >= boundary) return safeSize;
  return Math.min(safeSize, boundary - offset);
}

export function nextPageState({ offset = 0, total = 0, received = 0, startOffset = 0, wrapped = false }) {
  const safeOffset = Math.max(0, Number(offset) || 0);
  const safeTotal = Math.max(0, Number(total) || 0);
  const safeReceived = Math.max(0, Number(received) || 0);
  const endOffset = safeOffset + safeReceived;
  const shouldWrap = safeReceived > 0 && endOffset >= safeTotal && safeTotal > 0 && startOffset > 0 && !wrapped;
  const wrappedEnd = wrapped ? Math.min(endOffset, Math.max(0, Number(startOffset) || 0)) : endOffset;
  const nextOffset = shouldWrap ? 0 : wrappedEnd;
  const reachedBoundary = wrapped ? nextOffset >= Math.max(0, Number(startOffset) || 0) : nextOffset >= safeTotal;
  return {
    nextOffset,
    wrapped: wrapped || shouldWrap,
    hasMore: safeReceived > 0 && (shouldWrap || !reachedBoundary),
  };
}
