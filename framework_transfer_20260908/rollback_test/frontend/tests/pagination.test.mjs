import test from "node:test";
import assert from "node:assert/strict";
import {
  mergePageItems,
  nextPageState,
  pageLimitForState,
} from "../src/lib/pagination.js";

test("mergePageItems appends pages once by stable item id", () => {
  const first = [{ id: "a" }, { id: "b" }];
  const second = [{ id: "b" }, { id: "c" }, { id: "d" }];
  assert.deepEqual(mergePageItems(first, second), [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "d" }]);
});

test("nextPageState stops exactly at the API total", () => {
  assert.deepEqual(nextPageState({ offset: 0, limit: 30, total: 61, received: 30 }), { nextOffset: 30, wrapped: false, hasMore: true });
  assert.deepEqual(nextPageState({ offset: 30, limit: 30, total: 61, received: 30 }), { nextOffset: 60, wrapped: false, hasMore: true });
  assert.deepEqual(nextPageState({ offset: 60, limit: 30, total: 61, received: 1 }), { nextOffset: 61, wrapped: false, hasMore: false });
  assert.deepEqual(nextPageState({ offset: 50, limit: 30, total: 61, received: 11, startOffset: 20 }), { nextOffset: 0, wrapped: true, hasMore: true });
  assert.deepEqual(nextPageState({ offset: 0, limit: 20, total: 61, received: 20, startOffset: 20, wrapped: true }), { nextOffset: 20, wrapped: true, hasMore: false });
  assert.deepEqual(nextPageState({ offset: 0, limit: 30, total: 0, received: 0 }), { nextOffset: 0, wrapped: false, hasMore: false });
});

test("a randomized start traverses the entire stable result set exactly once", () => {
  const all = Array.from({ length: 61 }, (_, index) => ({ id: String(index) }));
  const requests = [];
  let loaded = [];
  let state = {
    nextOffset: 20,
    startOffset: 20,
    wrapped: false,
    hasMore: true,
  };

  while (state.hasMore) {
    const limit = pageLimitForState(state, 30);
    const offset = state.nextOffset;
    requests.push({ offset, limit });
    const incoming = all.slice(offset, offset + limit);
    loaded = mergePageItems(loaded, incoming);
    state = {
      ...state,
      ...nextPageState({
        offset,
        total: all.length,
        received: incoming.length,
        startOffset: state.startOffset,
        wrapped: state.wrapped,
      }),
    };
  }

  assert.deepEqual(requests, [
    { offset: 20, limit: 30 },
    { offset: 50, limit: 30 },
    { offset: 0, limit: 20 },
  ]);
  assert.equal(loaded.length, all.length);
  assert.deepEqual(new Set(loaded.map((item) => item.id)), new Set(all.map((item) => item.id)));
});
