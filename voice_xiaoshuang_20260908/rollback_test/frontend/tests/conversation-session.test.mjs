import assert from "node:assert/strict";
import test from "node:test";

import { createConversationSessionId } from "../src/lib/conversationSession.js";

test("a page conversation uses a newly generated id", () => {
  const values = ["page-one", "page-two"];
  const cryptoLike = { randomUUID: () => values.shift() };

  assert.equal(createConversationSessionId(cryptoLike), "page-one");
  assert.equal(createConversationSessionId(cryptoLike), "page-two");
});

test("the fallback id is built from browser entropy", () => {
  const cryptoLike = {
    getRandomValues(words) {
      words.set([1, 2, 3, 4]);
      return words;
    },
  };

  assert.equal(
    createConversationSessionId(cryptoLike),
    "00000001000000020000000300000004",
  );
});
