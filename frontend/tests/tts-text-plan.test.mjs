import test from "node:test";
import assert from "node:assert/strict";
import { TtsTextPlan } from "../src/lib/ttsTextPlan.js";

test("commits exactly one first sentence and one complete remainder", () => {
  const plan = new TtsTextPlan();
  assert.equal(plan.append("第一句。第二"), "第一句。");
  assert.equal(plan.append("句。第三句。尾"), null);
  assert.equal(plan.append("巴。"), null);
  assert.equal(plan.finish(), "第二句。第三句。尾巴。");
  assert.equal(plan.finish(), null);
  assert.equal(plan.append("迟到文本。"), null);
});

test("uses the whole answer as the only segment when no early sentence arrives", () => {
  const plan = new TtsTextPlan();
  assert.equal(plan.append("没有标点的回答"), null);
  assert.equal(plan.finish(), "没有标点的回答");
});

test("commits an English full stop without waiting for the complete answer", () => {
  const plan = new TtsTextPlan("en-US");
  assert.equal(plan.append("The first sentence. The sec"), "The first sentence.");
  assert.equal(plan.append("ond sentence. A third sentence."), null);
  assert.equal(plan.finish(), "The second sentence. A third sentence.");
});

test("does not split decimals, common abbreviations, or initialisms", () => {
  const plan = new TtsTextPlan("en-US");
  assert.equal(
    plan.append("The value is 3.14 and Dr. Zhang used e.g. paper cutting. Next."),
    "The value is 3.14 and Dr. Zhang used e.g. paper cutting.",
  );
  assert.equal(plan.finish(), "Next.");
});

test("keeps a closing quote with the completed English sentence", () => {
  const plan = new TtsTextPlan("en-US");
  assert.equal(plan.append('She said “Hello.” Then left.'), 'She said “Hello.”');
  assert.equal(plan.finish(), "Then left.");
});
