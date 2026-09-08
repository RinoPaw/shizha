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
