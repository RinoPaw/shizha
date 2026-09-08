import test from "node:test";
import assert from "node:assert/strict";
import { TtsScheduler } from "../src/lib/ttsScheduler.js";

class FakeAudio extends EventTarget {
  constructor(url) {
    super();
    this.url = url;
    this.src = url;
    this.currentTime = 0;
    this.preload = "";
    this.playCalls = 0;
    this.pauseCalls = 0;
    this.loadCalls = 0;
  }

  play() { this.playCalls += 1; return Promise.resolve(); }
  pause() { this.pauseCalls += 1; }
  load() { this.loadCalls += 1; }
  emit(type) { this.dispatchEvent(new Event(type)); }
}

function makeScheduler() {
  const audios = [];
  const events = [];
  const terminals = [];
  const playingChanges = [];
  const scheduler = new TtsScheduler({
    createAudio: (url) => {
      const audio = new FakeAudio(url);
      audios.push(audio);
      return audio;
    },
    onEvent: (event) => events.push(event),
    onPlayingChange: (playing) => playingChanges.push(playing),
    onTerminal: (event) => terminals.push(event),
  });
  return { scheduler, audios, events, terminals, playingChanges };
}

test("preloads the second segment before the first segment ends", async () => {
  const { scheduler, audios, events, terminals, playingChanges } = makeScheduler();
  scheduler.begin();

  assert.equal(scheduler.enqueue("第一句。", { url: "/tts/0", reason: "first_sentence" }), true);
  assert.equal(audios.length, 1);
  assert.equal(audios[0].playCalls, 1);
  audios[0].emit("loadeddata");
  audios[0].emit("playing");

  assert.equal(scheduler.enqueue("剩余内容。", { url: "/tts/1", reason: "text_complete" }), true);
  assert.equal(audios.length, 2);
  assert.equal(audios[1].loadCalls, 1);
  assert.equal(audios[1].playCalls, 0);

  scheduler.complete();
  audios[0].emit("ended");
  assert.equal(audios[1].playCalls, 1);
  audios[1].emit("loadeddata");
  audios[1].emit("playing");
  audios[1].emit("ended");
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(scheduler.isPlaying, false);
  assert.deepEqual(playingChanges, [true, false, true, false]);
  assert.equal(terminals.length, 1);
  assert.equal(terminals[0].failed, false);
  assert.deepEqual(events.filter((event) => event.type === "request.start").map((event) => event.segment), [0, 1]);
  assert.deepEqual(events.filter((event) => event.type === "first_audio_chunk").map((event) => event.segment), [0, 1]);
});

test("scheduler has no tentative pause path; candidate energy cannot stop playback", () => {
  const { scheduler, audios, playingChanges } = makeScheduler();
  scheduler.begin();
  scheduler.enqueue("播报内容。", { url: "/tts/0" });
  audios[0].emit("playing");

  assert.equal("pauseTentative" in scheduler, false);
  assert.equal("resume" in scheduler, false);
  assert.equal("isPaused" in scheduler, false);
  assert.equal(scheduler.isPlaying, true);
  assert.deepEqual(playingChanges, [true]);
});

test("stop releases current and prefetched audio and ignores late callbacks", () => {
  const { scheduler, audios, terminals, playingChanges } = makeScheduler();
  scheduler.begin();
  scheduler.enqueue("第一句。", { url: "/tts/0" });
  audios[0].emit("playing");
  scheduler.enqueue("剩余内容。", { url: "/tts/1" });

  scheduler.stop();
  assert.equal(scheduler.isPlaying, false);
  assert.equal(audios[0].pauseCalls, 1);
  assert.equal(audios[0].loadCalls, 1);
  assert.equal(audios[1].pauseCalls, 1);
  assert.equal(audios[1].loadCalls, 2);

  audios[0].emit("playing");
  audios[0].emit("ended");
  audios[1].emit("playing");
  audios[1].emit("ended");
  assert.equal(scheduler.isPlaying, false);
  assert.deepEqual(playingChanges, [true, false]);
  assert.equal(terminals.length, 0);
});

test("a new generation isolates stale audio and never creates a third segment", () => {
  const { scheduler, audios, terminals } = makeScheduler();
  scheduler.begin();
  scheduler.enqueue("旧第一句。", { url: "/tts/old-0" });
  const oldAudio = audios[0];
  scheduler.stop();

  scheduler.begin();
  scheduler.enqueue("新第一句。", { url: "/tts/new-0" });
  scheduler.enqueue("新剩余内容。", { url: "/tts/new-1" });
  assert.equal(scheduler.enqueue("不应出现。", { url: "/tts/new-2" }), false);

  oldAudio.emit("playing");
  oldAudio.emit("ended");
  assert.equal(scheduler.isPlaying, false);
  assert.equal(audios[1].playCalls, 1);
  assert.equal(audios[2].playCalls, 0);
  scheduler.complete();
  audios[1].emit("ended");
  audios[2].emit("ended");
  assert.equal(terminals.length, 1);
});

test("completion with no generated segment terminates cleanly", () => {
  const { scheduler, terminals } = makeScheduler();
  scheduler.begin();
  assert.equal(scheduler.complete(), true);
  assert.equal(terminals.length, 1);
  assert.equal(terminals[0].failed, false);
});
