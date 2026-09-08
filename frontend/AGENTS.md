# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

## Confirmed visual direction

- The current durable direction is a light, full-screen three-column workspace; do not restore a wide
  top header or make the page depend on body scrolling.
- The left column is reserved for the digital human, the center column for the conversation, and the
  right column for searchable project material and detail views.
- Keep the composer at the bottom in a restrained GPT-like form. The realtime voice control sits to the
  right as a genuinely round button; do not turn it into a square or a speech-to-text-looking control.
- Language and dialect support is automatic. Do not add a language selector, language button, or any
  other locale control; infer the current turn from speech/text with browser locale only as a hint.
- Use restrained xuan-paper texture and a quiet light palette. Readable type and compact, consistent
  conversation bubbles take priority over decorative chrome, large empty states, or redundant labels.
- Project imagery is intentionally not displayed for now; do not add empty image regions, “待上传” text,
  or visual placeholders. The digital human itself remains a real video surface.
- Treat `.superdesign/references/shizha-high-fidelity-direction.png` as historical exploration, not a
  license to restore the superseded dark-wood/high-decoration treatment.

## Supported voice path

The only supported voice architecture is:

```text
browser VAD → /api/voice → Xunfei streaming ASR
→ AssistantService / DeepSeek → Edge TTS
→ interruptible browser playback
```

Do not add or document an OpenAI Realtime/WebRTC path. Text chat and realtime voice must continue to
share the same session, turn, cancellation, retrieval, and assistant core. Voice UI state should be
derived from one canonical lifecycle: listening, transcribing, thinking, responding, interrupted, or
error; recognition belongs to the user side and must not fight the assistant state or digital-human
video lifecycle.

- A disconnected voice transport is always visually idle; stale voice status must never show a prompt,
  suppress the normal empty state, or coexist with the text-chat progress indicator.
- Every new user submission atomically invalidates the previous turn and stops current, queued, and
  prefetched TTS before the new turn starts.
- While realtime voice is connected, the former text-input area contains only the live spectrum. Keep
  all send/stop controls out of that area; the round realtime control remains separate on the right.
- A high-confidence VAD onset during playback pauses local TTS immediately but is only a tentative
  interruption. Confirm it when ASR produces non-empty text; if ASR rejects the sound, resume the same
  audio element from its preserved position. Consecutive utterances that overlap provider finalization
  are combined in order, never cancelled, overwritten, or answered as a detached fragment.
