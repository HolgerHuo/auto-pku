# Pipeline internals

Stages run in this order: `frames`, `slides`, `theme`, `asr`, `merge`, and
`notes`. A stage is cached when its manifest entry matches the video
fingerprint (size, modification time, and samples from the start and end).

| Stage | Work | Cache |
|---|---|---|
| frames | local thumbnail scan and frame extraction | `frames.json`, `frames/` |
| slides | one vision request per retained frame | `slides.json`, `slides_kept.json` |
| theme | one text request over early slide text | `theme.json` |
| asr | one transcription request per audio chunk | `chunks/`, `asr_chunks.json` |
| merge | local timeline alignment | `sections.json` |
| notes | one request per note unit plus a summary | `notes_units.json` |

## Frames and slides

Frames are scanned at `SCAN_FPS` in bounded `SCAN_WINDOW` intervals. A visual
change must exceed `MAD_THRESHOLD` and remain stable for `DWELL_SECONDS`.
Retained full-resolution frames are sent as `image_url` content. The response
contract is:

```text
kind: slide | blackboard | demo_video | chrome
text: <near-verbatim text>
content: <description of formulas, diagrams, and figures>
```

Chrome-only frames are excluded from downstream notes and ASR vocabulary.
Classification is biased toward retaining content because a false exclusion
would silently remove lecture material.

## Theme and ASR

The theme stage reads the course name, lecture name, and first six usable slide
texts. It returns `language`, `theme`, and up to ten `key_terms`. Explicit CLI
values take precedence. Slide terms and inferred terms provide transcription
context.

ASR uses fixed `ASR_CHUNK_SECONDS` spans. Media extraction tries an M4A stream
copy, AAC encoding, then 16 kHz mono WAV. Requests contain `model`, `language`,
`hotwords`, `prompt`, and `response_format=json`. Since the pipeline consumes
flat text, transcript timestamps are chunk-level.

Each completed span is cached by its boundaries. The quality filter removes
known filler and collapses runs of more than three identical sentences. If it
changes a span larger than `ASR_MIN_RETRY_CHUNK`, that span is split and retried.

HTTP 429, 5xx, and transport errors retry with capped exponential backoff.
Other client errors fail immediately.

## Merge and notes

Slide times define sections. Sentences receive estimated times based on their
character positions within each ASR chunk, then attach to exactly one section.
Long slide-free intervals are split at `MAX_SECTION`.

Section material is split on sentence or clause boundaries and packed into
units no larger than `NOTES_UNIT_CHARS`. This bounds request size even when a
slide or transcript contains an unusually long unpunctuated block.

Each unit is synthesized into study notes. Failed synthesis falls back to raw
material so content is not lost. Unit cache keys include prompt version, span,
and material digest. A final bounded request builds the overview, main thread,
key points, requirements, homework, and open questions.

`notes.md` and `transcription.md` are rendered from the same cached timeline.
