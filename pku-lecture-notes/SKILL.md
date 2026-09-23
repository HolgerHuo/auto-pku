---
name: pku-lecture-notes
description: Convert a lecture recording into synthesized notes.md and a timestamped transcription.md. Use for course notes, class notes, or transcripts from recordings.
---

# Lecture notes

Convert a lecture recording into:

- `notes.md`: detailed study notes organized by what the lecture TAUGHT (a
  content-based outline, not time or slide order): a lecture-level summary,
  then topic sections that fold the slide content into the prose, with
  requirements and homework surfaced up front. Reads like a careful student's
  after-class notes.
- `transcription.md`: the full ASR transcript with chunk-level timestamps.

The pipeline calls configurable OpenAI-compatible vision and transcription
services. Local processing is limited to frame scanning and media slicing with
ffmpeg.

## Run

```bash
cd <repo>/pku-lecture-notes
export VISION_BASE_URL=https://vision.example.com VISION_API_KEY=<key>
export ASR_BASE_URL=https://asr.example.com ASR_API_KEY=<key>
uv run python scripts/run_pipeline.py ../courses/<course>/lec1
```

Process every lecture directory under a course with `--course`. Cached stages
resume automatically. To rerun selected work:

```bash
uv run python scripts/run_pipeline.py ../courses/<course>/lec1 \
  --force-stage asr --stages asr,merge,notes
```

Optional overrides include `--language zh`, `--theme "..."`, `--prompt "..."`,
and `--video path`.

## Pipeline

Run stages in this order:

1. `frames`: scan low-resolution thumbnails and retain stable visual changes.
2. `slides`: read each retained frame and exclude desktop chrome.
3. `theme`: infer language, topic, and vocabulary from early slides.
4. `asr`: transcribe fixed-size chunks using slide vocabulary as context.
5. `merge`: align sentences with slide-defined sections.
6. `notes`: synthesize bounded note units, then integrate them into a
   content-organized lecture document (a planning call proposes a topic
   outline over the units; one call per outline section merges that
   section's units into detailed notes), plus a lecture-level summary.

Do not skip dependencies when forcing a stage. Changes to slides affect theme,
ASR hints, merging, and notes; changes to ASR affect merging and notes.

## Verification

Before finishing, inspect both generated files. Confirm that proper nouns agree
with slides, requirements and deadlines retain exact details, filler and loops
are absent, and the notes do not invent unsupported material. For `notes.md`,
confirm the body is organized by content (topic headings, no time-based or
slide-by-slide sections), that the slide content is folded into the prose
instead of dumped as raw `[Slide ...]` lines, and that no fallback labels like
"Part of the lecture" or "Lecture segment" remain.

## Recording downloader

`course-download/pku_video.py` downloads PKU classroom recordings using cookies
exported from the user's logged-in browser. Keep cookie files under
`course-download/tmp/`; that directory is ignored by Git. Run the downloader
with `uv run --with requests python course-download/pku_video.py --help`.

## References

- `references/pipeline.md`: stages, cache artifacts, and configuration.
- `references/troubleshooting.md`: provider-neutral failure diagnosis.
