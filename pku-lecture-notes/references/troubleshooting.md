# Troubleshooting

## Requests fail or time out

Confirm that `VISION_BASE_URL`, `VISION_API_KEY`, `ASR_BASE_URL`, and
`ASR_API_KEY` are set for OpenAI-compatible services. HTTP 429 responses,
server errors, and transport failures are retried automatically. Other client
errors fail immediately so invalid requests do not consume the retry budget.

For repeated timeouts, lower `ASR_CHUNK_SECONDS` or the relevant worker count.
Completed spans and frames remain cached, so rerunning resumes unfinished work.

## ASR rejects an audio chunk

Check the service's file-size, duration, codec, and request-field limits. The
pipeline first attempts an M4A stream copy, then AAC, then 16 kHz mono WAV.
Lower `ASR_CHUNK_SECONDS` if the service imposes a smaller request limit.

## Desktop or watermark text appears in notes

Extend `CHROME_VOCAB` or `CHROME_MARKERS` in `scripts/slides.py`, then rerun:

```bash
uv run python scripts/run_pipeline.py <lecture> \
  --force-stage slides --stages slides,theme,asr,merge,notes
```

For recordings with a persistent side pane, set `CROP=w:h:x:y` to exclude it
from frame-change detection.

## Transcript contains filler or repeated sentences

The ASR stage removes known filler phrases and collapses consecutive duplicate
sentences. When cleanup triggers, it retries that span as smaller chunks. Add a
new recurring filler phrase to `FILLER_PATTERNS` in `scripts/asr.py`, or reduce
`ASR_CHUNK_SECONDS`, then force the `asr` stage.

## No ASR text and no slides

Verify that the selected video has a decodable audio or video stream. For very
static slides, lower `MAD_THRESHOLD` or `DWELL_SECONDS`, run only the `frames`
stage, and inspect `.cache/frames.json` before continuing.
