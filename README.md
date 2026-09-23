# auto-pku

Tools for downloading PKU classroom recordings and converting lecture videos
into study notes and timestamped transcripts.

## Components

- `course-download/pku_video.py`: downloads authenticated PKU classroom HLS
  recordings using browser-exported cookies.
- `pku-lecture-notes/`: a Codex skill and standalone pipeline for frame
  extraction, slide reading, transcription, alignment, and note synthesis.

## Lecture pipeline

Requirements: `uv`, `ffmpeg`, and OpenAI-compatible vision and transcription
services.

Copy `.env.sample` to `.env` and fill in the service URLs and API keys, or
export the same variables in your shell. The `.env` file is ignored by Git.

```bash
cd pku-lecture-notes
uv sync --locked
export VISION_BASE_URL=<openai-compatible-base-url>
export VISION_API_KEY=<key>
export ASR_BASE_URL=<openai-compatible-base-url>
export ASR_API_KEY=<key>
uv run python scripts/run_pipeline.py ../courses/<course>/lec1
```

Recordings may use `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`, `.ts`, or
`.flv`. Generated caches and course media are ignored by Git.

See [the skill instructions](pku-lecture-notes/SKILL.md) and
[pipeline reference](pku-lecture-notes/references/pipeline.md) for stage and
configuration details.

## Downloader

Keep exported cookies in `course-download/tmp/`, which is ignored by Git.

```bash
uv run --with requests python course-download/pku_video.py --help
```

Only download recordings you are authorized to access and retain.

## License

See [LICENSE](LICENSE).
