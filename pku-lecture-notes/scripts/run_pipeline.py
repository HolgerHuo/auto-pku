#!/usr/bin/env python3
"""Convert a lecture recording into study notes (notes.md) + transcription.md.

  export VISION_BASE_URL=... VISION_API_KEY=... VISION_MODEL=qwen3.8-27b
  export ASR_BASE_URL=...    ASR_API_KEY=...    ASR_MODEL=qwen3-asr-1.7b

  # single lecture
  uv run python scripts/run_pipeline.py courses/games-change-world/lec1
  # whole course (every lecN under the course dir)
  uv run python scripts/run_pipeline.py courses/games-change-world --course

  # resume is the default (fingerprint-gated cache). Redo a stage:
  uv run python scripts/run_pipeline.py courses/.../lec1 --force-stage asr --stages asr,merge,notes

Every stage is cached under <lecture>/.cache keyed by a video fingerprint, so a
run can be interrupted and resumed, and a single stage redone without paying
for the others. Vision+slides run before ASR so the audio leg has a theme +
hotwords prompt ready.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import STAGE_ORDER, Cache, Config, log, srtot, video_fingerprint  # noqa: E402
import asr as asr_mod   # noqa: E402
import frames as frames_mod  # noqa: E402
import merge as merge_mod    # noqa: E402
import notes as notes_mod    # noqa: E402
import slides as slides_mod  # noqa: E402
import theme as theme_mod    # noqa: E402


def find_video(lecture_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = lecture_dir / p
        if not p.exists():
            raise SystemExit(f"video not found: {p}")
        return p
    candidates = sorted(
        [p for p in lecture_dir.iterdir()
         if p.is_file() and p.suffix.lower() in
         (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv")],
        key=lambda p: p.stat().st_size, reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no video file in {lecture_dir}")
    return candidates[0]


def title_for(lecture_dir: Path) -> str:
    course = lecture_dir.parent.name.replace("-", " ").replace("_", " ")
    return f"{course.title()} — {lecture_dir.name}"


def run_lecture(lecture_dir: Path, video: Path, cfg: Config, stages: list[str],
                force: list[str], language: str, theme_override: str,
                prompt_override: str) -> None:
    cache = Cache(lecture_dir)
    fingerprint = video_fingerprint(video)
    log(f"video: {video.name} ({video.stat().st_size / 1e6:.0f} MB) fp={fingerprint}")
    for st in force:
        cache.clear_stage(st)

    duration = frames_mod.ffprobe_duration(video)
    log(f"  duration {srtot(duration)}")

    def want(st: str) -> bool:
        return st in stages

    # ---- frames -------------------------------------------------------------
    if want("frames"):
        if cache.stage_done("frames", fingerprint):
            log("stage frames: cached")
            times = cache.load_json("frames.json")["times"]
            paths = [Path(p) for p in cache.load_json("frames.json")["frames"]]
        else:
            log("stage frames: scanning for slide changes")
            times = frames_mod.candidate_frames(video, duration, cache, cfg)
            paths = frames_mod.extract_frames(video, times, cache, cfg)
            cache.save_json("frames.json", {"times": times, "frames": [str(p) for p in paths]})
            cache.record_stage("frames", fingerprint, n_frames=len(times))
            log(f"  {len(times)} frames for {srtot(duration)}")
    else:
        times = (cache.load_json("frames.json", {}) or {}).get("times", [])
        paths = [Path(p) for p in (cache.load_json("frames.json", {}) or {}).get("frames", [])]
    if not times:
        log("  no frames detected (check MAD_THRESHOLD / DWELL_SECONDS)")

    # ---- slides (vision + OCR-equivalent) -----------------------------------
    if want("slides"):
        if cache.stage_done("slides", fingerprint):
            log("stage slides: cached")
            slide_recs = cache.load_json("slides_kept.json")
        else:
            log("stage slides: reading frames with the vision model")
            slide_recs = slides_mod.run_slides(times, paths, cache, cfg)
            cache.save_json("slides_kept.json", slide_recs)
            cache.record_stage("slides", fingerprint, n=len(slide_recs))
    else:
        slide_recs = cache.load_json("slides_kept.json", []) or []
    words = slides_mod.hotwords(slide_recs, limit=cfg.hotword_limit)
    log(f"  hotwords: {len(words)} chars")

    # ---- theme (seeds language + prompt for ASR) ----------------------------
    if want("theme"):
        if cache.stage_done("theme", fingerprint) and not (language or theme_override or prompt_override):
            log("stage theme: cached")
            theme_rec = cache.load_json("theme.json")
        else:
            log("stage theme: inferring language + key terms")
            first_texts = [r["text"] for r in slide_recs if r["kind"] != "chrome"][:6]
            theme_rec = theme_mod.infer(
                lecture_dir.parent.name, lecture_dir.name, first_texts, cfg,
                override_language=language)
            cache.save_json("theme.json", theme_rec)
            cache.record_stage("theme", fingerprint, language=theme_rec.get("language"))
    else:
        theme_rec = cache.load_json("theme.json", {}) or {}
    if theme_override:
        theme_rec = {"language": language, "theme": theme_override, "key_terms": []}
    asr_language = language or theme_rec.get("language", "")
    asr_prompt = prompt_override or theme_mod.build_prompt(theme_rec)

    # ---- asr ----------------------------------------------------------------
    if want("asr"):
        if cache.stage_done("asr", fingerprint):
            log("stage asr: cached")
            asr_chunks = cache.load_json("asr.json")
        else:
            log("stage asr: transcribing")
            asr_chunks = asr_mod.transcribe_all(
                video, duration, cache, cfg,
                language=asr_language, hotwords=words, prompt=asr_prompt)
            cache.save_json("asr.json", asr_chunks)
            cache.record_stage("asr", fingerprint, n_chunks=len(asr_chunks),
                               chars=sum(len(c["text"]) for c in asr_chunks))
    else:
        asr_chunks = cache.load_json("asr.json", []) or []

    if not asr_chunks and not slide_recs:
        raise SystemExit("nothing to merge: no ASR text and no slides")

    # ---- merge + notes ------------------------------------------------------
    # merge: sections (slide boundaries + sentence-level speech attribution)
    # and note units (boundary-aligned groups capped at notes_unit_chars).
    # notes: one vision-model call per unit synthesizes study notes from the
    # unit's slide material + lecture speech; a final cheap call writes the
    # summary. Unit synthesis is cached by content hash (see notes._unit_key),
    # so changing slides/ASR text only re-synthesizes the affected units.
    if want("merge") or want("notes"):
        sections = merge_mod.build_sections(slide_recs, asr_chunks, duration)
        cache.save_json("sections.json", sections)
        units = merge_mod.build_units(sections, cfg.notes_unit_chars)
        if want("notes"):
            if cache.stage_done("notes", fingerprint):
                log("stage notes: cached")
            else:
                context = {
                    "course": lecture_dir.parent.name,
                    "lecture": lecture_dir.name,
                    "theme": theme_rec.get("theme", ""),
                    "duration": duration,
                }
                unit_mds = notes_mod.synthesize_units(
                    sections, units, context, cache, cfg)
                summary_md = notes_mod.synthesize_summary(
                    sections, units, unit_mds, context, cfg)
                meta = {
                    "title": title_for(lecture_dir),
                    "duration": duration,
                    "language": asr_language or next(
                        (c.get("language") for c in asr_chunks if c.get("language")), ""),
                    "theme": theme_rec.get("theme", ""),
                    "asr_model": cfg.asr_model,
                    "vision_model": cfg.vision_model,
                    "chunk_seconds": cfg.asr_chunk_seconds,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M"),
                }
                notes_mod.write_outputs(lecture_dir, sections, asr_chunks,
                                        units, unit_mds, summary_md, meta)
                cache.record_stage("notes", fingerprint, n_sections=len(sections),
                                   n_units=len(units))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path,
                    help="lecture dir (courses/<course>/lecN) or course dir with --course")
    ap.add_argument("--video", help="explicit video path inside the lecture dir")
    ap.add_argument("--course", action="store_true",
                    help="target is a course dir; process every lecN under it")
    ap.add_argument("--stages", help="comma list subset of " + ",".join(STAGE_ORDER))
    ap.add_argument("--force-stage", action="append", default=[],
                    help="invalidate a cached stage before running (repeatable)")
    ap.add_argument("--language", default="", help="pin ASR language (zh|en|...)")
    ap.add_argument("--theme", default="", help="override inferred theme sentence")
    ap.add_argument("--prompt", default="", help="override the ASR context prompt")
    args = ap.parse_args()

    cfg = Config.from_env()
    target = args.target.expanduser().resolve()
    if not target.is_dir():
        raise SystemExit(f"not a directory: {target}")

    stages = ([s.strip() for s in args.stages.split(",") if s.strip()]
              if args.stages else list(STAGE_ORDER))
    unknown = [s for s in stages if s not in STAGE_ORDER]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; valid: {STAGE_ORDER}")

    if args.course:
        lecs = sorted(d for d in target.iterdir()
                      if d.is_dir() and any(
                          p.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi",
                                               ".webm", ".m4v", ".ts", ".flv")
                          for p in d.iterdir() if p.is_file()))
        if not lecs:
            raise SystemExit(f"no lecture subdirs with videos under {target}")
        log(f"course mode: {len(lecs)} lectures")
    else:
        lecs = [target]

    rc = 0
    for lec in lecs:
        log(f"=== {lec.name} ===")
        try:
            video = find_video(lec, args.video)
            t0 = time.time()
            run_lecture(lec, video, cfg, stages, args.force_stage,
                        args.language, args.theme, args.prompt)
            log(f"=== {lec.name} done in {time.time() - t0:.0f}s ===")
        except (RuntimeError, SystemExit) as exc:
            log(f"=== {lec.name} FAILED: {exc} ===")
            rc = 1
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted; re-run to resume from cached stages")
        raise SystemExit(130)
