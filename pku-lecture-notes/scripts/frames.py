"""Slide-change detection: a single cheap, model-free video decode.

Frame selection does not trust raw pixel differences. On real PKU screen-capture
lectures the per-second change signal is bimodal because slides embed demo video
and a moving cursor: naive thresholding reports hundreds of "changes" where only
~150 are real slide transitions. Dwell time (a run must be stable for a few
seconds) is what makes segmentation usable: a slide held and talked over for ten
minutes becomes one representative frame (the last one of the run, which is the
most complete state -- important for blackboard writing that accumulates strokes).

Decoding happens in bounded windows to low-res grayscale so the scan stays
light on boxes with a couple of GB free. This is the only compute-expensive-ish
local step in the whole pipeline, and it is a single decode pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import Cache, Config, log, run, srtot


def ffprobe_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path,
    ]).stdout.strip()
    return float(out)


def _scan_thumbs(video: Path, duration: float, cache: Cache, cfg: Config) -> np.ndarray:
    """Decode low-res grayscale thumbnails over the whole video, window by window."""
    raw = cache.path("scan.raw")
    side = cfg.scan_thumb
    frame_bytes = side * side
    if raw.exists():
        arr = np.fromfile(raw, dtype=np.uint8)
        if arr.size % frame_bytes == 0:
            log(f"  reusing {arr.size // frame_bytes} scan thumbnails from cache")
            return arr.reshape(-1, side, side)

    n_windows = max(1, int(np.ceil(duration / cfg.scan_window)))
    chunks: list[np.ndarray] = []
    total = 0
    for w in range(n_windows):
        t0 = w * cfg.scan_window
        part = cache.path(f"scan_{w:03d}.raw")
        if not (part.exists() and part.stat().st_size > 0):
            run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                "-ss", f"{t0:.3f}", "-t", f"{cfg.scan_window:.3f}", "-i", video,
                "-vf", f"fps={cfg.scan_fps},scale={side}:{side}",
                "-f", "rawvideo", "-pix_fmt", "gray",
                "-threads", str(cfg.ffmpeg_threads), part, "-y",
            ])
        a = np.fromfile(part, dtype=np.uint8)
        a = a[: a.size - a.size % frame_bytes]
        if a.size:
            chunks.append(a.reshape(-1, side, side))
            total += a.size // frame_bytes
        log(f"  scan window {w + 1}/{n_windows} -> {total} frames (~{srtot(total / cfg.scan_fps)})")
    out = np.concatenate(chunks) if chunks else np.zeros((0, side, side), np.uint8)
    out.tofile(raw)
    _prune_scan_parts(cache)
    return out


def candidate_frames(video: Path, duration: float, cache: Cache, cfg: Config) -> list[float]:
    """Timestamps of settled frames: stable for >= dwell_seconds, one per run.

    Representative is the last frame of a stable run (most complete state).
    """
    thumbs = _scan_thumbs(video, duration, cache, cfg)
    if len(thumbs) < 2:
        return []
    fps = cfg.scan_fps
    mad = np.abs(thumbs[1:].astype(np.int16) - thumbs[:-1].astype(np.int16)).mean((1, 2))
    changed = mad > cfg.mad_threshold

    reps: list[float] = []
    run_start = 0
    n = len(mad)
    i = 0
    while i < n:
        if changed[i]:
            if (i - run_start) * fps >= cfg.dwell_seconds:
                reps.append((i + 0.5) / fps)
            run_start = i + 1
        i += 1
    if (n - run_start) * fps >= cfg.dwell_seconds:
        reps.append((n - 0.5) / fps)

    merged: list[float] = []
    for t in reps:
        if merged and t - merged[-1] < cfg.dwell_seconds:
            merged[-1] = t
        else:
            merged.append(t)
    log(f"  {len(merged)} candidate frames (dwell>={cfg.dwell_seconds}s, "
        f"raw change points={int(changed.sum())})")
    return merged


def source_width(video: Path) -> int:
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width", "-of", "default=nw=1:nk=1", video,
    ]).stdout.strip()
    return int(out.splitlines()[0])


def _video_filter(video: Path, cfg: Config) -> str:
    """Crop (optional) then downscale. Never upscale."""
    parts = []
    if cfg.crop:
        parts.append(f"crop={cfg.crop}")
    w = source_width(video)
    target = min(w, cfg.vision_max_width)
    if target != w or cfg.crop:
        parts.append(f"scale={target}:-2")
    return ",".join(parts) if parts else "null"


def extract_frame(video: Path, t: float, target: Path, cfg: Config) -> Path:
    """Seek-extract one frame, cropped and downscaled for API cost."""
    if target.exists() and target.stat().st_size > 500:
        return target
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
        "-ss", f"{max(0.0, t):.3f}", "-i", video, "-frames:v", "1",
        "-vf", _video_filter(video, cfg), "-q:v", "3",
        "-threads", str(cfg.ffmpeg_threads), target, "-y",
    ])
    return target


def extract_frames(video: Path, times: list[float], cache: Cache, cfg: Config) -> list[Path]:
    frame_dir = cache.path("frames")
    targets = [frame_dir / f"f{int(round(t * 100)):07d}.jpg" for t in times]
    todo = [(t, p) for t, p in zip(times, targets) if not (p.exists() and p.stat().st_size > 500)]
    log(f"  extracting {len(todo)} of {len(times)} frames ({len(times) - len(todo)} cached)")
    if todo:
        with ThreadPoolExecutor(max_workers=max(2, min(6, cfg.ffmpeg_threads // 2))) as ex:
            list(ex.map(lambda tp: extract_frame(video, tp[0], tp[1], cfg), todo))
    return targets


def _prune_scan_parts(cache: Cache) -> int:
    """Delete per-window scan files once folded into scan.raw."""
    n = 0
    for part in sorted(cache.root.glob("scan_*.raw")):
        if part.is_file():
            part.unlink()
            n += 1
    return n
