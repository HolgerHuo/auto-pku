"""Shared configuration, caching, and HTTP helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

CACHE_DIRNAME = ".cache"

# Stage order is the pipeline order. Vision+slides run before asr so the ASR
# stage has a theme + hotwords prompt ready (see theme stage).
STAGE_ORDER = ["frames", "slides", "theme", "asr", "merge", "notes"]

_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw) -> subprocess.CompletedProcess:
    """Run a command capturing text output; failures raise with stderr attached."""
    cmd = [str(c) for c in cmd]
    kw.setdefault("check", True)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    try:
        return subprocess.run(cmd, **kw)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n" + "\n".join(tail)) from exc


def srtot(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def video_fingerprint(path: Path) -> str:
    """Cheap stable identity so a stale cache can never be silently reused."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > (1 << 21):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return f"{size}-{int(path.stat().st_mtime)}-{h.hexdigest()[:12]}"


def assistant_text(data: dict) -> str:
    """Return text from an OpenAI-compatible chat completion."""
    for ch in data.get("choices") or []:
        msg = ch.get("message") or {}
        c = msg.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        if isinstance(c, str) and c.strip():
            return c
        r = msg.get("reasoning")
        if isinstance(r, str) and r.strip():
            return r
    return ""


@dataclass
class Config:
    # --- vision (chat/completions, reads slide frames) ---
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = "qwen3.8-27b"
    vision_workers: int = 16          # per-frame slide calls, 16-way
    vision_max_width: int = 1280
    vision_max_retries: int = 4
    vision_max_tokens: int = 2048
    request_timeout: int = 600

    # --- ASR (qwen3-asr via /v1/audio/transcriptions) ---
    asr_base_url: str = ""
    asr_api_key: str = ""
    asr_model: str = "qwen3-asr-1.7b"
    asr_workers: int = 1
    asr_chunk_seconds: float = 1800.0
    asr_max_retries: int = 5
    asr_min_retry_chunk: float = 300.0  # quality-net split floor
    hotword_limit: int = 900

    # --- notes (study-notes synthesis, vision chat calls) ---
    notes_workers: int = 6
    notes_unit_chars: int = 1500
    notes_max_tokens: int = 10000
    notes_summary_max_tokens: int = 12000

    # --- local processing (cheap, model-free) ---
    ffmpeg_threads: int = 8
    scan_window: float = 600.0        # decode window, bounds memory
    scan_fps: float = 1.0
    scan_thumb: int = 64
    mad_threshold: float = 5.0        # thumbnail MAD above this = visual change
    dwell_seconds: float = 3.0        # a frame must stay stable this long
    crop: str = ""                    # optional ffmpeg crop expr w:h:x:y

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ.get

        def num(key, default, cast=float):
            raw = env(key, "")
            try:
                return cast(raw) if raw not in ("", None) else default
            except ValueError:
                return default

        return cls(
            vision_base_url=(env("VISION_BASE_URL") or env("OPENAI_BASE_URL") or "").rstrip("/"),
            vision_api_key=env("VISION_API_KEY") or env("OPENAI_API_KEY") or "",
            vision_model=env("VISION_MODEL", "qwen3.8-27b"),
            vision_workers=num("SLIDE_WORKERS", 16, int),
            vision_max_width=num("VISION_MAX_WIDTH", 1280, int),
            vision_max_retries=num("VISION_MAX_RETRIES", 4, int),
            vision_max_tokens=num("VISION_MAX_TOKENS", 2048, int),
            request_timeout=num("REQUEST_TIMEOUT", 600, int),
            asr_base_url=(env("ASR_BASE_URL") or "").rstrip("/"),
            asr_api_key=env("ASR_API_KEY") or "",
            asr_model=env("ASR_MODEL", "qwen3-asr-1.7b"),
            asr_workers=num("ASR_WORKERS", 1, int),
            asr_chunk_seconds=num("ASR_CHUNK_SECONDS", 1800.0),
            asr_max_retries=num("ASR_MAX_RETRIES", 5, int),
            asr_min_retry_chunk=num("ASR_MIN_RETRY_CHUNK", 300.0),
            ffmpeg_threads=num("FFMPEG_THREADS", 8, int),
            scan_window=num("SCAN_WINDOW", 600.0),
            scan_fps=num("SCAN_FPS", 1.0),
            scan_thumb=num("SCAN_THUMB", 64),
            mad_threshold=num("MAD_THRESHOLD", 5.0),
            dwell_seconds=num("DWELL_SECONDS", 3.0),
            notes_workers=num("NOTES_WORKERS", 6, int),
            notes_unit_chars=num("NOTES_UNIT_CHARS", 1500, int),
            notes_max_tokens=num("NOTES_MAX_TOKENS", 10000, int),
            notes_summary_max_tokens=num("NOTES_SUMMARY_MAX_TOKENS", 12000, int),
            crop=env("CROP", ""),
        )

    def require_vision(self) -> None:
        if not self.vision_base_url or not self.vision_api_key:
            raise SystemExit(
                "VISION_BASE_URL and VISION_API_KEY must be set "
                "(chat/completions endpoint with a vision-capable model)."
            )

    def require_asr(self) -> None:
        if not self.asr_base_url or not self.asr_api_key:
            raise SystemExit(
                "ASR_BASE_URL and ASR_API_KEY must be set "
                "(OpenAI-compatible /v1/audio/transcriptions, model qwen3-asr-1.7b)."
            )

    @property
    def vision_url(self) -> str:
        base = self.vision_base_url
        if base.endswith("/chat/completions"):
            return base
        return base + ("/chat/completions" if base.endswith("/v1")
                       else "/v1/chat/completions")

    @property
    def asr_url(self) -> str:
        base = self.asr_base_url
        if base.endswith("/audio/transcriptions"):
            return base
        return base + ("/audio/transcriptions" if base.endswith("/v1")
                       else "/v1/audio/transcriptions")


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    max_retries: int = 5,
    timeout: int = 600,
    log_label: str = "",
    **kw,
) -> requests.Response:
    """HTTP with exponential backoff on 429, 5xx, and transport errors."""
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        resp = None
        try:
            resp = requests.request(method, url, headers=headers, timeout=timeout, **kw)
        except requests.RequestException as exc:
            last = exc
        if resp is not None:
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            elif resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:600]}")
            else:
                return resp
        if attempt >= max_retries:
            break
        delay = min(60, 2 ** attempt * 3 + 5)
        log(f"  retry {attempt + 1}/{max_retries} {log_label or url}: {last}")
        time.sleep(delay)
    raise RuntimeError(f"exhausted retries for {log_label or url}: {last}")


class Cache:
    """Per-lecture cache with a stage manifest keyed by the video fingerprint."""

    def __init__(self, lecture_dir: Path):
        self.dir = lecture_dir
        self.root = lecture_dir / CACHE_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        (self.root / "chunks").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def path(self, name: str) -> Path:
        return self.root / name

    def load_json(self, name: str, default=None):
        p = self.path(name)
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  cache {name} is corrupt; ignoring")
            return default

    def save_json(self, name: str, obj) -> Path:
        p = self.path(name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
        return p

    def manifest(self) -> dict:
        return self.load_json("manifest.json", {}) or {}

    def stage_done(self, stage: str, fingerprint: str) -> bool:
        rec = self.manifest().get(stage)
        return bool(rec) and rec.get("fingerprint") == fingerprint and rec.get("ok")

    def record_stage(self, stage: str, fingerprint: str, **info) -> None:
        with self._lock:
            man = self.manifest()
            man[stage] = {
                "fingerprint": fingerprint,
                "ok": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **info,
            }
            self.save_json("manifest.json", man)

    def clear_stage(self, stage: str) -> None:
        with self._lock:
            man = self.manifest()
            man.pop(stage, None)
            self.save_json("manifest.json", man)
