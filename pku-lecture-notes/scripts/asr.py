"""ASR processing over fixed-size audio chunks.

Chunk format: M4A via `ffmpeg -vn -c:a copy` -- no re-encode, lossless,
Fallback chain
if the source audio is not copyable: AAC re-encode, then 16 kHz WAV.

The configured API is expected to return flat text. Timestamps therefore come
from chunk boundaries.

The quality net is a safety net, not the primary defence: it collapses
verbatim loops and drops known filler-banner phrases. If a chunk trips it, only
THAT chunk is split in half and re-sent (min 300s), so a bad region never forces
a full re-run.

Concurrency is configurable with ``ASR_WORKERS`` and defaults to one.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import Cache, Config, log, request_with_retry, run, srtot

# Common non-lecture filler emitted by speech models.
FILLER_PATTERNS = [
    r"请不吝点赞", r"优优独播", r"YoYo\s*Television", r"中文字幕志愿者",
    r"感谢.{0,4}观看", r"请(订阅|关注|转发|打赏)",
]
_FILLER = [re.compile(p, re.I) for p in FILLER_PATTERNS]

MAX_LOOP_RUN = 3  # >3 consecutive identical sentences = hallucination loop
MAX_SENTENCE_CHARS = 2000

def _span_key(start: float, end: float) -> str:
    return f"{int(round(start))}_{int(round(end))}"


def _norm(text: str) -> str:
    return re.sub(r"[\s\u3000，。！？、；：,.!?;:]+", "", text)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping punctuation attached."""
    parts = re.split(r"(?<=[。！？.!?])\s*", (text or "").strip())
    return [p for p in (s.strip() for s in parts) if p]


def _has_filler(text: str) -> bool:
    return any(p.search(text) for p in _FILLER)


def _clean_text(text: str) -> tuple[str, bool]:
    """Apply the quality net. Returns (cleaned_text, triggered).

    1. Drop sentences matching a known filler banner.
    2. Collapse runs of >MAX_LOOP_RUN consecutive identical sentences.
    Triggered=True when either rule removed anything, so the caller can decide
    whether to split-retry the chunk.
    """
    text, char_runs = re.subn(r"([^\s])\1{5,}", lambda m: m.group(1) * 3, text or "")
    text, token_runs = re.subn(
        r"(?P<token>[^\W\d_]{1,20})(?:[\s,，]+(?P=token)){5,}",
        lambda m: ", ".join([m.group("token")] * 3),
        text,
        flags=re.UNICODE,
    )
    text, phrase_runs = re.subn(
        r"(?P<phrase>[^,，。！？.!?\n]{2,80})(?:[,，]+\s*(?P=phrase)){5,}",
        lambda m: ", ".join([m.group("phrase").strip()] * 3),
        text,
    )
    out: list[str] = []
    run = 0
    prev = None
    seen: dict[str, int] = {}
    triggered = bool(char_runs or token_runs or phrase_runs)
    for s in _split_sentences(text):
        if len(s) > MAX_SENTENCE_CHARS:
            s = s[:MAX_SENTENCE_CHARS]
            triggered = True
        if _has_filler(s):
            triggered = True
            continue
        k = _norm(s)
        run = run + 1 if k == prev and k else 1
        prev = k
        seen[k] = seen.get(k, 0) + 1 if k else 1
        if run > MAX_LOOP_RUN or (k and seen[k] > MAX_LOOP_RUN):
            triggered = True
            continue
        out.append(s)
    return "".join(out), triggered


def _strip_context_echo(text: str, prompt: str = "", hotwords: str = "") -> tuple[str, bool]:
    """Remove exact ASR-context banners echoed into the transcript.

    Some compatible providers prepend the prompt repeatedly without sentence
    punctuation, so the sentence-level loop filter cannot recognize it.
    Only caller-supplied context is removed; lecture speech is otherwise left
    untouched.
    """
    cleaned = text or ""
    triggered = False
    topic_marker = "Lecture topic:"
    if cleaned.count(topic_marker) >= 3:
        cleaned = cleaned.split(topic_marker, 1)[0]
        triggered = True
    elif topic_marker in cleaned:
        cleaned = cleaned.replace(topic_marker, "")
        triggered = True
    prompt = prompt.strip()
    candidates = [prompt]
    if prompt.startswith("Lecture topic: "):
        candidates.append(prompt.removeprefix("Lecture topic: "))
    marker = "Key terms (spell exactly): "
    candidates.append(marker.strip())
    if marker in prompt:
        candidates.append(prompt.split(marker, 1)[1].strip())
    if hotwords.strip():
        candidates.extend([
            hotwords.strip(),
            f"Key terms (spell exactly): {hotwords.strip()}",
        ])
    for candidate in sorted({c for c in candidates if c}, key=len, reverse=True):
        if candidate in cleaned:
            cleaned = cleaned.replace(candidate, "")
            triggered = True
    return cleaned.strip(), triggered


def cut_chunk(video: Path, start: float, end: float, cache: Cache,
              cfg: Config) -> Path:
    """Cut one [start, end) span to its own file, cached.

    Prefers a lossless copy-cut M4A (no re-encode, ~0.7 MB/60s). If the source
    audio is not copyable, falls back to an AAC re-encode, then 16 kHz WAV.
    Raises if none succeed.
    """
    d = cache.path("chunks")
    d.mkdir(parents=True, exist_ok=True)
    k = _span_key(start, end)
    dur = f"{max(0.1, end - start):.3f}"
    m4a = d / f"{k}.m4a"
    if not (m4a.exists() and m4a.stat().st_size > 200):
        try:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                 "-ss", f"{start:.3f}", "-t", dur, "-i", video,
                 "-vn", "-c:a", "copy", m4a, "-y"])
        except RuntimeError:
            pass
    if m4a.exists() and m4a.stat().st_size > 200:
        return m4a
    aac = d / f"{k}.aac"
    if not (aac.exists() and aac.stat().st_size > 200):
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
             "-ss", f"{start:.3f}", "-t", dur, "-i", video,
             "-vn", "-c:a", "aac", "-b:a", "192k", aac, "-y"])
    if aac.exists() and aac.stat().st_size > 200:
        return aac
    wav = d / f"{k}.wav"
    if not (wav.exists() and wav.stat().st_size > 200):
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
             "-ss", f"{start:.3f}", "-t", dur, "-i", video,
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav, "-y"])
    if not (wav.exists() and wav.stat().st_size > 200):
        raise RuntimeError(f"could not cut chunk {k} from {video.name}")
    return wav


def _transcribe(path: Path, cfg: Config, language: str = "", hotwords: str = "",
                prompt: str = "") -> str:
    """One POST -> transcription text."""
    fields = {"model": cfg.asr_model, "response_format": "json"}
    if language:
        fields["language"] = language
    if hotwords:
        fields["hotwords"] = hotwords
    if prompt:
        fields["prompt"] = prompt

    def send():
        with path.open("rb") as fh:
            return request_with_retry(
                "POST", cfg.asr_url,
                headers={"Authorization": f"Bearer {cfg.asr_api_key}"},
                data=fields, files={"file": (path.name, fh, "application/octet-stream")},
                max_retries=cfg.asr_max_retries, timeout=cfg.request_timeout,
                log_label=f"asr {path.name}",
            )
    resp = send()
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return resp.text
    return data.get("text", "") if isinstance(data, dict) else str(data)


def transcribe_chunk(video: Path, start: float, end: float, cache: Cache,
                     cfg: Config, language: str = "", hotwords: str = "",
                     prompt: str = "", done: dict | None = None,
                     depth: int = 0) -> dict:
    """Transcribe one [start, end) span. Returns {start, end, text, retries}.

    Each span is cached by its boundary key in `done` (when provided), so both
    top-level chunks and split sub-spans resume after a crash. If the quality
    net fires and the span is above the retry floor, only THIS span is split in
    half and redone -- never the whole lecture.
    """
    done = done if done is not None else {}
    k = _span_key(start, end)
    if k in done:
        return done[k]

    span = end - start
    media = cut_chunk(video, start, end, cache, cfg)
    raw = _transcribe(media, cfg, language=language, hotwords=hotwords, prompt=prompt)
    without_echo, echo_triggered = _strip_context_echo(raw, prompt, hotwords)
    cleaned, quality_triggered = _clean_text(without_echo)
    if echo_triggered:
        log(f"  asr: removed echoed context from {srtot(start)}-{srtot(end)}")
    triggered = quality_triggered

    if triggered and span > cfg.asr_min_retry_chunk + 1:
        mid = start + span / 2
        log(f"  asr: quality net fired on {srtot(start)}-{srtot(end)}; "
            f"splitting (depth {depth + 1})")
        left = transcribe_chunk(video, start, mid, cache, cfg,
                                language=language, hotwords=hotwords,
                                prompt=prompt, done=done, depth=depth + 1)
        right = transcribe_chunk(video, mid, end, cache, cfg,
                                 language=language, hotwords=hotwords,
                                 prompt=prompt, done=done, depth=depth + 1)
        rec = {"start": start, "end": end, "text": left["text"] + right["text"],
               "retries": left["retries"] + right["retries"], "split": True}
    else:
        if triggered:
            log(f"  asr: quality net fired on {srtot(start)}-{srtot(end)} at retry "
                f"floor; keeping cleaned {len(cleaned)} chars")
        rec = {"start": start, "end": end, "text": cleaned, "retries": 1,
               "split": False}
    done[k] = rec
    return rec


def plan_chunks(duration: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Build a fixed-size plan to bound requests and support resume."""
    out: list[tuple[float, float]] = []
    t = 0.0
    while t < duration - 0.5:
        e = min(t + chunk_seconds, duration)
        out.append((round(t, 3), round(e, 3)))
        t = e
    return out


def transcribe_all(video: Path, duration: float, cache: Cache, cfg: Config,
                   language: str = "", hotwords: str = "", prompt: str = "") -> list[dict]:
    """Transcribe the whole recording, chunk by chunk, with a small pool.

    Per-chunk (and split sub-span) results are cached by span so a crash
    resumes where it left off.
    """
    cfg.require_asr()
    plan = plan_chunks(duration, cfg.asr_chunk_seconds)
    cache_name = "asr_chunks.json"
    done = cache.load_json(cache_name, {}) or {}
    normalized_cache = False
    for rec in done.values():
        cleaned, echo_triggered = _strip_context_echo(
            rec.get("text", ""), prompt=prompt, hotwords=hotwords)
        cleaned, quality_triggered = _clean_text(cleaned)
        if echo_triggered or quality_triggered:
            rec["text"] = cleaned
            normalized_cache = True
    if normalized_cache:
        log("  asr: removed echoed context from cached transcript chunks")
        cache.save_json(cache_name, done)
    todo = [(s, e) for s, e in plan if _span_key(s, e) not in done]
    log(f"  asr: {len(todo)} chunks of {cfg.asr_chunk_seconds:.0f}s "
        f"({len(plan) - len(todo)} cached), language={language or '(auto)'}")

    def work(span):
        s, e = span
        return transcribe_chunk(video, s, e, cache, cfg,
                                language=language, hotwords=hotwords, prompt=prompt,
                                done=done)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, cfg.asr_workers)) as ex:
            for n, rec in enumerate(ex.map(work, todo), 1):
                log(f"  asr {n}/{len(todo)}: {srtot(rec['start'])}-{srtot(rec['end'])} "
                    f"-> {len(rec['text'])} chars ({rec['retries']} pass"
                    f"{'es' if rec['retries'] != 1 else ''})")
                cache.save_json(cache_name, done)
    cache.save_json(cache_name, done)

    segments = [done[_span_key(s, e)] for s, e in plan if _span_key(s, e) in done]
    segments.sort(key=lambda r: r["start"])
    total = sum(len(r["text"]) for r in segments)
    log(f"  asr total: {len(segments)} chunks, {total} chars")
    return segments
