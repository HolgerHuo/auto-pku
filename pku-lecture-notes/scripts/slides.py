"""Use a vision model to read each deduplicated slide frame.

One call per kept frame (base64 image_url). The model returns, per frame:
  * kind    -- slide | blackboard | demo_video | chrome
  * text    -- near-verbatim slide/board text (this is the OCR equivalent)
  * content -- a faithful description (diagrams, formulas, figures in words)

The text doubles as (a) the ASR hotword source and (b) the per-slide notes
content. Chrome/watermark frames are classified and excluded so desktop chrome
never becomes lecture content. Vision runs before ASR precisely so hotwords are
ready when the audio leg starts.

OCR text is treated as a hint, not gospel: the vision model's own reading is
trusted over any OCR-style misread, and chrome strings are dropped, so
misreads do not propagate into the notes.
"""
from __future__ import annotations

import base64
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import Cache, Config, assistant_text, log, request_with_retry, srtot

# Window chrome / watermark vocabulary. Screen recordings carry a PKU watermark,
# the meeting app, the Office ribbon, dialogs, a file manager. This must never
# become lecture content. Matched loosely because the small render + JPEG makes
# exact strings unreliable; classification is biased toward KEEPING content
# (mislabelling a real slide as chrome silently deletes lecture material).
CHROME_VOCAB = [
    "课堂教学视频受知识产权保护", "未经北京大学授权", "不得翻录转载", "违者追究法律责任",
    "腾讯会议", "讯会议", "powerpoint", "幻灯片", "放映", "排练计时", "演示者",
    "监视器", "产品激活失败", "安全警告", "外部图片", "启用此内容",
    "此电脑", "回收站", "快速访问", "资源管理器", "复制到桌面",
    "chrome", "acrobat", "microsoft", "edge", "potplayer", "播放器",
]
CHROME_MARKERS = [
    "回收站", "快速访问", "此电脑", "复制到桌面", "排练计时", "监视器",
    "安全警告", "安全选项", "外部图片", "启用此内容", "产品激活失败", "讯会议",
]

PROMPT = """You are transcribing lecture visuals for course notes. The attached image \
is one captured frame from a university lecture screen recording.

Classify it and report what is shown. Output EXACTLY this format, nothing else:

kind: slide | blackboard | demo_video | chrome
text: <near-verbatim slide/board text, preserving the original language of each line; \
empty if none>
content: <faithful concise description: formulas as inline LaTeX, diagrams and figures \
described in words>

Rules:
- kind=chrome ONLY if the frame is desktop, a file manager, the meeting app, or \
window chrome (no real teaching content). A real slide that happens to show a \
watermark is still kind=slide.
- Never invent content that is not visible. If unreadable, write content: unreadable.
- Ignore watermark/subscription banners; they are not teaching content.
"""


def _norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).lower()


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _parse_frame(text: str) -> dict:
    d: dict[str, str] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*\**\s*(kind|text|content)\s*\**\s*[:：]\s*(.*)$", line)
        if m:
            d[m.group(1)] = m.group(2).strip()
            continue
        key = next((k for k in ("content", "text") if d.get(k) is not None), None)
        # continuation only for content (multiline descriptions)
        if line.strip() and d.get("content") is not None and "kind:" not in line and "text:" not in line:
            d["content"] = (d["content"] + " " + line.strip()).strip()
    return d


def _is_chrome(parsed: dict) -> bool:
    txt = (parsed.get("text", "") + " " + parsed.get("content", "")).lower()
    if any(m in txt for m in CHROME_MARKERS):
        return True
    hits = sum(1 for v in CHROME_VOCAB if v in txt)
    # Only call it chrome if a large share of the text is chrome vocabulary.
    words = [w for w in re.split(r"[\s,，。、;；:：]+", txt) if len(_norm(w)) >= 2]
    if words and hits >= 2 and hits / max(1, len(words)) >= 0.4:
        return True
    return False


def describe_frame(rec: dict, cfg: Config) -> dict:
    """One vision call for one kept frame. Returns {kind, text, content}."""
    path = Path(rec["frame"])
    body = {
        "model": cfg.vision_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}},
        ]}],
        "max_tokens": cfg.vision_max_tokens,
        "temperature": 0.2,
    }
    resp = request_with_retry(
        "POST", cfg.vision_url,
        headers={"Authorization": f"Bearer {cfg.vision_api_key}",
                 "Content-Type": "application/json"},
        json=body, max_retries=cfg.vision_max_retries,
        timeout=cfg.request_timeout, log_label=f"slides {path.name}",
    )
    parsed = _parse_frame(assistant_text(resp.json()))
    kind = parsed.get("kind", "slide").strip().lower()
    if kind not in ("slide", "blackboard", "demo_video", "chrome"):
        kind = "slide"
    if _is_chrome(parsed):
        kind = "chrome"
    return {"kind": kind,
            "text": parsed.get("text", ""),
            "content": parsed.get("content", "")}


def run_slides(frame_times: list[float], frame_paths: list[Path], cache: Cache,
               cfg: Config) -> list[dict]:
    """Describe every kept frame; cached per frame so a crash resumes cleanly."""
    cfg.require_vision()
    cache_name = "slides.json"
    done = cache.load_json(cache_name, {}) or {}
    todo = [(i, p) for i, p in enumerate(frame_paths) if str(i) not in done]
    log(f"  slides: {len(todo)} frames to describe ({len(frame_paths) - len(todo)} cached)")

    def work(pair):
        i, p = pair
        try:
            res = describe_frame({"frame": str(p)}, cfg)
        except Exception as exc:  # a bad frame must not kill the run
            log(f"  slide {p.name} failed: {exc}")
            res = {"kind": "slide", "text": "", "content": ""}
        return str(i), res

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, cfg.vision_workers)) as ex:
            for n, (key, rec) in enumerate(ex.map(work, todo), 1):
                done[key] = rec
                if n % 8 == 0 or n == len(todo):
                    log(f"  slides {n}/{len(todo)}")
                    cache.save_json(cache_name, done)
    cache.save_json(cache_name, done)

    out = []
    for i, (t, p) in enumerate(zip(frame_times, frame_paths)):
        r = done.get(str(i), {})
        out.append({
            "t": t,
            "frame": str(p),
            "kind": r.get("kind", "slide"),
            "text": r.get("text", ""),
            "content": r.get("content", ""),
        })
    noise = sum(1 for r in out if r["kind"] == "chrome")
    log(f"  slides: {len(out)} frames, {noise} classified chrome (excluded from notes)")
    return out


def hotwords(slides: list[dict], limit: int = 900) -> str:
    """Build the ASR `hotwords` field from repeated slide text.

    Repeated English/technical terms and proper nouns are exactly what the model
    mishears in a Chinese lecture; filler would dilute the hint, so dedupe and cap.
    """
    counts: Counter[str] = Counter()
    seen_first: dict[str, float] = {}
    for rec in slides:
        if rec["kind"] == "chrome":
            continue
        for tok in re.split(r"[\s,，。、;；:：()（）\"']+|、", rec.get("text", "")):
            tok = tok.strip()
            key = _norm(tok)
            if len(key) < 2 or len(tok) > 60:
                continue
            counts[tok] += 1
            seen_first.setdefault(tok, rec["t"])
    ranked = sorted(counts, key=lambda t: (-counts[t], seen_first[t]))
    chosen: list[str] = []
    total = 0
    for tok in ranked:
        if total + len(tok) + 1 > limit:
            break
        chosen.append(tok)
        total += len(tok) + 1
    return ", ".join(chosen)
