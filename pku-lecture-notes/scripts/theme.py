"""Theme inference: give the ASR leg a language + hotword context.

A single cheap vision-model call reads the course/lecture directory names plus
the first few slide texts (already produced by the slides stage) and returns a
short theme sentence, a best-guess ISO-639-1 language, and a handful of key
terms. Those seed the ASR request's `language`, `hotwords`, and `prompt` fields,
which measurably reduce misheard proper nouns and topic-specific terms.

Manual overrides always win: --language pins the language, --theme / --prompt
replace the inferred prompt. When a manual prompt is given but not the language,
we still ask the model to detect the language (it is cheap and prevents
per-chunk flip-flopping in mixed-language lectures).
"""
from __future__ import annotations

import json
import re

from common import Cache, Config, assistant_text, log, request_with_retry

PROMPT = """You are preparing context for transcribing a university lecture.
Read the course/lecture names and the first few slide texts, then reply with
ONLY a JSON object (no prose, no markdown fences) of the form:

{{
  "language": "<ISO-639-1 code, e.g. zh or en>",
  "theme": "<one short sentence in the lecture language describing the topic>",
  "key_terms": ["<term>", "<term>", ...]
}}

Course: {course}
Lecture: {lecture}

First slide texts (chronological):
{slide_texts}

Rules:
- "language" is the language the lecturer speaks in (usually the slide body
  language, but use your judgement: a Chinese lecturer quoting English keeps zh).
- "key_terms" lists 3-10 high-information proper nouns / technical terms / course
  vocabulary a speech model is likely to mishear (people, places, products,
  course-specific words). Preserve original spelling. Do not include filler.
"""


def infer(course: str, lecture: str, slide_texts: list[str], cfg: Config,
          override_language: str = "") -> dict:
    cfg.require_vision()
    joined = "\n".join(t for t in slide_texts if t.strip()) or "(none)"
    prompt = PROMPT.format(course=course, lecture=lecture, slide_texts=joined[:1500])
    body = {
        "model": cfg.vision_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.0,
    }
    resp = request_with_retry(
        "POST", cfg.vision_url,
        headers={"Authorization": f"Bearer {cfg.vision_api_key}",
                 "Content-Type": "application/json"},
        json=body, max_retries=cfg.vision_max_retries,
        timeout=cfg.request_timeout, log_label="theme",
    )
    text = assistant_text(resp.json())
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log(f"  theme: could not parse JSON ({text[:80]!r}); falling back")
        data = {}
    language = (override_language or data.get("language") or "").strip().lower()
    theme = (data.get("theme") or "").strip()
    terms = [str(t).strip() for t in (data.get("key_terms") or []) if str(t).strip()]
    log(f"  theme: language={language or '(auto)'} terms={len(terms)} "
        f"theme={theme[:60]!r}")
    return {"language": language, "theme": theme, "key_terms": terms[:10]}


def build_prompt(theme: dict) -> str:
    """Compose the ASR `prompt` (system-turn context) from the inferred theme."""
    parts = []
    if theme.get("theme"):
        parts.append(f"Lecture topic: {theme['theme']}")
    if theme.get("key_terms"):
        parts.append("Key terms (spell exactly): " + ", ".join(theme["key_terms"]))
    return " ".join(parts)
