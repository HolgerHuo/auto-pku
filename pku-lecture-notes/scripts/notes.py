"""Render the final artifacts: study notes + transcription.

notes.md is NOT a dump of slides + audio. The lecture is cut into note units
(merge.build_units), and for each unit ONE vision-model call synthesizes
study notes from that unit's slide material + attributed lecture speech:
topic headings, key points, the lecturer's explanations and examples
folded in, formulas/terms kept verbatim. Faithfulness is enforced in the
prompt (never invent; say so when a part is unclear), and a failed unit
falls back to the raw material so no content is ever lost. A final cheap
call writes the lecture summary (main thread + takeaways) from the outline
and the unit notes.

transcription.md stays the mechanical, full transcript (chunk-stamped) --
the auditable source the notes were written from.
"""
from __future__ import annotations

import hashlib
import re as _re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import Cache, Config, assistant_text, log, request_with_retry, srtot


def _unit_material(unit: dict) -> str:
    """The material one synthesis call sees: slides + attributed speech per part."""
    out = []
    for _sec, part in unit["parts"]:
        st = (part.get("slides") or "").strip()
        sp = (part.get("speech") or "").strip()
        if st:
            out.append(f"[Slides {srtot(part['t_start'])}]\n{st}")
        if sp:
            out.append(f"[Lecture speech {srtot(part['t_start'])}-{srtot(part['t_end'])}]\n"
                       + " ".join(sp.split()))
    return "\n\n".join(out)


# Bump when the synthesis prompt changes: old cached answers were written
# under the old prompt and must not be silently reused.
PROMPT_VERSION = "v2-requirements"


def _unit_key(unit: dict) -> str:
    """Stable identity: version + span + material content. Any slide/ASR
    change (same boundaries, different text) or prompt change invalidates
    the cached synthesis."""
    mat = _unit_material(unit)
    h = hashlib.sha256(mat.encode("utf-8")).hexdigest()[:16]
    return f"{PROMPT_VERSION}_{unit['i_start']}_{int(unit['t_start'])}_{int(unit['t_end'])}_{h}"


UNIT_PROMPT = """You are writing study notes (课堂笔记) from a university lecture. \
Below is one time-ordered part of the lecture: the slides shown (near-verbatim \
text/descriptions) and the lecturer's speech (ASR transcript, may contain \
recognition errors and colloquial fillers).

Write the study notes for THIS part, in the language of the lecture, as GitHub \
markdown. Rules:
- These must be NOTES FOR STUDYING: topic sub-headings (###), key points as \
bullets, definitions/formulas/code/terms kept verbatim and exact, examples the \
lecturer gave preserved (shortened only when clearly verbose).
- Merge the two sources: the slides give structure; the speech gives \
explanations, context, and emphasis. Write what the lecture TAUGHT, not a \
side-by-side transcription of both.
- NEVER invent facts, numbers, or citations that are not in the material. If a \
part is garbled or unclear, write a brief "(unclear in recording)" note instead \
of guessing.
- Drop small talk, class management, and pure filler. Keep questions the \
lecturer posed and how they were answered -- they are study material.
- Do not repeat boilerplate like the course title or watermark text.
- No preamble, no "here are the notes" -- start directly with the content.
- Course requirements matter most for studying: if this part mentions
  requirements (grading scheme, final project/paper/exam expectations),
  homework or assignments (including for the NEXT class), quizzes, or
  deadlines, put them FIRST, under a heading "### 课程要求与作业" (or the
  lecture-language equivalent), as concrete bullets: what exactly is
  required, format, due date, weight/grade share. Never paraphrase away
  concrete details (numbers, dates, page counts, rubric items).

{course} / {lecture}
Lecture topic: {theme}
Part covers {t_start} to {t_end}.

{material}
"""

SUMMARY_PROMPT = """You are writing the top-level summary of a university lecture \
study-notes document, in the language of the lecture. Using the outline and the \
per-part notes below, produce GitHub markdown with:
1. A short overview (2-4 sentences: what the lecture is about).
2. "主要线索 / Main thread": the argument or sequence of the lecture, as bullets.
3. "核心要点 / Key points": the 5-12 most important facts, terms, or conclusions.
4. "课程要求与作业 / Requirements & homework": EVERY course requirement the
   lecturer mentioned -- grading scheme, final project/paper/exam (what it
   must include, format, due date), homework and assignments (including
   those for the NEXT class), quizzes -- as concrete bullets with the exact
   details given (dates, weights, deliverables). This is the section students
   will look for first: be complete and specific. Write "本课未布置作业"
   (or equivalent) only if the raw mentions below are empty.
5. "留疑 / Open questions" only if the lecturer explicitly left questions open.
Keep it faithful: no invented content. Start directly with the overview (## is \
reserved for the document; use ### for your sub-sections).

{course} / {lecture}
Lecture topic: {theme}
Duration: {duration}

Outline:
{outline}

Per-part notes (may be abbreviated):
{parts}

Raw mentions of requirements/homework extracted from the part notes:
{req_lines}
"""


def _chat(prompt: str, cfg: Config, *, max_tokens: int, label: str,
          timeout: int | None = None) -> tuple[str, str]:
    body = {
        "model": cfg.vision_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    # Note synthesis can exceed the shorter timeout used by other stages.
    resp = request_with_retry(
        "POST", cfg.vision_url,
        headers={"Authorization": f"Bearer {cfg.vision_api_key}",
                 "Content-Type": "application/json"},
        json=body, max_retries=cfg.vision_max_retries,
        timeout=timeout or max(cfg.request_timeout, 1800),
        log_label=f"notes {label}",
    )
    data = resp.json()
    choices = data.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning") or ""
    return (content or "").strip(), reasoning.strip()


# Signatures of model deliberation leaking into the answer field.
_COT_HEAD = _re.compile(
    r"^(we (need|should|must|are|will|are going)|let (me|'s)|i (need|should|think|'ll|am|will)|ok,|sure,|the user (is|wants|asks|is asking)|我们需要|我需要|让我们|好的，|首先，(让我|我需要|先)|分析(?:一下)?(?:用户|题目))",
    _re.I)
_COT_MID = _re.compile(r"need (maybe )?include|let me (now )?(write|check|structure|draft)|wait,|hmm,|as a (good|helpful)|i should (probably )?now|now (i|let) (write|produce|finalize)")


def is_cot_garbage(text: str) -> bool:
    """Return whether text looks like model deliberation rather than notes."""
    t = (text or "").strip()
    if not t:
        return False
    head, mid = t[:400], t[len(t) // 2:]
    if _COT_HEAD.match(t):
        return True
    return bool(_COT_MID.search(mid))


def salvage_from_reasoning(reasoning: str, content: str = "") -> str:
    """Recover a structured Markdown answer from a reasoning field."""
    t = reasoning or ""
    # candidate: last occurrence of a line starting with '### ' or '## '
    idx = -1
    for m in _re.finditer(r"(?m)^(#{2,3})\s+\S", t):
        idx = m.start()
    if idx < 0:
        return ""
    tail = t[idx:].strip()
    # the notes part must be substantive and mostly heading/bullet lines
    lines = [l for l in tail.splitlines() if l.strip()]
    if len(lines) < 5:
        return ""
    structured = sum(1 for l in lines if l.lstrip().startswith(("#", "-", "*", "1.", "2.", "3.", "4.", "5.")))
    if structured / len(lines) < 0.35:
        return ""
    if _COT_MID.search(tail[:600]):
        # planning notes, not the answer
        return ""
    if len(tail) < 120:
        return ""
    return tail


def _fallback_unit_md(unit: dict) -> str:
    """Raw material rendering, used only if the synthesis call fails."""
    return _unit_material(unit)


def synthesize_units(sections: list[dict], units: list[dict], context: dict,
                     cache: Cache, cfg: Config) -> list[str]:
    """One synthesis call per unit (8-way). Cached per unit content hash."""
    cfg.require_vision()
    cache_name = "notes_units.json"
    done = cache.load_json(cache_name, {}) or {}
    # Do not reuse cached model deliberation as user-facing notes.
    for k in [k for k, v in done.items() if is_cot_garbage(v)]:
        log(f"  notes: dropping cached unit {k[:32]}… (invalid note format)")
        del done[k]
    keys = [_unit_key(u) for u in units]
    todo = [i for i, k in enumerate(keys) if k not in done]
    log(f"  notes: {len(todo)} units to synthesize ({len(units) - len(todo)} cached)")

    def work(i: int) -> tuple[int, str]:
        u = units[i]
        material = _unit_material(u)
        prompt = UNIT_PROMPT.format(
            course=context.get("course", ""), lecture=context.get("lecture", ""),
            theme=context.get("theme") or "(not inferred)",
            t_start=srtot(u["t_start"]), t_end=srtot(u["t_end"]),
            material=material,
        )
        md = ""
        for attempt in range(3):
            try:
                content, reasoning = _chat(
                    prompt, cfg, max_tokens=cfg.notes_max_tokens,
                    label=f"unit {u['i_start']}")
                if content and not is_cot_garbage(content):
                    md = content
                elif not content:
                    md = salvage_from_reasoning(reasoning)
                if md:
                    break
            except Exception as exc:
                log(f"  notes: unit {u['i_start']} attempt {attempt + 1} "
                    f"failed: {exc}")
        if not md:
            log(f"  notes: unit {u['i_start']} synthesis empty/garbage after "
                f"3 tries; falling back to raw material")
            md = _fallback_unit_md(u)
        return i, md

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, cfg.notes_workers)) as ex:
            for n, (i, md) in enumerate(ex.map(work, todo), 1):
                done[keys[i]] = md
                if n % 4 == 0 or n == len(todo):
                    log(f"  notes {n}/{len(todo)}")
                    cache.save_json(cache_name, done)
    cache.save_json(cache_name, done)
    return [done[k] for k in keys]


_REQ_KEYWORDS = _re.compile(
    r"(作业|quiz|Quiz|考试|测验|期末|期中|论文|paper|Paper|final project|Final project|"
    r"deadline|Deadline|截止|due|Due|提交|submit|Submit|要求|考核|评分|打分|成绩|"
    r"grade|Grade|weight|占|百分|下次课|下节课|下周|next class|homework|Homework|"
    r"大作业|课程要求|平时分|平时成绩|rubric|Rubric|报告|report|Report|"
    r"展示|presentation|Presentation)",
)


def _requirement_lines(unit_mds: list[str], limit: int = 4000) -> str:
    """Lines that look like course requirements/homework, kept verbatim.

    The summary must quote concrete details (dates, weights, deliverables);
    the abbreviated per-part notes are too lossy for that, so pull the raw
    matching lines from the full unit notes instead.
    """
    out: list[str] = []
    for md in unit_mds:
        for line in md.splitlines():
            s = line.strip()
            if s and _REQ_KEYWORDS.search(s):
                out.append(s[:300])
    return "\n".join(out)[:limit]


def synthesize_summary(sections: list[dict], units: list[dict], unit_mds: list[str],
                       context: dict, cfg: Config) -> str:
    """Cheap final call: overview + main thread + key points."""
    cfg.require_vision()
    # Keep summary requests bounded across providers.
    outline = "\n".join(
        f"- {srtot(s['t_start'])}  {_section_label(s)}" for s in sections)[:2000]
    parts = "\n".join(
        f"[{srtot(u['t_start'])}] {md.strip()[:150]}"
        for u, md in zip(units, unit_mds))[:12000]
    req_lines = _requirement_lines(unit_mds)
    prompt = SUMMARY_PROMPT.format(
        course=context.get("course", ""), lecture=context.get("lecture", ""),
        theme=context.get("theme") or "(not inferred)",
        duration=srtot(context.get("duration", 0)),
        outline=outline[:6000],
        parts=parts[:24000],
        req_lines=req_lines or "(none mentioned)",
    )
    try:
        content, reasoning = _chat(
            prompt, cfg, max_tokens=cfg.notes_summary_max_tokens,
            label="summary", timeout=max(cfg.request_timeout, 1800))
        if content and not is_cot_garbage(content):
            return content
        salv = salvage_from_reasoning(reasoning, content)
        if salv:
            return salv
        log("  notes: summary had an invalid format; omitting summary section")
    except Exception as exc:
        log(f"  notes: summary failed ({exc}); omitting summary section")
    return ""


def _section_label(sec: dict) -> str:
    from merge import slide_text
    for f in sec["frames"]:
        t = (f.get("text") or "").strip()
        if t:
            return " ".join(t.split())[:70]
    st = slide_text(sec["frames"])
    if st:
        return " ".join(st.split())[:70]
    return "Lecture segment"


# ------------------------------------------------------------- rendering ---

def render_notes(title: str, sections: list[dict], units: list[dict],
                 unit_mds: list[str], summary_md: str, meta: dict) -> str:
    parts: list[str] = [f"# {title}\n"]
    meta_lines = [
        f"- **Duration:** {srtot(meta['duration'])} ({meta['duration'] / 60:.0f} min)",
        f"- **Language:** {meta.get('language') or '(auto)'}",
        f"- **ASR:** {meta.get('asr_model')}",
        f"- **Notes model:** {meta.get('vision_model')}",
        f"- **Generated:** {meta.get('generated_at')}",
    ]
    if meta.get("theme"):
        meta_lines.insert(1, f"- **Theme:** {meta['theme']}")
    parts.append("\n".join(meta_lines) + "\n")

    if summary_md:
        parts.append("## Summary\n")
        parts.append(summary_md + "\n")

    parts.append("## Outline\n")
    for sec in sections:
        parts.append(f"- `{srtot(sec['t_start'])}` {_section_label(sec)}")
    parts.append("")

    parts.append("## Notes\n")
    for u, md in zip(units, unit_mds):
        label = next((_section_label(s) for s in sections
                      if u["i_start"] <= s["idx"] <= u["i_end"]
                      and _section_label(s) != "Lecture segment"),
                     "Lecture segment")
        parts.append(f"### `{srtot(u['t_start'])} - {srtot(u['t_end'])}` {label[:60]}")
        parts.append("")
        parts.append(md)
        parts.append("")
    return "\n".join(parts)


def render_transcription(asr_chunks: list[dict], meta: dict) -> str:
    parts = [
        f"# Transcription — {meta['title']}",
        "",
        f"Duration {srtot(meta['duration'])} · model {meta.get('asr_model')} · "
        f"language {meta.get('language') or '(auto)'} · "
        f"chunk-level timestamps (flat text per {meta.get('chunk_seconds', 1800.0):.0f}s chunk)",
        "",
    ]
    for c in asr_chunks:
        parts.append(f"## [{srtot(c['start'])} - {srtot(c['end'])}]")
        parts.append("")
        parts.append(" ".join(c["text"].split()))
        parts.append("")
    return "\n".join(parts)


def write_outputs(lecture_dir: Path, sections: list[dict], asr_chunks: list[dict],
                  units: list[dict], unit_mds: list[str], summary_md: str,
                  meta: dict) -> list[Path]:
    notes_p = lecture_dir / "notes.md"
    trans_p = lecture_dir / "transcription.md"
    notes_p.write_text(render_notes(meta["title"], sections, units, unit_mds,
                                    summary_md, meta), encoding="utf-8")
    trans_p.write_text(render_transcription(asr_chunks, meta), encoding="utf-8")
    log(f"  wrote {notes_p.name} ({notes_p.stat().st_size // 1024} KB), "
        f"{trans_p.name} ({trans_p.stat().st_size // 1024} KB)")
    return [notes_p, trans_p]
