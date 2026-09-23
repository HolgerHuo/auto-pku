"""Render the final artifacts: study notes + transcription.

notes.md is NOT a dump of slides + audio. The lecture is cut into note units
(merge.build_units), and for each unit ONE vision-model call synthesizes
study notes from that unit's slide material + attributed lecture speech.
A second pass then reorganizes the unit notes into a lecture-organized
document: a planning call proposes a content-based outline (distinct topics,
practice blocks, announcements -- never time or slide boundaries), and one
call per outline section merges that section's unit notes into coherent
student-style study notes with the slide content folded into the prose.
Failed units fall back to the raw material and failed sections fall back to
their concatenated unit notes, so no content is ever lost. A final call
writes the lecture summary (main thread + takeaways).

transcription.md stays the mechanical, full transcript (chunk-stamped) --
the auditable source the notes were written from.
"""
from __future__ import annotations

import json as _json
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
        if st and is_cot_garbage(st):
            st = ""
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

# Bump when the integration prompts change: old cached plans/sections were
# written under the old prompts and must not be silently reused.
INTEGRATE_VERSION = "v2-loweffort"

INTEGRATE_PLAN_PROMPT = """You are organizing a university lecture into student \
study notes (课堂笔记), in the lecture's language.
Below are the time-ordered segment notes of the lecture (0-based indices with \
short previews). Create a CONTENT-based outline: 6 to 12 top-level sections \
that reflect what the lecture actually taught -- distinct topics, \
exercises/practice blocks, announcements or course requirements. Do NOT split \
by time or by slides; merge segments that cover the same topic, and keep each \
section focused.
Rules:
- Every segment 0..{n_minus_1} must be assigned to exactly one section, in \
chronological order (each section's segments form one contiguous run).
- Prefer 3-8 segments per section.
- Section titles are topic names in the lecture language: no timestamps, no \
"segment/part/continued" wording.
- Respond with ONLY one JSON object, no code fences, no commentary:
{{"sections": [{{"title": "...", "description": "one sentence on what this \
section covers", "segments": [0, 1, 2]}}]}}

{course} / {lecture}
Lecture topic: {theme}

Segments:
{listing}
"""

INTEGRATE_SECTION_PROMPT = """You are writing the study notes (课堂笔记) for one \
section of a university lecture, in the lecture's language. A careful student \
should be able to study this topic from your merged notes alone.
Below are the time-ordered segment notes for this section (each synthesized \
from the slides shown + the lecturer's speech). Merge them into ONE coherent, \
detailed set of notes for this section.
Rules:
- Write flowing study notes: prose plus bullets. Use ### sub-headings only \
where the topic genuinely shifts inside the section. No timestamps, no \
per-segment structure, no "slide N" references.
- Fold the slide content in: key terms, names, numbers, formulas, and \
exercise instructions belong where the lecture discussed them. Do NOT dump \
raw slide text as a separate list.
- Keep terms, definitions, numbers, dates, and grading/requirement details \
exact; preserve the lecturer's examples (shortened when verbose).
- Merge and deduplicate repeated content across segments; do not repeat a \
fact.
- Course requirements, homework, and deadlines: state them concretely, near \
the top of the section when they appear there.
- If a fragment is garbled or unclear, write a short "(unclear in \
recording)" note; never invent facts.
- Start directly with content, no preamble.

{course} / {lecture}
Section: {title}
{description}

Segment notes:
{material}
"""


def _chat(prompt: str, cfg: Config, *, max_tokens: int, label: str,
          timeout: int | None = None,
          reasoning_effort: str | None = None) -> tuple[str, str]:
    body = {
        "model": cfg.vision_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if reasoning_effort:
        # The vision model is a reasoning model whose default effort is very
        # high: on a long merge it can burn the entire max_tokens budget
        # producing `reasoning_content` and return empty `content`
        # (finish_reason=length). Mechanical merge/plan/summary work does not
        # need deep reasoning, so callers can lower the effort.
        body["reasoning_effort"] = reasoning_effort
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
    # Providers differ on where reasoning goes: `reasoning` (openrouter-style)
    # vs `reasoning_content` (qwen-style). A reasoning model can spend its
    # whole budget thinking and leave `content` empty, so this field must be
    # captured for salvage_from_reasoning.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return (content or "").strip(), reasoning.strip()


# Signatures of model deliberation leaking into the answer field.
_COT_HEAD = _re.compile(
    r"^(we (need|should|must|are|will|are going)|let (me|'s)|i (need|should|think|'ll|am|will)|ok,|sure,|the user (is|wants|asks|is asking)|我们需要|我需要|让我们|好的，|首先，(让我|我需要|先)|分析(?:一下)?(?:用户|题目))",
    _re.I)
_COT_MID = _re.compile(
    r"(?m:^\s*need\b)|"
    r"(?m:^\s*potential issue:)|"
    r"(?m:^\s*this is complete\.)|"
    r"(?m:^\s*open questions only if)|"
    r"let me (now )?(write|check|structure|draft)|wait,|hmm,|"
    r"as a (good|helpful)|i should (probably )?now|"
    r"now (i|let) (write|produce|finalize)",
    _re.I,
)


def is_cot_garbage(text: str) -> bool:
    """Return whether text looks like model deliberation rather than notes."""
    t = (text or "").strip()
    if not t:
        return False
    head = t[:400]
    if _COT_HEAD.match(t):
        return True
    return bool(_COT_MID.search(t))


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
                    if md and is_cot_garbage(md):
                        md = ""
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


def _integrate_max_tokens(cfg: Config, n_chars: int = 0) -> int:
    """Token budget for integration calls.

    Unit synthesis is kept small (NOTES_MAX_TOKENS) because each call covers
    one 1500-char unit. Integration calls cover an entire lecture (plan) or a
    whole topic (section merge), so they need a larger budget -- the JSON plan
    and merged notes are routinely longer than 2048 tokens.

    These calls run with `reasoning_effort=low` (see _chat), so the thinking
    stays short and the budget is dominated by the answer. A detailed merge
    of ~36k chars of material needs up to ~20k output tokens, so scale with
    the material, floored at 12k (enough for a full-lecture plan).
    """
    return max(12000, cfg.notes_max_tokens, n_chars // 2)


def synthesize_summary(sections: list[dict], units: list[dict], unit_mds: list[str],
                       context: dict, cfg: Config) -> str:
    """Cheap final call: overview + main thread + key points."""
    cfg.require_vision()
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
            prompt, cfg,
            max_tokens=_integrate_max_tokens(
                cfg, len(parts) + len(req_lines)),
            label="summary", timeout=max(cfg.request_timeout, 1800),
            reasoning_effort="low")
        if content and not is_cot_garbage(content):
            return content
        salv = salvage_from_reasoning(reasoning, content)
        if salv and not is_cot_garbage(salv):
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


# ------------------------------------------------------- lecture integrate --
# The unit notes are faithful but time-ordered fragments. Integration turns
# them into a content-organized lecture document: one planning call proposes
# a topic outline over the unit indices, then one call per outline section
# merges its units into detailed student-style notes.


def _parse_plan(text: str) -> list[dict]:
    """Extract the plan's sections from a model response.

    Tries the full JSON object first. The model sometimes truncates a long
    plan mid-object (returning `finish_reason: stop` with the text ending
    inside a section object), so on failure it salvages every COMPLETE section
    object with a raw JSON decoder. The salvage is validated by the caller
    (must be a chronological prefix), and any trailing units the plan dropped
    are back-filled into one continuation section.
    """
    s = text.find("{")
    if s < 0:
        raise ValueError("no JSON object in plan response")
    try:
        data = _json.loads(text[s:])
        sections = data.get("sections")
        if isinstance(sections, list) and sections:
            return sections
    except _json.JSONDecodeError:
        pass
    # Salvage complete section objects from a truncated / malformed response.
    dec = _json.JSONDecoder()
    sections: list[dict] = []
    for m in _re.finditer(r'\{"title"', text[s:]):
        start = m.start()
        try:
            obj, _end = dec.raw_decode(text[s + start:])
        except _json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "segments" in obj and "title" in obj:
            sections.append(obj)
    if not sections:
        raise ValueError("no complete sections in plan response")
    return sections


def _complete_plan(plan: list[dict], n_units: int) -> list[dict] | None:
    """Return a plan that covers all `n_units` exactly once, in order.

    Accepts a full plan, or a chronological PREFIX of one (the salvage case),
    back-filling the leftover units into a single continuation section.
    Returns None if the sections are out of order / have gaps / duplicates.
    """
    try:
        assigned: list[int] = []
        for sec in plan:
            segs = [int(x) for x in sec.get("segments", [])]
            if not segs or min(segs) != max(segs) + 1 - len(segs) \
                    or any(b < a for a, b in zip(segs, segs[1:])):
                return None
            for x in segs:
                if x < 0 or x >= n_units or x in assigned:
                    return None
            assigned.extend(segs)
    except (TypeError, ValueError):
        return None
    if not assigned or assigned != list(range(len(assigned))):
        # sections must start at 0 and be a contiguous chronological prefix
        return None
    k = len(assigned)
    if k < n_units:
        plan = list(plan)
        plan.append({"title": "后续内容（计划未覆盖）",
                     "description": "（planner 未覆盖到这部分，按顺序整理）",
                     "segments": list(range(k, n_units))})
    return plan


def _plan_key(context: dict, listing: str) -> str:
    h = hashlib.sha256(
        (INTEGRATE_VERSION + listing).encode("utf-8")).hexdigest()[:16]
    return f"{INTEGRATE_VERSION}_plan_{h}"


def _section_key(title: str, description: str, material: str) -> str:
    h = hashlib.sha256(
        (INTEGRATE_VERSION + title + description + material)
        .encode("utf-8")).hexdigest()[:16]
def integrate_notes(sections: list[dict], unit_mds: list[str], context: dict,
                    cache: Cache, cfg: Config) -> list[tuple[str, str]]:
    """Reorganize unit notes into content-based lecture sections.

    Returns [(title, description, merged_md)] per outline section.
    Falls back to concatenated unit notes when planning or section
    synthesis fails, so integration never loses content.
    """
    cfg.require_vision()
    cache_name = "notes_integrate.json"
    done = cache.load_json(cache_name, {}) or {}

    listing = "\n".join(
        f"{i}: {md.strip()[:200]}" for i, md in enumerate(unit_mds))
    pk = _plan_key(context, listing)
    plan = done.get(pk)
    if plan:
        log("  integrate: plan cached")
    else:
        prompt = INTEGRATE_PLAN_PROMPT.format(
            n_minus_1=len(unit_mds) - 1,
            course=context.get("course", ""),
            lecture=context.get("lecture", ""),
            theme=context.get("theme") or "(not inferred)",
            listing=listing[:24000],
        )
        plan = None
        for attempt in range(3):
            try:
                content, reasoning = _chat(
                    prompt, cfg, max_tokens=_integrate_max_tokens(cfg, len(listing)),
                    label="integrate plan", reasoning_effort="low")
                plan = _parse_plan(content or reasoning)
                break
            except Exception as exc:
                log(f"  integrate: plan attempt {attempt + 1} failed: {exc}")
        if plan:
            done[pk] = plan
            cache.save_json(cache_name, done)

    if plan:
        n_before = len(plan)
        plan = _complete_plan(plan, len(unit_mds))
        if plan is None:
            log("  integrate: plan invalid; using chronological fallback")
        elif len(plan) > n_before:
            log("  integrate: plan truncated; back-filled trailing units")

    out: list[tuple[str, str, str]] = []
    if plan:
        results: list[tuple[str, str, str]] = []
        todo: list[dict] = []
        for j, sec in enumerate(plan):
            title = str(sec.get("title") or f"Section {j + 1}").strip()[:120]
            desc = str(sec.get("description") or "").strip()[:300]
            segs = [int(s) for s in sec["segments"]]
            material = "\n\n".join(
                f"--- segment {k} ---\n{unit_mds[k]}" for k in segs)
            key = _section_key(title, desc, material)
            md = done.get(key)
            if md and not is_cot_garbage(md):
                results.append((title, desc, md))
            else:
                todo.append({"j": j, "title": title, "desc": desc,
                             "segs": segs, "material": material, "key": key})

        if todo:
            log(f"  integrate: {len(todo)} sections to merge "
                f"({len(plan) - len(todo)} cached)")

            def work(item: dict) -> tuple[int, str]:
                prompt = INTEGRATE_SECTION_PROMPT.format(
                    course=context.get("course", ""),
                    lecture=context.get("lecture", ""),
                    title=item["title"],
                    description=item["desc"] or "(no description)",
                    material=item["material"][:36000],
                )
                md = ""
                for attempt in range(3):
                    try:
                        content, reasoning = _chat(
                            prompt, cfg,
                            max_tokens=_integrate_max_tokens(
                                cfg, len(item["material"])),
                            label=f"integrate {item['j'] + 1}",
                            reasoning_effort="low")
                        if content and not is_cot_garbage(content):
                            md = content
                        elif not content:
                            md = salvage_from_reasoning(reasoning)
                            if md and is_cot_garbage(md):
                                md = ""
                        if md:
                            break
                    except Exception as exc:
                        log(f"  integrate: section {item['j'] + 1} attempt "
                            f"{attempt + 1} failed: {exc}")
                if not md:
                    log(f"  integrate: section {item['j'] + 1} failed; "
                        f"keeping merged unit notes")
                    md = "\n\n".join(unit_mds[k] for k in item["segs"])
                return item["j"], md

            filled: dict[int, str] = {}
            with ThreadPoolExecutor(
                    max_workers=max(1, cfg.notes_workers)) as ex:
                for n, (j, md) in enumerate(ex.map(work, todo), 1):
                    filled[j] = md
                    if n % 2 == 0 or n == len(todo):
                        log(f"  integrate {n}/{len(todo)}")
                        for item in todo:
                            if item["j"] in filled:
                                done[item["key"]] = filled[item["j"]]
                        cache.save_json(cache_name, done)
        for item in todo:
            results.append((item["title"], item["desc"], filled[item["j"]]))
        out = results
        cache.save_json(cache_name, done)
    else:
        # No usable plan: group units chronologically into 5-10 chunks.
        n = len(unit_mds)
        per = max(1, -(-n // 8))
        for a in range(0, n, per):
            segs = list(range(a, min(a + per, n)))
            out.append(("Part of the lecture",
                        "(planning failed; chronological grouping)",
                        "\n\n".join(unit_mds[segs[0]:segs[-1] + 1])))
    return out


# ------------------------------------------------------------- rendering ---
#
# notes.md layout (student study notes, organized by lecture CONTENT):
#   # title / meta
#   ## Summary                      (lecture-level, model-written)
#   ## Notes                        (content-organized sections)
#     ### <topic title>             (one per planned outline section;
#                                   slide content folded into the prose)
# No time or slide-based division: the integration plan (see
# INTEGRATE_PLAN_PROMPT) decides the structure from what was taught.

# Normalize heading levels inside one integrated section. The document's
# top-level sections are rendered as `### <topic>`; the model sometimes
# emits a `## ` or `# ` line inside a section body. Demote those so the
# nesting stays valid (a stray `##` becomes `####`, never a new top-level).
def _nest_headings(md: str) -> str:
    return _re.sub(r"^(#{1,2})(\s+\S.*)$", r"####\2", md, flags=_re.M)


def render_notes(title: str,
                 integrated: list[tuple[str, str, str]],
                 summary_md: str, meta: dict) -> str:
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

    parts.append("## Notes\n")
    for title, desc, md in integrated:
        parts.append(f"### {title}\n")
        if desc:
            parts.append(f"*{desc}*\n")
        parts.append("")
        parts.append(_nest_headings(md))
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
                  units: list[dict], unit_mds: list[str],
                  integrated: list[tuple[str, str, str]], summary_md: str,
                  meta: dict) -> list[Path]:
    notes_p = lecture_dir / "notes.md"
    trans_p = lecture_dir / "transcription.md"
    notes_p.write_text(render_notes(meta["title"],
                                    integrated, summary_md, meta),
                       encoding="utf-8")
    trans_p.write_text(render_transcription(asr_chunks, meta), encoding="utf-8")
    log(f"  wrote {notes_p.name} ({notes_p.stat().st_size // 1024} KB), "
        f"{trans_p.name} ({trans_p.stat().st_size // 1024} KB)")
    return [notes_p, trans_p]
