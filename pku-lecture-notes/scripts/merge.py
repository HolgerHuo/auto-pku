"""Merge the two legs onto one timeline and build sections + note units.

ASR text has chunk-level timestamps;
slide frames have exact transition timestamps. Sections are built primarily
from slide boundaries -- that is what makes the notes readable (one section
per slide/topic).

Speech attribution is sentence-level, not chunk-level: a chunk's sentences
are assigned absolute times by walking its start + character fractions (a
lecture's talking rate varies far less than its sentence boundaries move,
so the estimate is good to a few seconds), then each sentence lands in the
section whose span contains it. The old chunk-level "attach to every
overlapping section" duplicated a 30-minute chunk across ~50 sections.

Note units: sections are grouped so each unit carries at most
`notes_unit_chars` characters of slide+speech text (units are aligned to
section boundaries, so a unit is always a contiguous run of sections). The
notes stage synthesizes one coherent study-notes unit per group; a lone
section is its own unit.
"""
from __future__ import annotations

from common import log, srtot

MIN_SECTION = 30.0    # merge tiny gaps between slides
MAX_SECTION = 900.0   # split long slide-free stretches


def build_sections(slides: list[dict], asr_chunks: list[dict], duration: float) -> list[dict]:
    usable = [s for s in slides if s["kind"] != "chrome"]
    # Slide-anchored section boundaries: each kept slide starts a section.
    bounds = [0.0]
    for s in usable:
        if s["t"] - bounds[-1] >= MIN_SECTION:
            bounds.append(s["t"])
    if duration - bounds[-1] >= MIN_SECTION:
        bounds.append(duration)
    bounds.append(duration)

    # Split any inter-slide gap longer than MAX_SECTION.
    split: list[float] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        while b - a > MAX_SECTION:
            a += MAX_SECTION
            split.append(a)
    bounds = sorted(set(bounds + split))

    # Assign each sentence an estimated absolute time from its chunk, then
    # route it to the section containing that time.
    sent_times: list[tuple[float, str]] = []
    for c in asr_chunks:
        for s in _sentences(c.get("text", "")):
            sent_times.append((_estimate_time(c, s), s))
    sent_times.sort(key=lambda p: p[0])

    sections: list[dict] = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b - a < 5:
            continue
        frames = [s for s in usable if a <= s["t"] < b]
        speech = "".join(sent for t, sent in sent_times if a <= t < b)
        sections.append({
            "idx": i,
            "t_start": round(a, 3),
            "t_end": round(b, 3),
            "frames": frames,
            "speech": speech,
        })
    log(f"  sections: {len(sections)} sections from {len(usable)} slides + "
        f"{len(asr_chunks)} asr chunks ({len(sent_times)} sentences attributed)")
    return sections


def _sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the punctuation attached."""
    import re
    parts = re.split(r"(?<=[。！？!?；;])\s*", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _estimate_time(chunk: dict, sent: str) -> float:
    """Estimated absolute time of a sentence: chunk start + char fraction.

    Assumes a steady speaking rate within a chunk (a reasonable estimate for
    lecture audio; errors stay within the chunk and are small at 30-min
    chunk size).
    """
    return _estimate_in(chunk, sent)


def _estimate_in(chunk: dict, sent: str) -> float:
    start, end = float(chunk["start"]), float(chunk["end"])
    dur = max(end - start, 1.0)
    full = chunk.get("text", "")
    idx = full.find(sent)
    if idx < 0:
        # exact match not found (renderer/normalization differences): fall
        # back to the longest common prefix position, else chunk midpoint
        idx = _prefix_pos(full, sent)
    return start + dur * (idx + len(sent) / 2.0) / max(len(full), 1)


def _prefix_pos(full: str, sent: str) -> int:
    """Best start offset for `sent` inside `full`, scanning by prefix hits."""
    if not sent:
        return 0
    step = max(1, len(full) // 400)
    best, best_hits = 0, 0
    head = sent[: max(3, len(sent) // 4)]
    for i in range(0, len(full) - len(head), step):
        if full[i:i + len(head)] == head:
            hits = sum(1 for a, b in zip(sent, full[i:i + len(sent)]) if a == b)
            if hits > best_hits:
                best, best_hits = i, hits
            if hits > len(sent) * 0.9:
                break
    return best


def slide_text(frames: list[dict]) -> str:
    """One line per non-chrome frame: kind + text + content (the unit's slide material)."""
    lines = []
    for f in frames:
        if f["kind"] == "chrome":
            continue
        text = (f.get("text") or "").strip()
        content = (f.get("content") or "").strip()
        body = text if text else content
        if not body:
            continue
        label = {"slide": "Slide", "blackboard": "Board",
                 "demo_video": "Demo"}.get(f["kind"], "Visual")
        lines.append(f"[{label} {srtot(f['t'])}] {body}")
    return "\n".join(lines)


def _split_long_atom(a: str, limit: int) -> list[str]:
    """Break an atom longer than `limit` into <=limit pieces.

    Try sentence ends, then clause punctuation, then hard character chunks,
    so a wall of text (ASR with no sentence punctuation, one giant slide
    line) can never make a part exceed the cap.
    """
    import re
    if len(a) <= limit:
        return [a]
    for pat in (r"(?<=[。！？!?])", r"(?<=[，,；;、])"):
        bits = [b for b in re.split(pat, a) if b]
        if len(bits) > 1:
            return _pack_chars(bits, limit)
    return [a[i:i + limit] for i in range(0, len(a), limit)]


def _pack_chars(bits: list[str], limit: int) -> list[str]:
    """Greedily concatenate bits (keeping their own text) up to `limit`."""
    out: list[str] = []
    cur = ""
    for b in bits:
        if len(b) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_split_long_atom(b, limit))
            continue
        if cur and len(cur) + len(b) > limit:
            out.append(cur)
            cur = b
        else:
            cur += b
    if cur:
        out.append(cur)
    return out


def _section_parts(sec: dict, unit_chars: int) -> list[dict]:
    """Split a section's material into <= unit_chars pieces.

    A normal section is one piece. A section whose slide+speech material
    exceeds the cap (a long speech stretch with few slide changes, or one
    very dense slide) is cut along sentence/clause boundaries into
    contiguous pieces with proportionally estimated time ranges. No
    synthesis call may see oversized input.
    """
    st = slide_text(sec["frames"]).strip()
    sp = (sec.get("speech") or "").strip()
    t0, t1 = sec["t_start"], sec["t_end"]
    if len(st) + len(sp) <= unit_chars:
        return [{"slides": st, "speech": sp, "t_start": t0, "t_end": t1}]

    # Atoms in material order, each labeled slide/speech and hard-capped so a
    # single wall of text (one giant slide line, ASR without punctuation)
    # cannot defeat the cap.
    atoms: list[tuple[str, str]] = []  # (label, text)
    for line in st.splitlines():
        for a in _split_long_atom(line, unit_chars):
            atoms.append(("slide", a))
    for s in _sentences(sp):
        for a in _split_long_atom(s, unit_chars):
            atoms.append(("speech", a))
    total = sum(len(a) for _, a in atoms)
    if total == 0:
        return []

    # Greedy packing: (start_char, [atoms]). Budget includes the separators
    # re-assembly adds (one per atom), so reassembled parts honor the cap.
    packs: list[tuple[int, list]] = []
    cur: list = []
    cur_len = 0
    consumed = 0
    for label, a in atoms:
        if cur and cur_len + len(a) + 1 > unit_chars:
            packs.append((consumed, cur))
            cur, cur_len = [], 0
        cur.append((label, a))
        cur_len += len(a) + 1
        consumed += len(a)
    if cur:
        packs.append((consumed, cur))

    out = []
    for i, (start_char, pieces) in enumerate(packs):
        frac = start_char / total
        slides = "".join(a + "\n" for label, a in pieces if label == "slide").rstrip("\n")
        speech = "".join(a + " " for label, a in pieces if label == "speech").strip()
        t_start = t0 + (t1 - t0) * frac
        t_end = t1 if i == len(packs) - 1 else t0 + (t1 - t0) * (packs[i + 1][0] / total)
        out.append({"slides": slides, "speech": speech,
                    "t_start": t_start, "t_end": t_end})
    return out


def build_units(sections: list[dict], unit_chars: int = 3500) -> list[dict]:
    """Group section material into synthesis units (part-level packing).

    Each section contributes one or more parts (_section_parts); parts pack
    greedily into units up to `unit_chars`, so a unit never straddles a
    material boundary and a huge single section is split, not dumped whole
    at the model.
    """
    pieces: list[tuple[dict, dict]] = []  # (section, part)
    for sec in sections:
        for part in _section_parts(sec, unit_chars):
            pieces.append((sec, part))
    units: list[dict] = []
    cur: dict | None = None
    for sec, part in pieces:
        mat = len(part["slides"]) + len(part["speech"])
        if cur is None:
            cur = {"parts": [(sec, part)], "chars": mat}
            continue
        if cur["chars"] + mat > unit_chars:
            units.append(cur)
            cur = {"parts": [(sec, part)], "chars": mat}
        else:
            cur["parts"].append((sec, part))
            cur["chars"] += mat
    if cur is not None:
        units.append(cur)
    out = []
    for u in units:
        parts = [(s["idx"], p) for s, p in u["parts"]]
        out.append({
            "i_start": parts[0][0],
            "i_end": parts[-1][0],
            "t_start": min(p["t_start"] for _, p in parts),
            "t_end": max(p["t_end"] for _, p in parts),
            "parts": u["parts"],
            "chars": u["chars"],
        })
    log(f"  units: {len(out)} note units from {len(sections)} sections "
        f"({len(pieces)} parts, cap {unit_chars} chars)")
    return out
