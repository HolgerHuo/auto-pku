#!/usr/bin/env python3
"""Download authenticated PKU course recordings; see SKILL.md for browser login.

Use `uv run --project course-download course-download/pku_video.py --help`.
Recordings are identified by course_id/sub_id. A manifest tracks validated outputs;
HLS segments are cached for resume. Existing untracked videos require explicit import.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from download_engine import Downloader, DownloadError, atomic_json, probe

API = "https://yjapise.pku.edu.cn/courseapi/v2/schedule/search-live-course-list"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/140.0"
)
REFERER = "https://onlineroomse.pku.edu.cn/player"

COOKIE_KEYS = {
    "JWTUser",
    "_token",
    "login_cmc_id",
    "login_cmc_tid",
    "login_cmc_url",
    "login_cmc_type",
    "cmc_version",
    "group_code",
}


@dataclass
class Recording:
    sub_id: int
    course_id: int
    title: str
    begin: int  # unix ts of the session start
    duration_tick: int  # .NET ticks (100ns); 0 if unknown
    m3u8: str


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- cookies ---


def load_netscape_jar(path: Path) -> requests.cookies.RequestsCookieJar:
    jar = requests.cookies.RequestsCookieJar()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_") :]
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        dom, _flag, cpath, secure, expires, name, value = parts
        # 0 (or -1, as camofox exports session cookies) = session cookie
        exp = int(expires)
        jar.set(
            name,
            value,
            domain=dom,
            path=cpath,
            secure=secure.upper() == "TRUE",
            expires=exp if exp > 0 else None,
        )
    return jar


def platform_cookies(jar: requests.cookies.RequestsCookieJar) -> dict:
    """Cookies the platform API + media CDN actually need."""
    out: dict[str, str] = {}
    for c in jar:
        dom = c.domain.lstrip(".")
        if dom == "pku.edu.cn" or dom.endswith(".pku.edu.cn"):
            out[c.name] = c.value
    if not any(k in out for k in ("JWTUser", "_token")):
        log("WARNING: no JWTUser/_token in cookies — are you logged in?")
    return out


def export_cookies(storage_state_json: Path) -> None:
    """Convert a camofox/Playwright storage_state.json to Netscape format."""
    data = json.loads(storage_state_json.read_text())
    for c in data.get("cookies", []):
        dom = c["domain"]
        expires = int(c.get("expires") or 0)
        sys.stdout.write(
            f"{dom}\tTRUE\t{c.get('path', '/')}\t"
            f"{'TRUE' if c.get('secure') else 'FALSE'}\t"
            f"{expires if expires > 0 else 0}\t"
            f"{c['name']}\t{c['value']}\n"
        )


# ----------------------------------------------------------------- discovery


def session(jar: requests.cookies.RequestsCookieJar, proxy: str | None) -> requests.Session:
    s = requests.Session()
    s.cookies = jar
    s.headers.update(
        {"User-Agent": UA, "Referer": REFERER, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"}
    )
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    elif os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        s.proxies = {
            "http": os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", ""),
            "https": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY", ""),
        }
    return s


def list_recordings(s: requests.Session, course_id: int) -> list[Recording]:
    r = s.get(
        API,
        params={"all": 1, "course_id": course_id, "with_sub_data": 1, "with_room_data": 1},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("code") != 0:
        raise SystemExit(f"API error code={d.get('code')} msg={d.get('msg')}")
    recs: list[Recording] = []
    for it in d.get("list", []):
        sc = it.get("sub_content")
        sp = json.loads(sc).get("save_playback", {}) if sc else {}
        if sp.get("is_m3u8") != "yes" or not sp.get("contents"):
            continue  # not (yet) recorded
        recs.append(
            Recording(
                sub_id=int(it["sub_id"]),
                course_id=course_id,
                title=(it.get("sub_title") or it.get("title") or f"sub{it['sub_id']}").strip(),
                begin=int(it.get("course_begin") or 0),
                duration_tick=int(sp.get("contents_duration") or 0),
                m3u8=sp["contents"],
            )
        )
    recs.sort(key=lambda r: r.begin, reverse=True)
    return recs


def search_course(s: requests.Session, name: str) -> list[dict]:
    """Look up course ids by (exact) title via the platform course catalog."""
    r = s.get(
        "https://yjapise.pku.edu.cn/courseapi/v2/course",
        params={"page": 1, "pageSize": 50, "title": name},
        timeout=30,
    )
    r.raise_for_status()
    models = r.json()["result"]["data"]["models"]
    return [
        {
            "id": m["id"],
            "title": m["title"],
            "teacher": m.get("realname"),
            "start": m.get("start_at"),
            "end": m.get("end_at"),
        }
        for m in models
    ]


def import_inventory(engine, path, root, course_id, course_slug, recordings):
    """Import an explicit legacy identity mapping; never infer identity from duration."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise DownloadError("Legacy inventory must be a list")
    known = {r.sub_id: r for r in recordings}
    used = {
        str((engine.manifest.parent / e["path"]).resolve()): key
        for key, e in engine.state["recordings"].items()
    }
    imported = {}
    for item in data:
        if item.get("course") != course_slug:
            continue
        sub_id = int(item["sub_id"])
        if sub_id not in known:
            raise DownloadError(f"Inventory sub_id {sub_id} is not in course {course_id}")
        recording = known[sub_id]
        if item.get("date") != recording.title[:10]:
            raise DownloadError(f"Inventory date mismatch for sub_id {sub_id}")
        target = (root / item["path"]).resolve()
        key = engine.identity(recording)
        if key in imported or (str(target) in used and used[str(target)] != key):
            raise DownloadError("Duplicate recording identity or path in inventory")
        playlist = engine.playlist(recording.m3u8)
        expected = sum(s.duration for s in playlist.segments)
        entry = {
            "course_id": course_id,
            "sub_id": sub_id,
            "title": recording.title,
            "begin": recording.begin,
            "path": os.path.relpath(target, engine.manifest.parent),
            "playlist_duration": expected,
            "source_warning": (
                f"Source duration differs: catalog {recording.duration_tick / 1e7:.0f}s, HLS {expected:.0f}s"
                if recording.duration_tick and abs(recording.duration_tick / 1e7 - expected) > 60
                else None
            ),
            "status": "complete",
            **probe(target, expected),
        }
        imported[key] = entry
        used[str(target)] = key
    if not imported:
        raise DownloadError("No matching inventory entries; check --inventory-course")
    engine.state["recordings"].update(imported)
    atomic_json(engine.manifest, engine.state)
    log(f"Imported {len(imported)} explicit identities")


# -------------------------------------------------------------------- main ---


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--course", help="course_id (number) or exact course name, e.g. 公共初级意大利语"
    )
    ap.add_argument(
        "--find", action="store_true", help="with --course <name>: list matching courses and exit"
    )
    ap.add_argument("--sub-id", type=int, help="download a specific session (sub_id)")
    ap.add_argument("--latest", type=int, help="download the N most recent recordings")
    ap.add_argument("--all", action="store_true", help="download every recorded session")
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt")
    ap.add_argument(
        "--export-cookies",
        type=Path,
        help="convert camofox storage_state.json → Netscape cookies.txt on stdout",
    )
    ap.add_argument("--out", type=Path, default=Path("media"), help="output dir")
    ap.add_argument(
        "--proxy",
        help="proxy for media downloads, e.g. http://host:port (default: HTTPS_PROXY env / direct)",
    )
    ap.add_argument("--workers", type=int, default=2, help="initial concurrent recordings")
    ap.add_argument("--max-workers", type=int, default=8, help="adaptive concurrency ceiling")
    ap.add_argument(
        "--fixed-workers", action="store_true", help="disable bandwidth-based increases"
    )
    ap.add_argument(
        "--interval", type=float, default=15, help="progress/sample interval in seconds"
    )
    ap.add_argument("--manifest", type=Path, help="identity manifest (default: OUT/downloads.json)")
    ap.add_argument(
        "--dry-run", action="store_true", help="verify existing files and list missing recordings"
    )
    ap.add_argument("--import-inventory", type=Path, help="explicit legacy inventory JSON")
    ap.add_argument("--inventory-course", help="course slug in legacy inventory")
    ap.add_argument("--inventory-root", type=Path, default=Path.cwd(), help="root for legacy paths")
    ap.add_argument("--prefix", help="deprecated; output names use course/sub IDs")
    ap.add_argument(
        "--keep-segs", action="store_true", help="retain encrypted HLS segments after success"
    )
    args = ap.parse_args()

    if not 1 <= args.workers <= args.max_workers or args.interval <= 0:
        ap.error("require 1 <= workers <= max-workers and interval > 0")
    if args.import_inventory and not args.inventory_course:
        ap.error("--import-inventory requires --inventory-course")
    if args.export_cookies:
        export_cookies(args.export_cookies)
        return

    if not args.cookies:
        ap.error("--cookies is required (see --help for cookie export)")
    jar = load_netscape_jar(args.cookies)
    s = session(jar, args.proxy)

    if args.sub_id and not args.course:
        ap.error("--sub-id requires --course")
    if not args.course:
        ap.error("--course is required")

    if not str(args.course).isdigit():
        import datetime

        def fmt(x):
            try:
                return datetime.datetime.fromtimestamp(int(x)).strftime("%Y-%m")
            except TypeError, ValueError, OverflowError, OSError:
                return "?"

        matches = search_course(s, args.course)
        if not matches:
            raise SystemExit(f"no course named {args.course!r}")
        if args.find:
            log(f"course {args.course!r} matches:")
            for m in matches:
                log(f"  id={m['id']}  {m['teacher']}  {fmt(m['start'])}..{fmt(m['end'])}")
            return
        if len(matches) > 1:
            # same course runs each term — default to the most recent one
            matches.sort(key=lambda m: int(m["start"] or 0), reverse=True)
            log(f"course {args.course!r} has {len(matches)} terms; using the latest:")
            for m in matches:
                mark = "*" if m is matches[0] else " "
                log(f" {mark} id={m['id']}  {m['teacher']}  {fmt(m['start'])}..{fmt(m['end'])}")
            args.course = matches[0]["id"]
        else:
            log(f"course {args.course!r} → id {matches[0]['id']}")
            args.course = matches[0]["id"]

    recs = list_recordings(s, int(args.course))
    if not recs:
        raise SystemExit(f"no recorded sessions found for course {args.course}")

    if not (args.sub_id or args.latest or args.all or args.dry_run or args.import_inventory):
        log(f"course {args.course}: {len(recs)} recorded sessions")
        for r in recs:
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.begin)) if r.begin else "?"
            log(f"  {dt}  sub={r.sub_id}  {r.title}  {r.duration_tick / 1e7 / 60:.0f}min")
            log(f"      {r.m3u8}")
        return

    if args.sub_id:
        chosen = [r for r in recs if r.sub_id == args.sub_id]
        if not chosen:
            raise SystemExit(f"sub_id {args.sub_id} not recorded / not found")
    elif args.all or args.dry_run or args.import_inventory:
        chosen = recs
    else:
        chosen = recs[: max(1, args.latest)]

    import fcntl

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.out / "downloads.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.with_suffix(manifest.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another downloader is using this manifest") from None
        engine = Downloader(
            lambda: session(load_netscape_jar(args.cookies), args.proxy),
            manifest,
            workers=args.workers,
            max_workers=args.max_workers,
            adaptive=not args.fixed_workers,
            interval=args.interval,
            keep_segments=args.keep_segs,
            notify=log,
        )
        if args.import_inventory:
            import_inventory(
                engine,
                args.import_inventory,
                args.inventory_root,
                int(args.course),
                args.inventory_course,
                recs,
            )
        result = engine.run(chosen, args.out, dry_run=args.dry_run or bool(args.import_inventory))
        if result["failed"]:
            raise SystemExit(f"{result['failed']} recordings failed; rerun to resume")


if __name__ == "__main__":
    try:
        main()
    except (DownloadError, requests.RequestException) as exc:
        # Do not print request URLs, signatures, cookie values or raw response bodies.
        if isinstance(exc, requests.RequestException):
            status = exc.response.status_code if exc.response is not None else "network error"
            raise SystemExit(
                f"Request failed ({status}); check session and resource availability"
            ) from None
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; completed HLS segments retained for resume") from None
