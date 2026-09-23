#!/usr/bin/env python3
"""Download PKU 课堂实录 (course.pku.edu.cn lecture recordings).

The 课堂实录 videos are HLS (m3u8) streams served by PKU's 智播课 platform
(onlineroomse.pku.edu.cn / yjapise.pku.edu.cn / resourcese.pku.edu.cn).
They are NOT plain download links — the m3u8 URL only exists in the platform
API response and the CDN expects a logged-in cookie.

How it works (reverse-engineered 2026-09):
  1. The platform API `yjapise.pku.edu.cn/courseapi/v2/schedule/
     search-live-course-list?all=1&course_id=<id>&with_sub_data=1` returns
     every session of a course. Each session's `sub_content` (JSON string)
     carries `save_playback.contents` = the m3u8 URL when `is_m3u8 == "yes"`.
     This endpoint needs ONLY the platform login cookies on `.pku.edu.cn`
     (JWTUser, _token, login_cmc_*, group_code, cmc_version) — no Blackboard
     JSESSIONID and no one-time auth_data.
  2. The m3u8 lives on `resourcese.pku.edu.cn` and must be fetched with the
     `.pku.edu.cn` / resourcese cookies + a browser-like Referer.
  3. Segments are downloaded and stitched with ffmpeg.

Cookie setup (one time, per login):
  The long-lived platform cookies last ~7 days. Export them from any logged-in
  browser session. Easiest with the local camofox server (already used for
  browsing course.pku.edu.cn):

      # camofox exports a Playwright storage_state; convert to Netscape format:
      curl -s http://localhost:9377/sessions/<userId>/storage_state \
        -H "Authorization: Bearer $CAMOFOX_ACCESS_KEY" > state.json
      uv run --with requests python pku_video.py --export-cookies state.json > cookies.txt

  Or export a Netscape cookies.txt from your normal browser (e.g. the
  "cookies.txt" extension) — it must contain the cookies for `.pku.edu.cn`.

Usage:
  # list all recorded sessions of a course (id or exact name)
  uv run --with requests python pku_video.py --course 203706 --cookies cookies.txt
  uv run --with requests python pku_video.py --course 公共初级意大利语 --find --cookies cookies.txt

  # download the two most recent recordings
  uv run --with requests python pku_video.py --course 203706 --latest 2 --cookies cookies.txt \
      --out ../courses/games-change-world/media

  # download every recorded session
  uv run --with requests python pku_video.py --course 人类沟通的起源与发展 --all --prefix comm \
      --cookies cookies.txt --out ../courses/origin-of-communication/media

  # download a specific session
  uv run --with requests python pku_video.py --course 203706 --sub-id 4578930 --cookies cookies.txt

Notes:
  * Needs: uv and ffmpeg on PATH.
  * The media CDN (resourcese.pku.edu.cn) appears to restrict egress IPs to
    the PKU campus. From outside campus the m3u8/segments 404 even with
    valid cookies — run the download from a PKU network (or a PKU-side proxy)
    if you get 404s here while the browser player works for you.
  * Proxy: honors HTTPS_PROXY/HTTP_PROXY env; pass --proxy to override.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API = "https://yjapise.pku.edu.cn/courseapi/v2/schedule/search-live-course-list"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Firefox/140.0")
REFERER = "https://onlineroomse.pku.edu.cn/player"

COOKIE_KEYS = {"JWTUser", "_token", "login_cmc_id", "login_cmc_tid",
               "login_cmc_url", "login_cmc_type", "cmc_version", "group_code"}


@dataclass
class Recording:
    sub_id: int
    course_id: int
    title: str
    begin: int          # unix ts of the session start
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
            line = line[len("#HttpOnly_"):]
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        dom, _flag, cpath, secure, expires, name, value = parts
        # 0 (or -1, as camofox exports session cookies) = session cookie
        exp = int(expires)
        jar.set(name, value, domain=dom, path=cpath,
                secure=secure.upper() == "TRUE",
                expires=exp if exp > 0 else None)
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
        sys.stdout.write(
            f"{dom}\tTRUE\t{c.get('path', '/')}\t"
            f"{'TRUE' if c.get('secure') else 'FALSE'}\t"
            f"{c.get('expires', 0) if c.get('expires', 0) > 0 else 0}\t"
            f"{c['name']}\t{c['value']}\n")


# ----------------------------------------------------------------- discovery

def session(jar: requests.cookies.RequestsCookieJar,
            proxy: str | None) -> requests.Session:
    s = requests.Session()
    s.cookies = jar
    s.headers.update({"User-Agent": UA, "Referer": REFERER,
                      "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"})
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    elif os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        s.proxies = {"http": os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", ""),
                     "https": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY", "")}
    return s


def list_recordings(s: requests.Session, course_id: int) -> list[Recording]:
    r = s.get(API, params={"all": 1, "course_id": course_id,
                           "with_sub_data": 1, "with_room_data": 1}, timeout=30)
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
        recs.append(Recording(
            sub_id=it["sub_id"], course_id=course_id,
            title=(it.get("sub_title") or it.get("title") or f"sub{it['sub_id']}").strip(),
            begin=int(it.get("course_begin") or 0),
            duration_tick=int(sp.get("contents_duration") or 0),
            m3u8=sp["contents"]))
    recs.sort(key=lambda r: r.begin, reverse=True)
    return recs


def search_course(s: requests.Session, name: str) -> list[dict]:
    """Look up course ids by (exact) title via the platform course catalog."""
    r = s.get("https://yjapise.pku.edu.cn/courseapi/v2/course",
              params={"page": 1, "pageSize": 50, "title": name}, timeout=30)
    r.raise_for_status()
    models = r.json()["result"]["data"]["models"]
    return [{"id": m["id"], "title": m["title"],
             "teacher": m.get("realname"),
             "start": m.get("start_at"), "end": m.get("end_at")} for m in models]


# ------------------------------------------------------------------ m3u8 ---

def http_get(s: requests.Session, url: str, retries: int = 3,
             timeout: int = 30) -> requests.Response:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 * (i + 1))
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (i + 1))
    raise SystemExit(f"GET failed after {retries} tries: {url} ({last})")


def m3u8_to_mp4(s: requests.Session, m3u8: str, out_dir: Path,
                name: str, save_playlist: bool = True) -> Path:
    """Download an (encrypted) HLS stream with ffmpeg, using the login cookies.

    ffmpeg fetches the master/media playlists, resolves any #EXT-X-KEY
    (AES-128), downloads + decrypts the segments, and remuxes to mp4 — far
    more robust than manual segment handling for encrypted HLS.
    """
    r = http_get(s, m3u8, retries=3, timeout=60)
    if r.status_code == 404:
        raise SystemExit(
            f"m3u8 404 for {m3u8}\n"
            "PKU's media CDN (resourcese.pku.edu.cn) rate-limits/blocks\n"
            "egress IPs. Wait a bit and retry, or run from a PKU network.")
    r.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_playlist:
        (out_dir / f"{name}.m3u8").write_text(r.text)
        log(f"saved playlist {out_dir / (name + '.m3u8')}")

    mp4 = out_dir / f"{name}.mp4"
    cookie = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
    headers = (f"User-Agent: {UA}\r\n"
               f"Referer: {REFERER}\r\n"
               f"Cookie: {cookie}\r\n")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-user_agent", UA,
           "-headers", headers,
           "-i", m3u8,
           "-c", "copy", "-bsf:a", "aac_adtstoasc",
           str(mp4)]
    log(f"ffmpeg → {mp4.name} (cookies attached; ffmpeg resolves key + decrypts)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))
    if not mp4.exists() or mp4.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output: {mp4}")
    log(f"saved {mp4} ({mp4.stat().st_size/1e6:.1f} MB)")
    return mp4


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course", help="course_id (number) or exact course name, e.g. 公共初级意大利语")
    ap.add_argument("--find", action="store_true", help="with --course <name>: list matching courses and exit")
    ap.add_argument("--sub-id", type=int, help="download a specific session (sub_id)")
    ap.add_argument("--latest", type=int, help="download the N most recent recordings")
    ap.add_argument("--all", action="store_true", help="download every recorded session")
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt")
    ap.add_argument("--export-cookies", type=Path,
                    help="convert camofox storage_state.json → Netscape cookies.txt on stdout")
    ap.add_argument("--out", type=Path, default=Path("media"), help="output dir")
    ap.add_argument("--proxy", help="proxy for media downloads, e.g. http://host:port "
                                    "(default: HTTPS_PROXY env / direct)")
    ap.add_argument("--workers", type=int, default=8, help="(kept for compatibility)")
    ap.add_argument("--prefix", help="extra filename prefix, e.g. a course slug")
    ap.add_argument("--keep-segs", action="store_true", help="keep the .segs_* dir")
    args = ap.parse_args()

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
            try: return datetime.datetime.fromtimestamp(int(x)).strftime("%Y-%m")
            except Exception: return "?"
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

    if not (args.sub_id or args.latest or args.all):
        log(f"course {args.course}: {len(recs)} recorded sessions")
        for r in recs:
            dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.begin)) if r.begin else "?"
            log(f"  {dt}  sub={r.sub_id}  {r.title}  {r.duration_tick/1e7/60:.0f}min")
            log(f"      {r.m3u8}")
        return

    if args.sub_id:
        chosen = [r for r in recs if r.sub_id == args.sub_id]
        if not chosen:
            raise SystemExit(f"sub_id {args.sub_id} not recorded / not found")
    elif args.all:
        chosen = recs
    else:
        chosen = recs[: max(1, args.latest)]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(chosen, 1):
        dt = time.strftime("%Y%m%d", time.localtime(r.begin)) if r.begin else "000000"
        safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", r.title).strip("_")[:60]
        pre = (args.prefix + "_" if args.prefix else "")
        name = f"lec_{pre}{dt}_{r.sub_id}_{safe}"
        log(f"=== [{i}/{len(chosen)}] {r.title} (sub {r.sub_id}) → {name}")
        m3u8_to_mp4(s, r.m3u8, args.out, name)


if __name__ == "__main__":
    main()
