"""Recover PKU media cookies through links in an existing Camofox session."""

from __future__ import annotations

import argparse
import os
import re
import time
import uuid
from pathlib import Path

import requests

from pku_video import API, load_netscape_jar, session


class LoginRequired(RuntimeError):
    pass


class BrowserSession:
    def __init__(self, base, api_key, user_id):
        self.base = base.rstrip("/")
        self.user_id = user_id
        self.client = requests.Session()
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    def call(self, method, path, data=None):
        response = self.client.request(
            method,
            self.base + path,
            params={"userId": self.user_id} if method != "POST" else None,
            json={"userId": self.user_id, **(data or {})} if method == "POST" else None,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("error"):
            raise RuntimeError("Camofox operation failed; inspect the browser tab")
        return result

    def tabs(self):
        return self.call("GET", "/tabs")["tabs"]

    def evaluate(self, tab, expression):
        return self.call("POST", f"/tabs/{tab}/evaluate", {"expression": expression})["result"]

    def click(self, tab, selector):
        return self.call("POST", f"/tabs/{tab}/click", {"selector": selector})

    def state(self):
        return self.call("GET", f"/sessions/{self.user_id}/storage_state")

    def export(self, cookies, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use a private file even when replacing an old export.
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.touch(mode=0o600, exist_ok=True)
        temporary.chmod(0o600)
        lines = ["# Netscape HTTP Cookie File"]
        for c in cookies:
            domain = c["domain"]
            if domain.lstrip(".") != "pku.edu.cn" and not domain.endswith(".pku.edu.cn"):
                continue
            lines.append(
                "\t".join(
                    [
                        domain,
                        "TRUE" if domain.startswith(".") else "FALSE",
                        c.get("path", "/"),
                        "TRUE" if c.get("secure") else "FALSE",
                        str(max(0, int(c.get("expires") or 0))),
                        c["name"],
                        c["value"],
                    ]
                )
            )
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(path)

    def recover(self, bb_id, cookies_path):
        original = {tab["tabId"] for tab in self.tabs()}
        group = f"course-download-{uuid.uuid4().hex[:8]}"
        tab = self.call(
            "POST", "/tabs", {"sessionKey": group, "url": "https://course.pku.edu.cn/"}
        )["tabId"]
        preserve = False
        try:
            selector = f'a[href*="{bb_id}"]'
            if not self.evaluate(tab, f"!!document.querySelector('{selector}')"):
                if self.evaluate(tab, "!!document.querySelector(\"a[href*='action=logout']\")"):
                    raise RuntimeError("Requested course link is not visible in the teaching site")
                preserve = True
                raise LoginRequired(
                    f"Teaching site login required; complete login in tab {tab}, then rerun"
                )
            self.click(tab, selector)
            self.click(tab, 'a[href*="tool_id=_1761_1"]')
            before_play = {t["tabId"] for t in self.tabs()}
            self.click(tab, '#listContainer_datatable tr:first-child a[href^="playVideo.action"]')
            play = None
            for _ in range(10):
                tabs = self.tabs()
                play = next(
                    (
                        t
                        for t in tabs
                        if "playVideo.action" in t.get("url", "")
                        and (t["tabId"] not in before_play or t["tabId"] == tab)
                    ),
                    None,
                )
                if play:
                    break
                time.sleep(0.5)
            if not play:
                raise RuntimeError("No playback tab opened; inspect course recording list")
            state = self.state()
            self.export(state["cookies"], cookies_path)
            platform = session(load_netscape_jar(cookies_path), None)
            # The browser click establishes the authorized playback entry. Re-fetch that
            # entry for a fresh one-time signed exchange if embedded login did not persist.
            page = platform.get(play["url"], timeout=30)
            page.raise_for_status()
            sources = re.findall(r'<iframe\b[^>]*\bsrc="([^"]+)"', page.text, re.I)
            source = next(
                (
                    s
                    for s in sources
                    if s.startswith("https://yjapise.pku.edu.cn/") and "login-with-sign" in s
                ),
                None,
            )
            if not source:
                raise RuntimeError("Playback page has no signed platform login frame")
            # Do not apply html.unescape: '&timestamp' can become '×tamp'.
            response = platform.get(source.replace("&amp;", "&"), timeout=30)
            response.raise_for_status()
            if not any(c.name == "_token" for c in platform.cookies):
                preserve = True
                raise LoginRequired(
                    f"Media platform login required; inspect playback tab {play['tabId']}"
                )
            media_id = re.search(r"course_id(?:%3D|=)(\d+)", source, re.I)
            if not media_id:
                raise RuntimeError("Signed entry has no media course ID")
            self.course_id = int(media_id.group(1))
            check = platform.get(
                API, params={"all": 1, "course_id": self.course_id, "with_sub_data": 1}, timeout=30
            )
            if check.status_code in (401, 403) or (
                check.status_code == 200 and check.json().get("code") != 0
            ):
                preserve = True
                raise LoginRequired(
                    f"Media API login required; inspect playback tab {play['tabId']}"
                )
            check.raise_for_status()
            fresh = [
                {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path,
                    "secure": c.secure,
                    "expires": c.expires or -1,
                }
                for c in platform.cookies
            ]
            # Persist fresh platform cookies in the same browser owner session as well.
            self.call("POST", f"/sessions/{self.user_id}/cookies", {"cookies": fresh})
            self.export(fresh, cookies_path)
            return cookies_path
        finally:
            if not preserve:
                for created in self.tabs():
                    if created["tabId"] not in original and created.get("listItemId") == group:
                        self.call("DELETE", f"/tabs/{created['tabId']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9377")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--course-bb-id", required=True, help="Blackboard ID, e.g. _101745_1")
    parser.add_argument("--cookies", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"_\d+_\d+", args.course_bb_id):
        parser.error("invalid Blackboard course ID")
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ.get("CAMOFOX_API_KEY")
    )
    if not key:
        parser.error("provide --api-key-file or existing CAMOFOX_API_KEY")
    try:
        browser = BrowserSession(args.base_url, key, args.user_id)
        browser.recover(args.course_bb_id, args.cookies)
    except LoginRequired as exc:
        raise SystemExit(str(exc)) from None
    except requests.RequestException, RuntimeError:
        raise SystemExit(
            "Session recovery failed; inspect existing browser and authentication"
        ) from None
    print(f"Validated media course {browser.course_id}; cookies saved privately to {args.cookies}")


if __name__ == "__main__":
    main()
