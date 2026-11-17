"""Resumable PKU HLS downloads. Identity is (course_id, sub_id), never duration."""

from __future__ import annotations

import concurrent.futures as futures
import copy
import fcntl
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import m3u8
import requests


class DownloadError(RuntimeError):
    pass


class Throttled(DownloadError):
    pass


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def probe(path: Path, expected: float | None = None) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise DownloadError(f"Cannot read video: {path}")
    data = json.loads(result.stdout)
    kinds = {stream.get("codec_type") for stream in data.get("streams", [])}
    duration = float(data.get("format", {}).get("duration", 0))
    if not {"video", "audio"} <= kinds or duration <= 0:
        raise DownloadError(f"Missing audio/video or duration: {path}")
    if expected is not None and abs(duration - expected) > max(2, expected * 0.001):
        raise DownloadError(f"Duration mismatch: {path} ({duration:.2f}s vs {expected:.2f}s)")
    return {"duration": duration, "size_bytes": path.stat().st_size}


@dataclass
class AdaptiveLimit:
    current: int
    maximum: int
    adaptive: bool = True
    previous_rate: float | None = None
    probing: bool = False
    frozen: bool = False
    cooldown: int = 0

    def sample(self, rate: float, saturated: bool) -> None:
        if not self.adaptive or self.frozen or not saturated:
            return
        if self.cooldown:
            self.cooldown -= 1
            return
        if self.probing and self.previous_rate is not None:
            if rate < self.previous_rate * 1.10:
                self.current = max(1, self.current - 1)
                self.frozen = True
                return
        self.previous_rate = rate
        if self.current < self.maximum:
            self.current += 1
            self.probing = True

    def throttle(self) -> None:
        self.current = max(1, self.current // 2)
        self.previous_rate = None
        self.probing = False
        self.frozen = False
        self.cooldown = 2


class Downloader:
    def __init__(
        self,
        make_session,
        manifest: Path,
        *,
        workers=2,
        max_workers=8,
        adaptive=True,
        interval=15,
        keep_segments=False,
        notify=print,
    ):
        if not 1 <= workers <= max_workers or interval <= 0:
            raise ValueError("require 1 <= workers <= max_workers and interval > 0")
        self.make_session = make_session
        self.manifest = manifest.resolve()
        self.limit = AdaptiveLimit(workers, max_workers, adaptive)
        self.interval = interval
        self.keep_segments = keep_segments
        self.notify = notify
        self.lock = threading.Lock()
        self.local = threading.local()
        self.cancel = threading.Event()
        self.bytes_received = 0
        self.throttles = 0
        self.state = (
            json.loads(manifest.read_text())
            if manifest.exists()
            else {"version": 1, "recordings": {}}
        )
        if self.state.get("version") != 1 or not isinstance(self.state.get("recordings"), dict):
            raise DownloadError(
                "Unsupported manifest; use --import-inventory for legacy inventories"
            )

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = self.make_session()
        return self.local.session

    def fetch(self, url: str) -> bytes:
        for attempt in range(4):
            if self.cancel.is_set():
                raise DownloadError("Cancelled; completed segments retained")
            try:
                response = self.session().get(url, timeout=(15, 60))
                if response.status_code in (429, 503):
                    with self.lock:
                        self.throttles += 1
                    if attempt == 3:
                        raise Throttled("Server throttled requests after four attempts")
                    try:
                        delay = min(
                            30, max(1, float(response.headers.get("Retry-After", 2**attempt)))
                        )
                    except ValueError:
                        delay = 2**attempt
                    if self.cancel.wait(delay):
                        raise DownloadError("Cancelled")
                    continue
                if response.status_code >= 500 and attempt < 3:
                    self.cancel.wait(2**attempt)
                    continue
                if response.status_code != 200:
                    raise DownloadError(
                        f"HTTP {response.status_code}; check session, resource availability and access"
                    )
                body = response.content
                if not body:
                    raise DownloadError("Empty media response")
                with self.lock:
                    self.bytes_received += len(body)
                return body
            except requests.RequestException:
                if attempt == 3:
                    raise DownloadError("Network failed after four attempts") from None
                self.cancel.wait(2**attempt)
        raise DownloadError("Request failed")

    def playlist(self, url: str):
        # Follow master playlists without including credential-bearing URLs in logs.
        for _ in range(4):
            body = self.fetch(url).decode("utf-8-sig")
            if not body.lstrip().startswith("#EXTM3U"):
                raise DownloadError("Expected HLS playlist; session may require renewal")
            parsed = m3u8.loads(body, uri=url)
            if not parsed.is_variant:
                if not parsed.is_endlist or not parsed.segments:
                    raise DownloadError("Only completed, nonempty recordings are supported")
                if parsed.segment_map or any(s.byterange for s in parsed.segments):
                    raise DownloadError("fMP4 maps and byte-range playlists are not supported")
                if any(k and k.method not in ("AES-128", "NONE") for k in parsed.keys):
                    raise DownloadError("Unsupported HLS encryption method")
                return parsed
            choices = [p for p in parsed.playlists if not p.stream_info.audio]
            if not choices:
                raise DownloadError("Separate audio renditions are not supported")
            url = urljoin(url, max(choices, key=lambda p: p.stream_info.bandwidth).uri)
        raise DownloadError("Too many nested HLS playlists")

    @staticmethod
    def identity(recording):
        return f"{recording.course_id}:{recording.sub_id}"

    def store(self, key: str, entry: dict):
        with self.lock:
            self.state["recordings"][key] = entry
            atomic_json(self.manifest, self.state)

    def prepare(self, recording, out: Path, *, persist=True):
        key = self.identity(recording)
        playlist = self.playlist(recording.m3u8)
        expected = sum(s.duration for s in playlist.segments)
        warning = None
        if recording.duration_tick and abs(recording.duration_tick / 1e7 - expected) > 60:
            warning = f"Source duration differs: catalog {recording.duration_tick / 1e7:.0f}s, HLS {expected:.0f}s"
            self.notify(f"WARNING {key}: {warning}")
        previous = self.state["recordings"].get(key, {})
        target = (
            (self.manifest.parent / previous["path"]).resolve()
            if previous.get("path")
            else (out / f"{recording.course_id}_{recording.sub_id}" / "recording.mp4").resolve()
        )
        entry = {
            "course_id": recording.course_id,
            "sub_id": recording.sub_id,
            "title": recording.title,
            "begin": recording.begin,
            "path": os.path.relpath(target, self.manifest.parent),
            "playlist_duration": expected,
            "source_warning": warning,
            "status": "pending",
        }
        # A file is trusted only when its identity was explicitly recorded in the manifest.
        if target.exists():
            if not previous:
                raise DownloadError(f"Untracked output exists; import its identity first: {target}")
            try:
                entry.update(probe(target, expected))
            except DownloadError:
                raise DownloadError(
                    f"Existing file failed verification; move it aside to retry: {target}"
                ) from None
            entry["status"] = "complete"
            self.store(key, entry)
            self.notify(f"SKIP {key} {recording.title}")
            return None
        if persist:
            self.store(key, entry)
        return recording, playlist, target, entry

    def transfer(self, job):
        target = job[2]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".lock").open("a") as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DownloadError("Another task is downloading this recording") from None
            return self._transfer(job)

    def _transfer(self, job):
        recording, source_playlist, target, entry = job
        playlist = copy.deepcopy(source_playlist)
        key = self.identity(recording)
        cache = target.parent / ".hls-cache"
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        index = cache / "index.json"
        # A changed playlist invalidates cached segments, even for the same recording ID.
        signature = {
            "sequence": playlist.media_sequence,
            "segments": [
                {
                    "url": s.absolute_uri,
                    "duration": s.duration,
                    "key": s.key.absolute_uri if s.key and s.key.uri else None,
                    "iv": s.key.iv if s.key else None,
                    "method": s.key.method if s.key else None,
                }
                for s in playlist.segments
            ],
        }
        if index.exists() and json.loads(index.read_text()).get("source") != signature:
            shutil.rmtree(cache)
            cache.mkdir(mode=0o700)
        progress = (
            json.loads(index.read_text()) if index.exists() else {"source": signature, "sizes": {}}
        )
        key_paths = {}
        self.notify(f"START {key} {recording.title}")
        try:
            for i, segment in enumerate(playlist.segments):
                path = cache / f"segment_{i:06d}.ts"
                saved_size = progress["sizes"].get(str(i))
                if not path.exists() or not saved_size or path.stat().st_size != saved_size:
                    body = self.fetch(segment.absolute_uri)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(body)
                    os.replace(temporary, path)
                    progress["sizes"][str(i)] = len(body)
                    atomic_json(index, progress)
                segment.uri = path.name
                if segment.key and segment.key.method == "AES-128":
                    uri = segment.key.absolute_uri
                    if uri not in key_paths:
                        data = self.fetch(uri)
                        if len(data) != 16:
                            raise DownloadError("Invalid AES-128 key")
                        keyfile = cache / f"key_{len(key_paths)}.bin"
                        keyfile.touch(mode=0o600, exist_ok=True)
                        keyfile.write_bytes(data)
                        key_paths[uri] = keyfile.name
            # Mutate shared key objects only after all absolute key URIs have been resolved.
            for key_object in playlist.keys:
                if key_object and key_object.method == "AES-128":
                    key_object.uri = key_paths[key_object.absolute_uri]
            local_playlist = cache / "local.m3u8"
            local_playlist.write_text(playlist.dumps())
            partial = target.with_name("recording.partial.mp4")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-protocol_whitelist",
                    "file,crypto,data",
                    "-allowed_extensions",
                    "ALL",
                    "-i",
                    str(local_playlist),
                    "-c",
                    "copy",
                    "-bsf:a",
                    "aac_adtstoasc",
                    str(partial),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise DownloadError("Local HLS remux failed; segments retained for retry")
            entry.update(probe(partial, entry["playlist_duration"]))
            if target.exists():
                raise DownloadError("Output appeared during download; refusing to overwrite")
            os.replace(partial, target)
            entry["status"] = "complete"
            self.store(key, entry)
            self.notify(f"COMPLETE {key} {entry['size_bytes'] / 1e6:.1f} MB")
            if not self.keep_segments:
                shutil.rmtree(cache)
            return True
        except Exception:
            entry["status"] = "failed"
            self.store(key, entry)
            raise
        finally:
            # Keep encrypted segments for resume, but never retain decryption keys.
            for keyfile in cache.glob("key_*.bin"):
                keyfile.unlink(missing_ok=True)
            (cache / "local.m3u8").unlink(missing_ok=True)

    def run(self, recordings, out: Path, *, dry_run=False):
        if len({self.identity(r) for r in recordings}) != len(recordings):
            raise DownloadError("Duplicate recording identity in download plan")
        jobs = []
        for recording in recordings:
            job = self.prepare(recording, out, persist=not dry_run)
            if job:
                jobs.append(job)
        self.notify(f"PLAN {len(jobs)} missing / {len(recordings)} recordings")
        if dry_run:
            return {"missing": len(jobs), "failed": 0}
        active = {}
        failed = []
        pending = iter(jobs)
        exhausted = False
        last_time, last_bytes, last_throttles = (
            time.monotonic(),
            self.bytes_received,
            self.throttles,
        )
        pool = futures.ThreadPoolExecutor(max_workers=self.limit.maximum)
        try:
            while active or not exhausted:
                while not exhausted and len(active) < self.limit.current:
                    job = next(pending, None)
                    if job is None:
                        exhausted = True
                        break
                    active[pool.submit(self.transfer, job)] = job
                if not active:
                    break
                done, _ = futures.wait(
                    active, timeout=min(1, self.interval), return_when=futures.FIRST_COMPLETED
                )
                for future in done:
                    job = active.pop(future)
                    try:
                        future.result()
                    except (DownloadError, OSError, ValueError) as exc:
                        failed.append(job)
                        self.notify(f"FAILED {self.identity(job[0])}: {exc}")
                now = time.monotonic()
                if now - last_time >= self.interval:
                    with self.lock:
                        received, throttles = self.bytes_received, self.throttles
                    rate = (received - last_bytes) / (now - last_time)
                    before = self.limit.current
                    if throttles > last_throttles:
                        self.limit.throttle()
                    else:
                        self.limit.sample(rate, len(active) >= self.limit.current and not exhausted)
                    self.notify(
                        f"PROGRESS complete={sum(e['status'] == 'complete' for e in self.state['recordings'].values())} "
                        f"active={len(active)} limit={self.limit.current} speed={rate / 1e6:.2f} MB/s"
                    )
                    if before != self.limit.current:
                        self.notify(
                            f"CONCURRENCY {before} -> {self.limit.current}; active transfers continue"
                        )
                    last_time, last_bytes, last_throttles = now, received, throttles
            # A bounded serial retry resumes cached segments, including after throttling.
            errors = []
            for job in failed:
                try:
                    self.notify(f"RETRY serial {self.identity(job[0])}")
                    self.transfer(job)
                except (DownloadError, OSError, ValueError) as exc:
                    errors.append(self.identity(job[0]))
                    self.notify(f"FAILED {errors[-1]}: {exc}")
            return {"missing": len(jobs), "failed": len(errors)}
        except BaseException:
            self.cancel.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
