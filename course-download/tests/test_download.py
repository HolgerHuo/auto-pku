import json
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_engine import AdaptiveLimit, Downloader, DownloadError, probe
from pku_video import import_inventory


@pytest.fixture
def media(tmp_path):
    ts = tmp_path / "sample.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-f",
            "mpegts",
            str(ts),
        ],
        check=True,
    )
    plain = ts.read_bytes()
    key = b"0123456789abcdef"
    files = {"/key": key}
    for i in range(3):
        padder = padding.PKCS7(128).padder()
        padded = padder.update(plain) + padder.finalize()
        encryptor = Cipher(algorithms.AES(key), modes.CBC((42 + i).to_bytes(16, "big"))).encryptor()
        files[f"/s{i}.ts"] = encryptor.update(padded) + encryptor.finalize()
    # Repeat a single segment: validate decryption and nonzero sequence-derived IVs.
    files["/playlist.m3u8"] = (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:42\n"
        '#EXT-X-KEY:METHOD=AES-128,URI="key"\n'
        + "".join(f"#EXTINF:1.0,\ns{i}.ts\n" for i in range(3))
        + "#EXT-X-ENDLIST\n"
    ).encode()
    counts = Counter()
    failures = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            counts[self.path] += 1
            if failures.get(self.path):
                code = failures[self.path].pop(0)
                if code != 200:
                    self.send_response(code)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
            body = files.get(self.path)
            self.send_response(200 if body else 404)
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield SimpleNamespace(url=url, files=files, counts=counts, failures=failures, root=tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def recording(media, sub=1):
    return SimpleNamespace(
        course_id=10,
        sub_id=sub,
        title="2026-11-03 test",
        begin=0,
        duration_tick=7000 * 10_000_000,
        m3u8=media.url + "/playlist.m3u8",
    )


def engine(media, **kwargs):
    return Downloader(
        requests.Session, media.root / "manifest.json", notify=lambda _: None, **kwargs
    )


def test_encrypted_download_skip_and_source_warning(media):
    rec = recording(media)
    d = engine(media)
    result = d.run([rec], media.root / "out")
    assert result == {"missing": 1, "failed": 0}
    entry = d.state["recordings"]["10:1"]
    assert entry["source_warning"]
    assert probe(d.manifest.parent / entry["path"], 3)["duration"] > 0
    before = media.counts.copy()
    assert engine(media).run([rec], media.root / "out")["missing"] == 0
    assert media.counts["/s0.ts"] == before["/s0.ts"]
    assert not list(media.root.rglob("key_*.bin"))
    assert not list(media.root.rglob("recording.partial.mp4"))


def test_failed_download_resumes_completed_segments(media):
    d = engine(media)
    job = d.prepare(recording(media), media.root / "out")
    media.failures["/s1.ts"] = [404]
    with pytest.raises(DownloadError):
        d.transfer(job)
    assert media.counts["/s0.ts"] == 1
    assert not list(media.root.rglob("key_*.bin"))
    assert engine(media).run([recording(media)], media.root / "out")["failed"] == 0
    assert media.counts["/s0.ts"] == 1
    assert media.counts["/s1.ts"] == 2


def test_remux_retry_does_not_mutate_original_playlist(media, monkeypatch):
    d = engine(media)
    job = d.prepare(recording(media), media.root / "out")
    original = subprocess.run

    def fail_remux(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 1, "", "fixture error")
        return original(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_remux)
    with pytest.raises(DownloadError):
        d.transfer(job)
    monkeypatch.setattr(subprocess, "run", original)
    assert d.transfer(job)
    assert media.counts["/s0.ts"] == 1


def test_throttle_retries_and_policy(media):
    d = engine(media)
    media.failures["/key"] = [429, 200]
    assert d.fetch(media.url + "/key") == media.files["/key"]
    assert d.throttles == 1
    limit = AdaptiveLimit(2, 8)
    limit.sample(100, True)
    assert limit.current == 3
    limit.sample(120, True)
    assert limit.current == 4
    limit.sample(125, True)
    assert limit.current == 3 and limit.frozen
    limit.throttle()
    assert limit.current == 1


def test_dry_run_does_not_write_or_guess_identity(media):
    d = engine(media)
    out = media.root / "out"
    assert d.run([recording(media)], out, dry_run=True)["missing"] == 1
    assert not d.manifest.exists()
    assert not out.exists()
    assert not media.counts["/s0.ts"]
    # Equal duration is never enough to match another sub_id.
    d.run([recording(media)], out)
    assert engine(media).run([recording(media, 2)], out, dry_run=True)["missing"] == 1


def test_playlist_change_invalidates_cache(media):
    d = engine(media)
    job = d.prepare(recording(media), media.root / "out")
    media.failures["/s1.ts"] = [404]
    with pytest.raises(DownloadError):
        d.transfer(job)
    media.files["/playlist.m3u8"] = media.files["/playlist.m3u8"].replace(
        b"#EXTINF:1.0", b"#EXTINF:1.1"
    )
    assert engine(media).run([recording(media)], media.root / "out")["failed"] == 0
    assert media.counts["/s0.ts"] == 2


def test_reject_corrupt_existing_and_duplicate_import(media):
    d = engine(media)
    rec = recording(media)
    d.run([rec], media.root / "out")
    path = d.manifest.parent / d.state["recordings"]["10:1"]["path"]
    inventory = media.root / "legacy.json"
    inventory.write_text(
        json.dumps([{"course": "test", "sub_id": 1, "date": "2026-11-03", "path": str(path)}] * 2)
    )
    with pytest.raises(DownloadError, match="Duplicate"):
        import_inventory(d, inventory, media.root, 10, "test", [rec])
    path.write_bytes(b"broken")
    with pytest.raises(DownloadError, match="failed verification"):
        engine(media).run([rec], media.root / "out")


def test_scheduler_bounds_and_no_duplicate_transfers(media):
    import time

    d = engine(media, workers=1, max_workers=3, interval=0.01)
    active = 0
    peak = 0
    seen = Counter()

    def transfer(job):
        nonlocal active, peak
        with d.lock:
            active += 1
            peak = max(peak, active)
            seen[job[0].sub_id] += 1
        try:
            for _ in range(5):
                with d.lock:
                    d.bytes_received += 10000
                time.sleep(0.01)
        finally:
            with d.lock:
                active -= 1
        return True

    d.transfer = transfer
    result = d.run([recording(media, i) for i in range(12)], media.root / "out")
    assert result["failed"] == 0
    assert 1 < peak <= 3
    assert seen == Counter({i: 1 for i in range(12)})


def test_recording_lock_preserves_other_task(media):
    import fcntl

    d = engine(media)
    job = d.prepare(recording(media), media.root / "out")
    target = job[2]
    target.parent.mkdir(parents=True)
    with target.with_suffix(".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DownloadError, match="Another task"):
            d.transfer(job)
    assert not media.counts["/s0.ts"]


def test_key_rotation_and_explicit_iv(media):
    plain = (media.root / "sample.ts").read_bytes()
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:2", "#EXT-X-MEDIA-SEQUENCE:42"]
    for i in range(3):
        key = bytes([10 + i]) * 16
        iv = (100 + i).to_bytes(16, "big")
        padder = padding.PKCS7(128).padder()
        padded = padder.update(plain) + padder.finalize()
        encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
        media.files[f"/rotated{i}.ts"] = encryptor.update(padded) + encryptor.finalize()
        media.files[f"/key{i}"] = key
        lines.extend(
            [
                f'#EXT-X-KEY:METHOD=AES-128,URI="key{i}",IV=0x{iv.hex()}',
                "#EXTINF:1.0,",
                f"rotated{i}.ts",
            ]
        )
    media.files["/playlist.m3u8"] = ("\n".join(lines) + "\n#EXT-X-ENDLIST\n").encode()
    assert engine(media).run([recording(media)], media.root / "out")["failed"] == 0
    assert all(media.counts[f"/key{i}"] == 1 for i in range(3))
    assert not list(media.root.rglob("key_*.bin"))
