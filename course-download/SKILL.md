---
name: course-download
description: Download and synchronize authorized PKU teaching-site classroom recordings, reusing an existing Camofox login and skipping recordings already verified locally.
---

# PKU course recordings

Use `pku_video.py` for discovery, identity-based synchronization, resumable HLS
transfers and verification. Run commands with `uv run --project <skill-directory>`;
the project locks its dependencies. `ffmpeg` and `ffprobe` must be available.
Resolve the scripts relative to this file; output paths belong to the user's workspace.

## Session and navigation

Connect to the user's already running Camofox instance. Never start or stop the
browser server. Read its authenticated `/openapi.json` before using its API.
Use one persistent `userId` on **every** request, including `GET /tabs`; omitting it
can misleadingly return an empty list. Credentials belong in private ignored files
or an existing secret environment variable, never in skill files or logs.
When importing browser session cookies, preserve `expires: -1`; `0` means expired
in the browser import API. Netscape cookie files use `0` for session cookies.

Enter `course.pku.edu.cn` normally and click the course, **课堂实录**, then **观看**.
Watch for a new playback tab. Blackboard login and the media-platform login are
separate sessions; a successful teaching-site login does not prove media access.
Do not navigate directly to a guessed player URL or reuse a consumed signed link.

`browser_session.py` implements this click flow and the signed media cookie exchange:

```bash
uv run --project course-download course-download/browser_session.py \
  --user-id <persistent-owner> --course-bb-id <Blackboard-ID> \
  --api-key-file <private-key-file> --cookies course-download/tmp/media-cookies.txt
```

The helper preserves a tab when human login is required; let the user finish there,
then rerun. It closes only tabs it created on successful recovery. If a CAPTCHA
appears, request human intervention. Do not import old exports over a fresh login.
The helper re-fetches the clicked playback entry for a fresh signed exchange if
embedded login failed. Preserve query strings: generic `html.unescape` corrupts
`&timestamp` into `×tamp`. Cookie exports are restricted to PKU domains and mode 0600.

## Scope, identities and existing files

Resolve the requested course and term, and keep downloads within that scope.
Blackboard IDs (`_…_1`) differ from numeric media course IDs. The clicked playback
entry or the platform catalog supplies the latter. `--course <name> --find` lists
matching terms; ordinary name lookup selects the latest term, so use an explicit
numeric ID when the user specifies a term.

A recording is identified by **(course_id, sub_id)**. Never infer identity from a
`lecN` directory, similar duration, file size or lecture order. New recordings use
`OUT/<course_id>_<sub_id>/recording.mp4`; `OUT/downloads.json` tracks their identities.
Existing files require an explicit trusted mapping. For legacy inventories shaped
like `[{"course":"slug","date":"YYYY-MM-DD","sub_id":123,"path":"…"}]`:

```bash
uv run --project course-download course-download/pku_video.py \
  --course <media-ID> --cookies <cookies-file> --out <course-directory> \
  --import-inventory <inventory.json> --inventory-course <slug> \
  --inventory-root <root-for-inventory-paths>
```

Import verifies membership, date, unique identity/path, audio/video tracks and HLS
playlist duration. It cannot prove that a legacy mapping is semantically correct;
establish that mapping from provenance or content, not duration matching.

## Synchronize and monitor

```bash
# Verify tracked outputs and report missing recordings without downloading media.
uv run --project course-download course-download/pku_video.py \
  --course <media-ID> --cookies <cookies-file> --out <course-directory> --all --dry-run

# Download missing recordings; resume completed encrypted segments after failure.
uv run --project course-download course-download/pku_video.py \
  --course <media-ID> --cookies <cookies-file> --out <course-directory> --all \
  --workers 2 --max-workers 8
```

Use `--manifest <shared-file>` across courses when they should share one adaptive
queue. The CLI currently discovers one course per invocation; to run multiple courses
in one queue, construct their `Recording` objects with `list_recordings` and call
`Downloader.run(recordings, output_root)` once. Separate CLI processes have separate
concurrency budgets; cap their combined load deliberately.

The scheduler increases concurrent recordings when measured transfer bandwidth
improves, stops increasing after a gain below 10%, and reduces admission after
429/503 responses. Running transfers continue; do not restart them to change
concurrency. `--fixed-workers` disables bandwidth-based increases, while throttling
still reduces admission. Concurrency cannot exceed the number of missing recordings.
Progress reports actual received bytes, active tasks and completion count.

After network failure, rerun the same command and manifest. Completed encrypted
segments are reused; the partial MP4 itself is regenerated locally. Changed playlists
invalidate cached segments. Files become final only after ffprobe validation.
No automatic overwrite of untracked or invalid existing recordings is allowed.

## Exceptions and completion

Compare actual HLS duration with catalog duration. Preserve and report a source
warning if the recording is shorter; a complete 12-minute source is not a full
117-minute classroom session. Track download completeness and source completeness
separately. HTTP 403/404 alone does not establish campus-network blocking: check
session renewal, signed exchange and resource availability before diagnosing access.

The engine supports completed MPEG-TS HLS, unencrypted or AES-128, including key
rotation and sequence-derived IVs. It rejects unsupported encryption, byte ranges,
fMP4 maps and separate audio renditions explicitly rather than silently losing data.

Report downloaded/skipped/failed counts, output locations and source warnings.
Keep useful manifests and failed segment caches for resume; remove temporary cookie
exports and key files when no longer needed. Do not print process command lines or
raw URLs that could expose credentials. The engine remuxes local files so ffmpeg
process arguments contain no remote Cookie headers.
