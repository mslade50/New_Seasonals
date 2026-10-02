"""Idea Check poller: picks up queued trade ideas from R2, runs the /idea-check
skill headlessly, and uploads the verdict. Local half only; see
docs/claude_ref/idea_check.md."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "scratch" / "idea_checks"
STATE = WORK / "_state"
QUEUE_KEY = "idea_check/queue.json"
RESULT_KEY = "idea_check/results/{id}.json"
ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$")
MAX_TEXT = 2000
MAX_AGE = timedelta(hours=24)
DAILY_CAP = 20
AGENT_TIMEOUT_S = 1200
LOCK_STALE = timedelta(minutes=30)
VERDICTS = {"KILL", "SURVIVES", "NEAR-MISS", "NEEDS-INFO"}

Download = Callable[[str, str], bool]
Upload = Callable[[str, str], bool]
Runner = Callable[[str], int]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def acquire_lock(path: Path, now: datetime) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    held = _read_json(path, None)
    if path.exists():
        ts = _parse_ts(held.get("at")) if isinstance(held, dict) else None
        if ts is not None and now - ts < LOCK_STALE:
            return False
        try:
            path.unlink()
        except OSError:
            return False
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "at": _iso(now)}, fh)
    return True


def refresh_lock(path: Path, now: datetime) -> None:
    _write_json(path, {"pid": os.getpid(), "at": _iso(now)})


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def validate_verdict(obj: object) -> Optional[str]:
    if not isinstance(obj, dict):
        return "verdict.json is not an object"
    if obj.get("verdict") not in VERDICTS:
        return "verdict not in allowed set"
    head = obj.get("headline")
    if not isinstance(head, str) or not head.strip():
        return "headline missing or empty"
    for key in ("numbers", "tweaks"):
        val = obj.get(key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            return f"{key} must be a list of strings"
    if not isinstance(obj.get("body_md"), str):
        return "body_md must be a string"
    return None


def _result(rid: str, status: str, started: datetime, now: datetime,
            verdict: Optional[dict] = None, error: Optional[str] = None) -> dict:
    v = verdict or {}
    return {
        "id": rid,
        "status": status,
        "verdict": v.get("verdict"),
        "headline": v.get("headline", ""),
        "numbers": v.get("numbers", []),
        "tweaks": v.get("tweaks", []),
        "body_md": v.get("body_md", ""),
        "started_at": _iso(started),
        "finished_at": None if status == "running" else _iso(now),
        "error": error,
    }


def default_runner(rid: str) -> int:
    model = os.environ.get("IDEA_CHECK_MODEL", "opus")
    effort = os.environ.get("IDEA_CHECK_EFFORT", "high")
    exe = Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "claude.exe"
    cmd = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "scripts" / "invoke_idea_check_agent.ps1"),
        "-ClaudeExe", str(exe) if exe.exists() else "claude",
        "-Model", model, "-Effort", effort, "-RequestId", rid,
        "-TimeoutSeconds", str(AGENT_TIMEOUT_S),
    ]
    try:
        return subprocess.run(cmd, cwd=str(ROOT), timeout=AGENT_TIMEOUT_S + 90,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode
    except subprocess.TimeoutExpired:
        return 124


def _upload(upload: Upload, obj: dict, rid: str, tries: int = 1) -> bool:
    local = WORK / rid / "result.json"
    _write_json(local, obj)
    for _ in range(tries):
        if upload(str(local), RESULT_KEY.format(id=rid)):
            return True
    return False


def process_request(req: dict, upload: Upload, runner: Runner,
                    now_fn: Callable[[], datetime]) -> str:
    rid = req["id"]
    started = now_fn()
    folder = WORK / rid
    verdict_path = folder / "verdict.json"
    _write_json(folder / "request.json", req)
    if verdict_path.exists():
        verdict_path.unlink()
    _upload(upload, _result(rid, "running", started, started), rid)
    try:
        rc = runner(rid)
    except Exception as exc:  # noqa: BLE001
        rc, err = -1, f"agent runner failed: {type(exc).__name__}"
    else:
        err = None
    verdict: Optional[dict] = None
    if err is None:
        if rc == 124:
            err = f"timed out after {AGENT_TIMEOUT_S} seconds"
        elif rc != 0:
            err = f"agent exited with code {rc}"
        elif not verdict_path.exists():
            err = "agent produced no verdict.json"
        else:
            obj = _read_json(verdict_path, None)
            bad = validate_verdict(obj)
            if bad:
                err = f"invalid verdict: {bad}"
            else:
                verdict = obj
    finished = now_fn()
    if err is None:
        final = _result(rid, "done", started, finished, verdict=verdict)
    else:
        final = _result(rid, "error", started, finished, error=err)
    if not _upload(upload, final, rid, tries=3):
        print(f"[idea_check] final upload FAILED for {rid}", file=sys.stderr)
    return final["status"]


def run(download: Download, upload: Upload, runner: Runner, *,
        dry_run: bool = False,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    lock = STATE / "poller.lock"
    if not acquire_lock(lock, now_fn()):
        return 0
    try:
        queue_path = STATE / "queue.json"
        if queue_path.exists():
            queue_path.unlink()
        if not download(QUEUE_KEY, str(queue_path)) or not queue_path.exists():
            print("[idea_check] no queue")
            return 0
        queue = _read_json(queue_path, {})
        reqs = queue.get("requests", []) if isinstance(queue, dict) else []
        if not isinstance(reqs, list):
            return 0
        proc_path = STATE / "processed.json"
        processed = _read_json(proc_path, {})
        if not isinstance(processed, dict):
            processed = {}
        today = now_fn().astimezone().strftime("%Y-%m-%d")
        used = sum(1 for v in processed.values()
                   if isinstance(v, dict) and v.get("day") == today)
        todo = []
        for req in reqs:
            if not isinstance(req, dict):
                continue
            rid, text = req.get("id"), req.get("text")
            if not isinstance(rid, str) or not ID_RE.match(rid) or rid in processed:
                continue
            if not isinstance(text, str) or not (1 <= len(text) <= MAX_TEXT):
                continue
            sub = _parse_ts(req.get("submitted_at"))
            if sub is None or now_fn() - sub > MAX_AGE:
                continue
            todo.append(req)
        todo.sort(key=lambda r: r["submitted_at"])
        for req in todo:
            if used >= DAILY_CAP:
                print(f"[idea_check] daily cap of {DAILY_CAP} reached")
                break
            if dry_run:
                print(f"[idea_check] dry-run: would review {req['id']}")
                continue
            refresh_lock(lock, now_fn())
            status = process_request(req, upload, runner, now_fn)
            processed[req["id"]] = {"at": _iso(now_fn()), "day": today,
                                    "status": status}
            _write_json(proc_path, processed)
            used += 1
            print(f"[idea_check] {req['id']}: {status}")
        return 0
    finally:
        release_lock(lock)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass (default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no upload, no agent")
    args = ap.parse_args(argv)
    import cache_io
    return run(cache_io.download_to_local, cache_io.upload_from_local,
               default_runner, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
