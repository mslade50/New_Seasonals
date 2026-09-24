"""Deterministic morning capture and frozen review queue; no news judgments or email."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from episodic_pivot import bulk_premarket as bulk
from episodic_pivot.morning_completion import NY, checkpoint, inspect_morning
from research_io import file_lock, write_json


def run_step(script, args, log, *, timeout=300):
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Morning preparation step failed; inspect retained local log")


def checkpoint_active(root, target, stage, paths, now_fn):
    # Share the send/claim lock for the check-and-write boundary. A pause or
    # terminal receipt arriving while network work runs must not be overwritten.
    with file_lock(root / "morning_sessions" / f"{target}.jsonl"):
        if inspect_morning(root, target, now=now_fn())["status"] != "RESUME":
            return False
        checkpoint(root, target, stage, paths, now=now_fn())
        return True


def prepare(root, *, now_fn=None, capture_fn=None, step_fn=None):
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    capture_fn = capture_fn or bulk.capture
    step_fn = step_fn or run_step
    target = now_fn().astimezone(NY).date().isoformat()
    state = inspect_morning(root, target, now=now_fn())
    if state["status"] != "RESUME":
        return state
    with file_lock(root / "morning_sessions" / f"{target}-prepare"):
        state = inspect_morning(root, target, now=now_fn())
        if state["status"] != "RESUME":
            return state
        progress = state["progress"]
        # Resume the same frozen queue, never replace it with a smaller new scan.
        if progress["stage"] == "INVALID_CHECKPOINT" or progress.get("changed_artifacts"):
            raise ValueError("Saved morning evidence requires inspection")
        if "queue" in progress["artifacts"]:
            return {**state, "preparation": "QUEUE_ALREADY_FROZEN"}
        attempt = root / "preparation" / f"{target}-{uuid.uuid4().hex[:12]}"
        attempt.mkdir(parents=True)
        if not checkpoint_active(root, target, "DISCOVERY", {}, now_fn):
            return inspect_morning(root, target, now=now_fn())
        try:
            evidence = capture_fn()
            bulk.validate_capture(evidence)
            if evidence["target_session_date"] != target:
                raise ValueError("Capture changed target session")
            raw_path, daily_path, queue_path = (attempt / name for name in ("discovery.json", "daily.json", "queue.json"))
            write_json(raw_path, evidence)
            step_fn("capture_ep_daily_yfinance.py", ["--snapshot", raw_path, "--output", daily_path, "--capture"], attempt / "daily.log")
            coverage = json.loads(daily_path.read_text(encoding="utf-8"))["coverage"]
            if coverage["broad_nomination_count"] and not coverage["daily_metrics_verified_count"]:
                raise ValueError("Daily provider outage; zero verified histories is not an empty screen")
            if inspect_morning(root, target, now=now_fn())["status"] != "RESUME":
                return inspect_morning(root, target, now=now_fn())
            step_fn("run_episodic_pivot_shadow.py", ["--snapshot", daily_path,
                    "--prepare-google-review", queue_path, "--run-research"], attempt / "queue.log")
            if not checkpoint_active(root, target, "RESEARCH", {"snapshot_0": daily_path, "queue": queue_path}, now_fn):
                return inspect_morning(root, target, now=now_fn())
            return {**inspect_morning(root, target, now=now_fn()), "preparation": "QUEUE_FROZEN"}
        except (Exception, SystemExit) as exc:
            # An unsuccessful attempt does not create a delivery receipt. The next
            # scheduled run can try again, preserving every failed attempt.
            write_json(attempt / "failure.json", {"session_date": target, "stage": "PREPARATION",
                       "error_type": type(exc).__name__, "recorded_at": now_fn().isoformat(), "retryable": True})
            checkpoint_active(root, target, "RETRY_PENDING", {}, now_fn)
            return {**inspect_morning(root, target, now=now_fn()), "preparation": "RETRY_PENDING",
                    "failure_note": str(attempt / "failure.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args(argv)
    root = (ROOT / "artifacts" / "episodic_pivot").resolve()
    if not args.capture:
        print(json.dumps({"status": "DRY_RUN", "network": False, "email": False}))
        return 0
    try:
        result = prepare(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "PREPARATION_UNAVAILABLE", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(result))
    return 2 if result.get("preparation") == "RETRY_PENDING" else 0


if __name__ == "__main__":
    raise SystemExit(main())
