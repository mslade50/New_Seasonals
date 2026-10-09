"""Read-only critical-window monitor for an OpenBreakout live launch.

This process never imports the broker adapter and never connects to IBKR.  A
launcher can wait on it so Task Scheduler does not report success after only
the initial connection check while the real session later dies before arming.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
TERMINAL = {"FAILED", "HALTED", "HALTED_PREFLIGHT", "STOPPED", "STOPPED_BY_REQUEST"}
HEALTHY_AFTER_OPEN = {"ARMED_LIVE", "RUNNING_LIVE"}


def read_runtime(path: Path) -> dict:
    """Read current committed metadata, including an active writer's WAL."""
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=1.) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA busy_timeout=1000")
        return {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM meta")}


def evaluate_runtime(meta: dict, now: datetime, opening: datetime, stale_seconds: float) -> tuple[str, str]:
    """Return (WAIT|OK|FAIL, reason) for one immutable runtime snapshot."""
    phase = str(meta.get("phase") or "UNKNOWN")
    if (meta.get("finished_at") or phase in TERMINAL or phase.startswith("HALTED")
            or phase.startswith("SESSION_COMPLETE")):
        return "FAIL", f"live runtime ended before launch verification: phase={phase} error={meta.get('last_error')}"
    heartbeat = meta.get("heartbeat") or {}
    stamp = heartbeat.get("at") if isinstance(heartbeat, dict) else None
    if not stamp:
        return "WAIT", f"no heartbeat yet (phase={phase})"
    try:
        heartbeat_at=datetime.fromisoformat(stamp).astimezone(NY)
        age = (now - heartbeat_at).total_seconds()
    except (TypeError, ValueError):
        return "FAIL", f"invalid heartbeat timestamp: {stamp!r}"
    if age < -5.:
        return "FAIL", f"heartbeat timestamp is materially in the future: {stamp!r}"
    if age > stale_seconds:
        return "FAIL", f"stale live heartbeat: age={age:.1f}s phase={phase}"
    if now >= opening:
        if heartbeat_at < opening:
            return "WAIT", f"waiting for first post-open heartbeat (last={heartbeat_at.isoformat()} phase={phase})"
        if phase not in HEALTHY_AFTER_OPEN:
            return "FAIL", f"not armed/running after opening: phase={phase} error={meta.get('last_error')}"
        if heartbeat.get("connected") is not True or heartbeat.get("healthy") is not True:
            return "FAIL", f"unhealthy after opening: connected={heartbeat.get('connected')} healthy={heartbeat.get('healthy')}"
        if heartbeat.get("orders_open") is not True:
            return "FAIL", "order gate is not open after opening"
        return "OK", f"verified phase={phase} connected=true healthy=true orders_open=true"
    return "WAIT", f"healthy heartbeat before opening (phase={phase})"


def live_db(runs: Path, session: str) -> Path | None:
    pattern=re.compile(rf"^{re.escape(session)}-live(?:-(\d+))?$")
    candidates=[]
    for path in runs.glob(f"{session}-live*/runtime.sqlite"):
        match=pattern.fullmatch(path.parent.name)
        if match:candidates.append((int(match.group(1) or 1),path))
    candidates.sort(key=lambda item:item[0])
    candidates=[path for _,path in candidates]
    return candidates[-1] if candidates else None


def monitor(runs: Path, session: str, opening: datetime, poll_seconds: float,
            stale_seconds: float, timeout_seconds: float, clock=lambda: datetime.now(NY), sleep=time.sleep) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_seconds
    last = "live runtime.sqlite not found"
    while time.monotonic() < deadline:
        db = live_db(runs, session)
        if db:
            try:
                state, last = evaluate_runtime(read_runtime(db), clock(), opening, stale_seconds)
            except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
                state, last = "WAIT", f"runtime read not ready: {type(exc).__name__}: {exc}"
            if state == "OK":return 0, last
            if state == "FAIL":return 2, last
        sleep(poll_seconds)
    return 3, f"critical-window verification timed out: {last}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--opening", default="09:30:00")
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--stale-seconds", type=float, default=30.)
    parser.add_argument("--timeout-seconds", type=float, default=5400.)
    args = parser.parse_args(argv)
    day = datetime.strptime(args.session, "%Y-%m-%d").date()
    at = datetime.strptime(args.opening, "%H:%M:%S").time()
    opening = datetime.combine(day, at, NY)
    code, reason = monitor(args.runs, args.session, opening, args.poll_seconds,
                           args.stale_seconds, args.timeout_seconds)
    print(f"OPEN_BREAKOUT_LAUNCH_MONITOR {'OK' if code == 0 else 'FAILED'}: {reason}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
