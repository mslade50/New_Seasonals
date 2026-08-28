"""Deterministic health battery for the New_Seasonals pipeline.

Run by the /repo-health-check skill (and fine to run by hand):

    python scripts/repo_health_check.py [--skip-tests] [--skip-automation]

Prints one OK / WARN / FAIL line per check and a summary. Exit 1 when any
check FAILed, else 0. The skill layer investigates failures; this script only
detects them, so every check must be cheap, offline-safe and side-effect free
(the single exception: it maintains its own tripwire state file).

Checks:
  1. Automation       - latest verified R2 supervisor receipt + age for each
                        weekday-critical local-primary / GitHub-backup job
  2. Local data       - master_prices / rd2_fragility / cboe_putcall recency,
                        stray partial-write temp files in data/
  3. Fragility PIT    - tripwire: frozen rows of rd2_fragility.parquet must
                        never change between runs (rewrite = the drifted
                        recompute vintage reaching live sizing)
  4. Journals         - pitch/context/posts JSONL parse cleanly
  5. Delivery         - check_pitch_delivered / check_context_delivered for
                        the most recent expected run date
  6. Trigger logs     - pinned Task Scheduler runtime log recency
  7. Guard tests      - pytest --collect-only: collection errors FAIL, guard
                        files contributing zero tests WARN
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cache_io  # noqa: E402

STATE_PATH = ROOT / "data" / "health_check_state.json"
_DEFAULT_AUTOMATION_RUNTIME = (
    ROOT if ROOT.name == "New_Seasonals-automation-runtime"
    else ROOT.parent / "New_Seasonals-automation-runtime"
)
AUTOMATION_STATE_ROOT = Path(os.environ.get(
    "NEW_SEASONALS_AUTOMATION_STATE_ROOT",
    str(_DEFAULT_AUTOMATION_RUNTIME / "artifacts" / "automation"),
))
AUTOMATION_LOG_DIR = AUTOMATION_STATE_ROOT / "logs"
AUTOMATION_RECEIPT_SCHEMA = "automation-receipt.v1"

# Supervisor job id -> max business days the latest success may be old.
CRITICAL_AUTOMATION_JOBS: dict[str, int] = {
    "cboe_am": 1,
    "master_prices_am": 1,
    "risk_am": 1,
    "event_sleeve_am": 1,
    "scan_am": 1,
    "private_site_am": 1,
    "shared_site_am": 1,
    "discretionary_focus": 1,
    "execution_report": 2,
    "master_prices_pm": 1,
    "risk_pm": 1,
    "verify_fills": 2,
    "earnings_and_grades": 2,
    "portfolio_report": 1,
    "cboe_pm": 2,
    "trend_sleeve": 2,
    "intraday_prices": 3,
    "scan_pm": 1,
    "macro_releases": 2,
    "private_site_pm": 1,
    "shared_site_pm": 1,
    "indicator_cache": 8,
    "weekly_rundown": 8,
}

LOCAL_PIPELINE_MAX_BD: dict[str, int] = {
    "premarket": 2,
    "discretionary": 2,
    "execution": 2,
    "postclose": 2,
    "indicator": 8,
    "weekly-rundown": 8,
}

RESULTS: list[tuple[str, str, str]] = []  # (tier, check, detail)


def report(tier: str, check: str, detail: str) -> None:
    RESULTS.append((tier, check, detail))
    print(f"[{tier}] {check}: {detail}")


def bdays_behind(d: dt.date, today: dt.date) -> int:
    return int(np.busday_count(np.datetime64(d), np.datetime64(today)))


def prev_weekday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


# ---------------------------------------------------------------- 1. automation receipts
def _automation_receipt(job_id: str, run_date: dt.date) -> dict | None:
    key = f"automation/receipts/v1/{run_date.isoformat()}/{job_id}/latest.json"
    local = (AUTOMATION_STATE_ROOT / "health-receipts" /
             run_date.isoformat() / f"{job_id}.json")
    if not cache_io.download_to_local(key, str(local)):
        return None
    try:
        receipt = json.loads(local.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (receipt.get("schema_version") != AUTOMATION_RECEIPT_SCHEMA
            or receipt.get("job_id") != job_id
            or receipt.get("run_date_et") != run_date.isoformat()):
        return None
    return receipt


def _latest_receipt(job_id: str, today: dt.date, fetch) -> dict | None:
    # Ten calendar days covers the longest critical-job allowance plus a
    # weekend while keeping R2 reads bounded when a job has never run.
    for days_back in range(11):
        candidate = today - dt.timedelta(days=days_back)
        receipt = fetch(job_id, candidate)
        if receipt is not None:
            return receipt
    return None


def check_gha(fetch=None, today: dt.date | None = None) -> None:
    """Check the cross-runtime receipt contract (legacy name kept for CLI/API)."""
    today = today or dt.date.today()
    fetch = fetch or _automation_receipt
    for job_id, max_bd in CRITICAL_AUTOMATION_JOBS.items():
        receipt = _latest_receipt(job_id, today, fetch)
        check = f"automation:{job_id}"
        if receipt is None:
            report("FAIL", check, "no valid R2 receipt found in the last 10 days")
            continue
        run_date = dt.date.fromisoformat(receipt["run_date_et"])
        age_bd = bdays_behind(run_date, today)
        status = receipt.get("status")
        source = receipt.get("source", "unknown")
        updated = receipt.get("updated_at_utc", "unknown time")
        if status == "failure":
            report("FAIL", check,
                   f"latest receipt FAILED via {source} ({updated})")
        elif status == "indeterminate":
            detail = receipt.get("detail") or "external side effect could not be confirmed"
            report(
                "FAIL",
                check,
                f"latest receipt INDETERMINATE via {source}; manual resolution required: {detail}",
            )
        elif status == "running":
            lease_raw = receipt.get("lease_expires_at_utc")
            expired = False
            if lease_raw:
                try:
                    lease = dt.datetime.fromisoformat(str(lease_raw).replace("Z", "+00:00"))
                    if lease.tzinfo is None:
                        lease = lease.replace(tzinfo=dt.timezone.utc)
                    expired = dt.datetime.now(tz=dt.timezone.utc) >= lease
                except ValueError:
                    expired = True
            if expired:
                report(
                    "FAIL",
                    check,
                    f"latest receipt has an EXPIRED running lease via {source} ({updated})",
                )
            else:
                report("WARN", check,
                       f"latest receipt still running via {source} ({updated})")
        elif status != "success":
            report("FAIL", check, f"invalid receipt status {status!r}")
        elif age_bd > max_bd:
            report("FAIL", check,
                   f"latest success via {source} is {age_bd} bd old "
                   f"(max {max_bd})")
        else:
            report("OK", check,
                   f"success via {source}, {age_bd} bd old ({updated})")


# ---------------------------------------------------------------- 2. data
def _last_index_date(path: Path) -> dt.date | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    candidates = []
    if isinstance(df.index, pd.MultiIndex):
        candidates += [df.index.get_level_values(i)
                       for i in range(df.index.nlevels)]
    elif not isinstance(df.index, pd.RangeIndex):
        candidates.append(df.index)
    candidates += [df[c] for c in df.columns
                   if str(c).lower() in ("date", "datetime")]
    for cand in candidates:
        try:
            return pd.to_datetime(cand, format="mixed").max().date()
        except (ValueError, TypeError):
            continue
    raise ValueError("no datetime index or Date column found")


def check_local_data() -> None:
    today = dt.date.today()
    for name, warn_bd, fail_bd in [
        ("master_prices.parquet", 2, 3),
        ("rd2_fragility.parquet", 2, 4),   # FRAG_STALE_TD=3: >3 means live
        ("cboe_putcall.parquet", 2, 4),    # sizing already fell back silently
    ]:
        path = ROOT / "data" / name
        try:
            last = _last_index_date(path)
        except Exception as exc:
            report("FAIL", f"data:{name}", f"unreadable: {exc}")
            continue
        if last is None:
            report("FAIL", f"data:{name}", "file missing")
            continue
        behind = bdays_behind(last, today)
        tier = "FAIL" if behind >= fail_bd else "WARN" if behind >= warn_bd else "OK"
        note = "" if tier == "OK" else " (the pinned runtime may need a canonical R2 pull)"
        report(tier, f"data:{name}", f"last row {last}, {behind} bd behind{note}")

    strays = [p for p in (ROOT / "data").glob("*.parquet.*")
              if p.suffix != ".parquet"]
    if strays:
        report("WARN", "data:stray-temp-files",
               "partial-write artifacts: " + ", ".join(p.name for p in strays))
    else:
        report("OK", "data:stray-temp-files", "none")


# ---------------------------------------------------------------- 3. PIT
def check_fragility_pit() -> None:
    path = ROOT / "data" / "rd2_fragility.parquet"
    if not path.exists():
        return
    try:
        df = pd.read_parquet(path).sort_index()
        df.index = pd.to_datetime(df.index).normalize()
    except Exception as exc:
        report("FAIL", "pit:rd2_fragility", f"unreadable: {exc}")
        return
    state = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    frozen_through = state.get("frag_frozen_through")
    if frozen_through:
        frozen = df.loc[:pd.Timestamp(frozen_through)]
        digest = hashlib.sha256(
            frozen.round(6).to_csv().encode("utf-8")).hexdigest()
        if digest != state.get("frag_frozen_sha"):
            report("FAIL", "pit:rd2_fragility",
                   f"frozen history (<= {frozen_through}) CHANGED since last "
                   f"check - possible full-rewrite with the drifted recompute "
                   f"vintage; live frag_risk_bands sizing is compromised")
        else:
            report("OK", "pit:rd2_fragility",
                   f"frozen rows <= {frozen_through} unchanged")

    new_frozen = min(
        df.index.max(),
        pd.Timestamp(np.busday_offset(np.datetime64(dt.date.today()), -5,
                                      roll="backward")))
    frozen = df.loc[:new_frozen]
    state["frag_frozen_through"] = str(new_frozen.date())
    state["frag_frozen_sha"] = hashlib.sha256(
        frozen.round(6).to_csv().encode("utf-8")).hexdigest()
    state["last_check_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")
    if not frozen_through:
        report("OK", "pit:rd2_fragility",
               f"tripwire baseline set (frozen through {new_frozen.date()})")


# ---------------------------------------------------------------- 4. journals
def check_journals() -> None:
    for name in ("pitch_journal.jsonl", "context_journal.jsonl",
                 "posts_journal.jsonl"):
        path = ROOT / "data" / name
        if not path.exists():
            report("WARN", f"journal:{name}", "missing")
            continue
        bad = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        tier = "FAIL" if bad else "OK"
        report(tier, f"journal:{name}",
               f"{bad} unparseable line(s)" if bad else "parses clean")


# ---------------------------------------------------------------- 5. delivery
def check_delivery() -> None:
    now = dt.datetime.now()

    pitch_day = now.date()
    if now.hour < 6:
        pitch_day -= dt.timedelta(days=1)
    pitch_day = prev_weekday(pitch_day)

    ctx_day = now.date()
    if now.hour < 20:
        ctx_day -= dt.timedelta(days=1)
    while ctx_day.weekday() in (4, 5):  # no Fri/Sat run
        ctx_day -= dt.timedelta(days=1)

    for label, script, flag, day in [
        ("pitch", "check_pitch_delivered.py", "--asof", pitch_day),
        ("context", "check_context_delivered.py", "--run-date", ctx_day),
    ]:
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), flag, str(day)],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        msg = (out.stdout + out.stderr).strip().splitlines()
        detail = msg[0] if msg else f"exit {out.returncode}"
        report("OK" if out.returncode == 0 else "FAIL",
               f"delivery:{label}", f"{day}: {detail}")


# ---------------------------------------------------------------- 6. local Task Scheduler logs
def check_trigger_logs() -> None:
    if not AUTOMATION_LOG_DIR.exists():
        report("WARN", "triggers", f"{AUTOMATION_LOG_DIR} not found")
        return
    today = dt.date.today()
    for pipeline, max_bd in LOCAL_PIPELINE_MAX_BD.items():
        logs = list(AUTOMATION_LOG_DIR.glob(f"*/{pipeline}-*.log"))
        if not logs:
            report("WARN", f"triggers:{pipeline}", "no local runtime log found")
            continue
        log = max(logs, key=lambda path: path.stat().st_mtime)
        mtime = dt.datetime.fromtimestamp(log.stat().st_mtime).date()
        behind = bdays_behind(mtime, today)
        tier = "OK" if behind <= max_bd else "WARN"
        report(tier, f"triggers:{pipeline}",
               f"last wrote {mtime} ({behind} bd; {log.name})")


# ---------------------------------------------------------------- 7. tests
def check_test_collection() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    if out.returncode not in (0, 5):
        diagnostic = (out.stderr or out.stdout).strip().splitlines()
        tail = diagnostic[-1] if diagnostic else "no diagnostic output"
        report("FAIL", "tests:collect",
               f"collection process failed (exit {out.returncode}: {tail}); "
               f"run pytest --collect-only tests for detail")
        return
    collected = {line.split("::")[0].replace("\\", "/")
                 for line in out.stdout.splitlines() if "::" in line}
    empty = [p.name for p in sorted((ROOT / "tests").glob("test_*.py"))
             if f"tests/{p.name}" not in collected]
    if empty:
        report("WARN", "tests:collect",
               f"guard files with ZERO collectable tests (never run in CI): "
               + ", ".join(empty))
    else:
        report("OK", "tests:collect", f"{len(collected)} test files collect")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-gha", "--skip-automation",
                    dest="skip_automation", action="store_true")
    args = ap.parse_args()

    if not args.skip_automation:
        check_gha()
    check_local_data()
    check_fragility_pit()
    check_journals()
    check_delivery()
    check_trigger_logs()
    if not args.skip_tests:
        check_test_collection()

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n===== SUMMARY: {len(fails)} FAIL, {len(warns)} WARN, "
          f"{len(RESULTS) - len(fails) - len(warns)} OK =====")
    for tier, check, detail in fails + warns:
        print(f"  [{tier}] {check}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
