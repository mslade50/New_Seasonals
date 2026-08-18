"""Deterministic health battery for the New_Seasonals pipeline.

Run by the /repo-health-check skill (and fine to run by hand):

    python scripts/repo_health_check.py [--skip-tests] [--skip-gha]

Prints one OK / WARN / FAIL line per check and a summary. Exit 1 when any
check FAILed, else 0. The skill layer investigates failures; this script only
detects them, so every check must be cheap, offline-safe and side-effect free
(the single exception: it maintains its own tripwire state file).

Checks:
  1. GitHub Actions   - latest run conclusion + age for each weekday-critical
                        workflow (needs gh CLI; degrades to WARN without it)
  2. Local data       - master_prices / rd2_fragility / cboe_putcall recency,
                        stray partial-write temp files in data/
  3. Fragility PIT    - tripwire: frozen rows of rd2_fragility.parquet must
                        never change between runs (rewrite = the drifted
                        recompute vintage reaching live sizing)
  4. Journals         - pitch/context/posts JSONL parse cleanly
  5. Delivery         - check_pitch_delivered / check_context_delivered for
                        the most recent expected run date
  6. Trigger logs     - C:\\Scripts\\logs\\trigger_*.log recency (the local AM
                        dispatch chain)
  7. Guard tests      - pytest --collect-only: collection errors FAIL, guard
                        files contributing zero tests WARN
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "health_check_state.json"
TRIGGER_LOG_DIR = Path(r"C:\Scripts\logs")

# workflow file -> max business days the latest run may be old before FAIL
CRITICAL_WORKFLOWS: dict[str, int] = {
    "update_master_prices.yml": 1,
    "daily_screener.yml": 1,
    "risk_report.yml": 1,
    "portfolio_report.yml": 1,
    "build_earnings_calendar.yml": 2,
    "verify_fills.yml": 2,
    "update_cboe_putcall.yml": 1,
    "update_intraday_prices.yml": 3,
    "execution_report.yml": 2,
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


# ---------------------------------------------------------------- 1. GHA
def _github_repository() -> str | None:
    """Return OWNER/REPO without relying on gh's implicit git discovery."""
    from_env = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if from_env:
        return from_env
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "remote",
                "get-url",
                "origin",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", out.stdout.strip())
    return f"{match.group(1)}/{match.group(2)}" if match else None


def check_gha() -> None:
    repository = _github_repository()
    if not repository:
        report("WARN", "gha", "could not determine GitHub OWNER/REPO")
        return
    for wf, max_bd in CRITICAL_WORKFLOWS.items():
        try:
            out = subprocess.run(
                ["gh", "run", "list", f"--workflow={wf}", "--limit", "1",
                 "--json", "conclusion,status,updatedAt", "--repo", repository],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            report("WARN", f"gha:{wf}", f"gh CLI unavailable ({exc})")
            return
        if out.returncode != 0:
            report("WARN", f"gha:{wf}", out.stderr.strip()[:200] or "gh failed")
            continue
        runs = json.loads(out.stdout or "[]")
        if not runs:
            report("WARN", f"gha:{wf}", "no runs found")
            continue
        run = runs[0]
        updated = dt.datetime.fromisoformat(
            run["updatedAt"].replace("Z", "+00:00"))
        age_bd = bdays_behind(updated.date(), dt.date.today())
        concl = run.get("conclusion") or run.get("status")
        if concl == "failure":
            report("FAIL", f"gha:{wf}", f"latest run FAILED ({run['updatedAt']})")
        elif age_bd > max_bd:
            report("FAIL", f"gha:{wf}",
                   f"latest run is {age_bd} bd old (max {max_bd}) - cron "
                   f"likely shed and never backfilled")
        else:
            report("OK", f"gha:{wf}", f"{concl}, {age_bd} bd old")


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
        note = "" if tier == "OK" else " (local copy may just need a git pull / R2 pull)"
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


# ---------------------------------------------------------------- 6. triggers
def check_trigger_logs() -> None:
    if not TRIGGER_LOG_DIR.exists():
        report("WARN", "triggers", f"{TRIGGER_LOG_DIR} not found")
        return
    today = dt.date.today()
    for log in sorted(TRIGGER_LOG_DIR.glob("trigger_*.log")):
        mtime = dt.date.fromtimestamp(log.stat().st_mtime)
        behind = bdays_behind(mtime, today)
        tier = "OK" if behind <= 1 else "WARN"
        report(tier, f"triggers:{log.stem}", f"last wrote {mtime} ({behind} bd)")


# ---------------------------------------------------------------- 7. tests
def check_test_collection() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    if "error" in out.stdout.lower() and out.returncode not in (0, 5):
        report("FAIL", "tests:collect",
               f"collection errors (exit {out.returncode}); run pytest "
               f"--collect-only tests for detail")
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
    ap.add_argument("--skip-gha", action="store_true")
    args = ap.parse_args()

    if not args.skip_gha:
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
