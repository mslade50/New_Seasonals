"""Observe Windows sleeve deployment and optionally publish a status-only R2 object.

Never starts tasks, imports an executor, connects to IBKR, or changes live gates.
Run under the owner account. The only remote write, with --upload, is the fixed
ops/sleeve_runtime_status.json observability key. Without --upload the payload is
written only to the local preview file (and echoed with --print).
"""
from __future__ import annotations

import argparse
from contextlib import closing
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
R2_KEY = "ops/sleeve_runtime_status.json"
SCHEMA = "sleeve-runtime.v2"
TASK_NAMES = {
    "event": "IBKR Event Sleeve Auction Orders",
    "trend": "IBKR Trend Sleeve MOO",
    "chain": "IBKR Daily Order Chain",
    "legend": "IBKR Legend EMA",
    "legend_verify": "IBKR Legend EMA Verify",
}
_WANTED = ",".join("'" + name.replace("'", "''") + "'" for name in TASK_NAMES.values())
TASK_QUERY = r"""
$ErrorActionPreference = 'Stop'
$wanted = @(__WANTED__)
function Utc($t) { if ($t -and $t.Year -gt 2000) { $t.ToUniversalTime().ToString('o') } else { $null } }
$rows = @(Get-ScheduledTask | Where-Object {$_.TaskName -in $wanted} | ForEach-Object {
 $info = $_ | Get-ScheduledTaskInfo
 [pscustomobject]@{name=$_.TaskName;state=[string]$_.State;last_run_at=(Utc $info.LastRunTime);
  next_run_at=(Utc $info.NextRunTime);last_result=[long]$info.LastTaskResult}
})
ConvertTo-Json -InputObject $rows -Depth 4 -Compress
""".replace("__WANTED__", _WANTED)

LEGEND_STRATEGY = "Legend_EMA"
RUN_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})-(shadow|live)(?:-(\d+))?$")
# Not \b: an id glued to an underscore ("acct_U1234567") has no word boundary.
ACCOUNT = re.compile(r"(?<![A-Za-z0-9])(?:DU|DF|U|F)\d{5,}(?!\d)")


def clean_text(value: Any, limit: int = 200) -> str | None:
    if value is None:
        return None
    return ACCOUNT.sub("[account]", str(value))[:limit]


def finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def whole(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _journal(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _outcome(records: list[dict], ref: str) -> str:
    mine = [r for r in records if r.get("order_ref") == ref]
    for rec in reversed(mine):
        legs = rec.get("legs") if rec.get("kind") == "exit_fill" else None
        if isinstance(legs, dict):
            for leg, label in (("target", "target hit"), ("time", "time exit"), ("residual", "residual exit")):
                if isinstance(legs.get(leg), dict) and (finite(legs[leg].get("qty")) or 0) > 0:
                    return label
    kinds = {r.get("kind") for r in mine}
    if "target_filled" in kinds:
        return "target hit"
    if "entry_failed" in kinds:
        return "entry failed"
    verifies = [r for r in mine if r.get("kind") == "verify"]
    if verifies:
        return f"verify {clean_text(verifies[-1].get('result'), 40)}"
    return "no exit record yet"


def read_legend(root: Path) -> dict:
    """Last Legend EMA session from the runner's own result file and journal (read-only)."""
    result_path = root / "legend_ema_last_result.json"
    journal_path = root / "legend_ema_journal.jsonl"
    notes: list[str] = []
    result: dict | None = None
    records: list[dict] = []
    try:
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        result = loaded if isinstance(loaded, dict) else None
        if result is None:
            notes.append("last-result file invalid")
    except FileNotFoundError:
        notes.append("last-result file missing")
    except (OSError, ValueError):
        notes.append("last-result file unreadable")
    try:
        records = [r for r in _journal(journal_path) if r.get("strategy") == LEGEND_STRATEGY]
    except FileNotFoundError:
        notes.append("journal missing")
    except (OSError, UnicodeDecodeError):
        notes.append("journal unreadable")
    dates = [d for d in [result.get("date") if result else None, *(r.get("date") for r in records)]
             if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    if not dates:
        return unavailable("; ".join(notes) or "no Legend session recorded")
    session = max(dates)
    day = [r for r in records if r.get("date") == session]
    trades = [{"symbol": clean_text(r.get("symbol"), 12), "side": clean_text(r.get("side"), 8),
               "qty": whole(r.get("qty")), "status": clean_text(r.get("status"), 40),
               "outcome": _outcome(day, str(r.get("order_ref")))}
              for r in day if r.get("kind") == "entry"]
    skips: list[dict] = []
    fresh_result = result if result and result.get("date") == session else None
    if fresh_result and isinstance(fresh_result.get("symbols"), list):
        entered = {t["symbol"] for t in trades}
        for row in fresh_result["symbols"]:
            if not isinstance(row, dict):
                continue
            symbol = clean_text(row.get("Symbol"), 12)
            qty = whole(row.get("Qty")) or 0
            if qty > 0 and row.get("Side") in {"BUY", "SELL"}:
                if symbol not in entered:
                    trades.append({"symbol": symbol, "side": row.get("Side"), "qty": qty,
                                   "status": clean_text(row.get("Status"), 40), "outcome": "no journal record"})
            else:
                skips.append({"symbol": symbol, "status": clean_text(row.get("Status"), 40),
                              "note": clean_text(row.get("Note"))})
    else:
        for r in day:
            if r.get("kind") in {"decision", "abort"} and not r.get("side"):
                skips.append({"symbol": clean_text(r.get("symbol"), 12), "status": clean_text(r.get("kind"), 40),
                              "note": clean_text(r.get("reason"))})
    return {"available": True, "session_date": session,
            "result_generated": clean_text(fresh_result.get("generated"), 40) if fresh_result else None,
            "result_error": clean_text(fresh_result.get("error"), 80) if fresh_result else None,
            "trades": trades, "skips": skips, "notes": notes}


def _prior_range(meta: dict) -> dict | None:
    summary = meta.get("inputs", {}).get("prior_range") if isinstance(meta.get("inputs"), dict) else None
    if not isinstance(summary, dict):
        return None
    return {clean_text(market, 12): {"status": clean_text(row.get("prior_range_status"), 40),
                                     "ratio": finite(row.get("ratio")),
                                     "skip": row.get("skip_prior_range") is True,
                                     "half": row.get("half_prior_range") is True,
                                     "reason": clean_text(row.get("prior_range_reason"))}
            for market, row in summary.items() if isinstance(row, dict)}


def read_breakout_session(path: Path, run_dir: str) -> dict:
    """Allow-listed meta from one session's runtime.sqlite, opened read-only."""
    if not path.is_file():
        return {**unavailable("runtime.sqlite missing"), "run_dir": run_dir}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            raw = db.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error as exc:
        reason = "sqlite locked" if "locked" in str(exc).lower() else f"sqlite unreadable ({type(exc).__name__})"
        return {**unavailable(reason), "run_dir": run_dir}
    meta: dict[str, Any] = {}
    for key, value in raw:
        try:
            meta[key] = json.loads(value)
        except (TypeError, ValueError):
            continue
    beat = meta.get("heartbeat") if isinstance(meta.get("heartbeat"), dict) else {}
    settings = meta.get("settings") if isinstance(meta.get("settings"), dict) else {}
    risk = settings.get("risk_bps") if isinstance(settings.get("risk_bps"), dict) else {}
    execution = settings.get("execution") if isinstance(settings.get("execution"), dict) else {}
    return {
        "available": True, "run_dir": run_dir,
        "session": clean_text(meta.get("session"), 10), "mode": clean_text(meta.get("mode"), 10),
        "pid": whole(meta.get("pid")), "phase": clean_text(meta.get("phase"), 40),
        "heartbeat_at": clean_text(beat.get("at"), 40), "events": whole(beat.get("events")),
        "connected": beat.get("connected") if isinstance(beat.get("connected"), bool) else None,
        "healthy": beat.get("healthy") if isinstance(beat.get("healthy"), bool) else None,
        "finished_at": clean_text(meta.get("finished_at"), 40),
        "halted": meta.get("halted") is True,
        "last_error": clean_text(meta.get("last_error")), "feed_error": clean_text(meta.get("feed_error")),
        "settings": {"risk_bps": {clean_text(k, 12): finite(v) for k, v in risk.items()},
                     "execution": {clean_text(k, 12): clean_text(v, 12) for k, v in execution.items()}},
        "prior_range": _prior_range(meta),
    }


def read_breakout(runs_root: Path) -> dict:
    """Newest shadow and newest live session under artifacts/open_breakout_runs."""
    if not runs_root.is_dir():
        return unavailable("runs directory missing")
    newest: dict[str, tuple[tuple[str, int], Path]] = {}
    try:
        dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    except OSError:
        return unavailable("runs directory unreadable")
    for folder in dirs:
        match = RUN_DIR.match(folder.name)
        if not match:
            continue
        key = (match.group(1), int(match.group(3) or 1))
        mode = match.group(2)
        if mode not in newest or key > newest[mode][0]:
            newest[mode] = (key, folder)
    out: dict[str, Any] = {"available": True, "shadow": None, "live": None}
    for mode, (_, folder) in newest.items():
        try:
            out[mode] = read_breakout_session(folder / "runtime.sqlite", folder.name)
        except Exception as exc:  # one session must never fail the publish
            out[mode] = {**unavailable(f"read failed ({type(exc).__name__})"), "run_dir": folder.name}
    return out


def _guard(reader, *args) -> dict:
    try:
        return reader(*args)
    except Exception as exc:  # a card reader must never fail the whole publish
        return unavailable(f"read failed ({type(exc).__name__})")


def collect(executor_root: Path, runs_root: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("The machine status collector requires Windows")
    if not executor_root.is_dir():
        raise RuntimeError("Executor directory unavailable; no status will be published")
    shell = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(shell), "-NoProfile", "-NonInteractive", "-Command", TASK_QUERY],
                            capture_output=True, text=True, timeout=45, check=False,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Windows task inventory failed; no status will be published")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise RuntimeError("Invalid Windows task inventory")
    by_name = {r["name"]: r for r in rows}
    tasks = {}
    for key, name in TASK_NAMES.items():
        row = by_name.get(name)
        tasks[key] = ({"state": row["state"], "last_run_at": row["last_run_at"], "next_run_at": row.get("next_run_at"),
                       "last_result": row["last_result"]} if row
                      else {"state": "Missing", "last_run_at": None, "next_run_at": None, "last_result": None})
    return {
        "schema": SCHEMA, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tasks": tasks,
        "event_enabled": (executor_root / "event_moo_enabled.flag").is_file(),
        "trend_moo_enabled": (executor_root / "trend_moo_enabled.flag").is_file(),
        "legend_enabled": (executor_root / "legend_ema_enabled.flag").is_file(),
        "legend": _guard(read_legend, executor_root),
        "open_breakout": _guard(read_breakout, runs_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=ROOT)
    parser.add_argument("--executor-root", type=Path, default=Path.home() / "OneDrive/trading_ibkr")
    parser.add_argument("--breakout-runs", type=Path, default=ROOT / "artifacts/open_breakout_runs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sleeve-status/runtime-status.json")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--print", dest="echo", action="store_true", help="also print the payload to stdout")
    args = parser.parse_args()
    try:
        payload = collect(args.executor_root, args.breakout_runs)
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        if args.echo:
            print(text)
        if args.upload:
            from dotenv import load_dotenv
            load_dotenv(args.config_root / ".env", override=False)
            sys.path.insert(0, str(ROOT))
            import cache_io
            if not cache_io.upload_from_local(str(args.output), R2_KEY):
                raise RuntimeError("Status-only upload failed")
        print(f"Sleeve machine status observed at {payload['checked_at']}" + ("; published" if args.upload else "; local preview only"))
        return 0
    except Exception as exc:
        # Do not print subprocess output, environment contents, or broker details.
        print(f"Sleeve status collection failed ({type(exc).__name__}); previous report must age out", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
