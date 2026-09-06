"""Observe Windows sleeve deployment and optionally publish a status-only R2 object.

Never starts tasks, imports an executor, connects to IBKR, or changes live gates.
Run under the owner account. The only remote write, with --upload, is the fixed
ops/sleeve_runtime_status.json observability key.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
R2_KEY = "ops/sleeve_runtime_status.json"
TASK_NAMES = {
    "event": "IBKR Event Sleeve Auction Orders",
    "trend": "IBKR Trend Sleeve MOO",
    "chain": "IBKR Daily Order Chain",
    "legend_signals": "NewSeasonals-LegendETF-Signals",
    "legend_session": "NewSeasonals-LegendETF-Session",
    "legend_watchdog": "NewSeasonals-LegendETF-Watchdog",
}
TASK_QUERY = r"""
$ErrorActionPreference = 'Stop'
$wanted = @('IBKR Event Sleeve Auction Orders','IBKR Trend Sleeve MOO','IBKR Daily Order Chain',
 'NewSeasonals-LegendETF-Signals','NewSeasonals-LegendETF-Session','NewSeasonals-LegendETF-Watchdog')
$rows = @(Get-ScheduledTask | Where-Object {$_.TaskName -in $wanted} | ForEach-Object {
 $info = $_ | Get-ScheduledTaskInfo
 $last = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 2000) {$info.LastRunTime.ToUniversalTime().ToString('o')} else {$null}
 [pscustomobject]@{name=$_.TaskName;state=[string]$_.State;last_run_at=$last;last_result=[long]$info.LastTaskResult}
})
ConvertTo-Json -InputObject $rows -Depth 4 -Compress
"""


def collect(executor_root: Path, legend_root: Path) -> dict:
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
        tasks[key] = {"state": row["state"], "last_run_at": row["last_run_at"], "last_result": row["last_result"]} if row else {"state": "Missing", "last_run_at": None, "last_result": None}
    now = dt.datetime.now(dt.timezone.utc)
    live = {}
    env_path = legend_root / "runtime.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in {"LEGEND_ETF_LIVE_ENABLED", "LEGEND_ETF_LIVE_DATE"}:
                live[key.strip()] = value.strip().strip("\"'")
    return {
        "schema": "sleeve-runtime.v1", "checked_at": now.isoformat(), "tasks": tasks,
        "event_enabled": (executor_root / "event_moo_enabled.flag").is_file(),
        "trend_moo_enabled": (executor_root / "trend_moo_enabled.flag").is_file(),
        "legend_live_today": live.get("LEGEND_ETF_LIVE_ENABLED", "").lower() in {"1", "true", "yes"}
            and live.get("LEGEND_ETF_LIVE_DATE") == now.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=ROOT)
    parser.add_argument("--executor-root", type=Path, default=Path.home() / "OneDrive/trading_ibkr")
    parser.add_argument("--legend-runtime", type=Path, default=Path(os.environ.get("LOCALAPPDATA", "")) / "NewSeasonals/legend_etf")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sleeve-status/runtime-status.json")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    try:
        payload = collect(args.executor_root, args.legend_runtime)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
