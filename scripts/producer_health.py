"""Current-invocation coverage health, separate from completed side effects."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def expected_status_paths(commands):
    paths = set()
    for command in commands:
        args = list(command.argv)
        def flag(name, default):
            for index, argument in enumerate(args):
                if argument.startswith(name + "="):
                    return argument.split("=", 1)[1]
                if argument == name and index + 1 < len(args):
                    return args[index + 1]
            return default
        if "daily_scan.py" in args:
            paths.add(f"data/scan_coverage_{flag('--scope', 'liquid')}_{flag('--bookend', 'auto')}.json")
        if "verify_fills.py" in args:
            paths.add("data/fill_verification_status.json")
        if "scripts/update_master_prices.py" in args:
            paths.add("data/master_prices.parquet.status.json")
        if "scripts/build_earnings_calendar.py" in args:
            paths.add("data/earnings_calendar.parquet.status.json")
    return sorted(paths)


def inspect_health(commands, repo_root, started_at):
    problems = []
    started_at = started_at.astimezone(dt.timezone.utc)
    for relative in expected_status_paths(commands):
        try:
            receipt = json.loads((Path(repo_root) / relative).read_text(encoding="utf-8"))
            generated = dt.datetime.fromisoformat(str(receipt.get("generated_at", "")).replace("Z", "+00:00"))
            if generated.tzinfo is None or generated < started_at:
                raise ValueError("receipt does not belong to this invocation")
            if generated > dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30):
                raise ValueError("receipt is future-dated")
            if receipt.get("status") != "ok":
                counts = {key: len(receipt[key]) for key in ("unavailable", "failed_tickers", "empty_tickers", "unresolved_basis", "rejected_segments", "exceptions") if receipt.get(key)}
                problems.append(f"{relative}: degraded coverage {counts}; {receipt.get('fallback', '')}")
        except Exception as exc:
            problems.append(f"{relative}: coverage unverified ({type(exc).__name__})")
    return ("degraded", "; ".join(problems)) if problems else ("ok", None)
