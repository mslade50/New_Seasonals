"""Fail closed when a private-site build contains stale actionable data.

This is the final gate immediately before a Cloudflare Pages deployment.  It
checks the assembled artifact rather than trusting that each upstream command
ran, so skipped/best-effort steps and incremental dist leftovers cannot leak
an old Portfolio, Seasonal board, or staged-order sheet into production.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any


REQUIRED_PORTFOLIO_PAYLOADS = (
    "strategy_daily",
    "positions",
    "exposure",
    "trade_mtm",
)


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _same_build(left: Any, right: Any) -> bool:
    try:
        fmt = "%Y-%m-%d %H:%M UTC"
        a = dt.datetime.strptime(str(left), fmt)
        b = dt.datetime.strptime(str(right), fmt)
        return abs((a - b).total_seconds()) <= 300
    except Exception:
        return False


def validate_site(out_dir: str) -> list[str]:
    data_dir = os.path.join(out_dir, "data")
    meta = _read_json(os.path.join(data_dir, "meta.json"))
    health = _read_json(os.path.join(data_dir, "health.json"))
    positions = _read_json(os.path.join(data_dir, "positions.json"))
    ideas = _read_json(os.path.join(data_dir, "ideas.json"))
    signals = _read_json(os.path.join(data_dir, "signals.json"))
    fundamentals = _read_json(os.path.join(data_dir, "fundamentals.json"))
    problems: list[str] = []

    if meta is None:
        return ["data/meta.json is missing or unreadable"]
    if health is None:
        return ["data/health.json is missing or unreadable"]

    flags = meta.get("payloads") or {}
    artifacts = health.get("artifacts") or {}
    prev_td = health.get("prev_td")
    if flags.get("health") is not True:
        problems.append("current health payload is unavailable")
    if not _same_build(meta.get("built_at"), health.get("built_at")):
        problems.append("meta.json and health.json are from different builds")
    if not prev_td:
        problems.append("health.json has no previous-trading-day anchor")

    for name in REQUIRED_PORTFOLIO_PAYLOADS:
        if flags.get(name) is not True:
            problems.append(f"Portfolio payload {name!r} is unavailable")
    for name in ("ledger", "master_prices"):
        status = (artifacts.get(name) or {}).get("status")
        if status != "fresh":
            problems.append(f"{name} health is {status or 'missing'}")

    if positions is None:
        problems.append("data/positions.json is missing or unreadable")
    else:
        pos_asof = positions.get("asof")
        if prev_td and (not pos_asof or str(pos_asof) < str(prev_td)):
            problems.append(f"positions as-of {pos_asof or 'missing'} is before {prev_td}")
        expired = [
            row for row in (positions.get("positions") or [])
            if row.get("Days_To_Time_Stop") is not None
            and float(row["Days_To_Time_Stop"]) < 0
        ]
        if expired:
            symbols = ", ".join(str(row.get("Ticker") or "?") for row in expired[:5])
            problems.append(f"{len(expired)} open positions are past their time stop ({symbols})")

    idea_health = artifacts.get("ideas") or {}
    idea_meta = (ideas or {}).get("meta") or {}
    idea_asof = idea_meta.get("asof")
    if flags.get("ideas") is not True:
        problems.append("current Seasonal ideas payload is unavailable")
    if idea_health.get("status") != "fresh":
        problems.append(f"Seasonal ideas health is {idea_health.get('status') or 'missing'}")
    if idea_meta.get("unavailable"):
        problems.append("Seasonal ideas payload is an unavailable tombstone")
    if prev_td and (not idea_asof or str(idea_asof) < str(prev_td)):
        problems.append(f"Seasonal ideas as-of {idea_asof or 'missing'} is before {prev_td}")

    signal_health = artifacts.get("signals") or {}
    if flags.get("signals") is not True:
        problems.append("current staged-signals payload is unavailable")
    if signal_health.get("status") != "fresh":
        problems.append(f"staged-signals health is {signal_health.get('status') or 'missing'}")
    if (signals or {}).get("unavailable"):
        problems.append("staged-signals payload is an unavailable tombstone")

    if flags.get("fundamentals") is not True:
        problems.append("current Fundamentals payload is unavailable")
    if fundamentals is None:
        problems.append("data/fundamentals.json is missing or unreadable")
    else:
        fundamental_asof = fundamentals.get("as_of")
        if prev_td and (not fundamental_asof or str(fundamental_asof) < str(prev_td)):
            problems.append(
                f"Fundamentals as-of {fundamental_asof or 'missing'} is before {prev_td}"
            )
        if fundamentals.get("live_actions_enabled") is not False:
            problems.append("Fundamentals payload does not explicitly disable live actions")

    for rel in ("trades.json", "strategy_daily.json", "exposure.json", "trade_mtm.json"):
        if not os.path.isfile(os.path.join(data_dir, rel)):
            problems.append(f"data/{rel} is missing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist", help="assembled site directory")
    args = parser.parse_args()
    problems = validate_site(os.path.abspath(args.out))
    if problems:
        print("SITE DEPLOY BLOCKED: freshness validation failed")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Site freshness validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
