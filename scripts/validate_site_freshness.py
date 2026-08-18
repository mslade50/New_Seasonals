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


def validate_site(out_dir: str, *, require_r2_provenance: bool = False) -> list[str]:
    data_dir = os.path.join(out_dir, "data")
    meta = _read_json(os.path.join(data_dir, "meta.json"))
    health = _read_json(os.path.join(data_dir, "health.json"))
    positions = _read_json(os.path.join(data_dir, "positions.json"))
    ideas = _read_json(os.path.join(data_dir, "ideas.json"))
    signals = _read_json(os.path.join(data_dir, "signals.json"))
    risk = _read_json(os.path.join(data_dir, "risk.json"))
    seasonality_dir = os.path.join(data_dir, "seasonality")
    seasonality = _read_json(os.path.join(seasonality_dir, "manifest.json"))
    macro_seasonality = _read_json(os.path.join(seasonality_dir, "macro.json"))
    provenance = _read_json(os.path.join(data_dir, "provenance.json"))
    problems: list[str] = []

    if meta is None:
        return ["data/meta.json is missing or unreadable"]
    if health is None:
        return ["data/health.json is missing or unreadable"]

    if require_r2_provenance:
        meta_provenance = meta.get("data_provenance") or {}
        if meta_provenance.get("mode") != "r2-only":
            problems.append("meta.json does not certify an R2-only build")
        if provenance is None or provenance.get("mode") != "r2-only":
            problems.append("data/provenance.json is missing or not R2-only")
        elif (
            str(provenance.get("run_id")) != str(meta_provenance.get("run_id"))
            or provenance.get("source_sha") != meta_provenance.get("source_sha")
        ):
            problems.append("meta.json and provenance.json identify different builds")
        elif not provenance.get("entries"):
            problems.append("R2 provenance contains no materialized inputs")

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

    if flags.get("risk") is not True or risk is None:
        problems.append("current Risk payload is unavailable")
    else:
        if not _same_build(meta.get("built_at"), risk.get("built_at")):
            problems.append("meta.json and risk.json are from different builds")
        risk_asof = risk.get("asof")
        if prev_td and (not risk_asof or str(risk_asof) < str(prev_td)):
            problems.append(f"Risk payload as-of {risk_asof or 'missing'} is before {prev_td}")
        sizing_asof = (risk.get("sizing_state") or {}).get("asof")
        if prev_td and (not sizing_asof or str(sizing_asof) < str(prev_td)):
            problems.append(
                f"Risk sizing state as-of {sizing_asof or 'missing'} is before {prev_td}")
        detail = risk.get("signal_detail") or {}
        for name in ("Defensive Leadership", "Seasonal Rank Divergence"):
            current = (detail.get(name) or {}).get("current") or {}
            if current.get("value") is None:
                problems.append(f"Risk signal {name!r} has no current metric value")
        if not isinstance(risk.get("trade_console"), dict):
            problems.append("Risk trade console is unavailable")

    if flags.get("seasonality") is not True or seasonality is None:
        problems.append("Seasonality Lab payload is unavailable")
    else:
        seasonality_asof = seasonality.get("asof")
        if prev_td and (
            not seasonality_asof or str(seasonality_asof) < str(prev_td)
        ):
            problems.append(
                f"Seasonality Lab as-of {seasonality_asof or 'missing'} is before {prev_td}")
        if not os.path.isfile(os.path.join(seasonality_dir, "theses.json")):
            problems.append("Seasonality Lab theses payload is missing")
        missing_bins = []
        for ticker, ticker_meta in (seasonality.get("tickers") or {}).items():
            rel = str((ticker_meta or {}).get("file") or "")
            if not rel or not os.path.isfile(os.path.join(seasonality_dir, rel)):
                missing_bins.append(str(ticker))
        if missing_bins:
            problems.append(
                f"Seasonality Lab is missing {len(missing_bins)} ticker payload(s) "
                f"({', '.join(missing_bins[:5])})")

    if flags.get("macro_sznl") is not True or macro_seasonality is None:
        problems.append("Macro Seasonality payload is unavailable")
    else:
        macro_asof = macro_seasonality.get("asof")
        if prev_td and (not macro_asof or str(macro_asof) < str(prev_td)):
            problems.append(
                f"Macro Seasonality as-of {macro_asof or 'missing'} is before {prev_td}")
        if macro_seasonality.get("sznl_available") is not True:
            problems.append("Macro seasonal ranks are unavailable")
        missing_prices = [
            str(row.get("ticker") or "?")
            for row in (macro_seasonality.get("rows") or [])
            if not str(row.get("ticker") or "").startswith("^")
            and row.get("price") is None
        ]
        if missing_prices:
            problems.append(
                f"Macro Seasonality is missing prices for {len(missing_prices)} "
                f"ticker(s) ({', '.join(missing_prices[:5])})")

    for rel in ("trades.json", "strategy_daily.json", "exposure.json", "trade_mtm.json"):
        if not os.path.isfile(os.path.join(data_dir, rel)):
            problems.append(f"data/{rel} is missing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist", help="assembled site directory")
    parser.add_argument(
        "--require-r2-provenance",
        action="store_true",
        help="block any artifact not assembled from the R2-only cloud boundary",
    )
    args = parser.parse_args()
    problems = validate_site(
        os.path.abspath(args.out),
        require_r2_provenance=args.require_r2_provenance,
    )
    if problems:
        print("SITE DEPLOY BLOCKED: freshness validation failed")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Site freshness validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
