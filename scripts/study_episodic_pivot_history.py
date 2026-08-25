"""Run the EP v0 observed-panel census and optional diagnostic outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.config import DEFAULT_POLICY  # noqa: E402
from episodic_pivot.historical import (  # noqa: E402
    clustered_outcome_summary,
    diagnostic_summary,
    load_earnings_map,
    load_current_company_symbols,
    load_observed_panel,
    run_observed_panel_study,
)
from episodic_pivot.manifest import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EP survivor-biased observed-panel study")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--universe-path",
        type=Path,
        help="current FMP broad-universe parquet; defaults under --data-root",
    )
    parser.add_argument(
        "--mode",
        choices=("census", "diagnostic"),
        default="census",
        help="diagnostic adds ex-post fixed-horizon outcomes; it is not a backtest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    master = args.data_root / "master_prices.parquet"
    overflow = args.data_root / "overflow_prices.parquet"
    earnings = args.data_root / "earnings_calendar.parquet"
    universe = args.universe_path or (
        args.data_root / "fundamental" / "current" / "broad_universe_latest.parquet"
    )
    if not master.exists():
        raise SystemExit(f"missing {master}")

    output_root = args.output_root.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if output_root != artifact_root and artifact_root not in output_root.parents:
        raise SystemExit("--output-root must stay under this worktree's artifacts directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"historical-{args.mode}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    panel = load_observed_panel(master, overflow if overflow.exists() else None)
    benchmark_prices = panel[panel["ticker"].eq("SPY")].copy()
    if not universe.exists():
        raise SystemExit(
            "missing current operating-company universe; refusing to mix ETFs/FX/indices "
            f"into the EP census: {universe}"
        )
    company_symbols = load_current_company_symbols(universe)
    panel = panel[panel["ticker"].isin(company_symbols)].copy()
    earnings_map = load_earnings_map(earnings if earnings.exists() else None)
    events, anomalies, counts = run_observed_panel_study(
        panel,
        policy=DEFAULT_POLICY.historical,
        earnings_map=earnings_map,
        include_outcomes=args.mode == "diagnostic",
        benchmark_prices=benchmark_prices,
    )
    events.to_parquet(output / "historical_candidates.parquet", index=False)
    anomalies.to_parquet(output / "historical_anomalies.parquet", index=False)
    if args.mode == "diagnostic":
        diagnostic_summary(events).to_csv(output / "diagnostic_strata.csv", index=False)
        clustered = [
            clustered_outcome_summary(events, cluster_column="date"),
            clustered_outcome_summary(events, cluster_column="ticker"),
        ]
        import pandas as pd

        pd.concat(clustered, ignore_index=True).to_csv(
            output / "diagnostic_clustered.csv", index=False
        )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": args.mode,
        "policy": DEFAULT_POLICY.historical.__dict__,
        "counts": counts,
        "strict_events_with_earnings_date_match": int(events.get("earnings_date_match", []).sum()) if not events.empty else 0,
        "strict_events_clean_for_diagnostics": int(events.get("data_quality_clean", []).sum()) if not events.empty else 0,
        "input_hashes": {
            "master_prices": sha256_file(master),
            "overflow_prices": sha256_file(overflow) if overflow.exists() else None,
            "earnings_calendar": sha256_file(earnings) if earnings.exists() else None,
            "current_company_universe": sha256_file(universe),
        },
        "limitations": [
            "Current observed ticker panel is survivor-biased and lacks permanent security identifiers.",
            "The operating-company filter is a current FMP snapshot, not point-in-time membership.",
            "Daily OHLCV is adjusted and contains unresolved split/basis cliffs.",
            "Full-day volume confirmation is ex-post and cannot justify an event-open fill.",
            "Earnings dates lack reliable BMO/AMC timestamps; surprise fields are current-vintage and unused.",
            "The repo has no broad historical premarket bars or point-in-time news archive.",
            "Diagnostic outcomes are frictionless fixed-horizon observations, not a production backtest.",
        ],
    }
    (output / "historical_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))
    print(f"Study artifacts: {output}")
    if args.mode == "census":
        print("No forward returns were calculated. Resolve the anomaly table before diagnostics.")
    else:
        print("Diagnostic only: survivor-biased, ex-post volume-confirmed, and not tradeable as coded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
