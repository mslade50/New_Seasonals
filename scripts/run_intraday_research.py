"""Run the local-only intraday research lab and write non-production artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = (ROOT / "artifacts").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday import (
    CapitalReuseConfig,
    EligibilityConfig,
    GapFirstHourConfig,
    IntradayShockConfig,
    load_parquet_frames,
    prepare_metadata,
    run_intraday_research,
)


def _read_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sector map does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("--sector-map must be parquet or CSV")


def _default_output_dir(*, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return artifacts_root / "intraday_research" / stamp


def _resolve_artifact_output(
    requested: Path | None, *, artifacts_root: Path = ARTIFACTS_ROOT
) -> Path:
    """Resolve a fresh output directory confined to the ignored artifact root."""

    root = artifacts_root.resolve()
    output = (
        requested.resolve()
        if requested is not None
        else _default_output_dir(artifacts_root=root).resolve()
    )
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"--output-dir must stay under {root}") from exc
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"--output-dir is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"--output-dir must be empty: {output}")
    return output


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-only fixed-clock intraday template runner (local files only)."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "intraday")
    parser.add_argument(
        "--sector-map", type=Path, default=ROOT / "data" / "sector_map.parquet"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Target tickers; default is every local parquet except proxies.",
    )
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument(
        "--template",
        choices=("both", "gap_first_hour", "intraday_shock"),
        default="both",
    )
    parser.add_argument("--round-trip-cost-bps", type=float, default=5.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=25_000_000.0)
    parser.add_argument("--min-completeness", type=float, default=0.95)
    parser.add_argument("--per-trade-notional", type=float, default=None)
    parser.add_argument("--starting-settled-cash", type=float, default=None)
    parser.add_argument("--intraday-buying-power", type=float, default=None)
    parser.add_argument(
        "--same-day-reuse-allowed",
        action="store_true",
        help="Explicit research assumption; never inferred from account size.",
    )
    parser.add_argument("--max-concurrent-notional", type=float, default=None)
    parser.add_argument(
        "--capital-priority",
        choices=("signal_strength", "input_order"),
        default="signal_strength",
        help="Tie-break capacity for trades sharing an entry timestamp.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None, *, artifacts_root: Path = ARTIFACTS_ROOT
) -> int:
    args = parse_args(argv)
    market_ticker = args.market_ticker.upper()
    sector_map_path = args.sector_map
    if (
        sector_map_path == ROOT / "data" / "sector_map.parquet"
        and not sector_map_path.exists()
    ):
        sector_map_path = None
    metadata = _read_metadata(sector_map_path)

    candidates = (
        None if args.tickers is None else [ticker.upper() for ticker in args.tickers]
    )
    if candidates is None:
        frames = load_parquet_frames(args.data_dir)
    else:
        candidate_meta = prepare_metadata(
            candidates, metadata, market_ticker=market_ticker
        )
        required = (
            set(candidates) | {market_ticker} | set(candidate_meta["sector_proxy"])
        )
        frames = load_parquet_frames(args.data_dir, tickers=sorted(required))

    templates = (
        ("gap_first_hour", "intraday_shock")
        if args.template == "both"
        else (args.template,)
    )
    eligibility_config = EligibilityConfig(
        min_price=args.min_price,
        min_median_dollar_volume=args.min_dollar_volume,
        min_data_completeness=args.min_completeness,
    )
    gap_config = GapFirstHourConfig()
    shock_config = IntradayShockConfig()
    capital_values = (
        args.starting_settled_cash,
        args.intraday_buying_power,
        args.max_concurrent_notional,
    )
    if args.per_trade_notional is None and (
        any(value is not None for value in capital_values)
        or args.same_day_reuse_allowed
    ):
        raise ValueError("capital constraints require --per-trade-notional")
    if args.per_trade_notional is not None and not any(
        value is not None for value in capital_values
    ):
        raise ValueError("--per-trade-notional requires at least one capital ceiling")
    capital_config = None
    if args.per_trade_notional is not None:
        capital_config = CapitalReuseConfig(
            default_notional_per_trade=args.per_trade_notional,
            starting_settled_cash=args.starting_settled_cash,
            intraday_buying_power=args.intraday_buying_power,
            same_day_reuse_allowed=args.same_day_reuse_allowed,
            max_concurrent_notional=args.max_concurrent_notional,
        )
    result = run_intraday_research(
        frames,
        metadata,
        market_ticker=market_ticker,
        candidates=candidates,
        templates=templates,
        eligibility_config=eligibility_config,
        gap_config=gap_config,
        shock_config=shock_config,
        round_trip_cost_bps=args.round_trip_cost_bps,
        capital_config=capital_config,
        capital_priority_column=(
            "signal_strength" if args.capital_priority == "signal_strength" else None
        ),
    )

    output_dir = _resolve_artifact_output(
        args.output_dir, artifacts_root=artifacts_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.eligibility.to_parquet(output_dir / "eligibility.parquet", index=False)
    result.signals.to_parquet(output_dir / "signals.parquet", index=False)
    result.trades.to_parquet(output_dir / "trades.parquet", index=False)
    result.execution_rejections.to_parquet(
        output_dir / "execution_rejections.parquet", index=False
    )
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    if result.capital_audit is not None:
        result.capital_audit.to_parquet(
            output_dir / "capital_audit.parquet", index=False
        )
        result.capital_feasible_trades.to_parquet(
            output_dir / "capital_feasible_trades.parquet", index=False
        )
        result.capital_rejections.to_parquet(
            output_dir / "capital_rejections.parquet", index=False
        )
        result.capital_summary.to_csv(output_dir / "capital_summary.csv", index=False)
    manifest = {
        "schema_version": "intraday-research-run.v1",
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "sector_map": str(sector_map_path.resolve()) if sector_map_path else None,
        "market_ticker": market_ticker,
        "candidate_tickers": candidates,
        "loaded_tickers": sorted(frames),
        "templates": list(templates),
        "round_trip_cost_bps": args.round_trip_cost_bps,
        "eligibility_config": asdict(eligibility_config),
        "gap_config": asdict(gap_config),
        "shock_config": asdict(shock_config),
        "capital_config": asdict(capital_config) if capital_config else None,
        "capital_priority": args.capital_priority if capital_config else None,
        "n_eligibility_rows": len(result.eligibility),
        "n_signals": len(result.signals),
        "n_trades": len(result.trades),
        "n_execution_rejected": len(result.execution_rejections),
        "n_capital_feasible": (
            len(result.capital_feasible_trades)
            if result.capital_feasible_trades is not None
            else None
        ),
        "n_capital_rejected": (
            len(result.capital_rejections)
            if result.capital_rejections is not None
            else None
        ),
    }
    _write_manifest(output_dir / "run_manifest.json", manifest)

    print(f"Research artifacts: {output_dir}")
    print(
        f"Loaded {len(frames)} tickers | signals {len(result.signals)} | trades {len(result.trades)}"
    )
    if result.summary.empty:
        print("No trades passed the fixed v0 gates.")
    else:
        print(result.summary.to_string(index=False))
    if result.capital_rejections is not None:
        print(
            f"Capital feasibility: {len(result.capital_feasible_trades)} accepted | "
            f"{len(result.capital_rejections)} rejected"
        )
    print(
        "Research-only: no production state, orders, uploads, or strategy promotion occurred."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
