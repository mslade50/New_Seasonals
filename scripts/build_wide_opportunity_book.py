"""Build a local, research-only Opportunity Book across the broad universe.

This command never downloads data or touches any production surface.  It reads
one local parquet and writes a standalone artifact bundle to the explicitly
provided output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.opportunity_book import (
    OpportunityConfig,
    build_opportunity_book,
    default_universe,
    write_opportunity_book,
)


def _tickers_from_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        column = next(
            (c for c in frame.columns if c.lower() in {"ticker", "symbol"}),
            frame.columns[0],
        )
        return [str(v) for v in frame[column].dropna()]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sector_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("sector-map JSON must be an object of ticker: sector")
        return {str(k): str(v) for k, v in payload.items()}
    frame = pd.read_csv(path)
    columns = {c.lower(): c for c in frame.columns}
    ticker_col = columns.get("ticker") or columns.get("symbol")
    sector_col = (
        columns.get("sector")
        or columns.get("sector_ticker")
        or columns.get("benchmark")
    )
    if not ticker_col or not sector_col:
        raise ValueError("sector-map CSV needs ticker and sector/benchmark columns")
    return dict(zip(frame[ticker_col].astype(str), frame[sector_col].astype(str)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices",
        default=str(ROOT / "data" / "master_prices.parquet"),
        help="local long-form master-prices parquet",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="explicit local directory for JSON/CSV/HTML outputs",
    )
    parser.add_argument(
        "--asof",
        help=(
            "inclusive signal-date cutoff; defaults to the latest local bar for "
            "--market-ticker rather than the wall-clock date"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tickers", help="comma-separated explicit universe")
    group.add_argument("--tickers-file", help="text/CSV/TSV explicit universe")
    parser.add_argument("--sector-map", help="optional JSON/CSV ticker-to-sector map")
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument("--review-limit", type=int, default=75)
    parser.add_argument("--deep-test-limit", type=int, default=10)
    parser.add_argument("--audit-limit", type=int, default=10)
    parser.add_argument("--audit-seed", type=int, default=1729)
    parser.add_argument("--min-history", type=int, default=63)
    parser.add_argument("--max-stale-sessions", type=int, default=2)
    return parser.parse_args()


def _resolve_artifact_output(raw: str | Path) -> Path:
    """Resolve an output path and enforce the research-only artifacts boundary."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    artifacts_root = (ROOT / "artifacts").resolve()
    try:
        candidate.relative_to(artifacts_root)
    except ValueError as exc:
        raise SystemExit(
            f"Refusing non-artifact output: {candidate}. "
            f"Use a directory under {artifacts_root}."
        ) from exc
    return candidate


def _latest_market_asof(prices: pd.DataFrame, market_ticker: str) -> pd.Timestamp:
    """Infer the completed research cutoff from the local market anchor."""
    lower = {str(column).lower(): column for column in prices.columns}
    if {"date", "ticker"}.issubset(lower):
        ticker = prices[lower["ticker"]].astype(str).str.upper().str.strip()
        dates = pd.to_datetime(
            prices.loc[ticker.eq(market_ticker.upper()), lower["date"]],
            errors="coerce",
        ).dropna()
        if not dates.empty:
            return pd.Timestamp(dates.max()).normalize()
    raise ValueError(
        f"cannot infer latest {market_ticker.upper()} bar from local prices; "
        "pass --asof explicitly"
    )


def main() -> int:
    args = parse_args()
    prices_path = Path(args.prices)
    if not prices_path.exists():
        raise SystemExit(f"Local prices missing: {prices_path}")
    if args.tickers:
        tickers = [value.strip() for value in args.tickers.split(",") if value.strip()]
    elif args.tickers_file:
        tickers = _tickers_from_file(Path(args.tickers_file))
    else:
        tickers = default_universe()

    prices = pd.read_parquet(prices_path)
    asof = args.asof or _latest_market_asof(prices, args.market_ticker)
    config = OpportunityConfig(
        asof=asof,
        review_limit=args.review_limit,
        deep_test_limit=args.deep_test_limit,
        audit_limit=args.audit_limit,
        audit_seed=args.audit_seed,
        min_history=args.min_history,
        max_stale_sessions=args.max_stale_sessions,
        market_ticker=args.market_ticker,
    )
    result = build_opportunity_book(
        prices,
        tickers,
        config,
        sector_map=_sector_map(Path(args.sector_map) if args.sector_map else None),
    )
    paths = write_opportunity_book(result, _resolve_artifact_output(args.output_dir))
    coverage = result.manifest["coverage"]
    selection = result.manifest["selection"]
    print("RESEARCH ONLY - no production, staging, portfolio or broker action taken")
    print(
        f"Opportunity Book {result.manifest['asof']}: "
        f"{coverage['eligible_count']}/{coverage['requested_count']} eligible; "
        f"{selection['review_count']} review, "
        f"{selection['deep_test_count']} deep-test, "
        f"{selection['audit_count']} audit"
    )
    print(f"Local report: {paths['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
