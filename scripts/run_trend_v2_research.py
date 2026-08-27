"""Run the research-only Trend V2 harness from local adjusted data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.trend_v2.engine import load_adjusted_price_parquet
from research.trend_v2.runner import run_trend_v2_research, write_research_artifacts


def _local_table(path: str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"local input not found: {source}")
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError("sector history must be a local .parquet or .csv file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Trend benchmark, four preregistered ETF trials, "
            "and optionally the separate stock residual-trend family. No network or production writes."
        )
    )
    parser.add_argument("--prices", required=True, help="local adjusted OHLC parquet")
    parser.add_argument(
        "--sector-history",
        help="optional dated sector snapshots/intervals (.parquet or .csv) for stock mode",
    )
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument("--asof", help="optional inclusive data cutoff (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new explicit artifacts/ directory; non-empty directories are refused",
    )
    return parser.parse_args()


def _resolve_project_artifact_output(raw: str | Path) -> Path:
    """Confine command-line output to this isolated worktree's artifacts root."""
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


def main() -> int:
    args = parse_args()
    prices = load_adjusted_price_parquet(args.prices, asof=args.asof)
    sectors = _local_table(args.sector_history) if args.sector_history else None
    runs = run_trend_v2_research(
        prices=prices,
        sector_history=sectors,
        market_ticker=args.market_ticker,
    )
    output = write_research_artifacts(
        output_dir=_resolve_project_artifact_output(args.output_dir),
        prices=prices,
        runs=runs,
        stock_family_requested=sectors is not None,
    )
    print(f"Trend V2 research bundle: {output}")
    print(f"Rows: {len(runs)} (frozen benchmark + candidate trials); production writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
