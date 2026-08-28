"""Run the streaming intraday v0 study from explicit local research inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday.diagnostics import DEFAULT_COST_GRID_BPS
from research.intraday.eligibility import EligibilityConfig
from research.intraday.streaming import (
    run_streaming_intraday_research,
    write_streaming_research_artifacts,
)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"input table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"input table must be parquet or CSV: {path}")


def _read_universe(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"universe file does not exist: {path}")
    if path.suffix.lower() in {".csv", ".parquet"}:
        frame = _read_table(path)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        if "ticker" not in columns:
            raise ValueError("universe table must contain a ticker column")
        values = frame[columns["ticker"]].tolist()
    elif path.suffix.lower() in {".txt", ".list"}:
        values = path.read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError("universe file must be CSV, parquet, TXT, or LIST")
    tickers = sorted({str(value).upper().strip() for value in values if str(value).strip()})
    if not tickers:
        raise ValueError("universe file contains no tickers")
    return tickers


def _default_output_dir() -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts" / "intraday_streaming_research" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only streaming intraday v0 runner. Reads local parquets; "
            "never downloads, uploads, stages, or promotes."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Explicit directory containing {TICKER}_15min.parquet files.",
    )
    parser.add_argument("--sector-map", type=Path, required=True)
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument("--universe-file", type=Path)
    universe.add_argument("--tickers", nargs="+")
    parser.add_argument(
        "--cost-grid-bps",
        nargs="+",
        type=float,
        default=list(DEFAULT_COST_GRID_BPS),
        help="Round-trip cost grid; must include the locked 10 bps primary case.",
    )
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=25_000_000.0)
    parser.add_argument("--min-completeness", type=float, default=0.95)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = _read_table(args.sector_map)
    candidates = (
        _read_universe(args.universe_file)
        if args.universe_file is not None
        else sorted({str(ticker).upper().strip() for ticker in args.tickers})
    )
    eligibility_config = EligibilityConfig(
        min_price=args.min_price,
        min_median_dollar_volume=args.min_dollar_volume,
        min_data_completeness=args.min_completeness,
    )
    result = run_streaming_intraday_research(
        args.data_dir,
        metadata,
        candidates,
        eligibility_config=eligibility_config,
        cost_grid_bps=tuple(args.cost_grid_bps),
        bootstrap_reps=args.bootstrap_reps,
    )
    output_dir = write_streaming_research_artifacts(
        result, args.output_dir or _default_output_dir()
    )
    print(f"Research artifacts: {output_dir}")
    print(
        f"Requested {len(result.requested_tickers)} | "
        f"evaluated {len(result.loaded_candidate_tickers)} | "
        f"signals {len(result.signals)} | primary trades {len(result.trades)} | "
        f"execution rejected {len(result.execution_rejections)}"
    )
    primary = result.day_cluster_stats.loc[
        result.day_cluster_stats["primary_cost_case"]
    ]
    if primary.empty:
        print("No testable primary-cost day clusters.")
    else:
        print(primary.to_string(index=False))
    print(
        "Research-only: no network, R2 mutation, production write, broker action, "
        "order, schedule, or automatic promotion occurred."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
