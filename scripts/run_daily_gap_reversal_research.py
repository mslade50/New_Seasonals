"""Run the research-only daily-OHLC asymmetric gap-reversal screen."""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday.gap_reversal_daily import (
    DEFAULT_COST_GRID_BPS,
    freeze_universe,
    normalize_daily_prices,
    run_daily_gap_reversal_research,
    write_daily_gap_research_artifacts,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_universe(path: Path) -> tuple[list[str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"universe file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".txt", ".list"}:
        values = path.read_text(encoding="utf-8").splitlines()
        return values, len(values)
    else:
        raise ValueError("universe file must be CSV, parquet, TXT, or LIST")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "ticker" not in columns:
        raise ValueError("universe table must contain a ticker column")
    values = frame[columns["ticker"]].tolist()
    return values, len(frame)


def _default_output_dir() -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts" / "daily_gap_reversal_research" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimistic, research-only daily-OHLC gap-reversal screen. Reads one "
            "explicit local long-format parquet and never downloads, uploads, stages, "
            "places orders, or promotes a strategy."
        )
    )
    parser.add_argument(
        "--prices",
        type=Path,
        required=True,
        help="Long-format parquet with ticker,date,Open,High,Low,Close,Volume.",
    )
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument(
        "--universe-file",
        type=Path,
        help="Frozen CSV/parquet/TXT universe; duplicate ticker rows are deduped.",
    )
    universe.add_argument(
        "--all-tickers",
        action="store_true",
        help="Freeze all unique tickers in the supplied price parquet.",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Last completed session to admit (YYYY-MM-DD); required to exclude partial current days.",
    )
    parser.add_argument(
        "--cost-grid-bps",
        nargs="+",
        type=float,
        default=list(DEFAULT_COST_GRID_BPS),
        help="Round-trip cost grid; must include the locked 10 bps primary case.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    price_path = args.prices.resolve()
    if not price_path.is_file() or price_path.suffix.lower() != ".parquet":
        raise FileNotFoundError(f"--prices must be an existing parquet: {price_path}")
    price_hash = _sha256_file(price_path)
    raw = pd.read_parquet(price_path)
    normalized, original_row_count = normalize_daily_prices(raw, as_of=args.as_of)
    available = sorted(normalized["ticker"].unique())

    if args.universe_file is not None:
        universe_path = args.universe_file.resolve()
        source_tickers, source_row_count = _read_universe(universe_path)
        universe_hash = _sha256_file(universe_path)
        universe_source = "explicit_file"
    else:
        universe_path = None
        source_tickers = available
        source_row_count = len(source_tickers)
        universe_hash = None
        universe_source = "all_tickers_in_price_input"
    frozen = freeze_universe(
        source_tickers,
        available,
        source_row_count=source_row_count,
    )
    result = run_daily_gap_reversal_research(
        normalized,
        frozen,
        as_of=args.as_of,
        original_row_count=original_row_count,
        cost_grid_bps=tuple(args.cost_grid_bps),
        bootstrap_reps=args.bootstrap_reps,
    )
    provenance = {
        "price_input_path": str(price_path),
        "price_input_sha256": price_hash,
        "price_input_size_bytes": price_path.stat().st_size,
        "universe_source": universe_source,
        "universe_file_path": str(universe_path) if universe_path is not None else None,
        "universe_file_sha256": universe_hash,
    }
    target = write_daily_gap_research_artifacts(
        result,
        args.output_dir or _default_output_dir(),
        input_provenance=provenance,
    )
    print(f"Daily gap-reversal research bundle: {target}")
    print(
        result.primary_stats[
            [
                "arm_id",
                "n_candidate_orders",
                "n_selected_orders",
                "n_selected_fills",
                "mean_session_return",
                "holm_p_value_primary",
            ]
        ].to_string(index=False)
    )
    print("Daily OHLC range-touch results are optimistic and cannot override a causal intraday test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
