"""Fail-closed validation for a rebuilt ATR seasonal-rank artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atr_seasonal_contract import (  # noqa: E402
    RANK_METHOD_COLUMN,
    RANK_METHOD_VERSION,
    rank_artifact_version_error,
)


WINDOWS = (5, 10, 21, 63, 126, 252)
RANK_COLUMNS = [f"atr_sznl_{window}d" for window in WINDOWS]
REQUIRED_COLUMNS = ["Date", "ticker", *RANK_COLUMNS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    path: Path,
    baseline: Path | None,
    start_year: int,
    end_year: int,
    sources: list[Path] | None = None,
) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"rank artifact is missing or empty: {path}")

    frame = pd.read_parquet(path)
    missing_columns = [column for column in [*REQUIRED_COLUMNS, RANK_METHOD_COLUMN] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("rank artifact has no rows")
    version_error = rank_artifact_version_error(frame)
    if version_error:
        raise ValueError(version_error)

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"invalid Date values: {int(dates.isna().sum())}")
    tickers = frame["ticker"].astype("string").str.upper().str.strip()
    if tickers.isna().any() or tickers.eq("").any():
        raise ValueError("ticker contains null or blank values")
    duplicates = frame.assign(Date=dates, ticker=tickers).duplicated(["ticker", "Date"])
    if duplicates.any():
        raise ValueError(f"duplicate ticker/date rows: {int(duplicates.sum())}")

    ranks = frame[RANK_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = ranks.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("rank columns contain null or non-finite values")
    if ((values < 0) | (values > 100)).any():
        raise ValueError("rank values fall outside [0, 100]")

    present_years = set(dates.dt.year.unique().tolist())
    missing_years = sorted(set(range(start_year, end_year + 1)) - present_years)
    if missing_years:
        raise ValueError(f"artifact is missing target years: {missing_years}")

    output_tickers = set(tickers.tolist())
    baseline_count = 0
    baseline_ticker_years_count = 0
    baseline_sha256 = None
    if baseline is not None:
        if not baseline.is_file():
            raise ValueError(f"baseline artifact is missing: {baseline}")
        baseline_frame = pd.read_parquet(baseline, columns=["ticker", "Date"])
        baseline_frame["ticker"] = baseline_frame["ticker"].astype("string").str.upper().str.strip()
        baseline_frame["year"] = pd.to_datetime(
            baseline_frame["Date"], errors="coerce"
        ).dt.year
        if baseline_frame[["ticker", "year"]].isna().any().any():
            raise ValueError("baseline contains invalid ticker/date values")
        baseline_tickers = set(baseline_frame["ticker"].tolist())
        baseline_count = len(baseline_tickers)
        baseline_sha256 = _sha256(baseline)
        lost = sorted(baseline_tickers - output_tickers)
        if lost:
            raise ValueError(f"rebuilt artifact lost {len(lost)} baseline tickers: {lost[:20]}")
        baseline_ticker_years = set(
            baseline_frame[["ticker", "year"]].itertuples(index=False, name=None)
        )
        output_ticker_years = set(zip(tickers.tolist(), dates.dt.year.tolist()))
        lost_ticker_years = sorted(baseline_ticker_years - output_ticker_years)
        if lost_ticker_years:
            raise ValueError(
                "rebuilt artifact lost "
                f"{len(lost_ticker_years)} baseline ticker/year pairs: {lost_ticker_years[:20]}"
            )
        baseline_ticker_years_count = len(baseline_ticker_years)

    source_sha256 = {}
    for source in sources or []:
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"provenance source is missing or empty: {source}")
        source_sha256[str(source)] = _sha256(source)

    return {
        "method_version": RANK_METHOD_VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA") or "unknown",
        "artifact": path.name,
        "artifact_sha256": _sha256(path),
        "rows": int(len(frame)),
        "tickers": int(len(output_tickers)),
        "baseline_tickers": int(baseline_count),
        "baseline_ticker_years": int(baseline_ticker_years_count),
        "baseline_sha256": baseline_sha256,
        "source_sha256": source_sha256,
        "start_year": int(dates.dt.year.min()),
        "end_year": int(dates.dt.year.max()),
        "rank_columns": RANK_COLUMNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("atr_seasonal_ranks.parquet"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("atr_seasonal_ranks.meta.json"))
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    manifest = validate(
        args.path, args.baseline, args.start_year, args.end_year, sources=args.source,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
