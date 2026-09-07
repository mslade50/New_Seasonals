"""Regenerate corrected ATR-seasonal history from frozen cloud inputs.

This is the migration boundary for ``annual-outcome-cutoff-v2``.  It never
downloads prices and never publishes by itself.  The private-site workflow
materializes the canonical master-price and rank objects from R2, this command
rebuilds only when the canonical rank file is old/incomplete, and
``site_r2_pipeline.py promote-canonical`` performs the conditional R2 replace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_atr_seasonal_ranks import (
    ATR_WINDOW,
    FULL_START_YEAR,
    FWD_WINDOWS,
    SEASONAL_RANK_VERSION,
    compute_ranks_for_year,
    generate_trading_dates,
    prepare_ticker_data,
)
from research.strategy_discovery.contracts import (
    canonical_json,
    sha256_json,
)
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES

RANK_COLUMNS = tuple(f"atr_sznl_{window}d" for window in FWD_WINDOWS)
RECEIPT_SCHEMA = "atr-seasonal-regeneration.v1"


class RegenerationError(RuntimeError):
    """A canonical input cannot support a complete deterministic rebuild."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    return ticker if "-" in ticker else ticker.replace(".", "-")


def required_tickers(existing: pd.DataFrame | None) -> list[str]:
    configured = {normalize_ticker(value) for value in (*CSV_UNIVERSE, *LIQUID_PLUS_COMMODITIES)}
    observed: set[str] = set()
    if existing is not None and "ticker" in existing:
        observed = {normalize_ticker(value) for value in existing["ticker"].dropna().unique()}
    return sorted(configured | observed)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RegenerationError(f"canonical rank parquet is missing: {path}")
    try:
        # Read the full schema.  A projected read would make an object with
        # unexpected columns look contract-compliant.
        return pd.read_parquet(path)
    except Exception as exc:
        raise RegenerationError(f"canonical rank parquet is unreadable: {exc}") from exc


def _load_prices(path: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if not path.is_file():
        raise RegenerationError(f"canonical master-price parquet is missing: {path}")
    columns = ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]
    try:
        raw = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise RegenerationError(f"canonical master-price parquet is unreadable: {exc}") from exc
    if raw.empty:
        raise RegenerationError("canonical master-price parquet is empty")
    raw["ticker"] = raw["ticker"].map(normalize_ticker)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw["ticker"].isin(tickers)].dropna(subset=["date"])
    result: dict[str, pd.DataFrame] = {}
    for ticker, group in raw.groupby("ticker", sort=True):
        frame = group.drop(columns=["ticker"]).set_index("date").sort_index()
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.index = frame.index.normalize()
        frame = frame[~frame.index.duplicated(keep="last")]
        result[ticker] = frame
    return result


def _target_year(existing: pd.DataFrame, now: dt.datetime) -> int:
    # The production object covers through the current calendar year. Never
    # let a corrupt future-dated predecessor expand the rebuild horizon.
    return now.year


def current_health(
    existing: pd.DataFrame,
    *,
    metadata: dict,
    tickers: list[str],
    target_year: int,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if metadata.get("seasonal_rank_version") != SEASONAL_RANK_VERSION:
        problems.append(
            f"version={metadata.get('seasonal_rank_version')!r}; expected {SEASONAL_RANK_VERSION!r}"
        )
    expected_ticker_digest = sha256_json(sorted(tickers))
    expected_metadata = {
        "regeneration_schema": RECEIPT_SCHEMA,
        "ticker_count": len(tickers),
        "ticker_digest": expected_ticker_digest,
        "year_start": FULL_START_YEAR,
        "year_end": target_year,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            problems.append(f"{key}={metadata.get(key)!r}; expected {expected!r}")
    required_columns = {"Date", "ticker", *RANK_COLUMNS}
    if set(existing.columns) != required_columns:
        problems.append("rank columns do not match the production contract")
        return False, problems
    values = existing[list(RANK_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        problems.append("rank values contain missing or nonfinite observations")
    elif ((values < 0) | (values > 100)).any().any():
        problems.append("rank values fall outside 0..100")
    dates = pd.to_datetime(existing["Date"], errors="coerce")
    if dates.isna().any():
        problems.append("rank dates contain invalid observations")
    normalized = existing["ticker"].map(normalize_ticker)
    missing = sorted(set(tickers) - set(normalized))
    if missing:
        problems.append(f"rank coverage is missing {len(missing)} required ticker(s)")
    if not dates.empty:
        maximum_year = int(dates.dt.year.max())
        if maximum_year < target_year:
            problems.append(f"rank history ends before target year {target_year}")
        elif maximum_year > target_year:
            problems.append(f"rank history contains future year {maximum_year}")
        target_dates = pd.DatetimeIndex(generate_trading_dates(target_year)["Date"]).normalize()
        target_frame = pd.DataFrame({"ticker": normalized, "Date": dates})
        target_frame = target_frame[target_frame["Date"].dt.year == target_year]
        expected_dates = set(target_dates)
        observed_by_ticker = {
            ticker: set(group["Date"])
            for ticker, group in target_frame.groupby("ticker", sort=False)
        }
        incomplete_target = [
            ticker
            for ticker in tickers
            if observed_by_ticker.get(ticker, set()) != expected_dates
        ]
        if incomplete_target:
            problems.append(
                f"target-year trading-date coverage is incomplete for {len(incomplete_target)} ticker(s)"
            )
    if existing.assign(_ticker=normalized, _date=dates).duplicated(["_ticker", "_date"]).any():
        problems.append("rank history contains duplicate ticker/date rows")
    return not problems, problems


def rebuild(
    prices_path: Path,
    tickers: list[str],
    years: list[int],
) -> tuple[pd.DataFrame, dict]:
    prices = _load_prices(prices_path, tickers)
    missing = sorted(set(tickers) - set(prices))
    too_short = sorted(ticker for ticker, frame in prices.items() if len(frame) < ATR_WINDOW + 50)
    if missing or too_short:
        detail = []
        if missing:
            detail.append(f"missing={missing[:20]}{'...' if len(missing) > 20 else ''}")
        if too_short:
            detail.append(f"too_short={too_short[:20]}{'...' if len(too_short) > 20 else ''}")
        raise RegenerationError(
            "frozen master prices cannot rebuild every canonical rank ticker: " + "; ".join(detail)
        )

    # Calendar construction is relatively expensive; it is identical for
    # every ticker and therefore belongs outside the ticker loop.
    trading_dates = {year: generate_trading_dates(year) for year in years}
    results: list[pd.DataFrame] = []
    built: list[str] = []
    for index, ticker in enumerate(tickers, start=1):
        if index == 1 or index % 100 == 0:
            print(f"[atr-regen] {index}/{len(tickers)} {ticker}")
        prepared = prepare_ticker_data(prices[ticker])
        if prepared is None:
            raise RegenerationError(f"price history became unusable during preparation: {ticker}")
        ticker_rows = 0
        for year in years:
            ranks = compute_ranks_for_year(prepared, year)
            if ranks is None:
                continue
            dated = trading_dates[year].merge(
                ranks,
                left_on="day_count",
                right_index=True,
                how="left",
            )
            dated["ticker"] = ticker
            dated = dated.drop(columns=["day_count"])
            dated[list(RANK_COLUMNS)] = dated[list(RANK_COLUMNS)].fillna(50.0).round(1)
            ticker_rows += len(dated)
            results.append(dated)
        if ticker_rows == 0:
            raise RegenerationError(f"no rank rows were generated for required ticker: {ticker}")
        built.append(ticker)
    if not results:
        raise RegenerationError("corrected rank regeneration produced no rows")
    output = pd.concat(results, ignore_index=True)
    output["Date"] = pd.to_datetime(output["Date"])
    output = output[["Date", *RANK_COLUMNS, "ticker"]].sort_values(["ticker", "Date"])
    output = output.reset_index(drop=True)
    metadata = {
        "seasonal_rank_version": SEASONAL_RANK_VERSION,
        "regeneration_schema": RECEIPT_SCHEMA,
        "source_prices_sha256": file_sha256(prices_path),
        "ticker_count": len(built),
        "ticker_digest": sha256_json(built),
        "year_start": min(years),
        "year_end": max(years),
    }
    output.attrs.update(metadata)
    return output, metadata


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + ".candidate")
    candidate.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
    os.replace(candidate, path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + ".candidate")
    frame.to_parquet(candidate, index=False)
    check = pd.read_parquet(candidate)
    if check.attrs.get("seasonal_rank_version") != SEASONAL_RANK_VERSION:
        raise RegenerationError("serialized candidate lost the seasonal-rank version metadata")
    if len(check) != len(frame):
        raise RegenerationError("serialized candidate row count changed during round trip")
    os.replace(candidate, path)


def regenerate_if_needed(
    *,
    prices_path: Path,
    ranks_path: Path,
    receipt_path: Path,
    now: dt.datetime | None = None,
    force: bool = False,
) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise RegenerationError("regeneration clock must be timezone-aware")
    existing = _load_existing(ranks_path)
    metadata = dict(existing.attrs)
    tickers = required_tickers(existing)
    target_year = _target_year(existing, now)
    healthy, problems = current_health(
        existing,
        metadata=metadata,
        tickers=tickers,
        target_year=target_year,
    )
    before_sha = file_sha256(ranks_path)
    if healthy and not force:
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "CURRENT",
            "checked_at": now.isoformat(),
            "rank_version": SEASONAL_RANK_VERSION,
            "before_sha256": before_sha,
            "after_sha256": before_sha,
            "prices_sha256": file_sha256(prices_path),
            "ticker_count": len(tickers),
            "target_year": target_year,
            "reasons": [],
        }
        _write_json_atomic(receipt_path, receipt)
        return receipt

    years = list(range(FULL_START_YEAR, target_year + 1))
    frame, build_metadata = rebuild(prices_path, tickers, years)
    healthy_after, after_problems = current_health(
        frame,
        metadata=frame.attrs,
        tickers=tickers,
        target_year=target_year,
    )
    if not healthy_after:
        raise RegenerationError(
            "corrected rank candidate failed its own health check: " + "; ".join(after_problems)
        )
    generated_at = now.isoformat()
    # Keep wall-clock time in the receipt, not in the parquet bytes. The same
    # frozen prices/tickers/years should serialize to the same canonical data.
    _write_parquet_atomic(ranks_path, frame)
    after_sha = file_sha256(ranks_path)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "REGENERATED",
        "checked_at": generated_at,
        "rank_version": SEASONAL_RANK_VERSION,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "prices_sha256": build_metadata["source_prices_sha256"],
        "ticker_count": len(tickers),
        "row_count": len(frame),
        "year_start": years[0],
        "target_year": target_year,
        "reasons": problems if not force else ["operator requested a forced deterministic rebuild", *problems],
        "ticker_digest": build_metadata["ticker_digest"],
    }
    _write_json_atomic(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=ROOT / "data/master_prices.parquet")
    parser.add_argument("--ranks", type=Path, default=ROOT / "atr_seasonal_ranks.parquet")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "data/atr_seasonal_rank_regeneration.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = regenerate_if_needed(
            prices_path=args.prices.resolve(),
            ranks_path=args.ranks.resolve(),
            receipt_path=args.receipt.resolve(),
            force=bool(args.force),
        )
    except (OSError, ValueError, RegenerationError) as exc:
        print(f"ATR SEASONAL REGENERATION BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
