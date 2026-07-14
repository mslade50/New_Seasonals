"""Build the static, read-only price snapshot used by the private-site
seasonality lab.

The source parquet is only read.  Output is written beneath the caller's
build directory as a small manifest plus one compact binary payload per ticker.
No network clients, credentials, R2 helpers, or production data writers are
imported here.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("ticker", "date", "High", "Low", "Close")


def _ticker_id(ticker: str) -> str:
    """Return a reversible filename-safe identifier for a market symbol."""
    raw = base64.urlsafe_b64encode(ticker.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)


def _write_binary(frame: pd.DataFrame, path: Path) -> None:
    """Write SLB1: magic + count + int32 dates + float32 close + float32 ATR.

    Dates are days since 1970-01-01 and all arrays are little-endian.  The
    columnar 12-bytes/session representation is less than half the uncompressed
    JSON size while retaining the source parquet's practical precision.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    days = ((frame["date"].values.astype("datetime64[D]") - np.datetime64("1970-01-01"))
            .astype("<i4"))
    closes = frame["Close"].to_numpy(dtype="<f4", copy=True)
    atrs = frame["ATR"].to_numpy(dtype="<f4", copy=True)
    count = np.asarray([len(frame)], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(b"SLB1")
        handle.write(count.tobytes())
        handle.write(days.tobytes())
        handle.write(closes.tobytes())
        handle.write(atrs.tobytes())


def prepare_ticker_frame(frame: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Normalize one ticker and reproduce ``pages/user_input.py`` ATR exactly.

    That page uses a simple 14-session rolling mean of true range.  This is
    intentionally different from the Wilder ATR used by the trade sizer.
    """
    out = frame.loc[:, ["date", "High", "Low", "Close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("High", "Low", "Close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = (out.dropna(subset=["date", "High", "Low", "Close"])
           .sort_values("date")
           .drop_duplicates("date", keep="last")
           .reset_index(drop=True))
    if out.empty:
        return out

    previous_close = out["Close"].shift(1)
    upper = pd.concat([out["High"], previous_close], axis=1).max(axis=1)
    lower = pd.concat([out["Low"], previous_close], axis=1).min(axis=1)
    true_range = upper - lower
    out["ATR"] = true_range.rolling(window=atr_window).mean()
    return out


def export_seasonality_snapshot(
    source: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    min_year: int = 2000,
    atr_window: int = 14,
) -> dict:
    """Export static per-ticker inputs without modifying ``source``.

    Existing output files are overwritten in place, but stale files are not
    removed.  The manifest is authoritative, so an old unreferenced payload
    can never become selectable in the UI.
    """
    source_path = Path(source)
    output_path = Path(output_dir)
    prices_path = output_path / "t"

    prices = pd.read_parquet(source_path, columns=list(REQUIRED_COLUMNS))
    missing = [column for column in REQUIRED_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"seasonality source is missing columns: {', '.join(missing)}")

    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices[(prices["ticker"] != "") & prices["date"].notna()]
    prices = prices[prices["date"].dt.year >= int(min_year)]
    if prices.empty:
        raise ValueError("seasonality source has no usable rows")

    ticker_meta: dict[str, dict] = {}
    total_rows = 0
    global_asof: pd.Timestamp | None = None

    for ticker, group in prices.groupby("ticker", sort=True):
        prepared = prepare_ticker_frame(group, atr_window=atr_window)
        if prepared.empty:
            continue
        ticker_id = _ticker_id(ticker)
        _write_binary(prepared, prices_path / f"{ticker_id}.bin")

        start = prepared["date"].iloc[0]
        end = prepared["date"].iloc[-1]
        count = int(len(prepared))
        ticker_meta[ticker] = {
            "id": ticker_id,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "n": count,
            "file": f"t/{ticker_id}.bin",
        }
        total_rows += count
        if global_asof is None or end > global_asof:
            global_asof = end

    if not ticker_meta:
        raise ValueError("seasonality source has no exportable tickers")

    manifest = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "asof": global_asof.strftime("%Y-%m-%d") if global_asof is not None else None,
        "history_floor": f"{int(min_year):04d}-01-01",
        "price_basis": "adjusted OHLCV from master_prices.parquet",
        "encoding": "SLB1 columnar little-endian: int32 epoch-day, float32 close, float32 ATR",
        "atr": {
            "window": int(atr_window),
            "method": "simple rolling mean of true range",
        },
        "ticker_count": len(ticker_meta),
        "row_count": total_rows,
        "tickers": ticker_meta,
    }
    _write_json(manifest, output_path / "manifest.json")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build private-site seasonality inputs locally")
    parser.add_argument("--source", default=root / "data" / "master_prices.parquet")
    parser.add_argument("--out", default=root / "dist" / "data" / "seasonality")
    parser.add_argument("--min-year", type=int, default=2000)
    args = parser.parse_args()
    manifest = export_seasonality_snapshot(args.source, args.out, min_year=args.min_year)
    print(
        "seasonality snapshot: "
        f"{manifest['ticker_count']:,} tickers, {manifest['row_count']:,} rows, "
        f"as of {manifest['asof']}"
    )


if __name__ == "__main__":
    main()
