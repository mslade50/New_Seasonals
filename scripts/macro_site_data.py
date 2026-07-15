"""Build the static payload behind the private-site Macro Seasonality tab.

Replicates the table half of ``pages/macro_seasonality.py`` at build time:
per macro ticker, the last price, MA-extension percentile ranks (5/20/50/200
sessions over a trailing 2-year window of adjusted closes) and the ATR
seasonal ranks looked up as-of today from atr_seasonal_ranks.parquet.  The
chart half is rendered in the browser from the same per-ticker binaries the
Seasonality Lab uses, so this module only stamps each row with its bin path.

Both source parquets are only read.  No network clients or writers here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from macro_universe import SECTOR_ETFS, TICKER_INFO
from scripts.seasonality_site_data import _ticker_id

MA_WINDOWS = (5, 20, 50, 200)
SZNL_WINDOWS = (5, 10, 21, 63, 126, 252)
SORT_WINDOWS = (5, 10, 21, 63)
LOOKBACK_SESSIONS = 504  # matches the page's period="2y" yfinance pull


def percentile_rank(series: pd.Series, value: float) -> float | None:
    s = series.dropna().values
    if s.size == 0 or value is None or not np.isfinite(value):
        return None
    return float((s <= value).sum() / s.size * 100.0)


def extension_ranks(close: pd.Series) -> dict:
    """Price + percentile rank of today's distance from each MA, over the
    distances observed across the whole trailing window (mirrors
    ``load_sector_metrics``)."""
    out: dict = {"price": float(close.iloc[-1])}
    for window in MA_WINDOWS:
        ma = close.rolling(window).mean()
        dist = (close - ma) / ma * 100.0
        valid = dist.dropna()
        latest = float(valid.iloc[-1]) if not valid.empty else None
        out[f"r{window}"] = percentile_rank(dist, latest)
    return out


def load_sznl_asof(ranks_path: str | os.PathLike[str], asof: pd.Timestamp) -> dict[str, dict]:
    """{ticker: {'s5': rank, ...}} as-of the given date (last value <= asof)."""
    frame = pd.read_parquet(ranks_path)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    frame = frame[frame["Date"] <= asof.normalize()]
    frame = frame.sort_values("Date").groupby("ticker").tail(1)
    cols = {f"atr_sznl_{w}d": f"s{w}" for w in SZNL_WINDOWS}
    out: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        vals = {}
        for raw, key in cols.items():
            value = getattr(row, raw, None)
            vals[key] = float(value) if value is not None and np.isfinite(value) else None
        out[row.ticker] = vals
    return out


def sort_key(row: dict) -> float:
    devs = [abs(row[f"s{w}"] - 50.0) for w in SORT_WINDOWS if row.get(f"s{w}") is not None]
    return max(devs) if devs else -1.0


def export_macro_snapshot(
    prices_path: str | os.PathLike[str],
    ranks_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    asof: pd.Timestamp | None = None,
) -> dict:
    asof = pd.Timestamp(dt.date.today()) if asof is None else pd.Timestamp(asof)
    tickers = sorted(set(SECTOR_ETFS))

    prices = pd.read_parquet(prices_path, columns=["ticker", "date", "Close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices = prices[prices["ticker"].isin(tickers)]
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices = prices.dropna(subset=["date", "Close"])

    sznl = {}
    sznl_ok = False
    if ranks_path and Path(ranks_path).exists():
        try:
            sznl = load_sznl_asof(ranks_path, asof)
            sznl_ok = True
        except Exception as e:
            print(f"  macro: seasonal ranks unavailable ({e})")

    rows = []
    price_asof = None
    for ticker in tickers:
        info = TICKER_INFO.get(ticker, ("", ""))
        row: dict = {"ticker": ticker, "name": info[0], "ibkr": info[1],
                     "price": None}
        for window in MA_WINDOWS:
            row[f"r{window}"] = None
        group = (prices[prices["ticker"] == ticker]
                 .sort_values("date")
                 .drop_duplicates("date", keep="last")
                 .tail(LOOKBACK_SESSIONS))
        if not group.empty:
            close = group.set_index("date")["Close"]
            row.update(extension_ranks(close))
            row["file"] = f"t/{_ticker_id(ticker)}.bin"
            last = group["date"].iloc[-1]
            if price_asof is None or last > price_asof:
                price_asof = last
        for window in SZNL_WINDOWS:
            row[f"s{window}"] = sznl.get(ticker, {}).get(f"s{window}")
        rows.append(row)

    rows.sort(key=sort_key, reverse=True)
    payload = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "asof": price_asof.strftime("%Y-%m-%d") if price_asof is not None else None,
        "sznl_asof": asof.strftime("%Y-%m-%d"),
        "sznl_available": sznl_ok,
        "rows": rows,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
    return payload


def main() -> None:
    root = Path(_ROOT)
    parser = argparse.ArgumentParser(description="Build private-site macro seasonality payload")
    parser.add_argument("--prices", default=root / "data" / "master_prices.parquet")
    parser.add_argument("--ranks", default=root / "atr_seasonal_ranks.parquet")
    parser.add_argument("--out", default=root / "dist" / "data" / "seasonality" / "macro.json")
    args = parser.parse_args()
    payload = export_macro_snapshot(args.prices, args.ranks, args.out)
    with_prices = sum(1 for r in payload["rows"] if r.get("price") is not None)
    print(f"macro payload: {len(payload['rows'])} tickers ({with_prices} with prices), "
          f"sznl_available={payload['sznl_available']}")


if __name__ == "__main__":
    main()
