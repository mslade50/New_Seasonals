"""The ONE permitted yfinance pull: OLV liquid universe, 3 years, auto_adjust=True.

Cached to yf_adjusted_3y.parquet (long format) so 03/04 never re-pull.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

PARQUET = HERE / "yf_adjusted_3y.parquet"
META = HERE / "yf_pull_meta.json"
YEARS = 3


def universe() -> list[str]:
    from strategy_config import LIQUID_PLUS_COMMODITIES
    return sorted({t.replace(".", "-") for t in LIQUID_PLUS_COMMODITIES})


def load() -> pd.DataFrame:
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        return df
    import yfinance as yf
    tickers = universe()
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * YEARS + 1)
    print(f"yfinance pull: {len(tickers)} tickers {start} -> {end} auto_adjust=True")
    raw = yf.download(tickers, start=str(start), end=str(end + dt.timedelta(days=1)),
                      auto_adjust=True, group_by="column", threads=True, progress=False)
    frames = []
    failed = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                d = raw.xs(t, level="Ticker", axis=1)
            else:
                d = raw
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.copy()
            d.columns = [str(c).capitalize() for c in d.columns]
            d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if d.empty:
                failed.append(t)
                continue
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d = d.reset_index().rename(columns={d.index.name or "index": "date", "Date": "date"})
            d.insert(0, "ticker", t)
            frames.append(d)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{t}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(PARQUET, index=False)
    META.write_text(json.dumps({
        "pulled_utc": dt.datetime.utcnow().isoformat() + "Z",
        "yfinance_version": yf.__version__,
        "n_tickers_requested": len(tickers), "n_tickers_returned": df["ticker"].nunique(),
        "failed": failed, "start": str(start), "end": str(end),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "auto_adjust": True,
    }, indent=2))
    print(f"pulled {df['ticker'].nunique()} tickers, {len(df)} rows, failed {len(failed)}")
    return df


if __name__ == "__main__":
    load()
