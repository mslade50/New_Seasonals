"""
Intraday data loader — single source of truth for 15min bars used by the
backtester's intraday Day Trade mode.

Storage layout
--------------
Local: data/intraday/{TICKER}_{interval}.parquet
R2:    intraday/{interval}/{TICKER}.parquet
       intraday/{interval}/_meta.parquet  (index)

The first call to `available_tickers()` or `get_intraday()` lazily refreshes
local files from R2 when missing or stale (>18h old, matching the
master_prices convention). Subsequent calls within the staleness window
read from disk only.

Bars are regular-session only (09:30-15:45 ET, 26 bars/day for 15min) and
are not adjusted for splits/dividends — comparison vs yfinance auto_adjust=False
showed median drift of ~0.02% on close prices.
"""
import os
import time
from typing import Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(_ROOT, "data", "intraday")

_STALE_AFTER_SECONDS = 18 * 3600
_meta_cache: Optional[pd.DataFrame] = None


def _meta_local_path(interval: str) -> str:
    return os.path.join(INTRADAY_DIR, "_meta.parquet")


def _r2_key(interval: str, ticker: Optional[str] = None) -> str:
    if ticker is None:
        return f"intraday/{interval}/_meta.parquet"
    return f"intraday/{interval}/{ticker}.parquet"


def _local_path(interval: str, ticker: str) -> str:
    return os.path.join(INTRADAY_DIR, f"{ticker}_{interval}.parquet")


def _refresh_from_r2(local_path: str, r2_key: str) -> bool:
    """Pull from R2 when local copy is missing or stale. Quiet if R2 absent."""
    try:
        from cache_io import is_configured, download_to_local
    except ImportError:
        return False
    if not is_configured():
        return False
    needs_pull = not os.path.exists(local_path) or (
        time.time() - os.path.getmtime(local_path) > _STALE_AFTER_SECONDS
    )
    if not needs_pull:
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return download_to_local(r2_key, local_path)


def get_meta(interval: str = "15min") -> pd.DataFrame:
    """Return the meta index (ticker, first_ts, last_ts, n_bars, ...).

    Empty DataFrame if no local meta and R2 unavailable.
    """
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    local = _meta_local_path(interval)
    _refresh_from_r2(local, _r2_key(interval))
    if not os.path.exists(local):
        _meta_cache = pd.DataFrame()
        return _meta_cache
    _meta_cache = pd.read_parquet(local)
    return _meta_cache


def available_tickers(interval: str = "15min") -> set:
    """Set of tickers that have intraday data at this interval."""
    meta = get_meta(interval)
    if meta.empty:
        return set()
    return set(meta["ticker"].astype(str).str.upper())


def has_intraday(ticker: str, interval: str = "15min") -> bool:
    return ticker.upper() in available_tickers(interval)


def get_intraday(ticker: str, interval: str = "15min",
                 start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Return DataFrame[ts, open, high, low, close, volume] sorted by ts.

    Empty DataFrame if the ticker has no data. R2-pulls the parquet if local
    is missing/stale. `start`/`end` are inclusive ISO date strings (or None).
    """
    ticker = ticker.upper()
    local = _local_path(interval, ticker)
    if not _refresh_from_r2(local, _r2_key(interval, ticker)):
        if not os.path.exists(local):
            return pd.DataFrame()
    df = pd.read_parquet(local)
    if df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def get_intraday_for_date(ticker: str, date, interval: str = "15min") -> pd.DataFrame:
    """Bars for a single trading day. `date` is anything pd.Timestamp eats."""
    d = pd.Timestamp(date).normalize()
    df = get_intraday(ticker, interval=interval,
                      start=d.isoformat(), end=d.isoformat())
    if df.empty:
        return df
    # Filter to exactly this date (start/end above is inclusive of next-day buffer)
    same_day = df[df["ts"].dt.date == d.date()]
    return same_day.reset_index(drop=True)


def clear_meta_cache():
    """Force the next get_meta() call to re-read from disk/R2."""
    global _meta_cache
    _meta_cache = None
