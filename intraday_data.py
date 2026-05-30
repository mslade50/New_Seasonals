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
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl
    _POLARS_OK = True
except ImportError:
    _POLARS_OK = False

_ROOT = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(_ROOT, "data", "intraday")

_STALE_AFTER_SECONDS = 18 * 3600
_meta_cache: Optional[pd.DataFrame] = None

# Per-ticker in-memory cache populated by `prefetch_ticker()`.
# _bars_cache[ticker] is the full sorted DataFrame for that ticker.
# _date_index_cache[ticker] maps datetime.date -> (start_row, end_row_exclusive)
# so per-day slicing is an O(1) dict lookup + zero-copy iloc slice.
_bars_cache: dict = {}
_date_index_cache: dict = {}


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


def _read_parquet_fast(path: str) -> pd.DataFrame:
    """Read a parquet via polars when available (3-10x faster than pandas),
    fall back to pandas otherwise. Returns a pandas DataFrame for downstream
    compat (call sites use .iloc indexing)."""
    if _POLARS_OK:
        try:
            return pl.read_parquet(path).to_pandas()
        except Exception:
            pass
    return pd.read_parquet(path)


def _build_date_index(df: pd.DataFrame) -> dict:
    """Group row indices by trading date. Returns {date -> (start, end_excl)}.
    Assumes df is sorted by ts ascending (precondition enforced by caller)."""
    if df.empty:
        return {}
    dates = df["ts"].values.astype("datetime64[D]")
    # Find boundaries where date changes
    change_pts = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.concatenate(([0], change_pts))
    ends = np.concatenate((change_pts, [len(df)]))
    date_keys = dates[starts].astype("datetime64[D]").tolist()
    return {d: (int(s), int(e)) for d, s, e in zip(date_keys, starts, ends)}


def prefetch_ticker(ticker: str, interval: str = "15min") -> bool:
    """Load full intraday parquet for `ticker` into the in-memory cache.

    Call this once at the top of a backtest's per-ticker loop. Subsequent
    `get_intraday_for_date()` calls hit the cache (O(1) lookup) instead of
    re-reading the parquet for every signal day.

    Returns True on success (cache populated), False if the ticker has no
    intraday data (local + R2 both miss) or the parquet is empty.
    """
    ticker = ticker.upper()
    if ticker in _bars_cache:
        return True
    local = _local_path(interval, ticker)
    if not _refresh_from_r2(local, _r2_key(interval, ticker)):
        if not os.path.exists(local):
            return False
    df = _read_parquet_fast(local)
    if df.empty:
        return False
    df = df.sort_values("ts").reset_index(drop=True)
    _bars_cache[ticker] = df
    _date_index_cache[ticker] = _build_date_index(df)
    return True


def get_intraday(ticker: str, interval: str = "15min",
                 start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Return DataFrame[ts, open, high, low, close, volume] sorted by ts.

    Empty DataFrame if the ticker has no data. R2-pulls the parquet if local
    is missing/stale. `start`/`end` are inclusive ISO date strings (or None).
    Uses the cache when populated.
    """
    ticker = ticker.upper()
    if ticker in _bars_cache:
        df = _bars_cache[ticker]
    else:
        local = _local_path(interval, ticker)
        if not _refresh_from_r2(local, _r2_key(interval, ticker)):
            if not os.path.exists(local):
                return pd.DataFrame()
        df = _read_parquet_fast(local)
        if df.empty:
            return df
        df = df.sort_values("ts").reset_index(drop=True)
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def get_intraday_for_date(ticker: str, date, interval: str = "15min") -> pd.DataFrame:
    """Bars for a single trading day. `date` is anything pd.Timestamp eats.

    Fast path: if the ticker is in the prefetch cache, this is a dict lookup
    plus a zero-copy iloc slice (~microseconds). Otherwise falls back to
    re-reading the parquet (slow — milliseconds per call)."""
    ticker = ticker.upper()
    d = pd.Timestamp(date).normalize()
    if ticker in _bars_cache:
        idx = _date_index_cache.get(ticker, {})
        bounds = idx.get(d.date())
        if bounds is None:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        s, e = bounds
        return _bars_cache[ticker].iloc[s:e].reset_index(drop=True)
    # Fallback: legacy slow path
    df = get_intraday(ticker, interval=interval,
                      start=d.isoformat(), end=d.isoformat())
    if df.empty:
        return df
    same_day = df[df["ts"].dt.date == d.date()]
    return same_day.reset_index(drop=True)


def clear_meta_cache():
    """Force the next get_meta() call to re-read from disk/R2."""
    global _meta_cache
    _meta_cache = None


def clear_bars_cache(ticker: Optional[str] = None):
    """Drop prefetched ticker bars from memory. Pass None to clear all."""
    if ticker is None:
        _bars_cache.clear()
        _date_index_cache.clear()
    else:
        ticker = ticker.upper()
        _bars_cache.pop(ticker, None)
        _date_index_cache.pop(ticker, None)
