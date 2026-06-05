"""Single source of truth for OHLCV data — both backtesters read from here.

Data lives in data/master_prices.parquet (long format: ticker, date, OHLCV).
Build with scripts/build_master_prices.py; update daily with scripts/update_master_prices.py.
Audit with scripts/audit_master_prices.py.

When R2 is configured (R2_* env vars / .env) the parquet is auto-pulled from
the seasonals-cache bucket on first use if it's missing locally, and refreshed
when the local copy is older than ~18h. This lets Streamlit Cloud (and any
fresh checkout) read prices without rebuilding from yfinance.
"""
import os
import time
from typing import Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(_ROOT, "data", "master_prices.parquet")
# Isolated staging price cache for the new overflow candidate tickers. Only
# unioned in when a caller explicitly passes include_overflow=True (the
# backtester does, for the "Overflow (dynamic)" universe). Production callers
# (daily_portfolio_report) leave it False, so master_prices is the only source.
OVERFLOW_PATH = os.path.join(_ROOT, "data", "overflow_prices.parquet")

# Last reason _refresh_from_r2_if_needed bailed out without producing a fresh
# parquet (missing cache_io, missing creds, boto3 exception, etc.). The
# Streamlit pages read this via `last_r2_error()` and surface it in the
# "master parquet not found" error so the underlying R2 failure is visible.
_LAST_R2_ERROR: Optional[str] = None


def last_r2_error() -> Optional[str]:
    """Return the most recent R2 refresh failure reason, or None if the last
    attempt succeeded / was skipped because the local cache was fresh."""
    return _LAST_R2_ERROR

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Staleness threshold matches analyst_grades / earnings — the GHA updater
# writes R2 nightly around 22:00 UTC, so anything >18h old on disk means
# we should re-pull (the local Task Scheduler updater is disabled per the
# Phase-2 GHA migration, so R2 is the only source of fresh data).
_STALE_AFTER_SECONDS = 18 * 3600


def _refresh_from_r2_if_needed():
    """Pull master_prices.parquet from R2 when the local copy is missing or stale.

    Local copy is stale when its mtime is older than _STALE_AFTER_SECONDS.
    Fresh checkouts pull on first use; subsequent reruns within the window
    use the local file. R2 isn't queried at all when creds aren't set —
    callers fall back to the local file (or fail closed via has_master).

    Failure reasons are stashed in module-level `_LAST_R2_ERROR` so the
    calling layer (Streamlit) can surface them instead of the generic
    "master parquet not found" message.
    """
    global _LAST_R2_ERROR
    _LAST_R2_ERROR = None
    try:
        from cache_io import (
            is_configured,
            download_to_local,
            last_download_error,
            diagnose_creds,
        )
    except ImportError as e:
        _LAST_R2_ERROR = f"cache_io import failed: {e}"
        print(f"[data_provider] {_LAST_R2_ERROR}")
        return
    if not is_configured():
        _LAST_R2_ERROR = (
            "R2 credentials not present. "
            f"Diagnostics: {diagnose_creds()}. "
            "Local: set R2_* in .env. "
            "Streamlit Cloud: paste R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / "
            "R2_SECRET_ACCESS_KEY / R2_BUCKET into Manage app -> Settings "
            "-> Secrets as top-level TOML keys."
        )
        print(f"[data_provider] {_LAST_R2_ERROR}")
        return
    needs_pull = False
    if not os.path.exists(MASTER_PATH):
        needs_pull = True
        reason = "local cache missing"
    else:
        age = time.time() - os.path.getmtime(MASTER_PATH)
        if age > _STALE_AFTER_SECONDS:
            needs_pull = True
            reason = f"local cache stale ({age/3600:.1f}h > {_STALE_AFTER_SECONDS/3600:.0f}h)"
    if needs_pull:
        print(f"[data_provider] pulling master_prices.parquet from R2 ({reason})")
        ok = download_to_local("master_prices.parquet", MASTER_PATH)
        if not ok:
            _LAST_R2_ERROR = (
                last_download_error()
                or "R2 download returned False (unknown error - check cache_io stderr)"
            )


def has_master():
    _refresh_from_r2_if_needed()
    return os.path.exists(MASTER_PATH)


def _refresh_overflow_from_r2_if_needed():
    """Best-effort pull of overflow_prices.parquet from R2 (for Streamlit Cloud).
    Silent no-op if cache_io/creds are absent or the local copy is fresh."""
    try:
        from cache_io import download_to_local
    except Exception:
        return
    need = (not os.path.exists(OVERFLOW_PATH)) or (
        time.time() - os.path.getmtime(OVERFLOW_PATH) > _STALE_AFTER_SECONDS
    )
    if need:
        try:
            download_to_local("overflow_prices.parquet", OVERFLOW_PATH)
        except Exception:
            pass


def _load_full(include_overflow=False):
    df = pd.read_parquet(MASTER_PATH)
    if include_overflow:
        _refresh_overflow_from_r2_if_needed()
        if os.path.exists(OVERFLOW_PATH):
            try:
                odf = pd.read_parquet(OVERFLOW_PATH)
                # master wins on any ticker+date overlap (listed first)
                df = pd.concat([df, odf], ignore_index=True).drop_duplicates(
                    subset=["ticker", "date"], keep="first"
                )
            except Exception:
                pass
    return df


def get_history(tickers=None, start=None, end=None, include_overflow=False):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} indexed by Date.

    Mirrors the per-ticker df shape produced by yfinance after auto_adjust=True
    (no Adj Close column). Both backtesters consume this shape directly.

    include_overflow=True also unions data/overflow_prices.parquet (the isolated
    staging cache for new overflow names). Default False keeps production callers
    on master_prices only.
    """
    if not has_master():
        return {}
    df = _load_full(include_overflow=include_overflow)
    if tickers is not None:
        wanted = {str(t).upper().strip() for t in tickers}
        df = df[df["ticker"].isin(wanted)]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    out = {}
    for t, g in df.groupby("ticker", sort=False):
        g = g.drop(columns=["ticker"]).set_index("date").sort_index()
        g.index.name = "Date"
        # Cast back to float64 so consumers see the same dtype yfinance returns;
        # the parquet stores float32 for compactness only.
        for c in ["Open", "High", "Low", "Close"]:
            if c in g.columns:
                g[c] = g[c].astype("float64")
        out[t] = g[_OHLCV_COLS]
    return out


def get_universe():
    if not has_master():
        return set()
    df = pd.read_parquet(MASTER_PATH, columns=["ticker"])
    return set(df["ticker"].unique())


def get_last_dates(tickers=None):
    if not has_master():
        return {}
    df = pd.read_parquet(MASTER_PATH, columns=["ticker", "date"])
    if tickers is not None:
        df = df[df["ticker"].isin({str(t).upper().strip() for t in tickers})]
    return df.groupby("ticker")["date"].max().to_dict()
