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
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(_ROOT, "data", "master_prices.parquet")

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
    """
    try:
        from cache_io import is_configured, download_to_local
    except ImportError:
        return
    if not is_configured():
        return
    needs_pull = False
    if not os.path.exists(MASTER_PATH):
        needs_pull = True
    else:
        age = time.time() - os.path.getmtime(MASTER_PATH)
        if age > _STALE_AFTER_SECONDS:
            needs_pull = True
    if needs_pull:
        download_to_local("master_prices.parquet", MASTER_PATH)


def has_master():
    _refresh_from_r2_if_needed()
    return os.path.exists(MASTER_PATH)


def _load_full():
    return pd.read_parquet(MASTER_PATH)


def get_history(tickers=None, start=None, end=None):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} indexed by Date.

    Mirrors the per-ticker df shape produced by yfinance after auto_adjust=True
    (no Adj Close column). Both backtesters consume this shape directly.
    """
    if not has_master():
        return {}
    df = _load_full()
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
