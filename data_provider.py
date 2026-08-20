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
# Historical-only former constituents.  This surface is never loaded by the
# live scanner; full-history research opts in explicitly.
SURVIVORSHIP_PATH = os.path.join(_ROOT, "data", "survivorship_prices.parquet")
SURVIVORSHIP_MANIFEST_PATH = os.path.join(
    _ROOT, "data", "survivorship_prices.meta.json"
)

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


def _refresh_survivorship_from_r2_if_needed():
    """Pull the paired historical artifact/manifest when absent or stale.

    Unlike overflow staging, survivorship-enabled callers fail closed after
    this attempt; a missing or mismatched pair must never silently recreate the
    old current-universe backtest.
    """
    need = any(
        (not os.path.exists(path))
        or (time.time() - os.path.getmtime(path) > _STALE_AFTER_SECONDS)
        for path in (SURVIVORSHIP_PATH, SURVIVORSHIP_MANIFEST_PATH)
    )
    if not need:
        return
    try:
        from cache_io import download_to_local
    except Exception:
        return
    try:
        download_to_local("survivorship_prices.parquet", SURVIVORSHIP_PATH)
        download_to_local(
            "survivorship_prices.meta.json", SURVIVORSHIP_MANIFEST_PATH
        )
    except Exception:
        return


_SURVIVORSHIP_VALIDATION_CACHE = {}


def _validated_survivorship_manifest() -> dict:
    _refresh_survivorship_from_r2_if_needed()
    if not os.path.exists(SURVIVORSHIP_PATH) or not os.path.exists(
        SURVIVORSHIP_MANIFEST_PATH
    ):
        raise RuntimeError(
            "survivorship research surface is unavailable; refusing a "
            "current-universe-only historical backtest"
        )
    stamp = (
        os.path.getmtime(SURVIVORSHIP_PATH),
        os.path.getsize(SURVIVORSHIP_PATH),
        os.path.getmtime(SURVIVORSHIP_MANIFEST_PATH),
        os.path.getsize(SURVIVORSHIP_MANIFEST_PATH),
    )
    if _SURVIVORSHIP_VALIDATION_CACHE.get("stamp") == stamp:
        return _SURVIVORSHIP_VALIDATION_CACHE["manifest"]
    from survivorship_contract import validate_survivorship_artifact

    try:
        manifest = validate_survivorship_artifact(
            SURVIVORSHIP_PATH, SURVIVORSHIP_MANIFEST_PATH
        )
    except Exception as exc:
        raise RuntimeError(f"survivorship research surface failed closed: {exc}") from exc
    _SURVIVORSHIP_VALIDATION_CACHE.update(stamp=stamp, manifest=manifest)
    return manifest


def _read_price_source(path, *, wanted=None, start=None, end=None):
    filters = []
    if wanted:
        filters.append(("ticker", "in", sorted(wanted)))
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        return pd.read_parquet(path, filters=filters or None)
    except Exception:
        # Compatibility fallback for an older parquet engine. The predicate is
        # still applied immediately, but current pyarrow builds take the
        # memory-safe pushdown path above.
        frame = pd.read_parquet(path)
        if wanted:
            frame = frame[frame["ticker"].isin(wanted)]
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        return frame


def _load_full(
    include_overflow=False,
    include_survivorship=False,
    *,
    wanted=None,
    start=None,
    end=None,
):
    df = _read_price_source(MASTER_PATH, wanted=wanted, start=start, end=end)
    if include_overflow:
        _refresh_overflow_from_r2_if_needed()
        if os.path.exists(OVERFLOW_PATH):
            try:
                odf = _read_price_source(
                    OVERFLOW_PATH, wanted=wanted, start=start, end=end
                )
                # master wins on any ticker+date overlap (listed first)
                df = pd.concat([df, odf], ignore_index=True).drop_duplicates(
                    subset=["ticker", "date"], keep="first"
                )
            except Exception:
                pass
    if include_survivorship:
        _validated_survivorship_manifest()
        sdf = _read_price_source(
            SURVIVORSHIP_PATH, wanted=wanted, start=start, end=end
        )
        # Current primary sources win on overlap; the historical surface fills
        # only the symbols/dates today's sources no longer carry.
        df = pd.concat([df, sdf], ignore_index=True).drop_duplicates(
            subset=["ticker", "date"], keep="first"
        )
    return df


def get_history(
    tickers=None,
    start=None,
    end=None,
    include_overflow=False,
    include_survivorship=False,
):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} indexed by Date.

    Mirrors the per-ticker df shape produced by yfinance after auto_adjust=True
    (no Adj Close column). Both backtesters consume this shape directly.

    include_overflow=True also unions data/overflow_prices.parquet (the isolated
    staging cache for new overflow names). include_survivorship=True adds the
    separately validated historical-only former-constituent surface and fails
    closed when its paired manifest is unavailable or inconsistent. Both
    default False, so live production callers stay on master_prices only.
    """
    if not has_master():
        return {}
    wanted = (
        {str(t).upper().strip() for t in tickers}
        if tickers is not None else None
    )
    df = _load_full(
        include_overflow=include_overflow,
        include_survivorship=include_survivorship,
        wanted=wanted,
        start=start,
        end=end,
    )
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


def get_survivorship_manifest() -> dict:
    """Return the validated historical-surface manifest (fail closed)."""
    return dict(_validated_survivorship_manifest())


def get_survivorship_tickers() -> set:
    """The complete required catalog, including names primary prices cover."""
    manifest = _validated_survivorship_manifest()
    return {
        str(ticker).upper().strip()
        for ticker in manifest.get("required_tickers", [])
    }


def get_universe(include_overflow=False, include_survivorship=False):
    if not has_master():
        return set()
    paths = [MASTER_PATH]
    if include_overflow:
        _refresh_overflow_from_r2_if_needed()
        if os.path.exists(OVERFLOW_PATH):
            paths.append(OVERFLOW_PATH)
    if include_survivorship:
        _validated_survivorship_manifest()
        paths.append(SURVIVORSHIP_PATH)
    tickers = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["ticker"])
        tickers.update(frame["ticker"].dropna().astype(str).str.upper().str.strip())
    if include_survivorship:
        tickers.update(get_survivorship_tickers())
    return tickers


def get_last_dates(tickers=None):
    if not has_master():
        return {}
    df = pd.read_parquet(MASTER_PATH, columns=["ticker", "date"])
    if tickers is not None:
        df = df[df["ticker"].isin({str(t).upper().strip() for t in tickers})]
    return df.groupby("ticker")["date"].max().to_dict()
