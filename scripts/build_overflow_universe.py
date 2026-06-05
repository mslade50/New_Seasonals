"""Build the dynamic overflow universe (Layer C) by screening master_prices.

Reads ``data/master_prices.parquet`` (long format: ticker, date, OHLCV),
computes per-ticker liquidity + volatility metrics over the trailing window,
applies the screen, excludes the liquid tier, and writes:

    data/overflow_universe.parquet   ticker, addv_63d, atr_pct_63d, last_close,
                                     n_bars, last_bar_date, as_of
    data/overflow_universe.txt       one ticker per line (human diff)
    data/overflow_universe_config.json  the thresholds used (auditable)

…then uploads the parquet to R2. Thresholds come from ``overflow_universe.py``
so the loader and the builder agree.

The screen is computed from trailing rolling values only (no latest-only
look-ahead beyond the as-of bar) so the same function can later be reused for
point-in-time backtests (R-T1).

Usage:
    python scripts/build_overflow_universe.py            # build + write + R2 upload
    python scripts/build_overflow_universe.py --dry-run  # compute + print, NO write/upload
    python scripts/build_overflow_universe.py --held-tickers AAA BBB  # flag held drops (R-T6)

This script only READS master_prices and WRITES the overflow universe artifacts.
It never touches Google Sheets or the scan pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PARENT)

from overflow_universe import (  # noqa: E402
    OVERFLOW_UNIVERSE_PATH,
    OVERFLOW_UNIVERSE_R2_KEY,
    OVERFLOW_PRICES_PATH,
    MIN_ADDV_BASE,
    MIN_ATR_PCT,
    MIN_PRICE,
    MIN_BARS,
    FRESHNESS_TD,
    MAX_NAN_FRAC,
    ADDV_WINDOW,
    ATR_WINDOW,
)

MASTER_PRICES_PATH = os.path.join(_PARENT, "data", "master_prices.parquet")
_PRICE_COLS = ["ticker", "date", "High", "Low", "Close", "Volume"]


def _load_prices() -> pd.DataFrame:
    """Load production master_prices UNION the isolated overflow staging cache.

    During the build/backtest phase the new candidate prices live only in
    overflow_prices.parquet; master_prices is untouched. Reading both here lets
    the screen see the full candidate set without polluting production.
    """
    parts = []
    if os.path.exists(MASTER_PRICES_PATH):
        parts.append(pd.read_parquet(MASTER_PRICES_PATH, columns=_PRICE_COLS))
    if os.path.exists(OVERFLOW_PRICES_PATH):
        parts.append(pd.read_parquet(OVERFLOW_PRICES_PATH, columns=_PRICE_COLS))
        print(f"  + overflow staging prices: {OVERFLOW_PRICES_PATH}")
    if not parts:
        raise SystemExit(
            f"No price caches found (looked for {MASTER_PRICES_PATH} and "
            f"{OVERFLOW_PRICES_PATH})."
        )
    df = pd.concat(parts, ignore_index=True)
    # If a ticker appears in both, prefer the most recent bar (drop dup ticker+date).
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df
TXT_PATH = os.path.join(_PARENT, "data", "overflow_universe.txt")
CONFIG_PATH = os.path.join(_PARENT, "data", "overflow_universe_config.json")


def _is_tradeable_equity(tkr: str) -> bool:
    """Exclude non-equity symbols that live in master_prices for the dashboards
    (indices, crypto, futures/FX) — they have meaningless share 'Volume' and
    must never enter the tradeable overflow universe.
    """
    t = str(tkr).upper()
    if t.startswith("^"):                      # indices: ^IXIC, ^DJI, ^GSPC ...
        return False
    if t.endswith("-USD") or t.endswith("-USDT"):  # crypto pairs: BTC-USD, ETH-USD
        return False
    if "=" in t:                               # futures / FX: GC=F, EURUSD=X
        return False
    return True


def _atr_pct_series(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """Per-bar ATR% = (Wilder-free SMA ATR_window / Close) * 100.

    Mirrors indicators.py: ATR = mean(TrueRange, window); ATR_Pct = ATR/Close*100.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    return (atr / close) * 100.0


def screen_universe(
    prices: pd.DataFrame,
    liquid_set: set,
    *,
    as_of: pd.Timestamp,
    allowed: set = None,
    min_addv: float = MIN_ADDV_BASE,
    min_atr_pct: float = MIN_ATR_PCT,
    min_price: float = MIN_PRICE,
    min_bars: int = MIN_BARS,
    freshness_td: int = FRESHNESS_TD,
    max_nan_frac: float = MAX_NAN_FRAC,
    addv_window: int = ADDV_WINDOW,
) -> pd.DataFrame:
    """Pure screen. ``prices`` is long-format (ticker, date, OHLCV).

    Returns a DataFrame of passing names with columns: ticker, addv_63d,
    atr_pct_63d, last_close, n_bars, last_bar_date. Deterministic and free of
    I/O so it can be unit-tested and reused for point-in-time backtests.
    """
    liquid_norm = {str(t).upper().strip().replace(".", "-") for t in liquid_set}
    allowed_norm = (
        {str(t).upper().strip().replace(".", "-") for t in allowed}
        if allowed is not None else None
    )
    fresh_cutoff = as_of - pd.tseries.offsets.BDay(freshness_td)

    df = prices.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    # Only bars at or before the as-of date count (point-in-time safe).
    df = df[df["date"] <= as_of]

    rows = []
    for tkr, grp in df.groupby("ticker", sort=False):
        if tkr in liquid_norm:
            continue
        if not _is_tradeable_equity(tkr):
            continue
        if allowed_norm is not None and tkr not in allowed_norm:
            continue  # only screen the equity candidate sources, not master_prices extras
        grp = grp.sort_values("date")
        n_bars = len(grp)
        if n_bars < min_bars:
            continue
        last = grp.iloc[-1]
        last_date = last["date"]
        if last_date < fresh_cutoff:
            continue  # stale / likely delisted or halted
        last_close = float(last["Close"])
        if not (last_close >= min_price):
            continue

        win = grp.tail(addv_window)
        # Data-quality gate (R-T7): too many NaN closes in the window → skip.
        nan_frac = float(win["Close"].isna().mean())
        if nan_frac > max_nan_frac:
            continue

        dollar_vol = (win["Close"].astype(float) * win["Volume"].astype(float))
        addv = float(dollar_vol.mean())
        if not np.isfinite(addv) or addv < min_addv:
            continue

        atr_pct = _atr_pct_series(grp).tail(addv_window).mean()
        atr_pct = float(atr_pct) if np.isfinite(atr_pct) else np.nan
        if not (atr_pct >= min_atr_pct):
            continue

        rows.append(
            {
                "ticker": tkr,
                "addv_63d": addv,
                "atr_pct_63d": atr_pct,
                "last_close": last_close,
                "n_bars": int(n_bars),
                "last_bar_date": last_date.normalize(),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("addv_63d", ascending=False).reset_index(drop=True)
    return out


def _load_prior_tickers() -> set:
    if not os.path.exists(OVERFLOW_UNIVERSE_PATH):
        return set()
    try:
        prior = pd.read_parquet(OVERFLOW_UNIVERSE_PATH, columns=["ticker"])
        return {str(t).upper().strip().replace(".", "-") for t in prior["ticker"].tolist()}
    except Exception as e:  # noqa: BLE001
        print(f"[diff] could not read prior universe: {e}")
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the dynamic overflow universe.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print only — no parquet write, no R2 upload.")
    ap.add_argument("--held-tickers", nargs="*", default=None,
                    help="Tickers currently held — flag any that drop out (R-T6).")
    ap.add_argument("--as-of", default=None,
                    help="As-of date YYYY-MM-DD (default: max bar date in master_prices).")
    args = ap.parse_args()

    # master_prices UNION the isolated overflow staging cache (production
    # master_prices is never written by this pipeline pre-promote).
    prices = _load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    if prices["date"].dt.tz is not None:
        prices["date"] = prices["date"].dt.tz_localize(None)

    as_of = pd.Timestamp(args.as_of) if args.as_of else prices["date"].max().normalize()

    # Liquid set + equity candidate allow-list — imported here (not at module
    # top) so the screen function stays import-light and testable without
    # strategy_config. The allow-list constrains screening to actual equity
    # sources (CSV_UNIVERSE = sznl_ranks ∪ FMP symbol_master) so dashboard
    # ETFs / indices / crypto sitting in master_prices don't leak into the
    # tradeable universe.
    from strategy_config import LIQUID_PLUS_COMMODITIES, CSV_UNIVERSE
    liquid_set = set(LIQUID_PLUS_COMMODITIES)
    allowed = {str(t).upper().replace(".", "-") for t in CSV_UNIVERSE}
    sm_path = os.path.join(_PARENT, "data", "symbol_master.parquet")
    if os.path.exists(sm_path):
        try:
            _sm = pd.read_parquet(sm_path, columns=["ticker"])["ticker"]
            allowed |= {str(t).upper().replace(".", "-") for t in _sm}
            print(f"  allow-list: CSV_UNIVERSE + symbol_master = {len(allowed):,} symbols")
        except Exception as e:
            print(f"  warn: could not read symbol_master for allow-list: {e}")

    universe = screen_universe(prices, liquid_set, as_of=as_of, allowed=allowed)
    universe = universe.assign(as_of=as_of.normalize())

    print(f"As-of: {as_of.date()}  |  candidates screened: {prices['ticker'].nunique():,}")
    print(f"Overflow universe: {len(universe):,} names pass the screen")
    if not universe.empty:
        print(universe.head(15).to_string(index=False))

    # Diff vs prior + held-name drop flag (R-T6).
    prior = _load_prior_tickers()
    new_set = set(universe["ticker"].tolist())
    added = sorted(new_set - prior)
    dropped = sorted(prior - new_set)
    print(f"\nDelta vs prior: +{len(added)} added, -{len(dropped)} dropped")
    if added[:20]:
        print(f"  added sample:   {added[:20]}")
    if dropped[:20]:
        print(f"  dropped sample: {dropped[:20]}")
    if args.held_tickers:
        held = {str(t).upper().strip().replace(".", "-") for t in args.held_tickers}
        held_dropped = sorted(held & set(dropped))
        if held_dropped:
            print(f"  WARN: HELD names dropping out (review exits): {held_dropped}")

    if args.dry_run:
        print("\n[dry-run] no parquet written, no R2 upload.")
        return

    os.makedirs(os.path.dirname(OVERFLOW_UNIVERSE_PATH), exist_ok=True)
    universe.to_parquet(OVERFLOW_UNIVERSE_PATH, index=False)
    with open(TXT_PATH, "w") as f:
        f.write("\n".join(universe["ticker"].tolist()) + "\n")
    config = {
        "as_of": str(as_of.date()),
        "min_addv_base": MIN_ADDV_BASE,
        "min_atr_pct": MIN_ATR_PCT,
        "min_price": MIN_PRICE,
        "min_bars": MIN_BARS,
        "freshness_td": FRESHNESS_TD,
        "max_nan_frac": MAX_NAN_FRAC,
        "addv_window": ADDV_WINDOW,
        "atr_window": ATR_WINDOW,
        "n_names": int(len(universe)),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote {OVERFLOW_UNIVERSE_PATH} ({len(universe):,} names)")

    try:
        from cache_io import upload_from_local
        upload_from_local(OVERFLOW_UNIVERSE_PATH, OVERFLOW_UNIVERSE_R2_KEY)
    except Exception as e:  # noqa: BLE001
        print(f"[r2 upload] non-fatal error: {e}")


if __name__ == "__main__":
    main()
