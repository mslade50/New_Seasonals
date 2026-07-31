"""Engine parity check for the Monthly Weak Close pilot (2026-07-31).

Runs the production engine (precompute -> candidates -> process_signals_fast,
flat sizing, per-strategy cap) with ONLY the new strategy in the book and
compares signal dates / fills / exits against the scratch research backtest
(monthly_weak_close_mr.py limit-entry cell with the 200d-SMA gate).

Expected divergences (conventions, not bugs):
- engine anchors the hold to the SIGNAL date (a T+2 fill holds 4 more days,
  not 5) and applies 2 bps entry slippage;
- engine ATR is its own precompute (vs scratch Wilder ewm) so the limit
  price can differ by pennies -> a marginal fill may flicker;
- engine books nothing before BT_START 2003 and sizes with frag bands
  post-2016 (Size_Mult only, never the fill decision).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

import data_provider
from strategy_config import ACCOUNT_VALUE, STRATEGY_BOOK
from pages.strat_backtester import (
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    load_seasonal_map,
    load_atr_seasonal_map,
)

from monthly_weak_close_mr import load_data as scratch_load, month_signals, run_trades

BT_START = pd.Timestamp("2003-01-01").date()


def scratch_trades() -> pd.DataFrame:
    data = scratch_load()
    rows = []
    for tk in ["SPY", "QQQ"]:
        df = data[tk]
        sma200 = df["Close"].rolling(200).mean()
        for x in run_trades(df, month_signals(df, 0.15), "limit", 5, 2.0):
            sd = pd.Timestamp(x["sig_day"])
            if sd.date() < BT_START:
                continue
            if bool(df["Close"].loc[sd] > sma200.loc[sd]):
                x["ticker"] = tk
                rows.append(x)
    return pd.DataFrame(rows)


def main() -> None:
    book = [s for s in STRATEGY_BOOK if s["name"] == "Monthly Weak Close"]
    assert book, "strategy missing from STRATEGY_BOOK"

    md = data_provider.get_history(["SPY", "QQQ", "^VIX"], start="2000-01-01")
    vix = md.get("^VIX")
    vix_series = None
    if vix is not None and not vix.empty:
        vd = vix.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    sznl_map = load_seasonal_map()
    atr_map = load_atr_seasonal_map()
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_map)
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
    print(f"engine candidates: {len(candidates)}")

    trades = process_signals_fast(candidates, signal_data, processed, book,
                                  ACCOUNT_VALUE, cap_bps=250, flat_sizing=True)
    print(f"engine fills: {len(trades)}")

    sc = scratch_trades()
    print(f"scratch signals (gated, >= {BT_START}): "
          f"{len(set(zip(sc.ticker, pd.to_datetime(sc.sig_day).dt.date)))} fills expected {len(sc)}")

    cand_keys = sorted((c[1], pd.Timestamp(c[0]).date()) for c in candidates)
    print("\nengine candidate signal dates:")
    for t, d in cand_keys:
        print(f"  {t} {d}")

    if len(trades):
        cols = [c for c in ["Ticker", "Date", "Entry Date", "Price", "Exit Price",
                            "Exit Date", "Exit Type", "R_Multiple", "Size_Mult", "PnL"]
                if c in trades.columns]
        print("\nengine trades:")
        print(trades[cols].to_string(index=False))

    print("\nscratch fills:")
    print(sc[["ticker", "sig_day", "entry_day", "entry", "exit", "exit_kind", "ret"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
