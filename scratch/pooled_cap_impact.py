"""Pooled direction-cap impact probe (2026-07-10).

How much does modeling order_staging's pooled caps (long 500 / short 250 bps,
staged basis) change the ledger? Liquid book only (overflow pass skipped for
runtime — overflow adds OVS/OLV etc. rows to the same pools, so the true
impact is >= this probe). Flat $750k sizing, per-strategy cap 250 in both.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

START = pd.Timestamp("2003-01-01")

book = [s for s in STRATEGY_BOOK]
tickers = set()
for s in book:
    tickers.update(s["universe_tickers"])
tickers.update(["SPY", "^VIX"])

md = data_provider.get_history(sorted(tickers), start="2000-01-01")
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)
cands, sd = generate_candidates_fast(processed, book, sznl_map, START)
print(f"{len(cands)} candidates")

base = process_signals_fast(cands, sd, processed, book, ACCOUNT_VALUE,
                            cap_bps=250, flat_sizing=True)
capped = process_signals_fast(cands, sd, processed, book, ACCOUNT_VALUE,
                              cap_bps=250, flat_sizing=True,
                              max_long_risk_bps=500, max_short_risk_bps=250)

for name, df in (("base (per-strat cap only)", base), ("with pooled 500L/250S", capped)):
    print(f"{name}: {len(df)} trades, PnL ${df['PnL'].sum():,.0f}, "
          f"risk ${df['Risk $'].sum():,.0f}")

m = base.merge(
    capped[["Ticker", "Date", "Strategy", "Risk $", "PnL"]],
    on=["Ticker", "Date", "Strategy"], suffixes=("_b", "_c"), how="inner")
m["scale"] = m["Risk $_c"] / m["Risk $_b"].replace(0, np.nan)
trimmed = m[m["scale"] < 0.999]
print(f"\ntrades trimmed by pooled caps: {len(trimmed)}/{len(m)} "
      f"({len(trimmed)/max(len(m),1):.1%})")
if len(trimmed):
    trimmed = trimmed.copy()
    trimmed["Date"] = pd.to_datetime(trimmed["Date"])
    days = trimmed.groupby(trimmed["Date"].dt.normalize()).agg(
        n=("scale", "count"), min_scale=("scale", "min"),
        dPnL=("PnL_c", "sum"))
    days["dPnL"] -= trimmed.groupby(trimmed["Date"].dt.normalize())["PnL_b"].sum()
    print(f"days with a pooled-cap trim: {len(days)}")
    print(f"total PnL delta on trimmed trades: "
          f"${(trimmed['PnL_c'] - trimmed['PnL_b']).sum():,.0f}")
    print("\nworst-trimmed days:")
    print(days.sort_values("min_scale").head(15).to_string())
