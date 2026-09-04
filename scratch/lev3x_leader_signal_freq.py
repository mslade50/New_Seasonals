"""3x Leader Gap Fade — % of trading days with zero signals (2026-07-10).

Signal days = distinct dates with >=1 staged candidate (pre-gap, what the
scanner would put on the sheet), and separately dates surviving the T+1
gap gate. Denominator = SPY trading days, shown for the full backtest
window and from the strategy's first-ever signal.
"""
import copy
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, LEV3X_ALL, LEV3X_BULL_EQ
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
)

START = pd.Timestamp("2003-01-01")
UNIVERSE = [t for t in LEV3X_ALL if t not in LEV3X_BULL_EQ]

book_entry = next(s for s in STRATEGY_BOOK if s["name"] == "3x Leader Gap Fade")

staged = copy.deepcopy(book_entry)   # scanner view: no gap gate at signal time
staged["name"] = "staged"
staged["settings"]["use_t1_open_filter"] = False
staged["settings"]["t1_open_filters"] = []
gated = copy.deepcopy(book_entry)    # gap-passed
gated["name"] = "gated"

variants = [staged, gated]
md = data_provider.get_history(UNIVERSE + ["SPY", "^VIX"], start="2000-01-01")
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
processed = precompute_all_indicators(md, variants, sznl_map, vix_series, atr_sznl_map)

spy = md["SPY"].copy()
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_days = pd.DatetimeIndex(spy.index)
spy_days = spy_days[spy_days >= START]

dates = {}
for v in variants:
    cands, _ = generate_candidates_fast(processed, [v], sznl_map, START)
    dates[v["name"]] = sorted({pd.Timestamp(c[0]).normalize() for c in cands})

first_sig = dates["staged"][0]
for label, d0 in (("full window 2003->now", START),
                  (f"since first signal ({first_sig.date()})", first_sig)):
    td = spy_days[spy_days >= d0]
    print(f"\n{label}: {len(td)} trading days")
    for name in ("staged", "gated"):
        ds = [d for d in dates[name] if d >= d0]
        pct = len(ds) / len(td)
        print(f"  {name:<7} signal days: {len(ds):>4}  ({pct:.2%} of days)  "
              f"-> ZERO-signal days: {1 - pct:.2%}")

# gap between signal days (staged), for a feel of the dry spells
sd = pd.Series(dates["staged"])
gaps = sd.diff().dt.days.dropna()
print(f"\nstaged signal-day gaps: median {gaps.median():.0f}d, "
      f"mean {gaps.mean():.0f}d, max {gaps.max():.0f}d "
      f"({sd[gaps.idxmax()-1].date() if gaps.idxmax()-1 in sd.index else '?'} -> "
      f"{sd[gaps.idxmax()].date()})")
per_year = pd.Series(1, index=pd.DatetimeIndex(dates['staged'])).resample('YE').sum()
print("staged signal days per year:",
      {d.year: int(n) for d, n in per_year.items() if n > 0})
