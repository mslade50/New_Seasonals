"""Whole-book signal frequency: % of trading days with ZERO staged signals
(2026-07-10). Liquid book (all 14 strategies, native universes), signals =
candidates post earnings-blackout (what daily_scan would stage), NOT fills.
Overflow tier excluded (same strategies on more tickers — would only raise
the signal-day count).
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK
from earnings_filter import load_earnings_dates_map, in_blackout
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
)

START = pd.Timestamp("2003-01-01")

book = list(STRATEGY_BOOK)
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
cands, _ = generate_candidates_fast(processed, book, sznl_map, START)
print(f"{len(cands)} raw candidates")

# earnings blackout, mirroring the engine pre-pass / daily_scan inline drop
eb = {i: s['execution'].get('earnings_blackout_td')
      for i, s in enumerate(book) if s.get('execution', {}).get('earnings_blackout_td')}
emap = load_earnings_dates_map() if eb else {}
rows = []
for c in cands:
    sig_ts, tkr, t_clean, strat_idx, _ = c
    w = eb.get(strat_idx)
    if w and emap and in_blackout(pd.Timestamp(sig_ts), emap.get(t_clean.upper()), window=w):
        continue
    rows.append((pd.Timestamp(sig_ts).normalize(), book[strat_idx]["name"]))
df = pd.DataFrame(rows, columns=["date", "strategy"])
print(f"{len(df)} staged signals post-blackout")

spy = md["SPY"].copy()
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
td = pd.DatetimeIndex(spy.index)
td = td[td >= START]

sig_days = df["date"].nunique()
print(f"\ntrading days {START.date()} -> {td.max().date()}: {len(td)}")
print(f"days with >=1 signal: {sig_days} ({sig_days/len(td):.1%})")
print(f"days with ZERO signals: {len(td) - sig_days} ({1 - sig_days/len(td):.1%})")

per_day = df.groupby("date").size()
print(f"\nsignals per signal-day: median {per_day.median():.0f}, "
      f"mean {per_day.mean():.1f}, p90 {per_day.quantile(0.9):.0f}, "
      f"max {per_day.max()} ({per_day.idxmax().date()})")
for k in (1, 2, 3, 5, 10):
    n = (per_day >= k).sum()
    print(f"  days with >={k:>2} signals: {n:>4} ({n/len(td):.1%} of all days)")

# per-year zero-signal %
print("\nper-year % of days with zero signals:")
td_s = pd.Series(1, index=td)
for y in sorted(set(td.year)):
    tdy = (td.year == y).sum()
    sdy = df[df["date"].dt.year == y]["date"].nunique()
    print(f"  {y}: {1 - sdy/tdy:.0%}  ({sdy}/{tdy} days signaled)")

# which strategies carry the signal days
print("\nsignal-days by strategy (days where it fired at least once):")
for s, g in sorted(df.groupby("strategy"), key=lambda x: -x[1]["date"].nunique()):
    print(f"  {s:<32} {g['date'].nunique():>4} days, {len(g):>5} signals")
