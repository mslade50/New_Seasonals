"""Signals vs fills + same-day de-rating rule for the proposed 3x Bear fade.

User proposal: at staging time, reduce each signal's risk by 10%% per ADDITIONAL
same-day signal in this strategy (2 signals -> 0.9x each, 3 -> 0.8x each, ...).
Signal count is known pre-market; fills are not.

This script:
  1. counts SIGNALS per day (candidates, pre-fill) vs FILLS per day
     (limit Open +0.5 ATR for shorts doesn't always fill),
  2. tabulates multi-signal days that got 0 / 1 / all fills,
  3. simulates the de-rating rule in R-contribution space:
       weighted totR = sum(mult_day * R_trade), mult = 1 - 0.10*(n_sig_day - 1)
     vs baseline (mult=1) and vs a full budget split (mult = 1/n_sig_day),
  4. reports worst-day / worst-episode aggregate exposure under each rule.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LEV3X_ALL
from pages.strat_backtester import (
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast,
)

START = pd.Timestamp("2003-01-01")
BEAR_EQ = ['SPXS', 'SQQQ', 'SDOW', 'TZA',
           'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ']

v = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])
v["name"] = "bear 80+consec1"
v["universe_tickers"] = BEAR_EQ
for i in range(4):
    v["settings"]["perf_filters"][i]["thresh"] = 80.0
v["settings"]["perf_filters"][3]["consecutive"] = 1

md = data_provider.get_history(list(LEV3X_ALL) + ["SPY", "^VIX"], start="2000-01-01")
vd = md["^VIX"].copy()
if isinstance(vd.columns, pd.MultiIndex):
    vd.columns = vd.columns.get_level_values(0)
vd.columns = [c.capitalize() for c in vd.columns]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, [v], sznl_map, vd["Close"], atr_sznl_map)

cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
# candidate tuple: (signal_ts_value, ticker, t_clean, strat_idx, signal_idx)
sig = pd.DataFrame(
    [(pd.Timestamp(c[0]), c[2]) for c in cands], columns=["Date", "Ticker"])
sig["Date"] = sig["Date"].dt.normalize()

tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
tr["Date"] = pd.to_datetime(tr["Date"]).dt.normalize()
tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)

n_sig = sig.groupby("Date").size().rename("n_sig")
n_fill = tr.groupby("Date").size().rename("n_fill")
days = pd.concat([n_sig, n_fill], axis=1).fillna(0).astype(int)

print(f"signal-days: {len(days)}   total signals: {days.n_sig.sum()}   "
      f"total fills: {days.n_fill.sum()} "
      f"(overall fill rate {days.n_fill.sum()/days.n_sig.sum():.0%})")

print("\nsignals-per-day vs fills:")
print(f"{'n_sig':>6} {'days':>5} {'fills: 0':>9} {'1':>4} {'2':>4} {'3+':>4}"
      f" {'avg fill rate':>14}")
for n, g in days.groupby("n_sig"):
    f0 = (g.n_fill == 0).sum()
    f1 = (g.n_fill == 1).sum()
    f2 = (g.n_fill == 2).sum()
    f3 = (g.n_fill >= 3).sum()
    fr = (g.n_fill / g.n_sig).mean()
    print(f"{n:>6} {len(g):>5} {f0:>9} {f1:>4} {f2:>4} {f3:>4} {fr:>13.0%}")

multi = days[days.n_sig >= 2]
print(f"\nmulti-SIGNAL days: {len(multi)}  -> of these, "
      f"0 fills: {(multi.n_fill==0).sum()}, exactly 1 fill: {(multi.n_fill==1).sum()}, "
      f"2+ fills: {(multi.n_fill>=2).sum()}")

# ---- de-rating simulation (R-contribution space, flat risk budget = 1R/trade)
tr = tr.merge(n_sig, left_on="Date", right_index=True, how="left")
tr["mult_user"] = (1.0 - 0.10 * (tr["n_sig"] - 1)).clip(lower=0.1)
tr["mult_split"] = 1.0 / tr["n_sig"]

print("\n" + "=" * 74)
print("DE-RATING SIMULATION (51 filled trades; contribution = mult x R)")
print("=" * 74)
for name, m in [("baseline (1.0x always)", pd.Series(1.0, index=tr.index)),
                ("user rule (-10% per extra signal)", tr["mult_user"]),
                ("full split (1/n)", tr["mult_split"])]:
    contrib = m * tr["R"]
    day = contrib.groupby(tr["Date"]).sum()
    # risk actually deployed on each day (in units of one full position)
    risk_day = m.groupby(tr["Date"]).sum()
    print(f"\n{name}")
    print(f"  totR-contribution: {contrib.sum():+7.1f}   "
          f"avg per unit risk: {contrib.sum()/m.sum():+.3f}")
    print(f"  worst day: {day.min():+.2f}   worst 2-day window: "
          f"{day.rolling(2).sum().min():+.2f}")
    print(f"  max risk deployed in one day: {risk_day.max():.2f} positions "
          f"({risk_day.max()*37.5:.0f} bps effective at 25 nominal)")

# what did the de-rating actually cost/save, day by day, on multi days
print("\nmulti-signal days, baseline vs user rule (fills only):")
mult_days = tr[tr.n_sig >= 2]
for d, g in mult_days.groupby("Date"):
    base = g["R"].sum()
    ruled = (g["mult_user"] * g["R"]).sum()
    print(f"  {d.date()}  n_sig={int(g.n_sig.iloc[0])}  fills={len(g)}  "
          f"baseline={base:+.2f}R  ruled={ruled:+.2f}R  delta={ruled-base:+.2f}R  "
          f"[{','.join(g.Ticker)}]")
