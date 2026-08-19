"""C1/C2 round 1, CORRECTED trigger definition.

a1_c1c2_round1.py used pitch_lab.pct_rank(spread, 252), which computes the
252-day PERCENT CHANGE of a difference series that crosses zero -- garbage.
This script ranks the daily spread LEVEL in a trailing 252-session window,
which is the only lookahead-free version of "99th percentile one-day gap".

Everything else is unchanged: one cell, both signs, per-leg attribution,
beta-neutral form, magnitude gradient at today's reading.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

TK = ["XLV", "XLK", "SPY", "QQQ"]
px = close_panel(TK)
r1 = px.pct_change()
spread = (r1["XLV"] - r1["XLK"]).dropna()
today = spread.index[-1]
live = spread.loc[today]

trail = spread.rolling(252).rank(pct=True) * 100.0
print(f"live spread {today.date()} = {100*live:+.3f}pp")
print(f"  full-sample pctile {100*(spread < live).mean():.2f}")
print(f"  TRAILING-252 pctile {trail.loc[today]:.2f}  <-- the tradeable one")
print(f"  trailing-252 max = {100*spread.iloc[-252:].max():+.3f}pp on "
      f"{spread.iloc[-252:].idxmax().date()}")

masks = {
    "trail252 rank>=99.5": (trail >= 99.5),
    "trail252 rank>=99": (trail >= 99.0),
    "trail252 rank>=97.5": (trail >= 97.5),
    "trail252 rank>=95": (trail >= 95.0),
    "abs >= 3.0pp": (spread >= 0.030),
    "abs >= 3.5pp": (spread >= 0.035),
    "abs >= 4.07pp (today)": (spread >= live),
}
print()
for k, m in masks.items():
    mm = m.reindex(px.index, fill_value=False)
    fires = bool(mm.loc[today])
    print(f"  {k:24s} n_days={int(mm.sum()):5d}   FIRES TODAY: {fires}")

BASE = masks["trail252 rank>=99"].reindex(px.index, fill_value=False)

for h in (1, 3, 5, 10):
    battery(px, BASE, [("XLV", 1.0), ("XLK", -1.0)], h,
            "C1 CELL (corrected): XLV-XLK, trailing-252 rank>=99 one-day gap",
            cost_bps=2.0,
            variants={k: v.reindex(px.index, fill_value=False)
                      for k, v in masks.items()},
            event_kinds=("cpi", "fomc"))

print("\n\n########## PER-LEG ATTRIBUTION (episodes) ##########")
rows = []
for h in (1, 2, 3, 5, 10):
    ret_sp = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    valid = ret_sp.dropna().index
    sig = px.index[BASE.values].intersection(valid)
    epi = declusters(sig, h, valid)
    row = {"h": h, "n_epi": len(epi)}
    for tkr in ("XLV", "XLK", "SPY"):
        leg = fwd_lag(px[tkr], h, 1)
        row[f"{tkr}_cond"] = round(100 * leg.loc[epi].mean(), 3)
        row[f"{tkr}_base"] = round(100 * leg.dropna().mean(), 3)
        row[f"{tkr}_exc"] = round(100 * (leg.loc[epi].mean()
                                         - leg.dropna().mean()), 3)
    row["spread_cond"] = round(100 * ret_sp.loc[epi].mean(), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n\n########## BETA-NEUTRAL vs EQUAL-DOLLAR ##########")
beta = r1["XLV"].rolling(252).cov(r1["XLK"]) / r1["XLK"].rolling(252).var()
print(f"live PIT beta(XLV~XLK) = {beta.loc[today]:.3f}   "
      f"median hist {beta.median():.3f}   "
      f"mean over trigger days {beta[BASE.values].mean():.3f}")
for h in (1, 3, 5, 10):
    ret_eq = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    ret_bn = fwd_lag(px["XLV"], h, 1) - beta * fwd_lag(px["XLK"], h, 1)
    valid = ret_eq.dropna().index.intersection(ret_bn.dropna().index)
    sig = px.index[BASE.values].intersection(valid)
    epi = declusters(sig, h, valid)
    show([summarize(ret_eq.loc[epi].values, f"h={h} eq-$ COND"),
          summarize(ret_eq.loc[valid].values, f"h={h} eq-$ all days"),
          summarize(ret_bn.loc[epi].values, f"h={h} beta-neutral COND"),
          summarize(ret_bn.loc[valid].values, f"h={h} beta-neutral all days")],
         f"beta-neutral h={h}")

print("\n\n########## MAGNITUDE GRADIENT INSIDE THE TRIGGER SET ##########")
print("the 'loud state' rule: today is a 99.3rd-pctile print, so a pooled")
print("mean over a 95th-pctile set is not what this trade is entitled to.")
for h in (3, 5, 10):
    ret_sp = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    valid = ret_sp.dropna().index
    wide = px.index[(spread.reindex(px.index) >= 0.020).fillna(False).values]
    wide = wide.intersection(valid)
    epi = declusters(wide, h, valid)
    mag = spread.loc[epi].values
    y = ret_sp.loc[epi].values
    b1, b0 = np.polyfit(mag, y, 1)
    fit_today = b0 + b1 * live
    rr = np.corrcoef(mag, y)[0, 1]
    print(f"h={h:2d}  N_epi={len(epi):3d}  slope={b1:+.3f} (spread frac -> ret frac)  "
          f"corr={rr:+.3f}  fitted C1 return at today's {100*live:+.2f}pp = "
          f"{100*fit_today:+.3f}%")
    # bucketed
    q = pd.qcut(pd.Series(mag), 3, labels=["low", "mid", "high"])
    for lab, g in pd.Series(y).groupby(q.values, observed=True):
        print(f"      {lab:5s} spread bucket N={len(g):3d} mean C1 "
              f"{100*g.mean():+.3f}%  hit {100*(g>0).mean():.1f}%")

print("\n\n########## WHAT KIND OF DAY IS THE TRIGGER? ##########")
sig = px.index[BASE.values]
print(f"SPY same-day return on trigger days: mean {100*r1['SPY'].loc[sig].mean():+.3f}%  "
      f"median {100*r1['SPY'].loc[sig].median():+.3f}%  "
      f"frac down {100*(r1['SPY'].loc[sig] < 0).mean():.1f}%")
print(f"  all days: mean {100*r1['SPY'].mean():+.3f}%  frac down "
      f"{100*(r1['SPY'] < 0).mean():.1f}%")
print(f"  live 2026-08-18 SPY {100*r1['SPY'].loc[today]:+.3f}%")
sma200 = px["SPY"].rolling(200).mean()
above = px["SPY"] > sma200
print(f"trigger days above SPY 200d {100*above.loc[sig].mean():.1f}% vs base "
      f"{100*above.dropna().mean():.1f}%")
