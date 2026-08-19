"""SUPERSEDED BY a1b_c1c2_fixed.py -- the trigger in this file is WRONG.
pitch_lab.pct_rank(spread, 252) ranks the 252-day PERCENT CHANGE of a
difference series that crosses zero, not the spread level.  Kept on disk
because the per-leg attribution and beta-neutral blocks are still valid
reads on the (garbage-selected) day set, and because the corrected run
reversed the sign at h=3, which is itself part of the definition-
fragility evidence.

C1/C2 round 1: the XLV-minus-XLK one-day rotation cell, measured ONCE.

C1 = long XLV / short XLK (continuation).  C2 = the same cell with the sign
flipped (snap-back).  They are one measurement; the sign of the mean decides
which of the two is even a candidate, and neither is entitled to a "pick the
positive sign" report.

Registry trap this script exists to spring: "sector-vs-index pairs on a
crowding or leadership trigger" (2026-08-07) and the 2026-08-18 EFA/SPY leg
attribution finding.  So EVERY spread number here is accompanied by each
leg's excess over its OWN unconditional drift.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["XLV", "XLK", "SPY", "QQQ"]
px = close_panel(TK)
r1 = px.pct_change()

spread = (r1["XLV"] - r1["XLK"]).dropna()
today = spread.index[-1]
live = spread.loc[today]
print(f"live spread {today.date()}: {100*live:+.3f}pp")
print(f"full-sample pctile: {100*(spread < live).mean():.2f}  (N={len(spread)})")
tr = pct_rank(spread, 252)
print(f"trailing-252 pctile today: {tr.loc[today]:.1f}")

# ---------------------------------------------------------------------------
# trigger definitions.  The tradeable one is a TRAILING percentile (no
# lookahead).  The absolute-pp ladder is the definition-neighbour check.
# ---------------------------------------------------------------------------
masks = {
    "trail252 pctile>=99": (tr >= 99.0),
    "trail252 pctile>=97.5": (tr >= 97.5),
    "trail252 pctile>=95": (tr >= 95.0),
    "abs >= 3.0pp": (spread >= 0.030),
    "abs >= 3.5pp": (spread >= 0.035),
    "abs >= 4.07pp (today)": (spread >= live),
}
for k, m in masks.items():
    mm = m.reindex(px.index, fill_value=False)
    print(f"  {k:26s} n_days={int(mm.sum()):4d}")

BASE = masks["trail252 pctile>=99"].reindex(px.index, fill_value=False)

for h in (1, 3, 5, 10):
    battery(px, BASE, [("XLV", 1.0), ("XLK", -1.0)], h,
            f"C1 CELL: XLV-XLK after trailing-252 pctile>=99 one-day gap",
            cost_bps=2.0,
            variants={k: v.reindex(px.index, fill_value=False)
                      for k, v in masks.items()},
            event_kinds=("cpi", "fomc"))

# ---------------------------------------------------------------------------
# PER-LEG ATTRIBUTION -- the registry's #1 kill route for this shape.
# ---------------------------------------------------------------------------
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
        base = leg.dropna()
        cond = leg.loc[epi]
        row[f"{tkr}_cond"] = round(100 * cond.mean(), 3)
        row[f"{tkr}_base"] = round(100 * base.mean(), 3)
        row[f"{tkr}_exc"] = round(100 * (cond.mean() - base.mean()), 3)
    row["spread_cond"] = round(100 * ret_sp.loc[epi].mean(), 3)
    row["spread_base"] = round(100 * ret_sp.dropna().mean(), 3)
    row["exc_check"] = round(row["XLV_exc"] - row["XLK_exc"], 3)
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
print("\nRead: if XLV_exc ~ 0 and the whole 'spread' is -XLK_exc, this is a "
      "short-tech bet wearing a pair-trade costume.")

# ---------------------------------------------------------------------------
# beta-neutral form.  hedge ratio = trailing-252 OLS beta of XLV on XLK,
# point-in-time as of the signal date.
# ---------------------------------------------------------------------------
print("\n\n########## BETA-NEUTRAL vs EQUAL-DOLLAR ##########")
cov = r1["XLV"].rolling(252).cov(r1["XLK"])
var = r1["XLK"].rolling(252).var()
beta = (cov / var)
print(f"live trailing-252 beta(XLV on XLK) = {beta.loc[today]:.3f}  "
      f"median hist {beta.median():.3f}")

for h in (3, 5, 10):
    ret_eq = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    xlv_h = fwd_lag(px["XLV"], h, 1)
    xlk_h = fwd_lag(px["XLK"], h, 1)
    ret_bn = xlv_h - beta * xlk_h          # beta-neutral, PIT beta
    valid = ret_eq.dropna().index.intersection(ret_bn.dropna().index)
    sig = px.index[BASE.values].intersection(valid)
    epi = declusters(sig, h, valid)
    show([summarize(ret_eq.loc[epi].values, f"h={h} equal-dollar"),
          summarize(ret_eq.dropna().values, f"h={h} equal-dollar ALL DAYS"),
          summarize(ret_bn.loc[epi].values, f"h={h} beta-neutral"),
          summarize(ret_bn.dropna().values, f"h={h} beta-neutral ALL DAYS")],
         f"beta-neutral check h={h}")

# ---------------------------------------------------------------------------
# tape over-selection: what fraction of trigger days sit above SPY's 200d?
# ---------------------------------------------------------------------------
print("\n\n########## TAPE OVER-SELECTION ##########")
sma200 = px["SPY"].rolling(200).mean()
above = (px["SPY"] > sma200)
sig_days = px.index[BASE.values & above.notna().values]
print(f"trigger days above SPY 200d: {100*above.loc[sig_days].mean():.1f}%  "
      f"(N={len(sig_days)})   base rate {100*above.dropna().mean():.1f}%")
print(f"live: SPY above 200d = {bool(above.loc[today])}")
