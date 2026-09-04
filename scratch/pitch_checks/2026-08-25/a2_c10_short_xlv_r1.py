"""C10 round 1 - short XLV outright after a 63d rank of 97.6 into a 52-week high.

Registry precedent this owes a number to (2026-08-11, c7_xlu_xlv_dispersion.py):
"the XLV short is the LOSING leg, -0.516% at h=3 against its own -0.117% short
drift ... because XLV keeps rising after the trigger."  That was measured on a
DISPERSION trigger; this is a momentum-rank trigger, so it needs its own run,
plus the mandated single-sector reference class across the other ten SPDRs.

Structure:
  0. rung population at the literal state
  1. battery, short XLV outright, rung ladder on the 63d rank
  2. gate attribution: 63d rank alone / 52w-high proximity alone / both
  3. sector reference class - run the identical trigger on all 11 SPDRs and
     rank XLV inside it (permutation max-of-k, per 2026-08-19)
  4. horizon scan + tape over-selection
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
TK = SECTORS + ["SPY"]
px = close_panel(TK)
print(f"panel {px.index[0].date()} -> {px.index[-1].date()}  {px.shape}")

r63 = pct_rank(px["XLV"], 63)
r21 = pct_rank(px["XLV"], 21)
r5 = pct_rank(px["XLV"], 5)
hi252 = rolling_on_valid(px["XLV"], lambda x: x.rolling(252).max())
d52 = px["XLV"] / hi252 - 1.0
print(f"\ntoday XLV: r63={r63.iloc[-1]:.1f}  r21={r21.iloc[-1]:.1f}  r5={r5.iloc[-1]:.1f}  "
      f"off 52w high {d52.iloc[-1]*100:+.2f}%")

# ------------------------------------------------------------------ 0. population
print("\n########## 0. RUNG POPULATION ##########")
for rk in (90, 95, 97, 97.6, 99):
    for dd in (None, -0.01):
        m = (r63 >= rk).fillna(False)
        lbl = f"r63>={rk}"
        if dd is not None:
            m = m & (d52 >= dd).fillna(False)
            lbl += f" & within {abs(dd)*100:.0f}% of 52w high"
        n = int(m.sum())
        yrs = sorted(set(px.index[m.values].year))
        print(f"  {lbl:<44} N={n:>4d}  {len(yrs)} yrs {yrs[:2]}..{yrs[-2:] if yrs else []}")
# the literal joint state including the 21d rank
LIT = ((r63 >= 97) & (r21 >= 90) & (d52 >= -0.01)).fillna(False)
print(f"  LITERAL (r63>=97 & r21>=90 & within 1% of high): N={int(LIT.sum())} days ever")

# ------------------------------------------------------------------ 1. battery
MAIN = ((r63 >= 95) & (d52 >= -0.01)).fillna(False)
variants = {
    "r63>=90 & nearhigh": ((r63 >= 90) & (d52 >= -0.01)).fillna(False),
    "r63>=95 & nearhigh": MAIN,
    "r63>=97.6 & nearhigh": ((r63 >= 97.6) & (d52 >= -0.01)).fillna(False),
    "r63>=95 only": (r63 >= 95).fillna(False),
    "nearhigh only": (d52 >= -0.01).fillna(False),
    "LITERAL joint": LIT,
}
for h in (3, 5):
    battery(px, MAIN, [("XLV", -1.0)], h,
            f"C10 SHORT XLV | r63>=95 & within 1% of 52w high", cost_bps=5.0,
            variants=variants if h == 5 else None)

# ------------------------------------------------------------------ 2. attribution
print("\n########## 2. GATE ATTRIBUTION (short XLV, h=5) ##########")
ret = vehicle_ret(px, [("XLV", -1.0)], 5)
valid = ret.dropna().index
base = ret.loc[valid].mean() * 100
rows = []
for lbl, m in variants.items():
    d = px.index[m.values].intersection(valid)
    e = declusters(d, 5, valid)
    s = summarize(ret.loc[e].values, lbl)
    s["n_days"] = len(d)
    s["edge_vs_base"] = round(s.get("mean_pct", np.nan) - base, 3)
    rows.append(s)
show(rows, f"short-XLV all-days drift {base:+.3f}%  (this is the number to beat)")
print(f"  LONG XLV unconditional h=5 drift = {-base:+.3f}%")

# ------------------------------------------------------------------ 3. ref class
print("\n########## 3. SECTOR REFERENCE CLASS - identical trigger on 11 SPDRs ##########")
rows = []
ts = []
for s_ in SECTORS:
    rk = pct_rank(px[s_], 63)
    hi = rolling_on_valid(px[s_], lambda x: x.rolling(252).max())
    dd = px[s_] / hi - 1.0
    m = ((rk >= 95) & (dd >= -0.01)).fillna(False)
    rr = vehicle_ret(px, [(s_, -1.0)], 5)
    v = rr.dropna().index
    e = declusters(px.index[m.values].intersection(v), 5, v)
    r = summarize(rr.loc[e].values, f"SHORT {s_}")
    if r["n"]:
        r["own_drift_pct"] = round(100 * rr.loc[v].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * rr.loc[v].mean(), 3)
        ts.append((s_, r["t"]))
    rows.append(r)
show(rows, "short each sector on its own r63>=95 & near-52w-high, h=5")
ts_sorted = sorted(ts, key=lambda x: -abs(x[1]))
print(f"  |t| ranking: {[(s, round(t, 2)) for s, t in ts_sorted]}")
xlv_rank = [i for i, (s_, _) in enumerate(ts_sorted, 1) if s_ == "XLV"]
print(f"  XLV ranks {xlv_rank[0] if xlv_rank else '?'} of {len(ts_sorted)} by |t|")
neg = [s_ for s_, t in ts if t < 0]
print(f"  sectors where the SHORT is profitable (t>0): "
      f"{[s_ for s_, t in ts if t > 0]}   |  short loses on: {neg}")

# ------------------------------------------------------------------ 4. horizons
print("\n########## 4. HORIZON SCAN + TAPE OVER-SELECTION ##########")
d = px.index[MAIN.values]
show(horizon_scan(px, d, [("XLV", -1.0)], hs=(1, 2, 3, 5, 7, 10)), "SHORT XLV, r63>=95 & nearhigh")
show(horizon_scan(px, d, [("XLV", -1.0), ("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "SHORT XLV / LONG SPY (the residual read)")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = px["SPY"] > sma200
print(f"\n  trigger days above SPY 200d: {above.loc[d.intersection(above.dropna().index)].mean()*100:.1f}% "
      f"(base {above.dropna().mean()*100:.1f}%)")
