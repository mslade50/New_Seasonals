"""kD V1 round 2: the round-1 kill rests on (i) the no-damage half paying LESS
(mechanism falsified in its own window), (ii) cost, (iii) the VIX-up gate not
filtering. Check each is not definition-fragile: SPY-damage split at several
cutoffs, MOVE extremity dose (today is the 99.7th pctile), PIT trailing-252
threshold, concentration / ex-year, and cost multiples per horizon."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

POST = pd.Timestamp("2018-03-01")
TK = ["SVXY", "SPY", "^VIX", "^MOVE"]
IDX = load_prices(["SPY"])["SPY"].index
px = close_panel(TK).reindex(IDX)
rs = px["SPY"].pct_change()
rv = rolling_on_valid(px["^VIX"], lambda x: x.pct_change())
rm = rolling_on_valid(px["^MOVE"], lambda x: x.pct_change())
post = pd.Series(IDX >= POST, index=IDX)
q = {p: rm.dropna().quantile(p) for p in (0.80, 0.90, 0.95, 0.98, 0.99)}
rank252 = rolling_on_valid(rm, lambda x: x.rolling(252).rank(pct=True) * 100)
vup = rv > 0
cell = (rm >= q[0.90]) & vup
COST_BP = 7.0


def hb(h):
    a, b = fwd_lag(px["SVXY"], h, 1), fwd_lag(px["SPY"], h, 1)
    m = post & a.notna() & b.notna()
    return np.polyfit(b[m].values, a[m].values, 1)[0]


RET = {h: vehicle_ret(px, [("SVXY", 1.0), ("SPY", -hb(h))], h, 1) for h in (1, 2, 3, 5)}


def ep(mask, h):
    r = RET[h]
    ok = r.notna() & post
    d = IDX[(mask.reindex(IDX, fill_value=False) & ok).values]
    e = declusters(d, h, IDX)
    return e, r.loc[e].values


def rrow(mask, h, lbl):
    e, v = ep(mask, h)
    s = summarize(v, lbl)
    if s["n"]:
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = sign_test(w, len(v))
    return s


COLS = ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")

# 1. mechanism split (no-damage vs damage) at several SPY cutoffs
print("=== 1. no-damage vs damage half of the cell, SPY cutoffs, residual ===")
for h in (1, 2, 3, 5):
    rows = []
    for cut in (-0.0025, -0.005, -0.0075, -0.010):
        rows.append(rrow(cell & (rs > cut), h, f"h={h} no-damage SPY>{100*cut:.2f}%"))
        rows.append(rrow(cell & (rs <= cut), h, f"h={h} damage    SPY<={100*cut:.2f}%"))
    show([{k: r.get(k) for k in COLS} for r in rows])

# 2. MOVE extremity dose with VIX up (today = top 0.3%)
print("\n=== 2. MOVE extremity dose (VIX up), residual ===")
bands = [(0.80, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]
for h in (1, 2, 3, 5):
    rows = []
    for lo, hi in bands:
        hi_v = q[hi] if hi < 1 else np.inf
        m = (rm >= q[lo]) & (rm < hi_v) & vup
        rows.append(rrow(m, h, f"h={h} MOVE pctile [{int(100*lo)},{int(100*hi) if hi<1 else 100}] & VIX up"))
    rows.append(rrow((rm >= q[0.99]) & vup, h, f"h={h} MOVE top1% & VIX up"))
    rows.append(rrow((rm >= q[0.98]) & vup & (rs > -0.0075), h, f"h={h} MOVE top2% & VIX up & SPY>-0.75 (today-like)"))
    show([{k: r.get(k) for k in COLS} for r in rows])

# 3. PIT threshold: trailing-252 rank of the MOVE daily change
print("\n=== 3. PIT trailing-252 rank threshold ===")
for h in (1, 3, 5):
    rows = [rrow((rank252 >= 90) & vup, h, f"h={h} rank252>=90 & VIX up"),
            rrow((rank252 >= 95) & vup, h, f"h={h} rank252>=95 & VIX up"),
            rrow(rank252 >= 90, h, f"h={h} rank252>=90 alone")]
    show([{k: r.get(k) for k in COLS} for r in rows])

# 4. concentration, ex-year, cost multiples
print("\n=== 4. concentration / ex-year / cost (pre-specified cell) ===")
for h in (1, 2, 3, 5):
    e, v = ep(cell, h)
    allv = RET[h][post].dropna()
    ctl = declusters(allv.index, h, IDX)
    cm = RET[h].loc[ctl].mean()
    print(f"h={h}: N={len(v)} mean {100*v.mean():+.3f}% ctl {100*cm:+.3f}%  raw/cost {100*100*v.mean()/COST_BP:.1f}x  "
          f"edge/cost {100*100*(v.mean()-cm)/COST_BP:.1f}x")
    print("   ", cluster_note(e, v))
    srt = np.sort(v)[::-1]
    print(f"    drop-best-2 mean {100*srt[2:].mean():+.3f}%", end="")
    yrs = pd.DatetimeIndex(e).year
    for y in (2020, 2022, 2024):
        print(f" | ex-{y} {100*v[yrs != y].mean():+.3f}% (n={int((yrs != y).sum())})", end="")
    by = pd.Series(v, index=yrs).groupby(level=0).agg(["count", "mean"])
    by["mean"] *= 100
    print("\n    by year:", {int(k): (int(r["count"]), round(r["mean"], 2)) for k, r in by.iterrows()})
