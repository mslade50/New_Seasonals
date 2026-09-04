"""C9 round 1 - long XLI on a 5-day rank-2.0 washout inside an intact 63d trend.

The 2026-08-24 XLI kill (d1_c4_xli_pair.py) died because the pitched rung had
NO HISTORY (r5 rank <=3 with a peer at a 52w high: 0 days ever).  So rule 1
here is POPULATION FIRST, at the literal state, before anything else runs.

Second thing checked, because the registry says it decides this shape: the
generic 5-day-washout reversal (k=5 / h=3, +0.534%, t 4.17, 2018+ +0.709%) is
"the book's own dip-buy family and must not be re-dressed as a pitch"
(2026-08-14).  So the question is not "does XLI bounce" but "does the
INTACT-TREND gate add anything over the plain washout, and does XLI beat the
other ten sectors on the identical trigger".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
px = close_panel(SECTORS + ["SPY"])

r5 = pct_rank(px["XLI"], 5)
r21 = pct_rank(px["XLI"], 21)
r63 = pct_rank(px["XLI"], 63)
hi252 = rolling_on_valid(px["XLI"], lambda x: x.rolling(252).max())
d52 = px["XLI"] / hi252 - 1.0
print(f"today XLI: r5={r5.iloc[-1]:.1f}  r21={r21.iloc[-1]:.1f}  r63={r63.iloc[-1]:.1f}  "
      f"off 52w high {d52.iloc[-1]*100:+.2f}%  5d ret {px['XLI'].pct_change(5).iloc[-1]*100:+.2f}%")

# ------------------------------------------------------------------ 0. population
print("\n########## 0. RUNG POPULATION AT THE LITERAL STATE ##########")
LITERAL = ((r5 <= 2.0) & (r63 >= 40) & (r63 <= 50) & (d52 >= -0.05)).fillna(False)
print(f"  LITERAL (r5<=2.0 & r63 in [40,50] & within 5% of high): "
      f"N={int(LITERAL.sum())} days ever")
for r5c in (2, 3, 5, 10, 15):
    for lbl, extra in (("bare", pd.Series(True, index=px.index)),
                       ("& r63 in [30,60]", ((r63 >= 30) & (r63 <= 60))),
                       ("& r63>=30 & within 5% of high", ((r63 >= 30) & (d52 >= -0.05)))):
        m = ((r5 <= r5c) & extra).fillna(False)
        yrs = sorted(set(px.index[m.values].year))
        print(f"  r5<={r5c:<2} {lbl:<32} N={int(m.sum()):>4d}  ({len(yrs)} yrs)")

# ------------------------------------------------------------------ 1. battery
MAIN = ((r5 <= 5) & (r63 >= 30) & (r63 <= 60) & (d52 >= -0.05)).fillna(False)
variants = {
    "r5<=2 bare": (r5 <= 2).fillna(False),
    "r5<=5 bare": (r5 <= 5).fillna(False),
    "r5<=10 bare": (r5 <= 10).fillna(False),
    "r5<=5 & r63[30,60]": ((r5 <= 5) & (r63 >= 30) & (r63 <= 60)).fillna(False),
    "r5<=5 & r63[30,60] & nr high": MAIN,
    "r5<=5 & r63>=70 (strong trend)": ((r5 <= 5) & (r63 >= 70)).fillna(False),
    "r5<=5 & r63<=30 (broken)": ((r5 <= 5) & (r63 <= 30)).fillna(False),
    "LITERAL": LITERAL,
}
for h in (3, 5):
    battery(px, MAIN, [("XLI", 1.0)], h,
            "C9 long XLI | r5<=5, r63 in [30,60], within 5% of 52w high",
            cost_bps=5.0, variants=variants if h == 3 else None)

# ------------------------------------------------------------------ 2. attribution
print("\n########## 2. GATE ATTRIBUTION at h=3 (the registry's washout horizon) ##########")
ret = vehicle_ret(px, [("XLI", 1.0)], 3)
valid = ret.dropna().index
base = 100 * ret.loc[valid].mean()
rows = []
for lbl, m in variants.items():
    d = px.index[m.values].intersection(valid)
    e = declusters(d, 3, valid)
    s = summarize(ret.loc[e].values, lbl)
    s["n_days"] = len(d)
    s["edge_pct"] = round(s.get("mean_pct", np.nan) - base, 3)
    rows.append(s)
show(rows, f"long XLI h=3   (XLI all-days drift {base:+.3f}%)")

# ------------------------------------------------------------------ 3. ref class
print("\n########## 3. SECTOR REFERENCE CLASS - identical washout trigger on 9 SPDRs ##########")
rows, ts = [], []
for s_ in SECTORS:
    a5 = pct_rank(px[s_], 5)
    a63 = pct_rank(px[s_], 63)
    hi = rolling_on_valid(px[s_], lambda x: x.rolling(252).max())
    dd = px[s_] / hi - 1.0
    m = ((a5 <= 5) & (a63 >= 30) & (a63 <= 60) & (dd >= -0.05)).fillna(False)
    rr = vehicle_ret(px, [(s_, 1.0)], 3)
    v = rr.dropna().index
    e = declusters(px.index[m.values].intersection(v), 3, v)
    r = summarize(rr.loc[e].values, f"LONG {s_}")
    if r["n"]:
        r["own_drift_pct"] = round(100 * rr.loc[v].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * rr.loc[v].mean(), 3)
        ts.append((s_, r["t"], r["mean_pct"]))
    rows.append(r)
show(rows, "long each sector on its own washout-in-intact-trend, h=3")
rk = sorted(ts, key=lambda x: -x[2])
print(f"  mean ranking: {[(s, round(m, 3)) for s, t, m in rk]}")
print(f"  XLI ranks {[i for i,(s,_,_) in enumerate(rk,1) if s=='XLI'][0]} of {len(rk)} by mean")
tt = np.array([t for _, t, _ in ts])
print(f"  max |t| across the 9 = {np.abs(tt).max():.2f} on "
      f"{ts[int(np.argmax(np.abs(tt)))][0]}; XLI |t| = "
      f"{abs([t for s,t,_ in ts if s=='XLI'][0]):.2f}")

# permutation: max-of-9 null
rng = np.random.default_rng(7)
maxts = []
n_epi = [r["n"] for r in rows if r.get("n")]
for _ in range(2000):
    best = 0.0
    for s_ in SECTORS:
        rr = vehicle_ret(px, [(s_, 1.0)], 3).dropna()
        k = max(3, int(np.median(n_epi)))
        idx = rng.choice(len(rr), size=k, replace=False)
        v = rr.values[idx]
        t = v.mean() / (v.std(ddof=1) / np.sqrt(k)) if v.std(ddof=1) > 0 else 0.0
        best = max(best, abs(t))
    maxts.append(best)
maxts = np.array(maxts)
print(f"  permutation max-of-9 null: P(max|t| >= {np.abs(tt).max():.2f}) = "
      f"{(maxts >= np.abs(tt).max()).mean():.3f}   (2000 draws, k={max(3,int(np.median(n_epi)))})")

# ------------------------------------------------------------------ 4. horizons/tape
print("\n########## 4. HORIZON SCAN + TAPE OVER-SELECTION ##########")
d = px.index[MAIN.values]
show(horizon_scan(px, d, [("XLI", 1.0)], hs=(1, 2, 3, 5, 7, 10)), "long XLI, main gate")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = px["SPY"] > sma200
dd = d.intersection(above.dropna().index)
print(f"  trigger days above SPY 200d: {above.loc[dd].mean()*100:.1f}% (base {above.dropna().mean()*100:.1f}%)")
