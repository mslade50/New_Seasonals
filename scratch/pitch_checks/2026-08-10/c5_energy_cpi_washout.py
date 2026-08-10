"""C5 round 1 -- Energy's 5d washout into a CPI print.

HARD CONSTRAINT from the brief: the outright long is DOA because a 5d washout
inside 21d strength IS materially the book's LT Trend ST OS setup (2/5/10/21d
rank < 15, 252d rank 65-90, long-term uptrend, long, short hold). So this
script does four things:

  0. COUNT the joint state first (registry method trap: count before measuring)
  1. price the DOA outright anyway, so the kill is quantified not asserted
  2. gate attribution: does the CPI anchor add ANYTHING over the plain washout?
  3. the forms that are NOT the book: the SHORT side, and the XLE-vs-SPY
     relative leg (both legs priced separately first)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
TK = ["XLE", "SPY", "XOP", "USO", "EOG", "CVX"]
P = close_panel(TK).dropna(subset=["XLE", "SPY"])
ASOF = P.index[-1]
idx = P.index
x = P["XLE"]
rk5, rk21 = pct_rank(x, 5), pct_rank(x, 21)
print(f"sample {idx.min().date()} .. {ASOF.date()}  n={len(idx)}")
print(f"TODAY: XLE rank5={rk5.loc[ASOF]:.1f} rank21={rk21.loc[ASOF]:.1f}")

WASH = ((rk5 <= 15) & (rk21 >= 55)).fillna(False)
cpi = load_events(["cpi"])["date"]


def within(k: int) -> pd.Series:
    """True on sessions that are 1..k trading days BEFORE a CPI print."""
    flag = pd.Series(False, index=idx)
    for d in cpi:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        for j in range(1, k + 1):
            if 0 <= p - j < len(idx):
                flag.iloc[p - j] = True
    return flag


PRE = within(3)
JOINT = WASH & PRE

# ------------------------------------------------------------------ 0. COUNT
print(f"\n### 0. COUNT FIRST ###")
print(f"washout days              : {int(WASH.sum())}")
print(f"pre-CPI (1..3 td) days    : {int(PRE.sum())}")
print(f"JOINT (today's state)     : {int(JOINT.sum())}")
for H in (2, 3, 5):
    e = declusters(idx[JOINT.values], H, idx)
    print(f"  joint episodes at min_gap={H}: {len(e)}")
print("joint dates:", [str(d.date()) for d in idx[JOINT.values]][:40])

# --------------------------------------------------- 1. the DOA outright long
print("\n### 1. the DOA outright long, quantified ###")
for H in (1, 2, 3, 5):
    fw = fwd_lag(x, H, LAG)
    ok = fw.notna()
    rows = []
    for lbl, m in (("washout ONLY", WASH), ("washout AND pre-CPI", JOINT),
                   ("pre-CPI ONLY", PRE)):
        t = idx[m.values & ok.values]
        if len(t) == 0:
            rows.append({"label": lbl, "n": 0})
            continue
        e = declusters(t, H, idx[ok.values])
        v = fw.loc[e].values
        r = summarize(v, f"{lbl} h={H}")
        r["n_days"] = len(t)
        r["edge_vs_drift"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
    rows.append(summarize(fw[ok].values, f"CTRL XLE all days h={H}"))
    show(rows, f"long XLE (h={H}) -- gate attribution")

# --------------------------------------------------- 2. full battery, best h
battery(P, JOINT, [("XLE", 1.0)], 3, "C5-LONG XLE washout into CPI", cost_bps=2.0,
        variants={"washout only": WASH, "joint (rk5<=15)": JOINT,
                  "joint rk5<=10": ((rk5 <= 10) & (rk21 >= 55) & PRE).fillna(False),
                  "joint rk5<=25": ((rk5 <= 25) & (rk21 >= 55) & PRE).fillna(False),
                  "joint, no rk21 gate": ((rk5 <= 15) & PRE).fillna(False)},
        lag=LAG, event_kinds=("cpi",))

# --------------------------------------------------- 3. forms that are NOT the book
print("\n### 3. NOT-THE-BOOK forms ###")
print("3a. relative: LEGS PRICED SEPARATELY before the spread")
for H in (2, 3, 5):
    fx, fs = fwd_lag(x, H, LAG), fwd_lag(P["SPY"], H, LAG)
    ok = fx.notna() & fs.notna()
    t = idx[JOINT.values & ok.values]
    e = declusters(t, H, idx[ok.values])
    show([summarize(fx.loc[e].values, f"XLE leg h={H}"),
          summarize(fs.loc[e].values, f"SPY leg h={H}"),
          summarize((fx - fs).loc[e].values, f"XLE-SPY spread h={H}"),
          summarize(fx[ok].values, "XLE all days"),
          summarize(fs[ok].values, "SPY all days"),
          summarize((fx - fs)[ok].values, "XLE-SPY all days")],
         f"relative form h={H}")

print("\n3b. the SHORT side (washout continues through the print)")
for H in (2, 3, 5):
    fw = fwd_lag(x, H, LAG)
    ok = fw.notna()
    t = idx[JOINT.values & ok.values]
    e = declusters(t, H, idx[ok.values])
    v = -fw.loc[e].values
    r = summarize(v, f"SHORT XLE h={H}")
    r["ctrl_short_all"] = round(-100 * fw[ok].mean(), 3)
    show([r], f"short side h={H}")

print("\n3c. other energy instruments on the same joint state")
H = 3
for tk in ("XOP", "USO", "EOG", "CVX"):
    if tk not in P.columns:
        continue
    s = P[tk].dropna()
    fw = fwd_lag(s, H, LAG)
    ok = fw.notna()
    t = pd.DatetimeIndex(idx[JOINT.values]).intersection(s.index[ok.values])
    if len(t) == 0:
        continue
    e = declusters(t, H, s.index[ok.values])
    r = summarize(fw.loc[e].values, f"{tk} long h={H}")
    r["own_drift"] = round(100 * fw[ok].mean(), 3)
    r["edge"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
    show([r], f"{tk}")

# --------------------------------------------------- 4. midterm
print("\n### 4. midterm split (2026 is midterm; the book's own LT Trend ST OS "
      "is -14.0pp win rate in midterms) ###")
for H in (2, 3, 5):
    fw = fwd_lag(x, H, LAG)
    ok = fw.notna()
    t = idx[JOINT.values & ok.values]
    e = declusters(t, H, idx[ok.values])
    v = fw.loc[e].values
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(v[mt], f"MIDTERM h={H} (N={int(mt.sum())})"),
          summarize(v[~mt], f"non-midterm h={H}")], f"midterm h={H}")
