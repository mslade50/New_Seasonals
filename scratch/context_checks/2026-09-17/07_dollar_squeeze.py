"""The dollar squeeze: three fired triggers that are one fact.

USDSEK printed a 5th straight up close (P7, one of 3 BH survivors: 52-86 down
next session, t -2.61, sign p 0.0024). USD/CHF printed its first 52w high in
30+ days (P1: 3-14 down, t -4.16). CHF, SEK, NOK, SGD, CAD, MXN and TRY are
all in the top 5% of their year on a 5-day basis. DXY 5d rank 92.1, z10 1.37.

Treat them as one dollar-breadth cell instead of seven, then check the
quad-witching DXY cell (65-41 up, sign p 0.0125) against it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, pct_rank, declusters,
    cluster_note,
)

PAIRS = ["CHF=X", "USDSEK=X", "USDNOK=X", "CAD=X", "USDSGD=X", "JPY=X",
         "USDMXN=X"]
px = close_panel(["DX-Y.NYB"] + PAIRS + ["SPY", "EEM"])
dxy = px["DX-Y.NYB"].dropna()

# how broad is the move: count pairs whose 5d return is in the top 5% of its year
counts = None
for p in PAIRS:
    s = px[p].dropna()
    r5 = s / s.shift(5) - 1.0
    rk = pct_rank(r5, 5, 252)
    c = (rk >= 95).astype(float).reindex(px.index)
    counts = c if counts is None else counts.add(c, fill_value=0)
counts = counts.dropna()
print(f"dollar-strength breadth today: {counts.iloc[-1]:.0f} of {len(PAIRS)} "
      f"pairs with a 5d return in the top 5% of their year")
print(f"  breadth history: >=5 of 7 on {int((counts >= 5).sum())} sessions, "
      f">=6 on {int((counts >= 6).sum())}")

f1 = dxy.shift(-1) / dxy - 1.0
f5 = dxy.shift(-5) / dxy - 1.0
common = counts.index.intersection(f5.dropna().index)
for h, f in [(1, f1), (5, f5)]:
    rows = []
    for lab, m in [(">=5 of 7 pairs extreme", counts.reindex(common) >= 5),
                   (">=6 of 7", counts.reindex(common) >= 6),
                   ("<5", counts.reindex(common) < 5)]:
        d = common[m.values]
        r = summarize(f.loc[d].values, lab)
        if r["n"]:
            k = int((f.loc[d] > 0).sum())
            r["record"] = f"{k}-{r['n'] - k} up"
            r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
        rows.append(r)
    rows.append(summarize(f.dropna().values, "CTL all sessions"))
    show(rows, f"DXY h={h} by dollar breadth")

d = common[(counts.reindex(common) >= 5).values]
epi = declusters(d, 10, common)
print(f"\n>=5-of-7 episodes (10td declustered): {len(epi)}")
print(f"  by year {dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")
for h, f in [(1, f1), (5, f5)]:
    v = f.loc[epi.intersection(f.dropna().index)]
    k = int((v > 0).sum())
    print(f"  DXY h={h}: n={len(v)} mean {100 * v.mean():+.3f}% "
          f"median {100 * v.median():+.3f}% {k}-{len(v) - k} up "
          f"sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")
print(f"  concentration h=5: {cluster_note(epi.intersection(f5.dropna().index), f5.loc[epi.intersection(f5.dropna().index)].values, k=2)}")

print("\n=== the two individual cells, verified ===")
# USDSEK 5-day up streak
s = px["USDSEK=X"].dropna()
r = s / s.shift(1) - 1.0
up = (r > 0)
streak = up & up.shift(1) & up.shift(2) & up.shift(3) & up.shift(4)
fs = s.shift(-1) / s - 1.0
d = s.index[streak.fillna(False).values].intersection(fs.dropna().index)
k = int((fs.loc[d] > 0).sum())
print(f"USDSEK 5+ up closes: n={len(d)} mean {100 * fs.loc[d].mean():+.3f}% "
      f"{k}-{len(d) - k} up, sign p(down) = {sign_test(len(d) - k, len(d)):.4f}, "
      f"ctl {100 * fs.dropna().mean():+.3f}%")
print(f"  today's streak length check: last 6 closes "
      f"{(100 * r.tail(6)).round(2).tolist()}")
for lab, m in [("pre-2018", d < pd.Timestamp("2018-01-01")), ("2018+", d >= pd.Timestamp("2018-01-01"))]:
    v = fs.loc[d[m]]
    kk = int((v > 0).sum())
    print(f"    {lab:<10} n={len(v):<4} mean {100 * v.mean():+.3f}%  {kk}-{len(v) - kk} up")
epi = declusters(d, 5, s.index)
v = fs.loc[epi.intersection(fs.dropna().index)]
kk = int((v > 0).sum())
print(f"    declustered(5td) n={len(v)} mean {100 * v.mean():+.3f}% {kk}-{len(v) - kk} up "
      f"sign p(down) = {sign_test(len(v) - kk, len(v)):.4f}")

# USD/CHF first 52w high in 30+ td
c = px["CHF=X"].dropna()
hi = c.rolling(252).max()
at = c >= hi - 1e-12
first = at & (~at.rolling(30).max().shift(1).fillna(0).astype(bool))
fc = c.shift(-1) / c - 1.0
d = c.index[first.fillna(False).values].intersection(fc.dropna().index)
k = int((fc.loc[d] > 0).sum())
print(f"\nUSD/CHF first 52w high in 30+ td: n={len(d)} "
      f"mean {100 * fc.loc[d].mean():+.3f}% {k}-{len(d) - k} up, "
      f"sign p(down) = {sign_test(len(d) - k, len(d)):.4f}, "
      f"ctl {100 * fc.dropna().mean():+.3f}%")
print(f"  dates {[str(x.date()) for x in d]}")
for lab, m in [("pre-2018", d < pd.Timestamp("2018-01-01")), ("2018+", d >= pd.Timestamp("2018-01-01"))]:
    v = fc.loc[d[m]]
    kk = int((v > 0).sum())
    print(f"    {lab:<10} n={len(v):<3} mean {100 * v.mean():+.3f}%  {kk}-{len(v) - kk} up")

print("\n=== DXY on the quad-witching session (engine: 65-41 up, p 0.0125) ===")
qw = pd.DatetimeIndex(load_events(["quad_witching"])["date"])
idx = dxy.index
pos = pd.Series(range(len(idx)), index=idx)
anch = pd.DatetimeIndex([idx[pos[x] - 1] for x in qw if x in pos.index and pos[x] > 0])
anch = anch.intersection(f1.dropna().index)
rows = [summarize(f1.loc[anch].values, "all quad witchings")]
for m, name in [(3, "March"), (6, "June"), (9, "September"), (12, "December")]:
    dd = anch[anch.month == m]
    r = summarize(f1.loc[dd].values, name)
    if r["n"]:
        k = int((f1.loc[dd] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["sign_p_up"] = round(sign_test(k, r["n"]), 4)
    rows.append(r)
rows.append(summarize(f1.dropna().values, "CTL all sessions"))
show(rows, "DXY on the witching session")
