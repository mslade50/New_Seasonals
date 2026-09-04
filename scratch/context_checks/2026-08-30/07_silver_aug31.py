"""Silver's Aug-31 seasonal cell, on a clean cut, with the roll check applied.

The engine's E:seasonal_doy|SI=F fired at n 25, +0.367%, 18-7 up, sign p
0.0216 for the all-years arm. That is a doy +/-2 window, which is a loose cut
of the same tape. Monday is unambiguously the last session of August, so the
right cell is the last session of August, not a five-day window around it.

Silver also printed -3.37% on Friday and sits 41.7% below its 52-week high,
so the same roll test drill 04 applied to the grains has to be applied here
before any silver claim is made. GC=F showed a 369x volume jump on Friday in
drill 04, which is exactly the signature that disqualifies a bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, summarize, show, sign_test, era_split, cluster_note,
)

px = load_prices(["SI=F", "GC=F", "^GSPC"])

print("=" * 78)
print("Q0. roll check on Friday's metals bars, same test as drill 04")
print("=" * 78)
for t in ["SI=F", "GC=F"]:
    d = px[t]
    i = d.index.get_loc(pd.Timestamp("2026-08-28"))
    last, prev = d.iloc[i], d.iloc[i - 1]
    o, c0 = last["Open"], prev["Close"]
    gap = 100 * (o / c0 - 1)
    intra = 100 * (last["Close"] / o - 1)
    v63 = d.iloc[max(0, i - 64):i]["Volume"].median()
    prior5 = d.iloc[max(0, i - 5):i]["Volume"]
    print(f"  {t}: session {100*(last['Close']/c0-1):+.2f}%  gap {gap:+.2f}%  "
          f"intraday {intra:+.2f}%  vol x{last['Volume']/v63:.1f}  "
          f"dup prior volume {bool(prior5.duplicated().any())}")
    print(f"       last 4 volumes: {list(d.iloc[i-3:i+1]['Volume'].values)}")

sil = px["SI=F"]["Close"].dropna()
cal = px["^GSPC"]["Close"].dropna().index
ym = pd.Series(cal.year * 100 + cal.month, index=cal)
COMPLETE = sorted(set(ym.values))[:-1]
finals = []
for key, grp in ym.groupby(ym.values):
    if key in COMPLETE:
        finals.append(list(cal).index(grp.index[-1]))
finals.sort()

print()
print("=" * 78)
print("Q1. clean cut: the last session of August, anchored on the session")
print("    before it, versus the doy +/-2 window the engine used")
print("=" * 78)
rows = []
for label, sel in [("last session of August", [p for p in finals if cal[p].month == 8]),
                   ("last session of any month", finals)]:
    ds, vs = [], []
    for p in sel:
        if p < 1:
            continue
        a, b = cal[p - 1], cal[p]
        if a in sil.index and b in sil.index:
            ds.append(a)
            vs.append(sil.loc[b] / sil.loc[a] - 1.0)
    v = np.asarray(vs, float)
    if len(v) < 5:
        continue
    r = summarize(v, label)
    u = int((v > 0).sum())
    r["record"] = f"{u}-{len(v) - u}"
    r["sign_p"] = round(sign_test(u, len(v)), 4)
    rows.append(r)
    if label.endswith("August"):
        aug_d, aug_v = pd.DatetimeIndex(ds), v
allr = (sil / sil.shift(1) - 1.0).dropna()
b = summarize(allr.values, "all sessions")
b["record"] = ""
b["sign_p"] = np.nan
rows.append(b)
show(rows, "SI=F month-end")

print("  era:", [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
                 for x in era_split(aug_d, aug_v) if x.get("n")])
print("  conc:", cluster_note(aug_d, aug_v))
print("  years:", {int(y): round(100 * val, 2) for y, val in zip(aug_d.year, aug_v)})

mid = np.array([y % 4 == 2 for y in aug_d.year])
if mid.sum() >= 3:
    u = int((aug_v[mid] > 0).sum())
    print(f"  midterm: n={int(mid.sum())} {u}-{int(mid.sum())-u} "
          f"mean {100*aug_v[mid].mean():+.3f}%")

print()
print("=" * 78)
print("today's reading")
print("=" * 78)
lo = sil.rolling(252, min_periods=200).min().iloc[-1]
hi = sil.rolling(252, min_periods=200).max().iloc[-1]
print(f"  SI=F {sil.iloc[-1]:.3f}  {100*(sil.iloc[-1]/lo-1):+.1f}% off 52w low  "
      f"{100*(sil.iloc[-1]/hi-1):+.1f}% off 52w high")
