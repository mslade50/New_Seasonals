"""A1 ROUND 2, part 4 -- horizon profile of the fresh-trigger cell.

The parked script (2026-08-12 a6b) named this as kill route (a): "the loose rung
was positive only at h=1. If the tight rung is the same shape, the 'edge' is one
bar wide and the rung was chosen off a ladder after the fact." It ran the scan
on the DECLUSTERS population. Run it on the population the ARM describes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

raw = load_prices(["TLT", "IEF", "LQD"])
IDX = raw["TLT"].index
POS = pd.Series(range(len(IDX)), index=IDX)
PX = pd.DataFrame({t: raw[t]["Close"].reindex(IDX) for t in raw})


def above_low(t, n=252):
    s = raw[t]["Close"]
    return ((s / s.rolling(n).min() - 1.0) * 100).reindex(IDX)


TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")
CELL = ((TL <= 0.5) & (IE <= 1.0) & (LQ <= 1.0)).fillna(False)


def first_in(mask, gap=10):
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    keep, last = [], -10 ** 9
    for d in days:
        p = int(POS[d])
        if p - last >= gap:
            keep.append(d)
        last = p
    return pd.DatetimeIndex(keep)


EPI = first_in(CELL)
DEC = declusters(IDX[CELL.values], 10, IDX)

for nm, A in (("FRESH-FIRST (the arm)", EPI), ("DECLUSTERS (the parked stat)", DEC)):
    print("=" * 78)
    print(nm)
    print("=" * 78)
    rows = []
    for h in (1, 2, 3, 4, 5, 7, 10, 15, 21):
        r = fwd_lag(PX["TLT"], h, 1)
        v = r.loc[A].dropna()
        b = r.dropna().mean()
        w = int((v > 0).sum())
        rows.append({"h": h, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "edge_pp": round(100 * (v.mean() - b), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "sign_p": round(sign_test(w, len(v)), 4),
                     "worst_pct": round(100 * v.min(), 2)})
    show(rows, "long TLT, lag=1")
    # lag=0 contrast: is the edge the signal day's own continuation?
    rows0 = []
    for h in (1, 2, 3, 5):
        r = fwd_lag(PX["TLT"], h, 0)
        v = r.loc[A].dropna()
        rows0.append({"h": h, "n": len(v), "lag0_mean_pct": round(100 * v.mean(), 3),
                      "hit": round(100 * (v > 0).mean(), 1)})
    show(rows0, "lag=0 contrast (untradeable)")
