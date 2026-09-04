"""The bond bid three sessions before a monthly opex.

The sweep has TLT +0.141% t 2.46 and IEF +0.064% t 2.12 off the k3 anchor,
with ^TNX at -0.236% t -1.51 on the other side. Two durations agreeing plus
the yield confirming is why this got a drill. Questions: is the bid the whole
opex week or only this one session, does it survive its own controls, is it
era-stable, and is it concentrated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, era_split,
    cluster_note, local_control,
)

TKRS = ["TLT", "IEF", "^TNX", "LQD"]
px = close_panel(TKRS)
idx = px.index

ev = load_events(["opex"])
rows = []
for d in pd.DatetimeIndex(ev["date"]):
    p = idx.searchsorted(d)
    if p < len(idx) and idx[p] == d:
        rows.append(p)
P = np.array(rows)
print(f"opex sessions matched: {len(P)}")


def leg(t, a, b):
    v = px[t].values
    out = []
    for p in P:
        if p + a < 0 or p + b >= len(v):
            continue
        out.append(v[p + b] / v[p + a] - 1.0)
    return np.array(out, float)


def dts(a):
    return pd.DatetimeIndex([idx[p + a] for p in P if 0 <= p + a < len(idx)])


print("\n" + "=" * 74)
print("A. every single session of opex week, per ticker")
print("=" * 74)
LEGS = [
    ("k4 -> k3", -4, -3),
    ("k3 -> k2  (TONIGHT's anchor, h1 = tomorrow)", -3, -2),
    ("k2 -> k1", -2, -1),
    ("k1 -> k0  (into the opex session)", -1, 0),
    ("k3 -> k0  (the whole run into opex)", -3, 0),
]
for t in TKRS:
    out = []
    for name, a, b in LEGS:
        v = leg(t, a, b)
        r = summarize(v, name)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        out.append(r)
    base = px[t].pct_change(fill_method=None).dropna()
    out.append(summarize(base.values, "CTRL all days"))
    show(out, f"{t}: opex week, session by session")

print("\n" + "=" * 74)
print("B. controls on the k3->k2 session (the one that is tomorrow)")
print("=" * 74)
for t in TKRS:
    v = leg(t, -3, -2)
    d = dts(-3)
    r1 = px[t].pct_change(fill_method=None)
    valid = r1.dropna().index
    anchors = pd.DatetimeIndex([idx[p - 2] for p in P if p - 2 >= 0])
    loc = local_control(valid, anchors.intersection(valid), win=126)
    out = [
        summarize(v, "opex k3->k2"),
        summarize(r1.loc[valid].values, "CTRL all days"),
        summarize(r1.loc[loc].values, "CTRL local +/-126td ex-trigger"),
    ]
    show(out, f"{t}: controls")

print("\n" + "=" * 74)
print("C. era split and concentration, TLT / IEF / ^TNX on k3->k2")
print("=" * 74)
for t in ["TLT", "IEF", "^TNX"]:
    v = leg(t, -3, -2)
    d = dts(-3)[: len(v)]
    show(era_split(d, v), f"{t}: era split at 2018")
    print(" ", cluster_note(d, v, k=2))
    by_yr = pd.Series(v).groupby(d.year.values).mean() * 100
    pos_yrs = int((by_yr > 0).sum())
    print(f"  positive in {pos_yrs} of {len(by_yr)} calendar years")

print("\n" + "=" * 74)
print("D. August opex only, and the midterm subset (TLT, IEF)")
print("=" * 74)
mon = np.array([idx[p].month for p in P])
yr = np.array([idx[p].year for p in P])
for t in ["TLT", "IEF"]:
    vals = []
    keep_m, keep_y = [], []
    v_all = px[t].values
    for p, m, y in zip(P, mon, yr):
        if p - 3 < 0 or p - 2 >= len(v_all):
            continue
        x = v_all[p - 2] / v_all[p - 3] - 1.0
        if np.isnan(x):
            continue
        vals.append(x)
        keep_m.append(m)
        keep_y.append(y)
    vals = np.array(vals)
    keep_m = np.array(keep_m)
    keep_y = np.array(keep_y)
    out = []
    for name, m in [
        ("all opex", np.ones(len(vals), bool)),
        ("AUGUST opex", keep_m == 8),
        ("August + midterm", (keep_m == 8) & (keep_y % 4 == 2)),
    ]:
        if m.sum() == 0:
            continue
        r = summarize(vals[m], name)
        r["rec"] = f"{int((vals[m] > 0).sum())}-{int((vals[m] <= 0).sum())}"
        r["sign_p"] = round(sign_test(int((vals[m] > 0).sum()), int(m.sum())), 4)
        out.append(r)
    show(out, f"{t}: k3->k2 by month bucket")
