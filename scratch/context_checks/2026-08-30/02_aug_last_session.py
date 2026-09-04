"""The equity side of the same month boundary, and the session after it.

Drill 01 found the final session of the month is where the bond bid lives and
that SPY is flat there (152-168, -0.007%) against a +0.039% all-day drift.
That contrast is the cross-asset claim. This drill checks it properly and asks
the follow-on question the boundary raises: Monday closes August, Tuesday
opens September, and September's reputation is entirely about what comes
after the turn, not about the turn itself.

Guard against the repetition problem: the last four briefs all led on an
August equity slot with a midterm split. Anything here has to be a MONTH
BOUNDARY claim, not a fifth August-slot claim, or it does not publish.

Anchor convention as in drill 01: anchor on the second-to-last session, so
h=1 is the final session itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
)

SUBJ = ["SPY", "QQQ", "IWM", "^GSPC"]
px = close_panel(SUBJ)
cal = px["^GSPC"].dropna().index
ym = pd.Series(cal.year * 100 + cal.month, index=cal)

# see drill 01: August 2026 is incomplete, so it cannot contribute a
# month-end observation. Exclude the trailing month key.
COMPLETE = sorted(set(ym.values))[:-1]

finals = []
for key, grp in ym.groupby(ym.values):
    if key not in COMPLETE:
        continue
    finals.append((key, list(cal).index(grp.index[-1])))
finals.sort(key=lambda x: x[1])

anchors = [(k, cal[p - 1], cal[p]) for k, p in finals if p >= 1]
# first session of the NEXT month, and its first five
firsts = []
for i, (k, p) in enumerate(finals):
    if i + 1 < len(finals) and p + 1 < len(cal):
        firsts.append((k, cal[p], cal[p + 1],
                       cal[min(p + 5, len(cal) - 1)]))


def leg(sub, trips):
    s = px[sub].dropna()
    ds, vs, keys = [], [], []
    for k, a, b in trips:
        if a in s.index and b in s.index:
            ds.append(a)
            vs.append(s.loc[b] / s.loc[a] - 1.0)
            keys.append(k)
    return pd.DatetimeIndex(ds), np.asarray(vs, float), np.asarray(keys)


print("=" * 78)
print("Q1. the final session of the month for equities, all months, against")
print("    the all-day drift. Is the bond bid's slot really an equity dead")
print("    spot, or is that a SPY artifact?")
print("=" * 78)
rows = []
for sub in SUBJ:
    d, v, k = leg(sub, [(a[0], a[1], a[2]) for a in anchors])
    s = px[sub].dropna()
    allr = (s / s.shift(1) - 1.0).dropna().values
    r = summarize(v, f"{sub} final session")
    up = int((v > 0).sum())
    r["record"] = f"{up}-{len(v) - up}"
    r["sign_p_up"] = round(sign_test(up, len(v)), 4)
    rows.append(r)
    b = summarize(allr, f"{sub} all sessions")
    b["record"] = ""
    b["sign_p_up"] = np.nan
    rows.append(b)
show(rows, "final session of the month vs baseline")

print()
print("=" * 78)
print("Q2. the same slot by month. Is the equity dead spot general or does it")
print("    sit in particular months?")
print("=" * 78)
for sub in ["SPY", "^GSPC"]:
    d, v, k = leg(sub, [(a[0], a[1], a[2]) for a in anchors])
    rows = []
    for m in range(1, 13):
        mm = d.month == m
        if mm.sum() < 5:
            continue
        r = summarize(v[mm], pd.Timestamp(2000, m, 1).strftime("%b"))
        up = int((v[mm] > 0).sum())
        r["record"] = f"{up}-{int(mm.sum()) - up}"
        rows.append(r)
    show(rows, f"{sub}: final session by month")

print()
print("=" * 78)
print("Q3. August's final session specifically, all years and midterm.")
print("=" * 78)
for sub in ["^GSPC", "SPY", "QQQ", "IWM"]:
    d, v, k = leg(sub, [(a[0], a[1], a[2]) for a in anchors])
    aug = d.month == 8
    if aug.sum() < 5:
        continue
    da, va = d[aug], v[aug]
    rows = []
    r = summarize(va, "Aug final, all years")
    up = int((va > 0).sum())
    r["record"] = f"{up}-{len(va) - up}"
    r["sign_p_dn"] = round(sign_test(len(va) - up, len(va)), 4)
    rows.append(r)
    mid = np.array([y % 4 == 2 for y in da.year])
    if mid.sum() >= 3:
        r = summarize(va[mid], "Aug final, midterm")
        up = int((va[mid] > 0).sum())
        r["record"] = f"{up}-{int(mid.sum()) - up}"
        r["sign_p_dn"] = round(sign_test(int(mid.sum()) - up, int(mid.sum())), 4)
        rows.append(r)
        r = summarize(va[~mid], "Aug final, non-midterm")
        up = int((va[~mid] > 0).sum())
        r["record"] = f"{up}-{int((~mid).sum()) - up}"
        r["sign_p_dn"] = round(sign_test(int((~mid).sum()) - up, int((~mid).sum())), 4)
        rows.append(r)
    show(rows, f"{sub}: last session of August")
    print("  era:", [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
                     for x in era_split(da, va)])
    print("  conc:", cluster_note(da, va))
    print("  years:", {int(y): round(100 * val, 2) for y, val in zip(da.year, va)})

print()
print("=" * 78)
print("Q4. the follow-on. Monday closes August; the next session opens")
print("    September. h1 = first session of the new month, h5 = its first")
print("    five, both anchored on the final session's close.")
print("=" * 78)
for sub in ["^GSPC", "SPY"]:
    s = px[sub].dropna()
    rows = []
    for label, month_key in [("all months", None), ("into September", 9),
                             ("into any month but September", -9)]:
        ds, v1, v5 = [], [], []
        for k, a, b, c in firsts:
            if a not in s.index or b not in s.index or c not in s.index:
                continue
            nxt = pd.Timestamp(b).month
            if month_key == 9 and nxt != 9:
                continue
            if month_key == -9 and nxt == 9:
                continue
            ds.append(a)
            v1.append(s.loc[b] / s.loc[a] - 1.0)
            v5.append(s.loc[c] / s.loc[a] - 1.0)
        if len(v1) < 5:
            continue
        v1 = np.asarray(v1, float)
        v5 = np.asarray(v5, float)
        r = summarize(v1, f"{label} h1")
        up = int((v1 > 0).sum())
        r["record"] = f"{up}-{len(v1) - up}"
        rows.append(r)
        r = summarize(v5, f"{label} h5")
        up = int((v5 > 0).sum())
        r["record"] = f"{up}-{len(v5) - up}"
        rows.append(r)
    show(rows, f"{sub}: from the final session's close into the new month")
