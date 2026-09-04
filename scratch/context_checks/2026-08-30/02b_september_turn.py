"""Era stability, concentration and controls for the one live finding in
drill 02: the turn-of-month bid is absent into September.

Drill 02 Q4: anchored on the final session's close, SPY's first session of the
new month runs 198-120 at +0.180% across 318 months, but the 26 turns into
September run 13-13 at -0.260%, and the first five sessions 11-15.

A 26-observation cell needs its era split and its concentration before it can
be quoted, and the claim is a DIFFERENCE so it needs the non-September arm
tested as the control rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
)

px = close_panel(["SPY", "^GSPC", "IWM", "QQQ"])
cal = px["^GSPC"].dropna().index
ym = pd.Series(cal.year * 100 + cal.month, index=cal)
COMPLETE = sorted(set(ym.values))[:-1]

finals = []
for key, grp in ym.groupby(ym.values):
    if key in COMPLETE:
        finals.append(list(cal).index(grp.index[-1]))
finals.sort()

# anchor on the final session's close; h1 = first session of the new month
turns = [(cal[p], cal[p + 1], cal[min(p + 5, len(cal) - 1)])
         for p in finals if p + 1 < len(cal)]


def arm(sub, want_sep):
    s = px[sub].dropna()
    ds, v1, v5 = [], [], []
    for a, b, c in turns:
        if a not in s.index or b not in s.index or c not in s.index:
            continue
        if (pd.Timestamp(b).month == 9) != want_sep:
            continue
        ds.append(a)
        v1.append(s.loc[b] / s.loc[a] - 1.0)
        v5.append(s.loc[c] / s.loc[a] - 1.0)
    return pd.DatetimeIndex(ds), np.asarray(v1, float), np.asarray(v5, float)


print("=" * 78)
print("Q1. the September arm, era split and concentration, h1 and h5")
print("=" * 78)
for sub in ["SPY", "^GSPC", "QQQ", "IWM"]:
    d, v1, v5 = arm(sub, True)
    up1 = int((v1 > 0).sum())
    up5 = int((v5 > 0).sum())
    print(f"\n  {sub}: n={len(v1)}  h1 {up1}-{len(v1)-up1} "
          f"mean {100*v1.mean():+.3f}%  |  h5 {up5}-{len(v5)-up5} "
          f"mean {100*v5.mean():+.3f}%")
    print("    h1 era:", [(r["label"], r["n"], round(r["mean_pct"], 3),
                           round(r["hit"], 1)) for r in era_split(d, v1)])
    print("    h5 era:", [(r["label"], r["n"], round(r["mean_pct"], 3),
                           round(r["hit"], 1)) for r in era_split(d, v5)])
    print("    h5 conc:", cluster_note(d, v5))

print()
print("=" * 78)
print("Q2. the difference, tested. September arm vs the other eleven months,")
print("    Welch t on the h1 and h5 return distributions.")
print("=" * 78)
rows = []
for sub in ["SPY", "^GSPC", "QQQ", "IWM"]:
    ds, s1, s5 = arm(sub, True)
    do, o1, o5 = arm(sub, False)
    for h, a, b in [("h1", s1, o1), ("h5", s5, o5)]:
        va = a.var(ddof=1) / len(a)
        vb = b.var(ddof=1) / len(b)
        tw = (a.mean() - b.mean()) / np.sqrt(va + vb)
        upa, upb = int((a > 0).sum()), int((b > 0).sum())
        rows.append({
            "subject": sub, "h": h,
            "sep_n": len(a), "sep_mean_pct": round(100 * a.mean(), 3),
            "sep_rec": f"{upa}-{len(a)-upa}",
            "other_n": len(b), "other_mean_pct": round(100 * b.mean(), 3),
            "other_rec": f"{upb}-{len(b)-upb}",
            "diff_pp": round(100 * (a.mean() - b.mean()), 3),
            "welch_t": round(tw, 2),
        })
show(rows, "September turn vs every other turn")

print()
print("=" * 78)
print("Q3. is it September the month, or is it the turn specifically? Compare")
print("    the turn window against the REST of September.")
print("=" * 78)
for sub in ["SPY"]:
    s = px[sub].dropna()
    r = (s / s.shift(1) - 1.0)
    sep = r[r.index.month == 9].dropna()
    d, v1, _ = arm(sub, True)
    firstsess = set()
    for a, b, c in turns:
        if pd.Timestamp(b).month == 9:
            firstsess.add(pd.Timestamp(b))
    rest = sep[[x not in firstsess for x in sep.index]]
    rows = [summarize(v1, "1st session of Sep"),
            summarize(rest.values, "every other Sep session"),
            summarize(r.dropna().values, "all sessions, all months")]
    for row, vals in zip(rows, [v1, rest.values, r.dropna().values]):
        up = int((np.asarray(vals) > 0).sum())
        row["record"] = f"{up}-{len(vals) - up}"
    show(rows, f"{sub}: September decomposed")

print()
print("=" * 78)
print("Q4. midterm years only, since 2026 is one")
print("=" * 78)
for sub in ["SPY", "^GSPC"]:
    d, v1, v5 = arm(sub, True)
    mid = np.array([y % 4 == 2 for y in d.year])
    for label, m in [("midterm", mid), ("non-midterm", ~mid)]:
        if m.sum() < 3:
            continue
        up1 = int((v1[m] > 0).sum())
        print(f"  {sub} {label:12s} n={int(m.sum())}  h1 {up1}-{int(m.sum())-up1} "
              f"{100*v1[m].mean():+.3f}%   h5 {100*v5[m].mean():+.3f}%  "
              f"years {sorted(set(d[m].year))}")
