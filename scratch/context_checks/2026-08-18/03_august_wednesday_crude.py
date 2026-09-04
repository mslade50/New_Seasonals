"""Crude on August Wednesdays.

The sweep has CL=F at -0.385%, t -2.06, n=114 on the bare "Wednesdays in
August" cell, the strongest |t| in that whole trigger group and the only
subject with a named mechanism: the EIA petroleum status report prints
Wednesday 10:30 ET, inside the bar.

The job here is to stop the August framing from being fake precision. If every
Wednesday is negative for crude then August is decoration; if only August
Wednesdays are, the month is doing work and the brief has to say which.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, summarize, show, sign_test, era_split, cluster_note  # noqa: E402

px = close_panel(["CL=F", "NG=F", "HG=F"])
r = px.pct_change(fill_method=None)
d = px.index
dow = d.dayofweek
mon = d.month
yr = d.year

print("=" * 74)
print("A. CL=F: the full weekday x (August / not) grid")
print("=" * 74)
names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
out = []
for wd in range(5):
    for label, mm in [("August", mon == 8), ("other months", mon != 8)]:
        v = r["CL=F"][(dow == wd) & mm].dropna().values
        row = summarize(v, f"{names[wd]} / {label}")
        row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        out.append(row)
show(out, "CL=F daily return by weekday and August membership")

print("\n" + "=" * 74)
print("B. the three competing explanations, head to head")
print("=" * 74)
cl = r["CL=F"]
cells = [
    ("August Wednesdays  (the claim)", (dow == 2) & (mon == 8)),
    ("Wednesdays, all year", dow == 2),
    ("Wednesdays ex-August", (dow == 2) & (mon != 8)),
    ("August, non-Wednesday", (dow != 2) & (mon == 8)),
    ("August, all sessions", mon == 8),
    ("CTRL all days", np.ones(len(d), bool)),
]
out = []
for name, m in cells:
    v = cl[m].dropna().values
    row = summarize(v, name)
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    out.append(row)
show(out, "CL=F: which slice carries it")

augwed = (dow == 2) & (mon == 8)
v_aug = cl[augwed].dropna()
v_wed = cl[(dow == 2) & (mon != 8)].dropna()
diff = v_aug.mean() - v_wed.mean()
se = np.sqrt(v_aug.var(ddof=1) / len(v_aug) + v_wed.var(ddof=1) / len(v_wed))
print(f"\n  August Wed minus other-month Wed: {100*diff:+.3f}pp, t = {diff/se:+.2f}")

v_augoth = cl[(dow != 2) & (mon == 8)].dropna()
diff2 = v_aug.mean() - v_augoth.mean()
se2 = np.sqrt(v_aug.var(ddof=1) / len(v_aug) + v_augoth.var(ddof=1) / len(v_augoth))
print(f"  August Wed minus August non-Wed: {100*diff2:+.3f}pp, t = {diff2/se2:+.2f}")

print("\n" + "=" * 74)
print("C. era split and concentration on the August Wednesday cell")
print("=" * 74)
dd = cl[augwed].dropna().index
vv = cl[augwed].dropna().values
show(era_split(dd, vv), "CL=F August Wednesdays, era split at 2018")
print(" ", cluster_note(dd, vv, k=2))
by_yr = pd.Series(vv).groupby(dd.year.values).mean() * 100
print(f"  negative in {int((by_yr < 0).sum())} of {len(by_yr)} Augusts")
print("  per-August mean %:", {int(k): round(v, 2) for k, v in by_yr.items()})

print("\n" + "=" * 74)
print("D. does the same shape appear in the other Wednesday-sensitive commodity")
print("=" * 74)
out = []
for t in ["NG=F", "HG=F"]:
    for name, m in [("Wed all year", dow == 2), ("August Wed", (dow == 2) & (mon == 8)),
                    ("all days", np.ones(len(d), bool))]:
        v = r[t][m].dropna().values
        out.append(summarize(v, f"{t} {name}"))
show(out, "NG=F prints Thursday 10:30 (gas storage), HG=F has no weekly print")

print("\n" + "=" * 74)
print("E. the specific cell for TOMORROW: Wednesday, August, midterm year")
print("=" * 74)
m = (dow == 2) & (mon == 8) & (yr % 4 == 2)
v = cl[m].dropna()
row = summarize(v.values, "CL=F August Wed, midterm years")
row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
show([row], "midterm subset")
