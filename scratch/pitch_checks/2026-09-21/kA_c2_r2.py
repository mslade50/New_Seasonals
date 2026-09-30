"""kA c2 round 2 - audit the whole-percent episode list and the placebo gap.

1. Every whole-percent approach/first-cross day since 2000 WITHOUT the parent gate, with
   rise63 and the gap to the 252 max, so the episode list is audited not trusted
   (e.g. why 2018's 3% does or does not appear).
2. Episode-level whole vs placebo on each individual grid, at the same gap, h=1..10
   (TLT, IEF, -dTNX), to see whether the whole-percent grid ever separates.
3. Midterm split on parent / whole / placebo.
4. The live cluster so far (09-11 first day): TLT from the 09-14 close.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "TLT", "IEF"])
idx = px.index
y = px["^TNX"]
yv = y.dropna()
rise63 = (yv - yv.shift(63)).reindex(idx)
max252 = rolling_on_valid(y, lambda x: x.rolling(252).max())
prior_max = rolling_on_valid(y, lambda x: x.shift(1).rolling(252).max())
parent = ((rise63 >= 0.50) & (y >= max252 - 0.05)).reindex(idx, fill_value=False)


def masks(f):
    lup = np.ceil(y - f - 1e-9) + f
    ldn = np.floor(y - f + 1e-9) + f
    below = (((lup - y) <= 0.05) & ((lup - y) >= 0)).fillna(False)
    xfirst = ((y >= ldn) & (prior_max < ldn)).fillna(False)
    return below, xfirst, lup, ldn


b, x, lup, ldn = masks(0.0)
print("=== 1. whole-percent FIRST CROSSES since 2000 (no parent gate) ===")
d = idx[x.values]
e = declusters(d, 21, idx)
rows = []
for a in e:
    rows.append({"date": a.date(), "TNX": round(y[a], 3), "level": ldn[a], "rise63bp": round(100 * rise63[a], 1),
                 "bp_below_max": round(100 * (max252[a] - y[a]), 1), "parent": bool(parent[a]),
                 "TLT5": round(100 * fwd_lag(px["TLT"], 5).get(a, np.nan), 2),
                 "TLT10": round(100 * fwd_lag(px["TLT"], 10).get(a, np.nan), 2),
                 "-dy10bp": round(-100 * (y.shift(-11) - y.shift(-1)).get(a, np.nan), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 1b. whole-percent APPROACH (within 5bp below) clusters with rise63 >= 30bp (no max gate) ===")
d = idx[(b & (rise63 >= 0.30)).reindex(idx, fill_value=False).values]
e = declusters(d, 21, idx)
rows = []
for a in e:
    rows.append({"date": a.date(), "TNX": round(y[a], 3), "level": lup[a], "rise63bp": round(100 * rise63[a], 1),
                 "bp_below_max": round(100 * (max252[a] - y[a]), 1), "parent": bool(parent[a]),
                 "TLT5": round(100 * fwd_lag(px["TLT"], 5).get(a, np.nan), 2),
                 "TLT10": round(100 * fwd_lag(px["TLT"], 10).get(a, np.nan), 2),
                 "-dy10bp": round(-100 * (y.shift(-11) - y.shift(-1)).get(a, np.nan), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 1c. spot audit 2018 (3%) ===")
s = pd.DataFrame({"TNX": y, "rise63": 100 * rise63, "max252": max252, "parent": parent, "below": b, "xfirst": x})
print(s.loc["2018-01-25":"2018-02-23"].round(3).to_string())
print(s.loc["2018-04-18":"2018-05-18"].round(3).to_string())

# ---------------------------------------------------------------- 2. horizon table
print("\n=== 2. whole vs each placebo grid across horizons (episodes, gap 21) ===")
cells = {f: (parent & (masks(f)[0] | masks(f)[1])).reindex(idx, fill_value=False) for f in (0, .25, .5, .75)}
plac = cells[.25] | cells[.5] | cells[.75]
rows = []
for h in (1, 2, 3, 5, 7, 10):
    rt = fwd_lag(px["TLT"], h)
    ri = fwd_lag(px["IEF"], h)
    rd = -100 * (y.shift(-(1 + h)) - y.shift(-1))
    for lbl, m in [("whole", cells[0]), ("q.25", cells[.25]), ("q.50", cells[.5]), ("q.75", cells[.75]),
                   ("placebo pooled", plac), ("parent", parent)]:
        dd = idx[m.values].intersection(rt.dropna().index)
        ee = declusters(dd, 21, idx)
        vt, vi, vd = rt.loc[ee].values, ri.loc[ee].values, rd.loc[ee].values
        rows.append({"h": h, "cell": lbl, "n": len(ee), "TLT%": round(100 * vt.mean(), 3),
                     "TLT_rec": f"{(vt>0).sum()}-{(vt<=0).sum()}", "IEF%": round(100 * vi.mean(), 3),
                     "-dy bp": round(np.nanmean(vd), 2),
                     "TLT_base%": round(100 * rt.dropna().mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 3. midterm
print("\n=== 3. midterm split, TLT h=5 / h=10 (episodes) ===")
mid = pd.Series(idx.year % 4 == 2, index=idx)
for h in (5, 10):
    r = fwd_lag(px["TLT"], h)
    out = []
    for lbl, m in [("parent", parent), ("whole", cells[0]), ("placebo", plac)]:
        for ml, mm in [("midterm", mid), ("non-mid", ~mid)]:
            dd = idx[(m & mm).values].intersection(r.dropna().index)
            ee = declusters(dd, 21, idx)
            out.append(summarize(r.loc[ee].values, f"{lbl} {ml} h={h}"))
    show(out)

# ---------------------------------------------------------------- 4. live cluster so far
print("\n=== 4. live cluster: TLT/IEF/TNX since the 09-11 first qualifying close ===")
for c in ("TLT", "IEF"):
    p = px[c].dropna()
    print(f"  {c}: 09-11 {p['2026-09-11']:.2f}  09-14 {p['2026-09-14']:.2f}  09-18 {p['2026-09-18']:.2f}  "
          f"(from 09-14 entry {100*(p['2026-09-18']/p['2026-09-14']-1):+.2f}%)")
print(f"  TNX 09-11 {y['2026-09-11']:.3f} -> 09-18 {y['2026-09-18']:.3f}")
