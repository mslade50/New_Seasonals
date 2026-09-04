"""Consolidation: the exact figures the brief quotes, each cell with the
control computed on ITS OWN definition rather than a neighbouring one.

Tonight VIX closes at the 13.1st percentile of its trailing year, i.e. the
bottom quartile but not the bottom decile, so the published cell is the
quartile one and its control has to match that definition exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^VIX", "SPY"])
vix = px["^VIX"]["Close"].dropna()
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))
d = vix.index
anc = []
for x in cpi:
    pos = d.searchsorted(x)
    if pos <= 0 or pos >= len(d) or d[pos] != x:
        continue
    anc.append(d[pos - 1])
anc = pd.DatetimeIndex(anc)

r = vix.pct_change()
nxt = r.shift(-1)
pct = vix.rolling(252, min_periods=252).rank(pct=True) * 100

print(f"today VIX {vix.iloc[-1]:.2f}  trailing-252 pctile {pct.iloc[-1]:.1f}")

q1 = pd.DatetimeIndex([x for x in anc if pd.notna(pct.get(x, np.nan)) and pct[x] < 25])
v = nxt.reindex(q1).dropna()
s = summarize(v.values, "cpi | vix bottom quartile")
dn, up = int((v < 0).sum()), int((v > 0).sum())
print(f"\nCELL  CPI print, VIX entering in the bottom quartile of its trailing year")
print(f"  n={s['n']}  mean {s['mean_pct']:+.3f}%  down {dn}-{up} up  ({dn/len(v)*100:.1f}% down)  "
      f"t={s['t']:+.2f}  sign p {sign_test(dn, len(v)):.4f}")
print(f"  era: {[(e['label'], e['n'], round(e['mean_pct'],3)) for e in era_split(v.index, v.values)]}")
for e in era_split(v.index, v.values):
    sub = v[v.index < '2018-01-01'] if e['label'].startswith('pre') else v[v.index >= '2018-01-01']
    print(f"    {e['label']}: {int((sub<0).sum())}-{int((sub>0).sum())} down  ({(sub<0).mean()*100:.1f}%)")
print(f"  concentration: {cluster_note(v.index, v.values)}")
print(f"  years: {sorted(set(v.index.year))}")

# CONTROL on the same definition: VIX in the bottom quartile, no print next session
same = pd.DatetimeIndex([x for x in d if pd.notna(pct.get(x, np.nan)) and pct[x] < 25])
ctrl = same.difference(anc)
vc = nxt.reindex(ctrl).dropna()
sc = summarize(vc.values, "ctrl")
dnc = int((vc < 0).sum())
print(f"  CONTROL, VIX bottom quartile with NO print the next session:")
print(f"    n={sc['n']}  mean {sc['mean_pct']:+.3f}%  down {dnc}-{int((vc>0).sum())}  ({dnc/len(vc)*100:.1f}% down)")
print(f"  EDGE {s['mean_pct']-sc['mean_pct']:+.3f}pp on the mean, "
      f"{dn/len(v)*100 - dnc/len(vc)*100:+.1f}pp on the down rate")

# and every CPI print for the top line
vall = nxt.reindex(anc).dropna()
sa = summarize(vall.values, "all")
print(f"\n  ALL CPI prints: n={sa['n']} mean {sa['mean_pct']:+.3f}% "
      f"down {int((vall<0).sum())}-{int((vall>0).sum())} t={sa['t']:+.2f}")
