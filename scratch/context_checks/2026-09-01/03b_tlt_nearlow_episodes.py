"""Name the 7 near-low sessions from 03, so the concentration and era of the
exception arm are on the record rather than asserted. Also nail down the
early-September midterm TLT arm with the engine's +/-2 doy window."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TLT", "IEF", "LQD", "SPY"]).dropna(subset=["TLT"])


def down_streak(s):
    dn = (s.diff() < 0).astype(int)
    out, run = [], 0
    for v in dn.values:
        run = run + 1 if v else 0
        out.append(run)
    return pd.Series(out, index=s.index)


st = down_streak(px["TLT"])
low252 = rolling_on_valid(px["TLT"], lambda x: x.rolling(252, min_periods=200).min())
near = (st >= 5) & (px["TLT"] <= low252 * 1.01)
dts = px.index[near.fillna(False).values]

print(f"=== the {len(dts)} sessions where a 5+ TLT down streak sat within 1% of a 252d low ===")
for d in dts:
    row = [f"  {d.date()}  streak {int(st[d])}",
           f"{100*(px['TLT'][d]/low252[d]-1):+.2f}% over the low"]
    for h in (1, 5, 21):
        v = fwd_ret(px["TLT"], h).get(d, np.nan)
        row.append(f"h{h} {100*v:+6.2f}%" if not np.isnan(v) else f"h{h}    n/a")
    print("  ".join(row))

print(f"\n  distinct episodes (21td gap): "
      f"{[str(x.date()) for x in declusters(dts, 21, px.index)]}")
print(f"  years: {sorted(set(dts.year))}")

for h in (5, 21):
    v = fwd_ret(px["TLT"], h).loc[dts].dropna()
    w = int((v.values > 0).sum())
    print(f"  h={h}: {w}-{len(v)-w}, mean {100*v.mean():+.2f}%, "
          f"sign p (losses) {sign_test(len(v)-w, len(v)):.4f}")

print("\n=== early-September TLT, midterm years, engine construction (doy +/-2) ===")
# trading-day-of-year of 2026-09-02, matched in prior years
idx = px.index
this_yr = idx[idx.year == 2026]
doy_pos = list(this_yr).index(pd.Timestamp("2026-09-01")) + 1  # next session's tdoy
print(f"  next session is trading day {doy_pos+1} of 2026")
rows = []
for y in sorted(set(idx.year)):
    if y == 2026:
        continue
    yr = idx[idx.year == y]
    if len(yr) <= doy_pos + 6:
        continue
    a = yr[doy_pos]           # same trading-day-of-year
    h5 = fwd_ret(px["TLT"], 5).get(a, np.nan)
    h1 = fwd_ret(px["TLT"], 1).get(a, np.nan)
    rows.append((y, a.date(), 100 * h1, 100 * h5, y % 4 == 2))
print("  year  anchor        h1       h5   midterm")
for y, a, h1, h5, mt in rows:
    print(f"  {y}  {a}  {h1:+6.2f}%  {h5:+6.2f}%   {'yes' if mt else ''}")

for label, sel in (("all years", rows), ("midterm only", [r for r in rows if r[4]])):
    v5 = np.array([r[3] for r in sel if not np.isnan(r[3])]) / 100
    if len(v5) == 0:
        continue
    w = int((v5 > 0).sum())
    print(f"  {label}: h5 n={len(v5)}, {w}-{len(v5)-w}, mean {100*v5.mean():+.2f}%, "
          f"median {100*np.median(v5):+.2f}%, sign p (losses) {sign_test(len(v5)-w, len(v5)):.4f}")
