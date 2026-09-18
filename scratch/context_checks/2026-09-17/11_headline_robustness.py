"""Era and concentration checks on the two cells I intend to publish first.

The honesty contract: any nugget whose sign flips across 2018 says so or dies,
and any nugget whose mean lives in its top two episodes says so. Neither check
was run inside 02 for the IWM-minus-SPY spread, which is the headline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, cluster_note,
)

px = close_panel(["IWM", "SPY", "QQQ", "^VIX"])
idx = px.index
qw = load_events(["quad_witching"])["date"]
sep = pd.DatetimeIndex([d for d in qw if d in set(idx) and d.month == 9])

i, s = px["IWM"].dropna(), px["SPY"].dropna()
f = (i.shift(-5) / i - 1.0) - (s.shift(-5) / s - 1.0)
d = sep.intersection(f.dropna().index)
v = f.loc[d]

print("=== IWM minus SPY, 5 sessions after September quad witching ===")
print(f"n={len(v)} mean {100 * v.mean():+.3f}pp median {100 * v.median():+.3f}pp "
      f"t {v.mean() / (v.std(ddof=1) / np.sqrt(len(v))):.3f}")
k = int((v > 0).sum())
print(f"IWM ahead {k} of {len(v)}, sign p(IWM lags) = {sign_test(len(v) - k, len(v)):.4f}")
for lab, m in [("pre-2018", d < pd.Timestamp("2018-01-01")),
               ("2018+", d >= pd.Timestamp("2018-01-01"))]:
    vv = f.loc[d[m]]
    kk = int((vv > 0).sum())
    print(f"  {lab:<10} n={len(vv):<3} mean {100 * vv.mean():+.3f}pp "
          f"median {100 * vv.median():+.3f}pp  IWM ahead {kk} of {len(vv)}")
print(f"  concentration: {cluster_note(d, v.values, k=2)}")
print(f"  worst {100 * v.min():+.2f}pp ({v.idxmin().year}), "
      f"best {100 * v.max():+.2f}pp ({v.idxmax().year})")
print(f"  year by year: {dict((x.year, round(100 * y, 2)) for x, y in v.items())}")
for phase, mod in [("midterm", 2)]:
    vv = f.loc[d[d.year % 4 == mod]]
    kk = int((vv > 0).sum())
    print(f"  {phase}: n={len(vv)} mean {100 * vv.mean():+.3f}pp "
          f"IWM ahead {kk} of {len(vv)}")

print("\n=== drop the two biggest episodes and re-measure ===")
order = np.argsort(-np.abs(v.values))[:2]
keep = np.ones(len(v), dtype=bool)
keep[order] = False
vv = v[keep]
kk = int((vv > 0).sum())
print(f"  n={len(vv)} mean {100 * vv.mean():+.3f}pp "
      f"t {vv.mean() / (vv.std(ddof=1) / np.sqrt(len(vv))):.3f} "
      f"IWM ahead {kk} of {len(vv)}, "
      f"sign p(IWM lags) = {sign_test(len(vv) - kk, len(vv)):.4f}")
print(f"  dropped: {[str(v.index[o].date()) for o in order]}")

print("\n=== SPY outright, same window, drop-two ===")
fs = s.shift(-5) / s - 1.0
ds = sep.intersection(fs.dropna().index)
vs = fs.loc[ds]
order = np.argsort(-np.abs(vs.values))[:2]
keep = np.ones(len(vs), dtype=bool)
keep[order] = False
w = vs[keep]
kw = int((w > 0).sum())
print(f"  full: n={len(vs)} mean {100 * vs.mean():+.3f}% "
      f"{int((vs > 0).sum())}-{len(vs) - int((vs > 0).sum())} up")
print(f"  drop-two: n={len(w)} mean {100 * w.mean():+.3f}% {kw}-{len(w) - kw} up "
      f"(dropped {[str(vs.index[o].date()) for o in order]})")

print("\n=== December witching, the mirror control ===")
dec = pd.DatetimeIndex([d for d in qw if d in set(idx) and d.month == 12])
for tkr in ["SPY", "IWM"]:
    ser = px[tkr].dropna()
    fd = ser.shift(-5) / ser - 1.0
    dd = dec.intersection(fd.dropna().index)
    k = int((fd.loc[dd] > 0).sum())
    print(f"  {tkr}: n={len(dd)} mean {100 * fd.loc[dd].mean():+.3f}% "
          f"{k}-{len(dd) - k} up, t "
          f"{fd.loc[dd].mean() / (fd.loc[dd].std(ddof=1) / np.sqrt(len(dd))):.2f}, "
          f"sign p(up) = {sign_test(k, len(dd)):.4f}")

print("\n=== VIX, 5 sessions after September witching: drop-two ===")
vx = px["^VIX"].dropna()
fv = vx.shift(-5) / vx - 1.0
dv = sep.intersection(fv.dropna().index)
vv = fv.loc[dv]
order = np.argsort(-np.abs(vv.values))[:2]
keep = np.ones(len(vv), dtype=bool)
keep[order] = False
w = vv[keep]
kw = int((w > 0).sum())
print(f"  full: n={len(vv)} mean {100 * vv.mean():+.2f}% "
      f"{int((vv > 0).sum())}-{len(vv) - int((vv > 0).sum())} up")
print(f"  drop-two: n={len(w)} mean {100 * w.mean():+.2f}% {kw}-{len(w) - kw} up "
      f"(dropped {[str(vv.index[o].date()) for o in order]})")
