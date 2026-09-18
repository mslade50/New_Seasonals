"""The curve at 52-week yield highs with equity vol asleep, into a CPI week.

No P-trigger fires on "near an extreme without printing one", so this state is
invisible to the sweep. Tonight: ^TNX 4.784% is 0.25% from its 52-week high,
^FVX 0.15%, ^IRX 63d rank 94; TLT sits 1.44% off its 52-week LOW, IEF 0.44%.
VIX meanwhile is in the 9th percentile of its 63-day range.

Three cells:
  A  ^TNX within 1% (relative) of its 52-week high -> forward TNX / TLT / SPY
  B  the same with VIX 63d rank <= 15, the divergence that is actually live
  C  the midterm-September doy claim for ^TNX h5 the engine reported 5 of 6
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, summarize, sign_test, era_split,  # noqa
                       cluster_note, pct_rank, declusters, local_control)

px = close_panel(["^TNX", "TLT", "IEF", "^GSPC", "SPY", "^VIX"])
tnx, tlt, spx, vix = (px[c].dropna() for c in ("^TNX", "TLT", "^GSPC", "^VIX"))
idx = tnx.index
MID = {y for y in range(1996, 2030) if y % 4 == 2}


def line(label, dates, s, h):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:46} h{h:<3} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum()); st = summarize(v, label)
    print(f"  {label:46} h{h:<3} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


hi52 = tnx.rolling(252).max()
dist = tnx / hi52 - 1.0
print(f"^TNX close 2026-09-04: {tnx.iloc[-1]:.3f}, 52w high {hi52.iloc[-1]:.3f}, "
      f"distance {100*dist.iloc[-1]:+.2f}%")
print(f"VIX 63d rank: {pct_rank(vix,63,252).iloc[-1]:.1f}")
print(f"TLT distance above its 52w low: "
      f"{100*(tlt.iloc[-1]/tlt.rolling(252).min().iloc[-1]-1):+.2f}%")

print("\n=== A) ^TNX within 1% of its 52-week high ===")
m = dist >= -0.01
trig = pd.DatetimeIndex([d for d in m.index[m.fillna(False)] if d < idx[-1]])
dec = declusters(trig, 10, idx)
print(f"  {len(trig)} sessions, {len(dec)} declustered episodes, "
      f"years {sorted(set(d.year for d in dec))}")
for h in (1, 5, 10, 21):
    line("^TNX", dec, tnx, h)
for h in (5, 21):
    line("TLT", dec, tlt, h)
for h in (1, 5, 21):
    line("^GSPC", dec, spx, h)
print("  controls")
ctrl = local_control(idx, dec, 126)
for h in (5, 21):
    line("  ^TNX local +/-126td", ctrl, tnx, h)
    line("  ^GSPC local +/-126td", ctrl, spx, h)
for h in (5, 21):
    r = fwd_ret(spx, h).dropna(); up = int((r.values > 0).sum())
    print(f"  {'  ^GSPC all sessions':46} h{h:<3} n={len(r):5d} "
          f"mean={100*r.values.mean():+7.3f}% hit={100*up/len(r):5.1f}%")

print("\n=== B) the divergence: yields at a 52w high AND VIX 63d rank <= 15 ===")
vr = pct_rank(vix, 63, 252)
m2 = m & (vr <= 15)
trig2 = pd.DatetimeIndex([d for d in m2.index[m2.fillna(False)] if d < idx[-1]])
dec2 = declusters(trig2, 21, idx)
print(f"  {len(trig2)} sessions, {len(dec2)} declustered episodes at 21 td")
print(f"  episodes: {[str(d.date()) for d in dec2]}")
for h in (1, 5, 10, 21):
    line("^GSPC after the divergence", dec2, spx, h)
for h in (5, 21):
    line("^TNX after the divergence", dec2, tnx, h)
    line("^VIX after the divergence", dec2, vix, h)
r21 = line("^GSPC after the divergence", dec2, spx, 21)
if r21 is not None and len(r21) >= 5:
    print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}%"
                     for e in era_split(r21.index, r21.values)])
    print("  concentration:", cluster_note(r21.index, r21.values, 2))

print("\n=== C) midterm-September doy for ^TNX, h5 ===")
doy = []
for yr in sorted(set(idx.year)):
    c = idx[(idx.year == yr) & (idx.month == 9)]
    c = c[(c >= pd.Timestamp(yr, 9, 6)) & (c <= pd.Timestamp(yr, 9, 10))]
    if len(c):
        doy.append(c[abs((c - pd.Timestamp(yr, 9, 8)).days).argmin()])
doy = pd.DatetimeIndex(doy)
mid = pd.DatetimeIndex([d for d in doy if d.year in MID])
r = fwd_ret(tnx, 5).reindex(mid).dropna()
print(f"  midterm Sep-08 anchors: {[str(d.date()) for d in r.index]}")
print(f"  ^TNX h5: {[f'{d.year}:{100*v:+.2f}%' for d, v in zip(r.index, r.values)]}")
line("^TNX midterm Sep-08", mid, tnx, 5)
line("^TNX all-year Sep-08", doy, tnx, 5)
line("TLT midterm Sep-08", mid, tlt, 5)
