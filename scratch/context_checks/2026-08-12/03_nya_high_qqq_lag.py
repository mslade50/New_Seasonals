"""^NYA closed at a 52-week high today. QQQ is 2.90% below its own. IWM and ^RUT also
printed 52-week highs. The broad tape is leading the Nasdaq to a high, which is the
inverse of the last three years' shape and is NOT the SPY/TLT divergence published on
08-10. Cell: ^NYA at a 252-day closing high while QQQ is >= 2.5% under its own.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, local_control,
    sign_test, summarize,
)

px = close_panel(["^NYA", "QQQ", "SPY", "IWM", "^GSPC"])
dates = px.index
nya, qqq, spy, iwm = (px[c] for c in ("^NYA", "QQQ", "SPY", "IWM"))

nya_hi = nya.rolling(252).max()
qqq_hi = qqq.rolling(252).max()
spy_hi = spy.rolling(252).max()
iwm_hi = iwm.rolling(252).max()

at_high = nya >= nya_hi * 0.9999
qqq_gap = qqq / qqq_hi - 1.0

print("tonight: ^NYA at high=%s (%.2f%% under)  QQQ %.2f%% under  SPY %.2f%%  IWM %.2f%%"
      % (bool(at_high.iloc[-1]), 100 * (nya.iloc[-1] / nya_hi.iloc[-1] - 1),
         100 * qqq_gap.iloc[-1], 100 * (spy.iloc[-1] / spy_hi.iloc[-1] - 1),
         100 * (iwm.iloc[-1] / iwm_hi.iloc[-1] - 1)))


def show(label, idx, h=1, tkr="^NYA"):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<48} {tkr:<5} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<48} {tkr:<5} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
    return a, v


print("\n=== how rare is the shape ===")
trig = at_high & (qqq_gap <= -0.025)
ti = trig[trig].index
print(f"  ^NYA 52wh sessions: {int(at_high.sum())}")
print(f"  of those, QQQ >= 2.5% under its own high: {len(ti)} "
      f"({100 * len(ti) / max(1, int(at_high.sum())):.1f}%)")
print(f"  declustered (30td): {len(declusters(ti, 30, dates))}")
yrs = pd.Series(ti.year).value_counts().sort_index()
print(f"  by year: {dict(yrs)}")

print("\n=== forward from the shape ===")
td = declusters(ti, 10, dates)
for tkr in ("^NYA", "QQQ", "SPY", "IWM"):
    for h in (1, 5, 21):
        show(f"NYA at high, QQQ >=2.5% under, h{h}", td, h, tkr)
    print()

print("=== control: ^NYA at a 52w high with QQQ ALSO within 2.5% ===")
trig2 = at_high & (qqq_gap > -0.025)
td2 = declusters(trig2[trig2].index, 10, dates)
for tkr in ("^NYA", "QQQ"):
    for h in (1, 5, 21):
        show(f"NYA at high, QQQ close behind, h{h}", td2, h, tkr)
    print()

print("=== the relative leg: QQQ minus ^NYA after the shape ===")
for h in (1, 5, 10, 21):
    fq = fwd_ret(qqq.dropna(), h)
    fn = fwd_ret(nya.dropna(), h)
    a = td.intersection(fq.dropna().index).intersection(fn.dropna().index)
    v = (fq.loc[a] - fn.loc[a]).values
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  QQQ - ^NYA  h{h:<3} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  {up}-{len(v) - up} QQQ ahead  "
          f"sign p={sign_test(up, len(v)):.4f}")
    # unconditional control for the same spread
    aa = fq.dropna().index.intersection(fn.dropna().index)
    vv = (fq.loc[aa] - fn.loc[aa]).values
    dd = summarize(vv)
    print(f"              control n={len(vv):<5} mean={dd['mean_pct']:+.3f}%  "
          f"med={dd['median_pct']:+.3f}%  {100 * (vv > 0).mean():.1f}% QQQ ahead")

print("\n=== era + concentration on the headline horizon ===")
r = show("NYA at high, QQQ lagging, h21", td, 21, "^NYA")
if r:
    for part in era_split(r[0], r[1]):
        print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
              f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
    print(f"  {cluster_note(r[0], r[1])}")
    print("  occurrences:")
    for dt in td:
        print(f"    {dt.date()}   QQQ {100 * qqq_gap.loc[dt]:+.2f}% under its high")

print("\n=== local control ===")
f1 = fwd_ret(nya.dropna(), 21).dropna()
ctrl = local_control(f1.index, td.intersection(f1.index), 126)
v = f1.loc[ctrl.intersection(f1.index)].values
d = summarize(v)
print(f"  ^NYA h21 local control n={len(v)} mean={d['mean_pct']:+.3f}% hit={d['hit']:.1f}%")
