"""^MOVE fell 7.48% today to 72.09 while ^TNX moved -0.04% and TLT -0.10%. Bond
volatility collapsed on a CPI print without the yield going anywhere. Off the trigger
inventory entirely (P10 covers VIX term structure, not MOVE).

Cell: ^MOVE 1-day drop <= -5% with |^TNX 1-day move| <= 0.5%, i.e. the vol came out
without the level moving. Forward for TLT, IEF, ^TNX and ^MOVE itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, load_events,
    local_control, sign_test, summarize,
)

px = close_panel(["^MOVE", "^TNX", "TLT", "IEF", "SPY"])
move = px["^MOVE"].dropna()
tnx = px["^TNX"].dropna()
dates = move.index

m1 = move.pct_change()
t1 = tnx.pct_change().reindex(move.index)

print("tonight: ^MOVE %.2f (%+.2f%%)  ^TNX %.3f (%+.2f%%)  "
      "MOVE 252d pctile %.1f"
      % (move.iloc[-1], 100 * m1.iloc[-1], tnx.iloc[-1], 100 * t1.iloc[-1],
         100 * move.rolling(252).rank(pct=True).iloc[-1]))


def show(label, idx, h=1, tkr="TLT"):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<46} {tkr:<6} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<46} {tkr:<6} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
    return a, v


print("\n=== A. MOVE 1d <= -5%, any yield move ===")
A = (m1 <= -0.05)
Ai = declusters(A[A].index, 5, dates)
print(f"  raw {int(A.sum())}, declustered(5td) {len(Ai)}")
for tkr in ("TLT", "IEF", "^TNX"):
    for h in (1, 5, 21):
        show(f"MOVE -5%+  h{h}", Ai, h, tkr)
    print()

print("=== B. MOVE 1d <= -5% AND |TNX 1d| <= 0.5% (the level did not move) ===")
B = (m1 <= -0.05) & (t1.abs() <= 0.005)
Bi = declusters(B[B].index, 5, dates)
print(f"  raw {int(B.sum())}, declustered {len(Bi)}")
for tkr in ("TLT", "IEF", "^TNX", "SPY"):
    for h in (1, 5, 21):
        show(f"MOVE -5%+, TNX flat  h{h}", Bi, h, tkr)
    print()

print("=== does the vol crush persist or snap back? ^MOVE itself ===")
for h in (1, 5, 10, 21):
    show(f"MOVE after its own -5% day  h{h}", Ai, h, "^MOVE")
print()
for h in (1, 5, 10, 21):
    show(f"MOVE after -5% with TNX flat  h{h}", Bi, h, "^MOVE")

fm = fwd_ret(move, 5).dropna()
d = summarize(fm.values)
print(f"\n  ^MOVE h5 all sessions control: n={len(fm)} mean={d['mean_pct']:+.3f}% "
      f"med={d['median_pct']:+.3f}% hit={d['hit']:.1f}%")

print("\n=== C. same, but only when the crush lands on a CPI print ===")
ev = load_events()
cpi = set(ev.loc[ev["event"] == "cpi", "date"])
Ci = pd.DatetimeIndex([d0 for d0 in A[A].index if d0 in cpi])
print(f"  MOVE -5% days that were CPI prints: {len(Ci)}")
for tkr in ("TLT", "IEF", "^MOVE"):
    for h in (1, 5):
        show(f"MOVE -5% on a CPI print  h{h}", Ci, h, tkr)
    print()

print("=== era + concentration, cell B, TLT h5 ===")
r = show("cell B TLT h5", Bi, 5, "TLT")
if r:
    for part in era_split(r[0], r[1]):
        print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
              f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
    print(f"  {cluster_note(r[0], r[1])}")
    print("  most recent occurrences:")
    for dt in list(Bi)[-10:]:
        print(f"    {dt.date()}  MOVE {100 * m1.loc[dt]:+.2f}%  TNX {100 * t1.loc[dt]:+.2f}%")

print("\n=== local control for cell B ===")
for tkr in ("TLT", "IEF"):
    f = fwd_ret(px[tkr].dropna(), 5).dropna()
    ctrl = local_control(f.index, Bi.intersection(f.index), 126)
    v = f.loc[ctrl.intersection(f.index)].values
    d = summarize(v)
    print(f"  {tkr:<5} h5 local control n={len(v)} mean={d['mean_pct']:+.3f}% "
          f"hit={d['hit']:.1f}%")
