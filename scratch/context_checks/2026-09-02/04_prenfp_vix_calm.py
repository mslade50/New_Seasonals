"""Tomorrow is the session before payrolls. The SPY arm of that cell was last
night's headline at k=3 and is a countdown re-telling tonight, so this drills
the vol arm instead, which has a mean and a median that disagree.

Engine E:nfp|^VIX|k2: n=317, h1 mean +0.728%, record 139-178 DOWN, sign p
0.0163, control edge +0.471pp, era-stable. VIX rises into the print on average
and falls more often than not. Live state: ^VIX 15.20, 63d return rank 7.1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^VIX", "SPY", "^VIX3M"])
vx = px["^VIX"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()

ev = load_events(["nfp"])["date"]
pos, kept = anchor_positions(vx.index, ev, offset=-2)
anch = vx.index[pos]
anch = anch[anch < vx.index[-1]]
print(f"{len(anch)} anchors, {anch[0].date()} -> {anch[-1].date()}")

r1 = fwd_ret(vx, 1)
r2 = fwd_ret(vx, 2)
s1 = fwd_ret(spy, 1)
s2 = fwd_ret(spy, 2)


def row(v, lab):
    d = summarize(v.values, lab)
    u = int((v > 0).sum())
    d["up"], d["down"] = u, len(v) - u
    d["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4) if len(v) else None
    return d


rows = [row(r1.reindex(anch).dropna(), "^VIX h=1 (Thu)"),
        row(r2.reindex(anch).dropna(), "^VIX h=2 (print day)"),
        row(r1.dropna(), "^VIX all days h=1"),
        row(s1.reindex(anch).dropna(), "SPY h=1 (Thu)"),
        row(s2.reindex(anch).dropna(), "SPY h=2 (print day)")]
show(rows, "the base cell, unconditioned")

# --- condition on the live state: VIX already calm going in ---
rk63 = pct_rank(vx, 63)
lvl = rolling_on_valid(vx, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"\nlive: VIX 63d return rank {rk63.iloc[-1]:.1f}, level rank {lvl.iloc[-1]:.1f}, "
      f"close {vx.iloc[-1]:.2f}")

for lab, m in [("63d rank <= 15", rk63.reindex(anch) <= 15),
               ("63d rank <= 25", rk63.reindex(anch) <= 25),
               ("63d rank > 25", rk63.reindex(anch) > 25),
               ("level rank <= 25", lvl.reindex(anch) <= 25),
               ("level rank <= 40", lvl.reindex(anch) <= 40)]:
    a = anch[m.fillna(False).values]
    if len(a) < 8:
        print(f"  {lab}: n={len(a)}, too few"); continue
    show([row(r1.reindex(a).dropna(), f"^VIX h=1 | {lab}"),
          row(r2.reindex(a).dropna(), f"^VIX h=2 | {lab}"),
          row(s1.reindex(a).dropna(), f"SPY  h=1 | {lab}"),
          row(s2.reindex(a).dropna(), f"SPY  h=2 | {lab}")])

# --- the mean/median split: what drives the positive mean ---
v = r1.reindex(anch).dropna()
print(f"\n=== the mean is a tail: h=1 VIX moves, {len(v)} anchors ===")
print(f"  mean {100*v.mean():+.3f}%  median {100*v.median():+.3f}%  "
      f"up {int((v>0).sum())} down {int((v<=0).sum())}")
q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
print("  quantiles:", {f"{int(100*k)}%": f"{100*x:+.2f}%" for k, x in q.items()})
trimmed = v[(v > v.quantile(0.05)) & (v < v.quantile(0.95))]
print(f"  5-95% trimmed mean {100*trimmed.mean():+.3f}% (n={len(trimmed)})")
big = v[v > 0.05]
print(f"  anchors with VIX +5% or more: {len(big)}, mean of those {100*big.mean():+.2f}%")
era = era_split(v.index, v.values)
show(era, "era split, ^VIX h=1")
print(cluster_note(v.index, v.values, k=2))

# September only
sep = anch[anch.month == 9]
if len(sep) >= 8:
    show([row(r1.reindex(sep).dropna(), "^VIX h=1 | September anchors"),
          row(s1.reindex(sep).dropna(), "SPY  h=1 | September anchors")],
         f"September only ({len(sep)} anchors)")
