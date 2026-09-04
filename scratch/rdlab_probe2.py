"""rdlab_probe2.py — is it the weights or the decay? Unweighted-but-decayed
count vs weighted composite. Recompute vintage (pit_signals.pkl) — PIT caveat."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pk = pickle.load(open(ROOT / "scratch" / "pit_signals.pkl", "rb"))
fires: pd.DataFrame = pk["fires"]
comp: pd.DataFrame = pk["frag_df_current"]
idx = fires.index.intersection(comp.index)
f = fires.loc[idx].astype(float)
comp = comp.loc[idx]

for h, win in [("5d", 5), ("21d", 21), ("63d", 63)]:
    # per-signal: linearly-decaying persistence since last fire, equal weight
    dec = pd.DataFrame(index=f.index, columns=f.columns, dtype=float)
    kern = np.linspace(1.0, 1.0 / win, win)  # fresh fire=1, decays to ~0 over win
    for c in f.columns:
        arr = f[c].values
        out = np.zeros(len(arr))
        for i in range(len(arr)):
            lo = max(0, i - win + 1)
            seg = arr[lo:i + 1][::-1]  # most recent first
            out[i] = float((seg * kern[:len(seg)]).max()) if seg.any() else 0.0
        dec[c] = out
    eqsum = dec.sum(axis=1)
    print(f"{h}: equal-weight decayed sum vs weighted composite: "
          f"pearson {eqsum.corr(comp[h]):.3f}, spearman {eqsum.corr(comp[h], method='spearman'):.3f}")
    # rolling any-fired count in the window (simplest possible)
    rollcnt = f.rolling(win).max().sum(axis=1)
    print(f"     rolling-{win}d any-fired count vs composite: "
          f"pearson {rollcnt.corr(comp[h]):.3f}, spearman {rollcnt.corr(comp[h], method='spearman'):.3f}")

# FAMILY4 gate replication with the decayed equal-weight 63d proxy
win = 63
dec = pd.DataFrame(index=f.index, columns=f.columns, dtype=float)
kern = np.linspace(1.0, 1.0 / win, win)
for c in f.columns:
    arr = f[c].values
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - win + 1)
        seg = arr[lo:i + 1][::-1]
        out[i] = float((seg * kern[:len(seg)]).max()) if seg.any() else 0.0
    dec[c] = out
eqsum63 = dec.sum(axis=1)
gate = (comp["63d"].rolling(10).mean() >= 50)
best = None
prox = eqsum63.rolling(10).mean()
for thr in np.arange(0.2, 3.0, 0.05):
    g2 = prox >= thr
    agr = (g2 == gate).mean()
    cap = (g2 & gate).sum() / max(gate.sum(), 1)
    if best is None or agr > best[1]:
        best = (thr, agr, cap, g2.mean())
print(f"\nFAMILY4 gate via 10dMA of equal-weight decayed sum: thr={best[0]:.2f} -> "
      f"{best[1]*100:.1f}% agreement, captures {best[2]*100:.1f}% of gate-ON days, "
      f"ON {best[3]*100:.1f}% vs gate {gate.mean()*100:.1f}%")
