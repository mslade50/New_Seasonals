"""SPY and TLT both down 50bp+, conditioned on whether the vol market actually bid.

Bare cell P9b: SPY h1 +0.125% (t=0.96), TLT +0.018% (t=0.31), both era_stable=False.
Dead as published. Today's version came with ^VIX +7.5% and the 10y up 9bp, so the
question is whether "stocks and bonds down together AND vol bid hard" separates from
the far more common version where nothing much happened in vol.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, declusters, era_split, sign_test, summarize  # noqa


def report(label, v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<50} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<50} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}  sign_p={sign_test(up, st['n']):.4f}"
    )


cp = close_panel(["SPY", "TLT", "^VIX"]).dropna()
spy, tlt, vix = cp["SPY"], cp["TLT"], cp["^VIX"]
idx = cp.index
rs, rt, rv = spy.pct_change(), tlt.pct_change(), vix.pct_change()
f1 = spy.pct_change().shift(-1)
f5 = spy.shift(-5) / spy - 1.0

both_down = (rs <= -0.005) & (rt <= -0.005)
volbid = rv >= 0.05
print(f"SPY/TLT/^VIX joint panel {idx[0].date()} to {idx[-1].date()}")
print(
    f"  live: SPY {rs.iloc[-1] * 100:+.2f}%, TLT {rt.iloc[-1] * 100:+.2f}%, "
    f"VIX {rv.iloc[-1] * 100:+.2f}% -> both_down={bool(both_down.iloc[-1])}, volbid={bool(volbid.iloc[-1])}"
)

report("all sessions, SPY h1 (control)", f1.values)
report("stocks and bonds both down 50bp+", f1[both_down].values)
report("  ... and VIX +5% or more", f1[both_down & volbid].values)
report("  ... and VIX up less than 5%", f1[both_down & ~volbid].values)
report("VIX +5% or more, without the bond leg", f1[volbid & ~both_down].values)
print()
report("both down + vol bid, SPY h5", f5[both_down & volbid].values)
report("both down, no vol bid, SPY h5", f5[both_down & ~volbid].values)
report("all sessions, SPY h5 (control)", f5.values)

cell = both_down & volbid & f1.notna()
d = idx[cell]
print(f"\n  cell size {int(cell.sum())}")
print("  era split:")
for e in era_split(d, f1[cell].values):
    print(f"    {e['label']:<9} n={e['n']:<4} mean={e['mean_pct']:+.3f}%  hit={e['hit']:.1f}%  t={e['t']:+.2f}")
print("  concentration:", cluster_note(d, f1[cell].values))
dc = declusters(d, min_gap_td=5, all_dates=idx)
report("  declustered at 5td, h1", f1.reindex(dc).values)
print(f"  declustered episodes: {len(dc)}; most recent: {[str(x.date()) for x in dc[-8:]]}")
