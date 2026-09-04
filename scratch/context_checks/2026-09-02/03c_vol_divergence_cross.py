"""The cross, with every rank computed on its own index and only then aligned.

Drill 03 built the ranks on a union-calendar panel and read VIX's 63d rank as
48.0 against the engine's 7.1, so its cross cell was measuring the wrong state.
Live tonight: ^MOVE 5d return rank 93.3, ^VIX 63d return rank 7.1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^MOVE", "^VIX", "SPY", "TLT", "^TNX"])
mv = px["^MOVE"]["Close"].dropna()
vx = px["^VIX"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()
tlt = px["TLT"]["Close"].dropna()

mv_rk = pct_rank(mv, 5)
vx_rk = pct_rank(vx, 63)
print(f"live: MOVE5 rank {mv_rk.iloc[-1]:.1f}, VIX63 rank {vx_rk.iloc[-1]:.1f}")

base = spy.index[spy.index >= mv.index[0]]
a = mv_rk.reindex(base).ffill(limit=3)
b = vx_rk.reindex(base).ffill(limit=3)

for mt, vt in [(90, 15), (90, 10), (85, 20)]:
    mask = (a >= mt) & (b <= vt)
    trig = base[mask.fillna(False)]
    trig = trig[trig < base[-1]]
    epi = declusters(trig, 10, base)
    print(f"\n=== MOVE5 rank >= {mt} AND VIX63 rank <= {vt}: "
          f"{len(trig)} sessions, {len(epi)} episodes ===")
    if len(epi) < 4:
        print("  too few"); continue
    print("  episodes:", [str(d.date()) for d in epi])
    rows = []
    for sub, s in [("SPY", spy), ("^VIX", vx), ("^MOVE", mv), ("TLT", tlt)]:
        for h in (1, 5, 21):
            r = fwd_ret(s, h).reindex(base)
            v = r.reindex(epi).dropna()
            if len(v) < 4:
                continue
            d = summarize(v.values, f"{sub} h={h}")
            bb = r.dropna()
            d["ctl_pct"] = round(100 * bb.mean(), 3)
            d["edge_pct"] = round(d["mean_pct"] - 100 * bb.mean(), 3)
            u = int((v > 0).sum())
            d["up"], d["down"] = u, len(v) - u
            d["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4)
            rows.append(d)
    show(rows, f"forward from the cross ({mt}/{vt})")

# how rare is the gap in plain terms
gap = a - b
print(f"\nlive rank gap (MOVE5 - VIX63) = {gap.iloc[-1]:.1f}")
print("percentile of that gap in its own history: "
      f"{100 * (gap.dropna() <= gap.iloc[-1]).mean():.1f}")
