"""Bond vol is climbing while equity vol collapses.

^MOVE closed at 79.71, a fifth straight up close, 5d return rank 93.3 of its
own year. ^VIX closed at 15.20, down 6.98%, with a 63d return rank of 7.1 and
-29.3% over that span. The engine fired P7:up_streak on ^MOVE (n=100, h5 mean
-3.619%) but never crosses the two vol surfaces. That cross is the cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["^MOVE", "^VIX", "SPY", "TLT", "IEF", "^TNX"]
px = close_panel(TK)
px = px.dropna(subset=["^MOVE", "^VIX", "SPY"])
idx = px.index
print("panel", idx[0].date(), "->", idx[-1].date(), len(idx), "sessions")

move_rk5 = pct_rank(px["^MOVE"], 5)
vix_rk63 = pct_rank(px["^VIX"], 63)
print(f"live: MOVE 5d rank {move_rk5.iloc[-1]:.1f}, VIX 63d rank {vix_rk63.iloc[-1]:.1f}")

# --- part 1: the ^MOVE up-streak base cell, drilled ---
up = px["^MOVE"] > px["^MOVE"].shift(1)
streak = up.groupby((~up).cumsum()).cumsum()
trig = idx[(streak >= 5) & up]
trig = trig[trig < idx[-1]]
print(f"\n=== ^MOVE 5+ consecutive up closes: {len(trig)} days ===")
for h in (1, 5, 10, 21):
    epi = declusters(trig, h, idx)
    rows = []
    for sub in ["^MOVE", "SPY", "TLT", "^VIX"]:
        r = fwd_ret(px[sub], h)
        v = r.reindex(epi).dropna()
        if not len(v):
            continue
        s = summarize(v.values, f"{sub}")
        base = r.dropna()
        s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        u = int((v > 0).sum())
        s["up"], s["down"] = u, len(v) - u
        s["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4)
        rows.append(s)
    show(rows, f"h={h}, {len(epi)} episodes (declustered {h} td)")

# --- part 2: the cross ---
print("\n\n=== the cross: MOVE 5d rank >= 90 AND VIX 63d rank <= 15 ===")
mask = (move_rk5 >= 90) & (vix_rk63 <= 15)
trig2 = idx[mask.fillna(False)]
trig2 = trig2[trig2 < idx[-1]]
epi2 = declusters(trig2, 10, idx)
print(f"{len(trig2)} sessions, {len(epi2)} episodes (10 td decluster)")
print("episodes:", [str(d.date()) for d in epi2])
if len(epi2) >= 5:
    for h in (1, 5, 21):
        rows = []
        for sub in ["SPY", "^VIX", "^MOVE", "TLT"]:
            r = fwd_ret(px[sub], h)
            v = r.reindex(epi2).dropna()
            if not len(v):
                continue
            s = summarize(v.values, sub)
            base = r.dropna()
            s["ctl_all_pct"] = round(100 * base.mean(), 3)
            s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
            u = int((v > 0).sum())
            s["up"], s["down"] = u, len(v) - u
            s["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4)
            rows.append(s)
        show(rows, f"forward h={h} from the cross")
    r21 = fwd_ret(px["SPY"], 21)
    print(cluster_note(epi2, r21.reindex(epi2).values, k=2))

# --- part 3: how unusual is the gap itself, and where has VIX gone from here ---
print("\n\n=== VIX 63d rank <= 10 on its own (the calm side alone) ===")
calm = idx[(vix_rk63 <= 10).fillna(False)]
calm = calm[calm < idx[-1]]
epic = declusters(calm, 21, idx)
rows = []
for h in (5, 21, 63):
    r = fwd_ret(px["^VIX"], h)
    v = r.reindex(epic).dropna()
    s = summarize(v.values, f"^VIX h={h}")
    base = r.dropna()
    s["ctl_all_pct"] = round(100 * base.mean(), 3)
    u = int((v > 0).sum())
    s["up"], s["down"] = u, len(v) - u
    rows.append(s)
    rs = fwd_ret(px["SPY"], h)
    vs = rs.reindex(epic).dropna()
    ss = summarize(vs.values, f"SPY h={h}")
    ss["ctl_all_pct"] = round(100 * rs.dropna().mean(), 3)
    u = int((vs > 0).sum())
    ss["up"], ss["down"] = u, len(vs) - u
    rows.append(ss)
show(rows, f"{len(epic)} episodes")
