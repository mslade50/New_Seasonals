"""Drill 06 turned up a number the engine's P7 cell buried: ^BVSP is on ELEVEN
consecutive up closes, not five. The trigger fires at 5+ and pools everything
above it. Eleven is a different object. How rare, and what follows?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^BVSP", "EWZ", "SPY", "^GSPC", "EEM", "^MXX"])
bv = px["^BVSP"]["Close"].dropna()
print(f"^BVSP {bv.index[-1].date()} {bv.iloc[-1]:,.0f}")

up = bv > bv.shift(1)
streak = up.groupby((~up).cumsum()).cumsum()
k = int(streak.iloc[-1])
print(f"live streak: {k} consecutive up closes")
print("the run:")
for d, c in bv.iloc[-(k + 1):].items():
    print(f"   {d.date()}  {c:>12,.0f}")
print(f"cumulative over the run: {100*(bv.iloc[-1]/bv.iloc[-(k+1)]-1):+.2f}%")


def rec(v, lab):
    d = summarize(v.values, lab)
    u = int((v > 0).sum())
    d["up"], d["down"] = u, len(v) - u
    d["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4) if len(v) else None
    return d


hist = streak[streak.index < bv.index[-1]]
print("\nstreak-length frequency since", bv.index[0].date())
for n in range(5, 15):
    days = int(((hist >= n) & up.reindex(hist.index)).sum())
    ends = int((hist == n).sum())
    print(f"  >= {n:2d} up closes: {days:4d} sessions, exactly {n}: {ends:3d} runs")

for n in (8, 10, 11):
    trig = hist.index[(hist >= n) & up.reindex(hist.index).fillna(False)]
    epi = declusters(trig, 10, bv.index)
    print(f"\n=== ^BVSP at {n}+ consecutive up closes: {len(trig)} sessions, "
          f"{len(epi)} episodes (10td decluster) ===")
    if len(epi) < 4:
        print("  too few"); continue
    print("  episodes:", [str(d.date()) for d in epi])
    rows = []
    for h in (1, 5, 21):
        r = fwd_ret(bv, h)
        v = r.reindex(epi).dropna()
        d = rec(v, f"^BVSP h={h}")
        d["ctl_pct"] = round(100 * r.dropna().mean(), 3)
        d["edge_pct"] = round(d["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(d)
    show(rows)
    r1 = fwd_ret(bv, 1)
    vv = r1.reindex(epi).dropna()
    if len(vv) >= 8:
        show(era_split(vv.index, vv.values), "era split h=1")

# how does an 11-run compare to every other index in the panel
print("\n=== longest current up-streak across the panel ===")
for t in ["^BVSP", "EWZ", "SPY", "^GSPC", "EEM", "^MXX"]:
    c = px[t]["Close"].dropna()
    u = c > c.shift(1)
    s = u.groupby((~u).cumsum()).cumsum()
    hi = int(s.max())
    print(f"  {t:<8} live {int(s.iloc[-1]):>3}  longest ever {hi:>3} "
          f"({s.idxmax().date()})")
