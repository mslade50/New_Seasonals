"""^BVSP is on an ELEVEN session losing streak, not the five the trigger fired on.

Drill 09 found the engine's cell (5+ consecutive down closes, +0.484% next
session, 48-30) does not describe today's state, and that the effect inverts
as the streak lengthens: at 6+ the h10 return is -1.028% on 8-17. So the
published number has to come from the streak length that is actually live.

First job is to verify the streak is real and not a stale-quote artifact.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note, declusters,
)

px = close_panel(["^BVSP", "EEM", "SPY"])
idx = px.index
b = px["^BVSP"].dropna()
r = b.pct_change(fill_method=None)

print("=" * 74)
print("A. verify the streak is real (stale quotes would show as zero changes)")
print("=" * 74)
tail = pd.DataFrame({"close": b.tail(16), "ret_pct": 100 * r.tail(16)})
print(tail.round(3).to_string())
print(f"\n  distinct closes in the last 12 sessions: {b.tail(12).nunique()} of 12")
print(f"  zero-change sessions in the last 12: {int((r.tail(12) == 0).sum())}")

runs = []
run = 0
for x in (r < 0).values:
    run = run + 1 if x else 0
    runs.append(run)
streak = pd.Series(runs, index=b.index)
cur = int(streak.iloc[-1])
print(f"\n  current streak: {cur} consecutive down closes")
print(f"  cumulative over the streak: {100*(b.iloc[-1]/b.iloc[-1-cur] - 1):+.2f}%")

print("\n" + "=" * 74)
print("B. how rare is a streak this long, and what followed")
print("=" * 74)
# a streak of exactly length L is marked at its final day: streak==L and next
# session is not a continuation of the same run
for L in (7, 8, 9, 10, 11):
    m = streak >= L
    trig = b.index[m.values]
    trig = trig[trig < b.index[-1]]
    epi = declusters(pd.DatetimeIndex(trig), 5, b.index)
    if len(epi) == 0:
        print(f"  streak >= {L}: no prior episodes")
        continue
    out = []
    for h in (1, 3, 5, 10, 21):
        f = b.shift(-h) / b - 1.0
        v = f.loc[f.index.intersection(epi)].dropna().values
        row = summarize(v, f"h={h}")
        row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        base = f.dropna()
        row["ctl_pct"] = round(100 * base.mean(), 3)
        row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
        out.append(row)
    show(out, f"^BVSP after a streak of {L}+ (episodes {len(epi)})")
    print("   dates:", [str(d.date()) for d in epi])

print("\n" + "=" * 74)
print("C. the monotonic picture: next-session return by streak length")
print("=" * 74)
f1 = b.shift(-1) / b - 1.0
out = []
for L in range(1, 12):
    m = (streak == L)
    v = f1[m].dropna().values
    if len(v) < 3:
        continue
    row = summarize(v, f"streak exactly {L}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    out.append(row)
show(out, "does the bounce strengthen or fade with streak length?")

print("\n" + "=" * 74)
print("D. era and concentration on the 8+ cell")
print("=" * 74)
m = streak >= 8
trig = b.index[m.values]
epi = declusters(pd.DatetimeIndex(trig[trig < b.index[-1]]), 5, b.index)
for h in (1, 10):
    f = b.shift(-h) / b - 1.0
    s = f.loc[f.index.intersection(epi)].dropna()
    if len(s) == 0:
        continue
    show(era_split(s.index, s.values), f"^BVSP h={h} after 8+ down closes")
    print(" ", cluster_note(s.index, s.values, k=2))

print("\n" + "=" * 74)
print("E. what a streak this long has meant for the drawdown already banked")
print("=" * 74)
m = streak >= 8
ep = declusters(pd.DatetimeIndex(b.index[m.values]), 5, b.index)
rows = []
for d in ep:
    p = b.index.get_loc(d)
    L = int(streak.loc[d])
    rows.append({"date": d.date(), "streak": L,
                 "streak_ret_pct": round(100 * (b.iloc[p] / b.iloc[p - L] - 1), 2),
                 "next1_pct": round(100 * (b.iloc[p + 1] / b.iloc[p] - 1), 2)
                 if p + 1 < len(b) else np.nan,
                 "next10_pct": round(100 * (b.iloc[p + 10] / b.iloc[p] - 1), 2)
                 if p + 10 < len(b) else np.nan})
print(pd.DataFrame(rows).to_string(index=False))
