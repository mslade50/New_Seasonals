"""Drills 02 and 08 both cover late August. How much do their windows share?

If the five sessions after August expiration usually CONTAIN the symposium,
the two cells are one observation wearing two labels and the brief has to say
so rather than present them as independent confirmation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, fwd_ret, summarize, sign_test

px = load_prices(["SPY"])
ASOF = pd.Timestamp("2026-08-21")
s = px["SPY"]["Close"].astype(float).loc[:ASOF]
idx = s.index
pos = pd.Series(range(len(idx)), index=idx)
opex = [d for d in pd.DatetimeIndex(load_events(["opex"])["date"])
        if d in pos.index and d <= ASOF and d.month == 8]
jh = {d.year: d for d in pd.DatetimeIndex(load_events(["jackson_hole"])["date"])}

inside, outside = [], []
for d in opex:
    j = jh.get(d.year)
    if j is None or j not in pos.index:
        continue
    lo, hi = pos[d], pos[d] + 5
    (inside if lo < pos[j] <= hi else outside).append((d, j, pos[j] - lo))
print(f"August expirations with a symposium inside the next 5 sessions: "
      f"{len(inside)} of {len(inside) + len(outside)}")
for d, j, k in inside:
    print(f"   {d.year}: opex {d.date()}, symposium {j.date()}, +{k} sessions")
print("   outside:", [(d.year, f"+{k}") for d, j, k in outside])

print("\n########## SPY h5 after August opex, split by whether JH is inside ##########")
f5 = fwd_ret(s, 5)
for lab, grp in [("symposium inside", inside), ("symposium outside", outside)]:
    v = f5.reindex(pd.DatetimeIndex([d for d, _, _ in grp])).dropna()
    if not len(v):
        continue
    up = int((v.values > 0).sum())
    st = summarize(v.values, lab)
    print(f"   {lab:18s} n={st['n']:2d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"record {up}-{st['n']-up} p={sign_test(up, st['n']):.4f}")

print("\n########## 2026: where does the symposium fall? ##########")
print(f"   opex 2026-08-21, symposium 2026-08-28, "
      f"5 sessions after opex = 2026-08-24..2026-08-28 -> symposium is the LAST bar")
