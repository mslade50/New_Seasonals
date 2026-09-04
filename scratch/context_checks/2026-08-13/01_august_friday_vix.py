"""August Fridays crush VIX 33-83. Is that August, or just Friday?

The sweep cell anchors on the session BEFORE the Friday, so h=1 is the Friday's
own close-to-close move. The obvious confound is the weekend effect: VIX drops
into a weekend because two calendar days of decay come out of the front month.
If all Fridays do this, August is decoration.

Also: does the Aug-14 trading-day-of-year cell (19-7 down) add anything, and
does VIX entering low (63d rank 20.6 tonight) change the picture.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, era_split, fwd_ret,  # noqa: E402
                       sign_test, summarize, cluster_note)

px = close_panel(["^VIX", "SPY"])
vix = px["^VIX"].dropna()
vix = vix[vix.index >= "1999-01-01"]
r1 = fwd_ret(vix, 1)

idx = vix.index
dow = idx.dayofweek
month = idx.month


def rep(name, mask):
    m = mask & r1.reindex(idx).notna().to_numpy()
    v = r1.to_numpy()[m]
    if len(v) == 0:
        return None
    d = summarize(v, name)
    down = int((v < 0).sum())
    p = sign_test(down, len(v))
    print(f"{name:<44} n={len(v):>5}  mean={d['mean_pct']:+7.3f}%  "
          f"med={d['median_pct']:+7.3f}%  down={down}-{len(v) - down} "
          f"({100 * down / len(v):4.1f}%)  t={d['t']:+5.2f}  signp={p:.4f}")
    return {"v": v, "dates": idx[m], "down": down, "n": len(v), "d": d, "p": p}


print("=== the anchor is the session BEFORE, so h1 is the named session's own move")
print("=== 'down' counts VIX FALLING, which is what the cell claims\n")

aug_fri = (month == 8) & (dow == 3)          # Thursday anchor -> Friday move
all_fri = (dow == 3)                          # every Thursday anchor
aug_not_fri = (month == 8) & (dow != 3)
not_aug_fri = (month != 8) & (dow == 3)

a = rep("August Fridays (the cell)", aug_fri)
f = rep("ALL Fridays", all_fri)
rep("Fridays outside August", not_aug_fri)
rep("August, not Friday", aug_not_fri)
rep("every session 1999+", np.ones(len(idx), dtype=bool))

print("\n--- does August add anything to Friday? ---")
print(f"August Fridays  {100 * a['down'] / a['n']:.1f}% down, mean {a['d']['mean_pct']:+.3f}%")
print(f"All Fridays     {100 * f['down'] / f['n']:.1f}% down, mean {f['d']['mean_pct']:+.3f}%")
print(f"difference in hit rate: {100 * a['down'] / a['n'] - 100 * f['down'] / f['n']:+.1f} pts")

# Two-proportion z on August-Friday vs other-Friday
nf = rep("(recompute) Fridays outside August", not_aug_fri)
p1, n1 = a["down"] / a["n"], a["n"]
p2, n2 = nf["down"] / nf["n"], nf["n"]
pp = (a["down"] + nf["down"]) / (n1 + n2)
z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
print(f"two-proportion z, August Fridays vs other Fridays: {z:+.2f}")

print("\n--- era split, August Fridays ---")
for e in era_split(a["dates"], a["v"]):
    print("   ", e)
print("   ", cluster_note(a["dates"], a["v"]))

print("\n--- magnitude: is the August Friday drop bigger, not just more frequent? ---")
for nm, msk in [("August Fridays", aug_fri), ("other Fridays", not_aug_fri)]:
    v = r1.to_numpy()[msk & r1.reindex(idx).notna().to_numpy()]
    print(f"   {nm:<16} mean {100*v.mean():+.3f}%  median {100*np.median(v):+.3f}%  "
          f"mean of down days {100*v[v < 0].mean():+.3f}%")

print("\n--- condition on VIX entering LOW (tonight: 63d rank 20.6, close 14.63) ---")
rank63 = vix.rolling(63).apply(lambda w: 100.0 * (w < w.iloc[-1]).sum() / (len(w) - 1), raw=False)
low = (rank63 <= 33).to_numpy()
rep("August Fridays, VIX in bottom third of 63d", aug_fri & low)
rep("other Fridays, VIX in bottom third of 63d", not_aug_fri & low)
rep("August Fridays, VIX NOT in bottom third", aug_fri & ~low)

print("\n--- the Aug-14 trading-day-of-year cell (sweep said 19-7 down) ---")
# same calendar slot, one anchor per year, +/- 2 calendar days of Aug 14
doy = (month == 8) & (np.abs(idx.day - 14) <= 2)
seen, keep = set(), []
for d in idx[doy]:
    if d.year not in seen:
        seen.add(d.year)
        keep.append(d)
km = idx.isin(pd.DatetimeIndex(keep))
rep("Aug-14 anchor, one per year", km)
overlap = int((km & aug_fri).sum())
print(f"    of those anchors, {overlap} are also August-Friday anchors "
      f"(the two cells are NOT independent)")
