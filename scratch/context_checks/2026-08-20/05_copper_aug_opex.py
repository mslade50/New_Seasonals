"""Copper on the August opex session: two independent cells pointing the same way.

E:opex|HG=F|k1     n=311, +0.263%, 170-141, t=2.89, era-stable  (pre-specified family)
E:seasonal_doy|HG=F  19-6 up, sign p 0.0073, +0.50% mean         (found BY the sweep)

The doy cell alone does not clear the 0.0052 BH bar, so it cannot stand on its own.
The question here is whether the two overlap into one August-opex cell that survives
the all-days control and the era split, or whether they are the same handful of days
counted twice.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, era_split, load_events, local_control, sign_test, summarize  # noqa


def report(label, v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<44} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<44} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}  sign_p={sign_test(up, st['n']):.4f}"
    )


px = close_panel(["HG=F"])["HG=F"].dropna()
idx = px.index
opex = set(pd.to_datetime(load_events(["opex"])["date"]).dt.normalize())
nxt = list(idx[1:]) + [pd.NaT]
next_is_opex = np.array([pd.notna(d) and d.normalize() in opex for d in nxt])
fwd1 = px.pct_change().shift(-1).values
fwd5 = (px.shift(-5) / px - 1.0).values
opex_month = np.array([d.month if pd.notna(d) else 0 for d in nxt])
opex_year = np.array([d.year if pd.notna(d) else 0 for d in nxt])

print(f"HG=F {idx[0].date()} to {idx[-1].date()}, live close {px.iloc[-1]:.4f}, 1d {px.pct_change().iloc[-1] * 100:+.2f}%")
ok = next_is_opex & ~np.isnan(fwd1)
report("all opex sessions", fwd1[ok])
report("August opex sessions", fwd1[ok & (opex_month == 8)])
report("non-August opex sessions", fwd1[ok & (opex_month != 8)])
report("August opex, midterm years", fwd1[ok & (opex_month == 8) & (opex_year % 4 == 2)])
print("  controls:")
report("  all sessions (own drift)", px.pct_change().shift(-1).values)
lc = local_control(idx, idx[ok & (opex_month == 8)], win=126)
report("  local +/-126td around Aug opex anchors", px.pct_change().shift(-1).reindex(lc).values)
print("  h5 follow-on:")
report("  all opex, h5", fwd5[ok])
report("  August opex, h5", fwd5[ok & (opex_month == 8)])

print("\n  era split, all opex h1:")
for e in era_split(idx[ok], fwd1[ok]):
    print(f"    {e['label']:<9} n={e['n']:<4} mean={e['mean_pct']:+.3f}%  hit={e['hit']:.1f}%  t={e['t']:+.2f}")
print("  concentration, all opex:", cluster_note(idx[ok], fwd1[ok]))

aug = ok & (opex_month == 8)
print(f"\n  August opex bars (n={int(aug.sum())}):")
for dt, v in zip(idx[aug], fwd1[aug] * 100):
    print(f"    anchor {dt.date()} -> opex {v:+.2f}%")
print("  concentration, Aug opex:", cluster_note(idx[aug], fwd1[aug]))

# does the doy cell overlap the opex cell, or are they distinct days?
doy_hits = [d for d in idx if d.month == 8 and 19 <= d.day <= 23]
overlap = [d for d in idx[aug] if d.month == 8 and 19 <= d.day <= 23]
print(f"\n  Aug-21 doy anchors in the panel: {len(doy_hits)}; of which also an opex anchor: {len(overlap)}")
