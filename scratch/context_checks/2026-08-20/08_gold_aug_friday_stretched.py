"""Gold's August-Friday cell, and whether it survives gold already being stretched.

E:weekday_month|GC=F: n=114, +0.225%, 66-48, t=2.33, era-stable. Gold closed today
at 4575 (+1.91%), up 10.3% over 21 sessions with z10 1.68, so the bare seasonal
cell is not the live cell. Splits the August Fridays by how extended gold was
walking in, and controls against every other Friday and against all days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, era_split, sign_test, summarize  # noqa


def report(label, v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<48} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<48} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}  sign_p={sign_test(up, st['n']):.4f}"
    )


px = close_panel(["GC=F"])["GC=F"].dropna()
idx = px.index
f1 = px.pct_change().shift(-1)
r21 = px.pct_change(21)
rank21 = r21.rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)

nxt = list(idx[1:]) + [pd.NaT]
next_is_fri = np.array([pd.notna(d) and d.weekday() == 4 for d in nxt])
next_is_aug = np.array([pd.notna(d) and d.month == 8 for d in nxt])
ok = ~np.isnan(f1.values) & ~np.isnan(rank21.values)

print(f"GC=F {idx[0].date()} to {idx[-1].date()}")
print(
    f"  live: close {px.iloc[-1]:.2f}, 1d {px.pct_change().iloc[-1] * 100:+.2f}%, "
    f"21d {r21.iloc[-1] * 100:+.2f}%, 21d rank {rank21.iloc[-1]:.1f}"
)

report("all sessions (control)", f1.values)
report("all Fridays", f1.values[next_is_fri & ok])
report("August Fridays", f1.values[next_is_fri & next_is_aug & ok])
report("non-August Fridays", f1.values[next_is_fri & ~next_is_aug & ok])
print()
hot = rank21.values > 85
report("August Fridays, gold 21d rank > 85", f1.values[next_is_fri & next_is_aug & ok & hot])
report("August Fridays, gold 21d rank <= 85", f1.values[next_is_fri & next_is_aug & ok & ~hot])
report("any Friday, gold 21d rank > 85", f1.values[next_is_fri & ok & hot])
report("any session, gold 21d rank > 85 (control)", f1.values[ok & hot])

cell = next_is_fri & next_is_aug & ok
d = idx[cell]
print("\n  August Fridays, era split:")
for e in era_split(d, f1.values[cell]):
    print(f"    {e['label']:<9} n={e['n']:<4} mean={e['mean_pct']:+.3f}%  hit={e['hit']:.1f}%  t={e['t']:+.2f}")
print("  concentration:", cluster_note(d, f1.values[cell]))

live = next_is_fri & next_is_aug & ok & hot
print(f"\n  live cell (August Friday, gold 21d rank > 85): n={int(live.sum())}")
for dt, v in zip(idx[live], f1.values[live] * 100):
    print(f"    anchor {dt.date()} rank={rank21.values[list(idx).index(dt)]:.0f} -> Friday {v:+.2f}%")
