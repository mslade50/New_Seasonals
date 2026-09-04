"""VIX on the monthly opex session.

Sweep cell E:opex|^VIX|k1: n=319, h1 -0.99%, 107-209 down, t=-2.58, BH pass.
Two problems to resolve before it publishes:
  1. the midterm-year subset is n=79 at -0.50%, t=-0.69
  2. the August subset is n=26 at -0.50%, but the RECORD there is 6-20
So the mean and the record disagree. This resolves which one carries, adds the
exact sign-test p at small N, and checks the state VIX is actually in tonight
(16.01, up 7.5% on the session, up 9.4% over 5 days).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, load_events, sign_test, summarize  # noqa


def report(label, v, dates=None):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<44} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<44} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}"
        f"  sign_p_down={sign_test(st['n'] - up, st['n']):.4f}"
    )


cp = close_panel(["^VIX", "SPY"])
vix = cp["^VIX"].dropna()
spy = cp["SPY"].reindex(vix.index)

opex = set(pd.to_datetime(load_events(["opex"])["date"]).dt.normalize())
idx = vix.index
nxt = list(idx[1:]) + [pd.NaT]
next_is_opex = np.array([pd.notna(d) and d.normalize() in opex for d in nxt])

fwd1 = vix.pct_change().shift(-1).values
vix_5d = vix.pct_change(5).values
vix_1d = vix.pct_change().values
month = np.array([d.month for d in idx])
year = np.array([d.year for d in idx])
midterm = (year % 4) == 2

live = f"live: VIX {vix.iloc[-1]:.2f}, 1d {vix_1d[-1] * 100:+.2f}%, 5d {vix_5d[-1] * 100:+.2f}%"
print(f"^VIX {idx[0].date()} to {idx[-1].date()}\n{live}\n")

ok = next_is_opex & ~np.isnan(fwd1)
report("all opex sessions", fwd1[ok])
report("August opex sessions", fwd1[ok & (month == 8)])
report("midterm-year opex sessions", fwd1[ok & midterm])
report("August opex, midterm years", fwd1[ok & (month == 8) & midterm])
print()
report("opex, VIX rose into it (1d > 0)", fwd1[ok & (vix_1d > 0)])
report("opex, VIX rose 5%+ on the anchor day", fwd1[ok & (vix_1d > 0.05)])
report("opex, VIX up over the prior 5d", fwd1[ok & (vix_5d > 0)])
report("opex, VIX fell into it (1d <= 0)", fwd1[ok & (vix_1d <= 0)])
print()
report("control: all non-opex sessions", fwd1[~next_is_opex & ~np.isnan(fwd1)])

aug = idx[ok & (month == 8)]
print("\n  August opex sessions, VIX move on the opex bar:")
for dt, v in zip(aug, fwd1[ok & (month == 8)] * 100):
    print(f"    {dt.date()} -> {v:+.2f}%")
print("  concentration:", cluster_note(aug, fwd1[ok & (month == 8)]))

# the live cell: August opex with VIX having risen 5%+ into it
cell = ok & (month == 8) & (vix_1d > 0.05)
print(f"\n  live cell (August opex, VIX +5% or more on the anchor session): n={int(cell.sum())}")
for dt, v in zip(idx[cell], fwd1[cell] * 100):
    print(f"    {dt.date()}  anchor VIX 1d {vix_1d[list(idx).index(dt)] * 100:+.1f}%  -> opex {v:+.2f}%")
