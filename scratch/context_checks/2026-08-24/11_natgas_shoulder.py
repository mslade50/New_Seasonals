"""Event lane. The sweep's seasonal_doy cell says natural gas fell in all six
midterm-year analogues of Aug 25, mean -5.38% over the following five sessions.
n=6 is an anecdote. Widen it: the same late-August window in ALL years, the
month-position version, and the mechanism check (is it a shoulder-season fact
or a midterm coincidence?).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test, fwd_ret, era_split, cluster_note  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["NG=F"])
c = px["NG=F"]["Close"].dropna()
print("panel", c.index[0].date(), "->", c.index[-1].date(), len(c))
print(f"live: close {c.iloc[-1]:.3f}, 21d {100*c.pct_change(21).iloc[-1]:.1f}%, "
      f"dist from 252d high {100*(c.iloc[-1]/c.rolling(252).max().iloc[-1]-1):.1f}%")


def per_year_window(month, day_lo, day_hi, h):
    """One anchor per year: the last session in [day_lo, day_hi] of `month`."""
    out = []
    for y, grp in c.groupby(c.index.year):
        sel = grp[(grp.index.month == month) & (grp.index.day >= day_lo) &
                  (grp.index.day <= day_hi)]
        if len(sel) == 0:
            continue
        a = sel.index[-1]
        f = fwd_ret(c, h).reindex([a]).iloc[0]
        if np.isnan(f):
            continue
        out.append((y, a, f))
    return out


for h in (5, 10, 21):
    rows = per_year_window(8, 22, 26, h)
    v = np.array([r[2] for r in rows])
    st = summarize(v, "")
    up = int((v > 0).sum())
    dn = len(v) - up
    print(f"\nAug 22-26 anchor, forward {h} sessions: n={st['n']} mean {st['mean_pct']:.2f}% "
          f"med {st['median_pct']:.2f}%  {up}-{dn} up  P(down) sign p {sign_test(dn, len(v)):.4f} "
          f" t {st['t']:.2f}")
    print("   era:", [(e['label'], e['n'], round(e.get('mean_pct', float('nan')), 2))
                      for e in era_split(pd.DatetimeIndex([r[1] for r in rows]), v)])
    print("  ", cluster_note(pd.DatetimeIndex([r[1] for r in rows]), v))
    mid = [r for r in rows if r[0] % 4 == 2]
    vm = np.array([r[2] for r in mid])
    if len(vm):
        upm = int((vm > 0).sum())
        print(f"   midterm only: n={len(vm)} mean {100*vm.mean():.2f}% {upm}-{len(vm)-upm} up "
              f"P(down) p {sign_test(len(vm)-upm, len(vm)):.4f}  years {[r[0] for r in mid]}")
    print(f"   year by year: {[(r[0], round(100*r[2], 1)) for r in rows]}")

print()
print("=== control: the same anchor-and-hold in every OTHER month ===")
for h in (5, 21):
    allr = []
    for m in range(1, 13):
        if m == 8:
            continue
        allr += per_year_window(m, 22, 26, h)
    v = np.array([r[2] for r in allr])
    st = summarize(v, "")
    up = int((v > 0).sum())
    print(f"  non-August day-22-26 anchor, h{h}: n={st['n']} mean {st['mean_pct']:.2f}% "
          f"{up}-{len(v)-up} up  t {st['t']:.2f}")
    allv = fwd_ret(c, h).dropna().values
    a = summarize(allv, "")
    print(f"  all NG=F days h{h}: n={a['n']} mean {a['mean_pct']:.2f}%  hit {a['hit']:.1f}%")
