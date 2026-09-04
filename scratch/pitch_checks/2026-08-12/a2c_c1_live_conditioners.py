"""C1 round 2c -- the verdict arithmetic.

Two conditioners describe TODAY and both land on the cell's weak side:
  * AUGUST print. a2b showed the month profile PERSISTS split-half (rank corr
    +0.71, permutation p 0.025) so month is a real conditioner here, not 12
    buckets of noise; August's persistent value is ~0.00-0.06% i.e. NO edge,
    and IEF/LQD/^TNX all agree (August PPI prints are yields-UP sessions).
  * MIDTERM year. The registry conditioner that has killed three ideas.

This script prices the cell AS TODAY IS, not as the headline average, and
checks whether the midterm leg is a cycle effect or the 2022 bond bear wearing
a cycle label (registry: 'an era cut that isolates one macro episode').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
base_hit = float((d1[~np.isnan(d1)] > 0).mean())

ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 1 <= p < N and not np.isnan(d1[p])]
cpi_all = set(sp("cpi"))

v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo, yr = dt.month.values, dt.year.values
ceve = np.array([(p - 1) in cpi_all for p in ppi_l])
mid = (yr % 4) == 2
aug = mo == 8


def rep(v_, lbl):
    if len(v_) == 0:
        return {"cell": lbl, "N": 0}
    w = int((v_ > 0).sum())
    sd = v_.std(ddof=1) if len(v_) > 1 else np.nan
    return {"cell": lbl, "N": len(v_), "mean_pct": round(100 * v_.mean(), 4),
            "hit": round(100 * w / len(v_), 1),
            "signp": round(sign_test(w, len(v_), base_hit), 4),
            "worst": round(100 * v_.min(), 2)}


print("=" * 104)
print("1. THE LIVE CELL PARTITIONED BY TODAY'S TWO CONDITIONERS")
print("   (live cell = PPI print with a CPI print on its eve, N=55)")
print("=" * 104)
L = ceve
rows = [rep(v[L], "LIVE CELL, headline (all months, all cycle years)"),
        rep(v[L & ~aug & ~mid], "clean: not August, not midterm"),
        rep(v[L & aug], "AUGUST  (today)"),
        rep(v[L & mid], "MIDTERM (today)"),
        rep(v[L & (aug | mid)], "AUGUST or MIDTERM = today's half"),
        rep(v[L & aug & mid], "AUGUST and MIDTERM = today exactly")]
print(pd.DataFrame(rows).to_string(index=False))
a = v[L & (aug | mid)]
b = v[L & ~aug & ~mid]
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"\n  today's-half vs clean-half diff = {100*(a.mean()-b.mean()):+.4f}pp  "
      f"welch t {(a.mean()-b.mean())/se:+.2f}")
print(f"  today's-half bootstrap P(mean<=0) = {bootstrap_p_le0(a):.3f}")

print("\n" + "=" * 104)
print("2. SAME PARTITION ON THE PARENT (all 286 PPI prints) -- is the")
print("   conditioner a property of the cell or of the whole event?")
print("=" * 104)
print(pd.DataFrame([
    rep(v, "PARENT headline"),
    rep(v[~aug & ~mid], "parent: not Aug, not midterm"),
    rep(v[aug], "parent AUGUST"),
    rep(v[mid], "parent MIDTERM"),
    rep(v[aug | mid], "parent Aug-or-midterm"),
    rep(v[aug & mid], "parent Aug AND midterm"),
]).to_string(index=False))

print("\n" + "=" * 104)
print("3. IS THE MIDTERM LEG A CYCLE EFFECT OR THE 2022 BOND BEAR?")
print("=" * 104)
for lbl, m in [("live cell midterm, ALL", L & mid),
               ("live cell midterm ex-2022", L & mid & (yr != 2022)),
               ("live cell 2022 only", L & (yr == 2022)),
               ("live cell 2026 only (this cycle)", L & (yr == 2026))]:
    print(f"  {lbl:36s} {rep(v[m], '')}")
print("\n  parent, per midterm year:")
for y in sorted(set(yr[mid])):
    s = v[mid & (yr == y)]
    print(f"    {y}: N={len(s):2d} {100*s.mean():+.4f}% hit {100*(s>0).mean():5.1f}%")
print("\n  -> if ex-2022 the midterm cell looks like the parent, MIDTERM is not")
print("     the kill; 2022 is, and 2022 is one macro episode.")

print("\n" + "=" * 104)
print("4. IS THE AUGUST LEG A REGIME EPISODE TOO? per-August-year detail")
print("=" * 104)
for y in sorted(set(yr[aug])):
    s = v[aug & (yr == y)]
    fl = bool(ceve[aug & (yr == y)][0]) if (aug & (yr == y)).sum() else False
    print(f"    {y}: {100*s.mean():+.4f}%   CPI-on-eve={fl}")
print(f"\n  August parent split-half: pre-2014 "
      f"{100*v[aug & (yr<2014)].mean():+.4f}% (N={int((aug&(yr<2014)).sum())})  "
      f"2014+ {100*v[aug & (yr>=2014)].mean():+.4f}% "
      f"(N={int((aug&(yr>=2014)).sum())})")
print("  Both halves near zero and the sign flips -> August is a persistent")
print("  NULL, not a persistent negative. A null month is a demotion, not a")
print("  short signal.")

print("\n" + "=" * 104)
print("5. WHAT THE TRADE IS WORTH TODAY, three honest readings")
print("=" * 104)
for lbl, m in [("headline live cell", L),
               ("August-conditioned live cell", L & aug),
               ("midterm-conditioned live cell", L & mid),
               ("today's-half (Aug or midterm)", L & (aug | mid)),
               ("parent August (any eve)", aug)]:
    s = v[m]
    bps = 100 * 100 * s.mean()
    print(f"  {lbl:32s} N={len(s):3d}  {bps:+7.2f} bps  "
          f"= {bps/2.5:+6.2f}x a 2.5 bps TLT round trip")
