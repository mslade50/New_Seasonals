"""C11 round 2b — Cochran-Q style homogeneity across the reference class, and
the DIAL support of the gate-off parent (the only object here with any edge).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)
ROOT = Path(__file__).resolve().parents[3]
REF = ["XLK", "XLV", "XLF", "XLI", "XLE", "XLP", "XLU", "XLY", "XLB",
       "SMH", "IWM", "QQQ", "DIA", "IBB", "XRT", "ITB", "IYT", "KRE", "GDX", "XBI"]
pxa = load_prices(sorted(set(REF + ["SPY"])))
CAL = pxa["SPY"]["Close"].dropna().index
px = pd.DataFrame({t: pxa[t]["Close"] for t in pxa}).reindex(CAL)
sr = pct_rank(pxa["SPY"]["Close"].dropna(), 63).reindex(CAL)

print("=" * 100)
print("H. HOMOGENEITY across the reference class (Cochran-Q style chi-sq on hit counts)")
print("=" * 100)
wins, ns, labs = [], [], []
for t in REF:
    r = pct_rank(pxa[t]["Close"].dropna(), 63).reindex(CAL)
    m = ((r <= 20) & (sr > 20)).fillna(False)
    ret = vehicle_ret(px, [(t, 1.0)], 5)
    v = ret.dropna().index
    e = declusters(CAL[m.values].intersection(v), 5, v)
    val = ret.loc[e].values
    if len(val) < 20:
        continue
    wins.append(int((val > 0).sum())); ns.append(len(val)); labs.append(t)
wins, ns = np.array(wins), np.array(ns)
p = wins.sum() / ns.sum()
Q = float((((wins - ns * p) ** 2) / (ns * p * (1 - p))).sum())
dfree = len(ns) - 1
from math import exp, lgamma
def chi_sf(x, k):
    # regularized upper incomplete gamma via series/continued fraction (k/2, x/2)
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1:
        s, term = 1.0 / a, 1.0 / a
        for i in range(1, 500):
            term *= xx / (a + i); s += term
            if abs(term) < abs(s) * 1e-14: break
        return 1.0 - s * exp(-xx + a * np.log(xx) - lgamma(a))
    b, c, d, h = xx + 1 - a, 1e300, 1.0 / (xx + 1 - a), 1.0 / (xx + 1 - a)
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-14: break
    return h * exp(-xx + a * np.log(xx) - lgamma(a))
print(f"  pooled hit rate {100*p:.1f}% over {len(ns)} vehicles, {ns.sum()} episodes")
print(f"  Q = {Q:.2f} on df={dfree} -> p = {chi_sf(Q, dfree):.3f}   "
      "(large p = the vehicles are INDISTINGUISHABLE; QQQ is not special)")
print("  hit rates: " + ", ".join(f"{l} {100*w/n:.0f}%" for l, w, n in zip(labs, wins, ns)))

print("\n" + "=" * 100)
print("I. DIAL SUPPORT OF THE GATE-OFF PARENT (QQQ r63<=20 alone) — today 89.5")
print("=" * 100)
frg = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frg.index = pd.to_datetime(frg.index)
ma10 = frg["63d"].rolling(10).mean()
qr = pct_rank(pxa["QQQ"]["Close"].dropna(), 63).reindex(CAL)
ret5 = vehicle_ret(px, [("QQQ", 1.0)], 5)
v5 = ret5.dropna().index
for lbl, m in (("parent QQQ r63<=20 alone", (qr <= 20)),
               ("PRE-SPEC QQQ<=20 & SPY>20", (qr <= 20) & (sr > 20))):
    e = declusters(CAL[m.fillna(False).values].intersection(v5), 5, v5)
    val = ret5.loc[e].values
    dv = ma10.reindex(e)
    have = dv.notna().values
    print(f"\n  {lbl}: {len(e)} episodes, {int(have.sum())} with a dial reading, "
          f"MAX dial = {dv.max():.1f}")
    rows = []
    for lo, hi in ((0, 50), (50, 70), (70, 85), (85, 200)):
        s = ((dv >= lo) & (dv < hi)).fillna(False).values
        rows.append(summarize(val[s], f"dial [{lo},{hi}) N={int(s.sum())}"))
    show(rows, "")
print("\n  ALL-DAYS dial distribution for scale:")
print(f"    days with ma10 >= 85 in the whole 2016+ series: "
      f"{int((ma10 >= 85).sum())} of {int(ma10.notna().sum())} "
      f"({100*(ma10>=85).mean():.2f}%)")
print(f"    days with ma10 >= 80: {int((ma10 >= 80).sum())}")
