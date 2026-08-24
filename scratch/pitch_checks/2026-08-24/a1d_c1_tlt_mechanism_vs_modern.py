"""C1 -- the last decisive check.

a1c found two things that cannot both be true of the same object:
  * the MECHANISM (month-end index-extension buying) has decayed hard. TLT's
    ME-1 session excess runs +25.65 bp (t 3.09) in 2002-2012, +14.89 (t 2.56)
    in 2013-2019 and +3.99 (t 0.37) in 2020-2026, and the same decay shows in
    IEF, LQD and AGG.
  * the CELL has not decayed at all: the 2014-2026 holdout pays +0.463% at
    t 3.56, better than the 2002-2013 in-sample +0.394%.

If the mechanism is gone but the cell still pays, then whatever is paying now
is NOT the thing I validated in round 2, and the mechanism argument that
answered the parked entry's grid debt does not carry to today's trade.

  (1) session-by-session excess inside the ME-5 hold, by era
  (2) the AUGUST subcell is a fossil (13-for-13 through 2014, 5-of-11 since):
      is August significantly worse than the rest of the modern-era cell?
  (3) modern-era cell with February and November removed
  (4) the yield-regime (bond-bull fossil) test restricted to the modern era
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from scipy import stats as st

pd.set_option("display.width", 230)
raw = load_prices(["TLT", "^TNX"])
cl = raw["TLT"]["Close"].dropna()
idx = cl.index
C = cl.values
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
isl = ym.ne(ym.shift(-1)).values
isl[-1] = False
pos = np.arange(len(idx))
lp = np.full(len(idx), -1)
cur = -1
for i in range(len(idx) - 1, -1, -1):
    if isl[i]:
        cur = i
    lp[i] = cur
me = np.where(lp >= 0, lp - pos, np.nan).astype(float)
E = pos[me == 5]
E = E[lp[E] < len(idx)]
r5 = C[lp[E]] / C[E] - 1.0
dts = idx[E]
r1 = np.full(len(idx), np.nan)
r1[:-1] = C[1:] / C[:-1] - 1.0

# ---------------------------------------------------------------------- (1)
print("(1) SESSION-BY-SESSION excess inside the ME-5 hold, BY ERA (TLT)")
rows = []
for lo, hi in ((2002, 2012), (2013, 2019), (2020, 2026), (2014, 2026)):
    sel = (idx.year >= lo) & (idx.year <= hi)
    u = np.nanmean(r1[sel])
    rec = {"era": f"{lo}-{hi}", "uncond_bp": round(100 * 100 * u, 2)}
    tot = 0.0
    for k in (5, 4, 3, 2, 1):
        v = r1[sel & (me == k) & ~np.isnan(r1)]
        e = 100 * 100 * (v.mean() - u)
        tot += e
        rec[f"ME-{k}"] = round(e, 2)
        rec[f"t{k}"] = round((v.mean() - u) / (v.std(ddof=1) / np.sqrt(len(v))), 2)
    rec["sum_bp"] = round(tot, 2)
    rec["flow_share"] = round(100 * (rec["ME-1"] + rec["ME-2"]) / tot, 0) if tot else np.nan
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))
print("  flow_share = how much of the hold's excess sits in the LAST TWO sessions,")
print("  i.e. in the index-extension window the mechanism argument rests on.")

# ---------------------------------------------------------------------- (2)
print("\n(2) is AUGUST adverse inside the modern-era cell?")
aug = dts.month == 8
for lo in (2002, 2013, 2015):
    m = dts.year >= lo
    a, b = r5[m & aug], r5[m & ~aug]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print("   %d+ : August N=%2d %+.3f%% (hit %.0f%%)   rest N=%3d %+.3f%% (hit %.0f%%)"
          "   diff %+.3fpp  Welch t %+.2f  p %.3f"
          % (lo, len(a), 100 * a.mean(), 100 * (a > 0).mean(),
             len(b), 100 * b.mean(), 100 * (b > 0).mean(),
             100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se,
             2 * (1 - st.norm.cdf(abs((a.mean() - b.mean()) / se)))))
print("   August ME-5 through 2014: %d-%d, mean %+.3f%%"
      % (int((r5[aug & (dts.year <= 2014)] > 0).sum()),
         int((r5[aug & (dts.year <= 2014)] <= 0).sum()),
         100 * r5[aug & (dts.year <= 2014)].mean()))
print("   August ME-5 2015+       : %d-%d, mean %+.3f%%  (sign p vs a coin %.3f)"
      % (int((r5[aug & (dts.year >= 2015)] > 0).sum()),
         int((r5[aug & (dts.year >= 2015)] <= 0).sum()),
         100 * r5[aug & (dts.year >= 2015)].mean(),
         sign_test(int((r5[aug & (dts.year >= 2015)] > 0).sum()),
                   int((aug & (dts.year >= 2015)).sum()))))
# is the 12-month profile stable across eras? rank correlation
p1 = pd.Series(r5[dts.year <= 2013], index=dts[dts.year <= 2013].month).groupby(level=0).mean()
p2 = pd.Series(r5[dts.year >= 2014], index=dts[dts.year >= 2014].month).groupby(level=0).mean()
rho = st.spearmanr(p1.reindex(range(1, 13)).values, p2.reindex(range(1, 13)).values)
print("   month-profile Spearman across the two eras: rho %+.2f p %.3f "
      "(a stable seasonal would be strongly positive)" % (rho.correlation, rho.pvalue))
print("   pre-2014 month means: %s" % {m: round(100 * v, 2) for m, v in p1.items()})
print("   2014+   month means: %s" % {m: round(100 * v, 2) for m, v in p2.items()})

# ---------------------------------------------------------------------- (3)
print("\n(3) modern-era cell with the two biggest months removed")
for lo in (2002, 2014):
    m = dts.year >= lo
    for drop, lbl in ((np.array([]), "all months"),
                      (np.array([2]), "ex-Feb"),
                      (np.array([2, 11]), "ex-Feb/Nov"),
                      (np.array([2, 11, 6]), "ex-Feb/Nov/Jun")):
        k = m & ~np.isin(dts.month, drop)
        v = r5[k]
        print("   %d+ %-16s N=%3d mean %+.4f%% t %+.2f hit %.1f%%"
              % (lo, lbl, len(v), 100 * v.mean(),
                 v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 100 * (v > 0).mean()))

# ---------------------------------------------------------------------- (4)
print("\n(4) yield-regime (bond-bull fossil) test, MODERN ERA only")
y = raw["^TNX"]["Close"].dropna().reindex(idx).ffill()
dy = (y - y.shift(63)).reindex(dts).values
for lo, hi in ((2002, 2013), (2014, 2026), (2020, 2026)):
    m = (dts.year >= lo) & (dts.year <= hi) & ~np.isnan(dy)
    f, r_ = r5[m & (dy < 0)], r5[m & (dy >= 0)]
    print("   %d-%d  yields FALLING N=%3d %+.3f%% (t %+.2f) | RISING N=%3d %+.3f%% (t %+.2f)"
          % (lo, hi, len(f), 100 * f.mean(),
             f.mean() / (f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else np.nan,
             len(r_), 100 * r_.mean(),
             r_.mean() / (r_.std(ddof=1) / np.sqrt(len(r_))) if len(r_) > 1 else np.nan))
print("   live: TLT 63d yield change = %+.3f pt (rising = the harder regime)"
      % float(y.iloc[-1] - y.iloc[-64]))
