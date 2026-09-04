"""C1 round 2. Three questions round 1 raised.

(a) Is the AUGUST inversion a real split or noise? Pool the four equity
    vehicles into one month-matched average overnight and split.
(b) Is ME+0 isolated on the ladder, or is it sitting on the turn-of-month
    plateau? ME+0 vs the pooled ME-5..ME-1 and ME+1..ME+5 neighbourhoods.
(c) The exact number that would turn C1 on: what excess over the
    unconditional overnight is needed for a 5x cost multiple.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

EQ = ["SPY", "QQQ", "IWM", "DIA"]
COST = {"SPY": 4.0, "QQQ": 4.0, "IWM": 6.0, "TLT": 5.0, "DIA": 6.0}
px = load_prices(EQ + ["TLT"])


def me_pos(idx):
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    return [int(g.index.max()) for _, g in per.groupby(per.values)][:-1]


# (a) pooled equity overnight, month-matched on SPY's calendar
cal = px["SPY"].index
pool = {}
for t in EQ:
    d = px[t].reindex(cal)
    o, c = d["Open"].values, d["Close"].values
    pool[t] = pd.Series(np.r_[o[1:] / c[:-1] - 1.0, np.nan], index=cal)
P = pd.DataFrame(pool)
mep = [p for p in me_pos(cal) if p + 1 < len(cal)]
mdates = cal[mep]
avg = P.mean(axis=1)                       # equal-weight the 4 vehicles
a_me = avg.loc[mdates].values
a_all = avg.loc[cal[mep[0]]:cal[mep[-1]]].dropna().values

aug = mdates.month == 8
mid = (mdates.year % 4) == 2
show([summarize(a_me, "pooled-4 equity overnight at ME-0 (all months)"),
      summarize(a_me[aug], "AUGUST turn only (today)"),
      summarize(a_me[~aug], "non-August"),
      summarize(a_me[aug & mid], "AUGUST turn in a MIDTERM year (today's exact cell)"),
      summarize(a_all, "CTRL unconditional overnight, same span")],
     "(a) pooled 4-vehicle equity overnight, August split")
x, y = a_me[aug], a_me[~aug]
se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
w = int((x > 0).sum())
print("  August MINUS non-August = %+.4fpp  welch t = %+.2f" % (100 * (x.mean() - y.mean()),
                                                                (x.mean() - y.mean()) / se))
print("  August record %d-%d, sign p (>=) = %.4f, sign p (<=, i.e. inversion) = %.4f"
      % (w, len(x) - w, sign_test(w, len(x)), sign_test(len(x) - w, len(x))))
print("  August years: %s"
      % [(d.year, round(100 * v, 2)) for d, v in zip(mdates[aug], x)])
print("  August excess over the unconditional overnight = %+.4fpp"
      % (100 * (x.mean() - a_all.mean())))

# (b) ladder plateau
print()
print("(b) is ME+0 isolated, or on the turn-of-month plateau?")
rows = []
for t in EQ + ["TLT"]:
    d = px[t]
    idx = d.index
    o, c = d["Open"].values, d["Close"].values
    mp = [p for p in me_pos(idx) if p + 1 < len(idx)]
    def mean_at(k):
        v = [o[p + k + 1] / c[p + k] - 1.0 for p in mp
             if 0 <= p + k and p + k + 1 < len(idx)]
        return 100 * float(np.mean(v))
    pre = float(np.mean([mean_at(k) for k in range(-5, 0)]))
    post = float(np.mean([mean_at(k) for k in range(1, 6)]))
    rows.append({"ticker": t, "ME+0": round(mean_at(0), 4),
                 "mean ME-5..-1": round(pre, 4),
                 "mean ME+1..+5": round(post, 4),
                 "ME0 minus pre": round(mean_at(0) - pre, 4)})
show(rows, "ladder plateau test")

# (c) the turn-on number
print()
print("(c) what would turn C1 on")
for t in EQ + ["TLT"]:
    d = px[t]
    idx = d.index
    o, c = d["Open"].values, d["Close"].values
    mp = [p for p in me_pos(idx) if p + 1 < len(idx)]
    on = np.array([o[p + 1] / c[p] - 1.0 for p in mp])
    lo, hi = mp[0], mp[-1]
    unc = (o[lo + 1:hi + 2] / c[lo:hi + 1] - 1.0).mean()
    exc = 1e4 * (on.mean() - unc)
    need = 5.0 * COST[t]
    print("  %-4s excess %+.2f bps; needs %.1f bps for 5x at %.1f bps rt "
          "-> short by %.2f bps (%.1fx of where it is)"
          % (t, exc, need, COST[t], need - exc, need / exc if exc > 0 else float("nan")))
