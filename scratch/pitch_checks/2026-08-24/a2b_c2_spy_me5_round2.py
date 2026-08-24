"""C2 ROUND 2 -- attribution, concentration, and the vehicle rescue attempt.

a2 found the equity ME-5 cell fails its own mechanism test. This closes it:
  (A) WHERE the excess sits, session by session, summed -- if the excess is
      one session out of 16 scanned it is a scan artifact wearing a flow label
  (B) the ME-1 -> ME-0 session, which is the ONLY session the month-end flow
      story predicts, measured on its own and split by era
  (C) concentration: top-2, drop-best-year, drop-November
  (D) the live subcell: August x midterm, and the sign test scored against
      SPY's OWN August 5-session up-rate rather than a coin
  (E) the IWM / QQQ rescue: does the small-cap vehicle carry a real month-end
      effect, or the same one-session scan artifact
  (F) turn-of-month overlap priced properly against the registry entry
  (G) the honest multiplicity: 16 offsets x 5 vehicles = the grid I walked
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from scipy import stats as st

pd.set_option("display.width", 230)
COST = {"SPY": 1.5, "IWM": 2.0, "QQQ": 1.5, "DIA": 2.0, "TLT": 3.0}
TK = ["SPY", "IWM", "QQQ", "DIA", "TLT"]
raw = load_prices(TK)


def build(t):
    c = raw[t]["Close"].dropna()
    ix = c.index
    ym = pd.Series(ix.year * 100 + ix.month, index=ix)
    isl = ym.ne(ym.shift(-1)).values
    isl[-1] = False
    pos = np.arange(len(ix))
    lp = np.full(len(ix), -1)
    cur = -1
    for i in range(len(ix) - 1, -1, -1):
        if isl[i]:
            cur = i
        lp[i] = cur
    return c.values, ix, np.where(lp >= 0, lp - pos, np.nan).astype(float), lp, pos


def cell(t, k):
    cv, ix, me, lp, pos = build(t)
    e = pos[me == k]
    e = e[lp[e] < len(ix)]
    return ix[e], cv[lp[e]] / cv[e] - 1.0


def alldays(t, k):
    cv, ix, *_ = build(t)
    return ix[:-k], cv[k:] / cv[:-k] - 1.0


def mo_matched(t, k, d, r):
    bd, br = alldays(t, k)
    b = pd.Series(br, index=bd).groupby(bd.month).mean()
    return r - b.reindex(d.month).values


# ---------------------------------------------------------------------- (A)
print("(A) SESSION-BY-SESSION excess inside the ME-5 hold (SPY)")
cv, ix, me, lp, pos = build("SPY")
r1 = np.full(len(ix), np.nan)
r1[:-1] = cv[1:] / cv[:-1] - 1.0
un = np.nanmean(r1)
tot = 0.0
for k in (5, 4, 3, 2, 1):
    m = (me == k) & ~np.isnan(r1)
    v = r1[m]
    e = 100 * 100 * (v.mean() - un)
    tot += e
    print("   session ME-%d -> ME-%d : mean %+.4f%%  excess %+6.2f bp  t %+.2f  hit %.1f%%"
          % (k, k - 1, 100 * v.mean(), e,
             v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 100 * (v > 0).mean()))
print("   -> summed excess over the 5-session hold = %+.2f bp (%.3f%%)" % (tot, tot / 100))
print("   the single ME-4 -> ME-3 session is %.0f%% of it." % (100 * 14.14 / tot))
# how many of the 16 month-position sessions clear |t| 2 by chance?
ts = []
for k in range(0, 16):
    m = (me == k) & ~np.isnan(r1)
    v = r1[m]
    ts.append(abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))))
print("   16 month-position sessions scanned; %d clear |t|>=2 "
      "(expected by chance at alpha .05: %.1f). max |t| = %.2f -> "
      "Sidak familywise p = %.3f"
      % (sum(t_ >= 2 for t_ in ts), 16 * 0.05, max(ts),
         1 - (1 - 2 * (1 - st.norm.cdf(max(ts)))) ** 16))

# ---------------------------------------------------------------------- (B)
print("\n(B) the ONLY session the month-end flow story predicts: ME-1 -> ME-0")
d1, r_1 = cell("SPY", 1)
show([summarize(r_1, "SPY ME-1 -> ME-0 (all)"),
      summarize(r_1[d1.year < 2013], "  pre-2013"),
      summarize(r_1[d1.year >= 2013], "  2013+"),
      summarize(r_1[d1.month == 8], "  August only"),
      summarize(alldays("SPY", 1)[1], "CTRL-b all 1-session")])
print("   for contrast, the same session on TLT: %s"
      % {k: round(v, 4) for k, v in summarize(cell("TLT", 1)[1], "").items()
         if k in ("n", "mean_pct", "hit", "t")})

# ---------------------------------------------------------------------- (C)
print("\n(C) CONCENTRATION of the SPY ME-5 cell")
d5, r5 = cell("SPY", 5)
ex5 = mo_matched("SPY", 5, d5, r5)
print("   " + cluster_note(d5, r5, k=2))
byy = pd.Series(r5, index=d5).groupby(d5.year).mean()
bad = byy.idxmax()
print("   drop the best YEAR (%d, %+.3f%%): mean %+.4f%% t %+.2f"
      % (bad, 100 * byy.max(), 100 * r5[d5.year != bad].mean(),
         r5[d5.year != bad].mean() / (r5[d5.year != bad].std(ddof=1) / np.sqrt((d5.year != bad).sum()))))
nn = d5.month != 11
print("   drop NOVEMBER (+1.691%%, the one month with t>2): mean %+.4f%% t %+.2f; "
      "month-matched %+.4f%% t %+.2f"
      % (100 * r5[nn].mean(),
         r5[nn].mean() / (r5[nn].std(ddof=1) / np.sqrt(nn.sum())),
         100 * mo_matched("SPY", 5, d5[nn], r5[nn]).mean(),
         mo_matched("SPY", 5, d5[nn], r5[nn]).mean() /
         (mo_matched("SPY", 5, d5[nn], r5[nn]).std(ddof=1) / np.sqrt(nn.sum()))))
top2 = np.argsort(-np.abs(r5))[:2]
keep = np.ones(len(r5), bool)
keep[top2] = False
print("   drop the 2 largest |moves| (%s): mean %+.4f%% t %+.2f"
      % (", ".join(str(d5[i].date()) for i in top2), 100 * r5[keep].mean(),
         r5[keep].mean() / (r5[keep].std(ddof=1) / np.sqrt(keep.sum()))))

# ---------------------------------------------------------------------- (D)
print("\n(D) the LIVE subcell: August in a MIDTERM year")
aug = d5.month == 8
mid = (d5.year % 4) == 2
_, b5 = alldays("SPY", 5)
bd, br = alldays("SPY", 5)
aug_up = float((br[bd.month == 8] > 0).mean())
show([summarize(r5[aug & mid], "SPY ME-5 AUGUST x MIDTERM (the live cell)"),
      summarize(r5[aug & ~mid], "SPY ME-5 August, non-midterm"),
      summarize(r5[mid], "SPY ME-5 all midterm months"),
      summarize(r5[~mid], "SPY ME-5 non-midterm")])
sel = aug & mid
print("   the 6 August-midterm observations: %s"
      % ", ".join("%d %+.2f%%" % (d.year, 100 * v) for d, v in zip(d5[sel], r5[sel])))
w = int((r5[sel] > 0).sum())
print("   record %d-%d; SPY's own August 5-session up-rate %.3f -> sign p (upside) %.4f"
      % (w, int(sel.sum()) - w, aug_up, sign_test(w, int(sel.sum()), aug_up)))

# ---------------------------------------------------------------------- (E)
print("\n(E) VEHICLE RESCUE: IWM / QQQ / DIA")
rows = []
for t in ("SPY", "IWM", "QQQ", "DIA"):
    d, r = cell(t, 5)
    e = mo_matched(t, 5, d, r)
    s = summarize(r, f"{t} ME-5 raw")
    s["mo_matched"] = round(100 * e.mean(), 3)
    s["mo_t"] = round(e.mean() / (e.std(ddof=1) / np.sqrt(len(e))), 2)
    s["x_cost"] = round(100 * s["mean_pct"] / COST[t], 1)
    rows.append(s)
    a = d.month == 8
    m2 = (d.year % 4) == 2
    rows.append(summarize(r[a & m2], f"  {t} August x MIDTERM"))
show(rows, "vehicles")
print("\n  per-session excess inside the hold, every vehicle (bp):")
out = []
for t in ("SPY", "IWM", "QQQ", "DIA", "TLT"):
    cvv, ixx, mee, lpp, poss = build(t)
    rr = np.full(len(ixx), np.nan)
    rr[:-1] = cvv[1:] / cvv[:-1] - 1.0
    u = np.nanmean(rr)
    rec = {"tkr": t}
    for k in (5, 4, 3, 2, 1):
        v = rr[(mee == k) & ~np.isnan(rr)]
        rec[f"ME-{k}"] = round(100 * 100 * (v.mean() - u), 2)
        rec[f"t{k}"] = round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)
    out.append(rec)
print(pd.DataFrame(out).to_string(index=False))

# ---------------------------------------------------------------------- (F)
print("\n(F) TURN-OF-MONTH overlap, priced")
cvv, ixx, mee, lpp, poss = build("SPY")
e5 = poss[mee == 5]
e5 = e5[lpp[e5] < len(ixx)]
legA = cvv[lpp[e5] - 1] / cvv[e5] - 1.0        # ME-5 -> ME-1
legB = cvv[lpp[e5]] / cvv[lpp[e5] - 1] - 1.0   # ME-1 -> ME-0 (classic ToM day 1)
dd = ixx[e5]
show([summarize(legA, "ME-5 -> ME-1 (outside the classic ToM window)"),
      summarize(legB, "ME-1 -> ME-0 (inside it)"),
      summarize(legA[dd.year >= 2013], "  ME-5 -> ME-1, 2013+"),
      summarize(legB[dd.year >= 2013], "  ME-1 -> ME-0, 2013+")])

# ---------------------------------------------------------------------- (G)
print("\n(G) the grid I walked: 15 entry offsets x 4 equity vehicles")
best = []
for t in ("SPY", "IWM", "QQQ", "DIA"):
    for k in range(1, 16):
        d, r = cell(t, k)
        e = mo_matched(t, k, d, r)
        best.append((abs(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))), t, k))
best.sort(reverse=True)
print("   top 5 by |t| on month-matched excess:")
for t_, tk, k in best[:5]:
    print("     |t| %.2f  %s ME-%d" % (t_, tk, k))
p = 2 * (1 - st.norm.cdf(best[0][0]))
print("   K=%d cells; best pointwise p %.4f -> Sidak familywise %.3f"
      % (len(best), p, 1 - (1 - p) ** len(best)))
print("   SPY ME-5 |t| rank in the grid: %d of %d"
      % (1 + [x[1:] for x in best].index(("SPY", 5)), len(best)))
