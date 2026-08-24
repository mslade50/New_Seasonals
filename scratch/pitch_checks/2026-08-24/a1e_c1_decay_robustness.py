"""C1 -- is the ME-1 session decay robust, or is it 2020's vol?

a1d's kill rests on the ME-1 session excess collapsing from +25.65 bp (t 3.09,
2002-2012) to +3.99 bp (t 0.37, 2020-2026) across four instruments. Two ways
that could be an artifact: (a) 2020's vol inflating the denominator, (b) a
mean dragged by one crash month. Both tested here, plus a rolling window so
the decay is visible as a trend rather than three arbitrary buckets.

Also confirms the live calendar: 2026-08-31 is August's last trading session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
TK = ["TLT", "IEF", "LQD", "AGG"]
raw = load_prices(TK)


def sessions(tkr):
    c = raw[tkr]["Close"].dropna()
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
    me = np.where(lp >= 0, lp - pos, np.nan).astype(float)
    v = c.values
    r = np.full(len(ix), np.nan)
    r[:-1] = v[1:] / v[:-1] - 1.0
    return ix, me, r


print("(1) ME-1 session excess, robustness of the decay")
for tkr in TK:
    ix, me, r = sessions(tkr)
    out = []
    for lbl, sel in (("2002-2012", (ix.year <= 2012)),
                     ("2013-2019", (ix.year >= 2013) & (ix.year <= 2019)),
                     ("2020-2026", (ix.year >= 2020)),
                     ("2020-2026 ex-2020", (ix.year >= 2021)),
                     ("2021-2026 ex-Mar", (ix.year >= 2021) & (ix.month != 3))):
        u = np.nanmean(r[sel])
        v = r[sel & (me == 1) & ~np.isnan(r)]
        if len(v) < 5:
            continue
        med = 100 * 100 * (np.median(v) - np.nanmedian(r[sel]))
        out.append("%s %+.2f bp (t %+.2f, med-excess %+.2f bp, n %d)"
                   % (lbl, 100 * 100 * (v.mean() - u),
                      (v.mean() - u) / (v.std(ddof=1) / np.sqrt(len(v))), med, len(v)))
    print("  %-4s %s" % (tkr, "  |  ".join(out)))

print("\n(2) rolling 8-year window on TLT's ME-1 session excess")
ix, me, r = sessions("TLT")
for y0 in range(2002, 2019):
    y1 = y0 + 7
    sel = (ix.year >= y0) & (ix.year <= y1)
    u = np.nanmean(r[sel])
    v = r[sel & (me == 1) & ~np.isnan(r)]
    print("   %d-%d  %+6.2f bp  t %+.2f  hit %.1f%%  n=%d"
          % (y0, y1, 100 * 100 * (v.mean() - u),
             (v.mean() - u) / (v.std(ddof=1) / np.sqrt(len(v))),
             100 * (v > 0).mean(), len(v)))

print("\n(3) hit rate of the ME-1 session (a mean-free read of the same decay)")
for tkr in TK:
    ix, me, r = sessions(tkr)
    s = []
    for lbl, sel in (("2002-2012", ix.year <= 2012),
                     ("2013-2019", (ix.year >= 2013) & (ix.year <= 2019)),
                     ("2020-2026", ix.year >= 2020)):
        v = r[sel & (me == 1) & ~np.isnan(r)]
        b = float((r[sel][~np.isnan(r[sel])] > 0).mean())
        w = int((v > 0).sum())
        s.append("%s %.1f%% vs base %.1f%% (%d-%d, sign p %.3f)"
                 % (lbl, 100 * (v > 0).mean(), 100 * b, w, len(v) - w,
                    sign_test(w, len(v), b)))
    print("  %-4s %s" % (tkr, "  |  ".join(s)))

print("\n(4) live calendar confirmation")
cal = pd.bdate_range("2026-08-24", "2026-09-08")
print("   business days 2026-08-24..09-08:", [str(d.date()) for d in cal])
print("   Labor Day 2026 = 2026-09-07, so August's last trading session is "
      "2026-08-31 and today (2026-08-24) is ME-5. Sessions in the hold: "
      "08-24 (entry MOC), 08-25, 08-26, 08-27, 08-28 (Jackson Hole), "
      "08-31 (exit MOC).")
