"""C1 round 1 -- long TLT MOC at ME-5 into the month-end close.

The parked entry (W12) measured ONE entry offset, ME-9, and reported +0.540%
over 288 anchors at t=3.88 with an EXIT-offset placebo ladder that decayed.
It never ran the ENTRY-offset ladder. Today is ME-5, not ME-9.

This script:
  (0) verifies the parked ME-9 headline from scratch on my own construction
  (1) walks the ENTRY offset k = 1..15 (entry MOC at ME-k close, exit at the
      month's last close, so h = k sessions) and reports RAW and EXCESS
      (excess = cell mean minus TLT's own all-days k-session drift, CTRL-b,
      and minus the same-span drift, CTRL-a)
  (2) decomposes the whole thing into PER-SESSION forward returns bucketed by
      me_offset -- the sharpest test of spike-vs-plateau, because a raw ladder
      that rises with k is indistinguishable from "TLT drifts up and k is the
      holding period"
  (3) month-demeans (the 2026-08-13 lesson) and splits era / midterm
  (4) applies the distance-from-the-52w-low gradient AT the ME-5 form, and
      reads the fitted value at today's +0.86%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

TK = "TLT"
COST_BPS = 3.0
raw = load_prices([TK])
c = raw[TK]["Close"].dropna()
idx = c.index
print("TLT panel %s .. %s  N=%d sessions" % (idx[0].date(), idx[-1].date(), len(idx)))

ymv = pd.Series(idx.year * 100 + idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)).values
is_last[-1] = False          # the final row's month is not known to be complete
pos = np.arange(len(idx))

# me_offset[i] = sessions from i to that month's last session (NaN if unknown)
last_pos = np.full(len(idx), -1)
cur = -1
for i in range(len(idx) - 1, -1, -1):
    if is_last[i]:
        cur = i
    last_pos[i] = cur
me_off = np.where(last_pos >= 0, last_pos - pos, np.nan).astype(float)

# sanity: where is today?
print("last 8 rows: ", [(str(d.date()), int(m) if m == m else None)
                        for d, m in zip(idx[-8:], me_off[-8:])])
aug26 = idx[(idx.year == 2026) & (idx.month == 8)]
print("Aug-2026 sessions on file: %d, last = %s" % (len(aug26), aug26[-1].date()))
print("NOTE: 2026-08-31 is not on the tape yet, so the live August month is EXCLUDED "
      "from every cell below by construction (is_last is unknown for it).")

cv = c.values


def cell(k):
    """entry at the ME-k close, exit at the month's last close. h = k."""
    ent = pos[(me_off == k)]
    ent = ent[last_pos[ent] < len(idx)]
    r = cv[last_pos[ent]] / cv[ent] - 1.0
    return idx[ent], r


def all_days_k(k):
    """TLT's own unconditional k-session close-to-close return, all days."""
    return (cv[k:] / cv[:-k] - 1.0)


# ---------------------------------------------------------------- (0) verify
d9, r9 = cell(9)
print("\n(0) VERIFY the parked ME-9 headline")
show([summarize(r9, "ME-9 -> ME-0, raw (parked: +0.540%%, t 3.88)")])
mo = pd.Series(r9, index=d9).groupby(d9.month).transform("mean")
print("    month-demeaned mean %+.4f%%  t %+.2f   (parked: +0.391%%, t 2.85)"
      % (100 * (r9 - mo.values).mean(),
         (r9 - mo.values).mean() / ((r9 - mo.values).std(ddof=1) / np.sqrt(len(r9)))))

# --------------------------------------------------- (1) ENTRY-offset ladder
print("\n(1) ENTRY-OFFSET LADDER  (the test the parked entry never ran)")
rows = []
for k in range(1, 16):
    d, r = cell(k)
    base = all_days_k(k)
    dm = pd.Series(r, index=d).groupby(d.month).transform("mean").values
    s = summarize(r, f"ME-{k:02d} -> ME-0")
    s["ctrl_b_pct"] = round(100 * base.mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    s["demean_mo_pct"] = round(100 * (r - dm).mean(), 3)
    s["excess_per_sess_bp"] = round(100 * 100 * (r.mean() - base.mean()) / k, 2)
    s["x_cost"] = round((100 * 100 * r.mean()) / COST_BPS, 1)
    rows.append(s)
show(rows, "raw / excess-over-own-drift / month-demeaned, by ENTRY offset")

# ------------------------------------------- (2) per-session decomposition
print("\n(2) PER-SESSION decomposition: 1-session forward return by me_offset")
r1 = np.full(len(idx), np.nan)
r1[:-1] = cv[1:] / cv[:-1] - 1.0
rows = []
uncond = np.nanmean(r1)
for k in range(0, 16):
    m = (me_off == k) & ~np.isnan(r1)
    v = r1[m]
    if len(v) < 5:
        continue
    s = summarize(v, f"session ME-{k:02d} -> ME-{k-1:02d}")
    s["excess_bp"] = round(100 * 100 * (v.mean() - uncond), 2)
    rows.append(s)
show(rows, f"unconditional 1-session mean = {100*uncond:+.4f}%")
print("  READ: the ME-k -> ME-0 cell is the SUM of sessions ME-k..ME-1. If the "
        "excess is spread flat across offsets it is month position, not a "
        "month-end flow event, and ME-5 buys 5/9 of the ME-9 number for 5/9 of "
        "the exposure -- no reason to prefer either.")

# ------------------------------------------------ (3) era / midterm on ME-5
print("\n(3) ME-5 form: era, midterm, month split")
d5, r5 = cell(5)
base5 = all_days_k(5)
show([summarize(r5, f"ME-5 all (N={len(r5)})"),
      summarize(base5, "CTRL-b all days 5-session"),
      summarize(r5[d5.year < 2013], "ME-5 pre-2013"),
      summarize(r5[(d5.year >= 2013) & (d5.year < 2019)], "ME-5 2013-2018"),
      summarize(r5[d5.year >= 2019], "ME-5 2019+"),
      summarize(r5[d5.year >= 2021], "ME-5 2021+"),
      summarize(r5[(d5.year % 4) == 2], "ME-5 MIDTERM years"),
      summarize(r5[(d5.year % 4) != 2], "ME-5 non-midterm"),
      summarize(r5[d5.month == 8], "ME-5 AUGUST only"),
      ], "ME-5 splits")

print("\n  ME-5 by calendar month:")
show([summarize(r5[d5.month == m], f"month {m:02d}") for m in range(1, 13)])

# the bond-bull fossil test the registry demands of any duration seasonal
tnx = load_prices(["^TNX"]).get("^TNX")
if tnx is not None:
    y = tnx["Close"].dropna().reindex(idx).ffill()
    dy63 = (y - y.shift(63))
    fall = dy63.reindex(d5).values < 0
    print("\n  BOND-BULL FOSSIL test (yield 63d change at the ME-5 entry):")
    show([summarize(r5[fall == True], "yields FALLING trailing 63d"),
          summarize(r5[fall == False], "yields RISING trailing 63d")])

# --------------------------------- (4) distance-from-52w-low gradient at ME-5
print("\n(4) DISTANCE-FROM-THE-52W-LOW GRADIENT, applied to the ME-5 form")
lo252 = pd.Series(cv, index=idx).rolling(252).min()
dist = (pd.Series(cv, index=idx) / lo252 - 1.0) * 100.0
dv = dist.reindex(d5).values
ok = ~np.isnan(dv)
X = dv[ok]
Y = r5[ok] * 100.0
b, a = np.polyfit(X, Y, 1)
res = Y - (a + b * X)
se = np.sqrt((res ** 2).sum() / (len(X) - 2) / ((X - X.mean()) ** 2).sum())
print("  within-cell OLS: fwd%% = %+.4f %+.4f * dist%%   (slope t = %+.2f, N=%d)"
      % (a, b, b / se, len(X)))
today_dist = 0.86
print("  fitted value at TODAY's +0.86%% off the low: %+.4f%%  -> %.1fx the %g bps cost"
      % (a + b * today_dist, (a + b * today_dist) * 100 / COST_BPS, COST_BPS))
print("  percentile of today's 0.86%% within the ME-5 trigger population: %.1f"
      % (100.0 * (X <= today_dist).mean()))
show([summarize(r5[ok][X <= 1.0], "ME-5, TLT within 1.0% of 52w low"),
      summarize(r5[ok][(X > 1.0) & (X <= 3.0)], "ME-5, 1-3% off low"),
      summarize(r5[ok][X > 3.0], "ME-5, >3% off low"),
      summarize(r5[ok][X > 10.0], "ME-5, >10% off low")],
     "ME-5 bucketed by distance from the 52w low")

# and the same for ME-9, to show the gradient is not an ME-5 artifact
d9d = dist.reindex(d9).values
ok9 = ~np.isnan(d9d)
X9, Y9 = d9d[ok9], r9[ok9] * 100
b9, a9 = np.polyfit(X9, Y9, 1)
res9 = Y9 - (a9 + b9 * X9)
se9 = np.sqrt((res9 ** 2).sum() / (len(X9) - 2) / ((X9 - X9.mean()) ** 2).sum())
print("\n  ME-9 gradient for contrast: slope %+.4f (t %+.2f); parked value +0.126 (t +2.18)"
      % (b9, b9 / se9))
show([summarize(r9[ok9][X9 <= 1.0], "ME-9, within 1.0% of low"),
      summarize(r9[ok9][X9 > 3.0], "ME-9, >3% off low")])
