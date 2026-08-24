"""C2 round 1 -- long SPY MOC at ME-5 into the month-end close.

Never measured on equities in this repo. Same anchor as C1, equity vehicle.
Owed by the registry, all in this one script:
  (0) the raw cell and the ENTRY-offset ladder (spike vs ramp)
  (1) the PER-SESSION decomposition by me_offset, extended THROUGH the turn
      into the next month's first 3 sessions -- this is what prices the
      "classic turn-of-month is last-1..first-3, arbitraged away post-2013"
      registry entry against the ME-5..ME-0 span, by MEASURING the overlap
      instead of asserting the difference
  (2) month-of-year control (the 2026-08-13 lesson), done properly: subtract
      the instrument's UNCONDITIONAL same-month k-session drift, not the
      cell's own month mean (which is zero by construction)
  (3) the era split at the registry's own 2013 cut, and the midterm split
  (4) the AUGUST subcell, which is what is live
  (5) vehicle alternatives IWM / QQQ, and the same cell on TLT for contrast
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

COST = {"SPY": 1.5, "IWM": 2.0, "QQQ": 1.5, "TLT": 3.0, "DIA": 2.0}
TK = ["SPY", "IWM", "QQQ", "TLT", "DIA"]
raw = load_prices(TK)


def build(tkr):
    c = raw[tkr]["Close"].dropna()
    idx = c.index
    ymv = pd.Series(idx.year * 100 + idx.month, index=idx)
    is_last = ymv.ne(ymv.shift(-1)).values
    is_last[-1] = False
    pos = np.arange(len(idx))
    last_pos = np.full(len(idx), -1)
    cur = -1
    for i in range(len(idx) - 1, -1, -1):
        if is_last[i]:
            cur = i
        last_pos[i] = cur
    me_off = np.where(last_pos >= 0, last_pos - pos, np.nan).astype(float)
    return c.values, idx, me_off, last_pos, pos


def cell(tkr, k):
    cv, idx, me_off, last_pos, pos = build(tkr)
    ent = pos[me_off == k]
    ent = ent[last_pos[ent] < len(idx)]
    return idx[ent], cv[last_pos[ent]] / cv[ent] - 1.0


def alldays_k(tkr, k):
    cv, idx, *_ = build(tkr)
    return idx[:-k], cv[k:] / cv[:-k] - 1.0


def month_matched_excess(tkr, k, d, r):
    """Subtract the instrument's UNCONDITIONAL k-session drift for the SAME
    calendar month. This is the month-of-year control the registry demands;
    demeaning by the cell's own month mean returns 0 by construction."""
    bd, br = alldays_k(tkr, k)
    base = pd.Series(br, index=bd).groupby(bd.month).mean()
    return r - base.reindex(d.month).values


for tkr in ["SPY"]:
    cv, idx, me_off, last_pos, pos = build(tkr)
    print("%s panel %s .. %s  N=%d" % (tkr, idx[0].date(), idx[-1].date(), len(idx)))

# --------------------------------------------------- (0) entry-offset ladder
print("\n(0) SPY ENTRY-OFFSET LADDER, exit always at the month's last close")
rows = []
for k in range(1, 16):
    d, r = cell("SPY", k)
    _, br = alldays_k("SPY", k)
    ex = month_matched_excess("SPY", k, d, r)
    s = summarize(r, f"ME-{k:02d} -> ME-0")
    s["ctrl_b_pct"] = round(100 * br.mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * br.mean(), 3)
    s["mo_matched_pct"] = round(100 * ex.mean(), 3)
    s["mo_t"] = round(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))), 2)
    s["x_cost"] = round(100 * 100 * r.mean() / COST["SPY"], 1)
    rows.append(s)
show(rows, "SPY: raw / excess over own drift / month-matched excess")

# ------------------------------------ (1) per-session, through the turn
print("\n(1) PER-SESSION decomposition around the turn (SPY)")
r1 = np.full(len(idx), np.nan)
r1[:-1] = cv[1:] / cv[:-1] - 1.0
uncond = np.nanmean(r1)
# forward month-day index: 1 = first session of the new month
ymv = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ymv.groupby(ymv.values).cumcount().values + 1
rows = []
for k in range(15, -1, -1):
    m = (me_off == k) & ~np.isnan(r1)
    if m.sum() < 5:
        continue
    v = r1[m]
    s = summarize(v, f"ME-{k:02d} -> ME-{k-1:02d}")
    s["excess_bp"] = round(100 * 100 * (v.mean() - uncond), 2)
    rows.append(s)
for t in range(1, 5):
    m = (tdom == t) & ~np.isnan(r1)
    v = r1[m]
    s = summarize(v, f"new month tdom {t} session")
    s["excess_bp"] = round(100 * 100 * (v.mean() - uncond), 2)
    rows.append(s)
show(rows, f"SPY unconditional 1-session mean = {100*uncond:+.4f}%")

print("\n  turn-of-month overlap, MEASURED not asserted:")
dl, rl = cell("SPY", 1)
d5, r5 = cell("SPY", 5)
# ME-5 -> ME-1 leg (the part of our hold OUTSIDE the classic ToM window)
cvv, idxx, meo, lastp, poss = build("SPY")
ent5 = poss[meo == 5]
ent5 = ent5[lastp[ent5] < len(idxx)]
r_5to1 = cvv[lastp[ent5] - 1] / cvv[ent5] - 1.0
r_1to0 = cvv[lastp[ent5]] / cvv[lastp[ent5] - 1] - 1.0
show([summarize(r5, "ME-5 -> ME-0 (the whole candidate)"),
      summarize(r_5to1, "  ME-5 -> ME-1 leg (OUTSIDE the classic ToM window)"),
      summarize(r_1to0, "  ME-1 -> ME-0 leg (INSIDE the classic ToM window)")],
     "does the candidate survive removing the arbitraged-away window?")

# ------------------------------------------ (2/3) era + midterm + months
print("\n(2/3) SPY ME-5 splits")
ex5 = month_matched_excess("SPY", 5, d5, r5)
_, b5 = alldays_k("SPY", 5)
show([summarize(r5, f"ME-5 all (N={len(r5)})"),
      summarize(b5, "CTRL-b all days 5-session"),
      summarize(ex5, "ME-5 month-matched EXCESS"),
      summarize(r5[d5.year < 2013], "ME-5 pre-2013 (registry cut)"),
      summarize(r5[d5.year >= 2013], "ME-5 2013+"),
      summarize(r5[d5.year >= 2019], "ME-5 2019+"),
      summarize(r5[(d5.year % 4) == 2], "ME-5 MIDTERM years"),
      summarize(r5[(d5.year % 4) != 2], "ME-5 non-midterm"),
      ], "SPY ME-5 era / cycle")
print("  ME-5 excess, era: %s" % [
    (lbl, round(100 * ex5[m].mean(), 3), len(ex5[m]))
    for lbl, m in [("pre2013", d5.year < 2013), ("2013+", d5.year >= 2013)]])

print("\n  SPY ME-5 by calendar month:")
show([summarize(r5[d5.month == m], f"month {m:02d}") for m in range(1, 13)])

# --------------------------------------------------------- (4) August cell
print("\n(4) the AUGUST subcell, which is what is live")
aug = d5.month == 8
show([summarize(r5[aug], f"SPY ME-5 AUGUST (N={int(aug.sum())})"),
      summarize(r5[aug & (d5.year < 2013)], "  August pre-2013"),
      summarize(r5[aug & (d5.year >= 2013)], "  August 2013+"),
      summarize(r5[aug & ((d5.year % 4) == 2)], "  August MIDTERM years"),
      summarize(r5[aug & ((d5.year % 4) != 2)], "  August non-midterm"),
      ], "August")
w = int((r5[aug] > 0).sum())
base_up = float((b5 > 0).mean())
augbd, augbr = alldays_k("SPY", 5)
aug_up = float((augbr[augbd.month == 8] > 0).mean())
print("  August ME-5 record %d-%d; sign p vs a coin %.4f; vs SPY's OWN August "
      "5-session up-rate %.3f -> p %.4f; vs all-days up-rate %.3f -> p %.4f"
      % (w, int(aug.sum()) - w, sign_test(w, int(aug.sum())), aug_up,
         sign_test(w, int(aug.sum()), aug_up), base_up,
         sign_test(w, int(aug.sum()), base_up)))
print("  August ME-5 years:",
      ", ".join("%d:%+.2f%%" % (y, 100 * v)
                for y, v in zip(d5[aug].year, r5[aug])))

# ------------------------------------------------------- (5) vehicles
print("\n(5) VEHICLE alternatives at ME-5")
rows = []
for tkr in TK:
    d, r = cell(tkr, 5)
    _, br = alldays_k(tkr, 5)
    ex = month_matched_excess(tkr, 5, d, r)
    s = summarize(r, f"{tkr} ME-5 (span {d[0].year}-{d[-1].year})")
    s["excess_pct"] = round(s["mean_pct"] - 100 * br.mean(), 3)
    s["mo_matched_pct"] = round(100 * ex.mean(), 3)
    s["x_cost"] = round(100 * 100 * r.mean() / COST[tkr], 1)
    rows.append(s)
    a = d.month == 8
    s2 = summarize(r[a], f"  {tkr} ME-5 AUGUST only")
    rows.append(s2)
show(rows, "vehicle comparison")
