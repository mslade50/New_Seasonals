"""C1 ROUND 2 -- decluster, concentration, definition neighbours, era/regime,
and GATE ATTRIBUTION run gate-OFF first.

Specific debts this settles:
  (A) the proper month-of-year control (a1's demean was degenerate: 24 anchors
      per month means subtracting the cell's own month mean returns 0 by
      construction). The right control is the instrument's UNCONDITIONAL
      same-month 5-session drift.
  (B) reproduce the parked +0.126pp / t+2.18 distance gradient. It does NOT
      reproduce on the UNGATED ME-9 cell (a1 got -0.008, t -0.51), so find
      which object it was measured on -- the oversold-GATED cell is the
      suspect, and if so the parked blocker does not bind the ungated form.
  (C) MECHANISM test that is not a grid re-draw: month-end index-extension
      buying predicts the excess sits in the LAST sessions and appears in
      EVERY duration instrument. Test TLT / IEF / LQD / AGG against SPY as
      the placebo. A mechanism confirmed cross-sectionally is the only real
      answer to the parked entry's own debt (the 2,415-cell grid, familywise
      p 0.90 for grid-max-t).
  (D) gate attribution: parent first, then today's actual state as the gate.
  (E) Jackson Hole contamination -- JH lands at ME-1 this year and the August
      cell embeds JH in most years. How much of the August ME-5 number is JH?
  (F) concentration, drop-best-year, drop-February, and the recent era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
COST_BPS = 3.0
TKS = ["TLT", "IEF", "LQD", "AGG", "SPY"]
raw = load_prices(TKS)


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
    return c.values, idx, np.where(last_pos >= 0, last_pos - pos, np.nan).astype(float), last_pos, pos


def cell(tkr, k):
    cv, idx, me, lp, pos = build(tkr)
    e = pos[me == k]
    e = e[lp[e] < len(idx)]
    return idx[e], cv[lp[e]] / cv[e] - 1.0


def alldays_k(tkr, k):
    cv, idx, *_ = build(tkr)
    return idx[:-k], cv[k:] / cv[:-k] - 1.0


def mo_matched(tkr, k, d, r):
    bd, br = alldays_k(tkr, k)
    base = pd.Series(br, index=bd).groupby(bd.month).mean()
    return r - base.reindex(d.month).values


d5, r5 = cell("TLT", 5)
d9, r9 = cell("TLT", 9)

# ------------------------------------------------------------------- (A)
print("(A) MONTH-OF-YEAR CONTROL done properly (unconditional same-month drift)")
ex5 = mo_matched("TLT", 5, d5, r5)
ex9 = mo_matched("TLT", 9, d9, r9)
show([summarize(r5, "TLT ME-5 raw"),
      summarize(ex5, "TLT ME-5 month-matched EXCESS"),
      summarize(r9, "TLT ME-9 raw"),
      summarize(ex9, "TLT ME-9 month-matched EXCESS (parked: +0.391, t 2.85)")])

# ------------------------------------------------------------------- (B)
print("\n(B) where the parked +0.126pp / t+2.18 GRADIENT actually lives")
cv, idx, me, lp, pos = build("TLT")
s = pd.Series(cv, index=idx)
dist = (s / s.rolling(252).min() - 1.0) * 100.0
t21 = s.pct_change(21) * 100.0


def grad(d, r, label):
    X = dist.reindex(d).values
    m = ~np.isnan(X) & ~np.isnan(r)
    X, Y = X[m], r[m] * 100
    if len(X) < 5:
        print("   %-42s N=%d too small" % (label, len(X)))
        return
    b, a = np.polyfit(X, Y, 1)
    res = Y - (a + b * X)
    se = np.sqrt((res ** 2).sum() / (len(X) - 2) / ((X - X.mean()) ** 2).sum())
    lo = Y[X <= 1.0]
    hi = Y[X > 3.0]
    print("   %-42s N=%3d slope %+.4f (t %+.2f)  fit@0.86%% %+.3f%%  "
          "<=1%%: N=%d %+.3f%% hit %.0f%%  >3%%: N=%d %+.3f%% hit %.0f%% t %+.2f"
          % (label, len(X), b, b / se, a + b * 0.86,
             len(lo), lo.mean() if len(lo) else np.nan,
             100 * (lo > 0).mean() if len(lo) else np.nan,
             len(hi), hi.mean(), 100 * (hi > 0).mean(),
             hi.mean() / (hi.std(ddof=1) / np.sqrt(len(hi)))))


g9 = (t21.reindex(d9).values <= -2.5)
g5 = (t21.reindex(d5).values <= -2.5)
grad(d9, r9, "ME-9 UNGATED (the parked parent)")
grad(d9[g9], r9[g9], "ME-9 + oversold gate TLT21d<=-2.5%")
grad(d5, r5, "ME-5 UNGATED (today's form)")
grad(d5[g5], r5[g5], "ME-5 + oversold gate")
print("   live state: TLT 21d = %+.2f%% (gate needs <=-2.5%%), dist from 52w low = "
      "%+.2f%%" % (t21.iloc[-1], dist.iloc[-1]))

# ------------------------------------------------------------------- (C)
print("\n(C) MECHANISM test: is the month-end excess a DURATION-COMPLEX fact "
      "or a TLT grid artifact?")
rows = []
for tkr in TKS:
    cvv, ix, mee, lpp, poss = build(tkr)
    r1 = np.full(len(ix), np.nan)
    r1[:-1] = cvv[1:] / cvv[:-1] - 1.0
    un = np.nanmean(r1)
    rec = {"ticker": tkr, "span": "%d-%d" % (ix[0].year, ix[-1].year),
           "uncond_bp": round(100 * 100 * un, 2)}
    for k in (1, 2, 3, 4, 5):
        m = (mee == k) & ~np.isnan(r1)
        v = r1[m]
        rec[f"ME-{k}_bp"] = round(100 * 100 * (v.mean() - un), 2)
        rec[f"ME-{k}_t"] = round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))
print("   READ: index-extension flow predicts a POSITIVE last-1/last-2 session "
      "excess in EVERY duration instrument and none in SPY.")

# ---- and the tradeable form on each duration vehicle
rows = []
for tkr in ["TLT", "IEF", "LQD", "AGG"]:
    d, r = cell(tkr, 5)
    e = mo_matched(tkr, 5, d, r)
    q = summarize(r, f"{tkr} ME-5 raw")
    q["mo_matched_pct"] = round(100 * e.mean(), 3)
    q["mo_t"] = round(e.mean() / (e.std(ddof=1) / np.sqrt(len(e))), 2)
    rows.append(q)
    d2, r2 = cell(tkr, 2)
    e2 = mo_matched(tkr, 2, d2, r2)
    q2 = summarize(r2, f"  {tkr} ME-2 raw (the mechanism's own window)")
    q2["mo_matched_pct"] = round(100 * e2.mean(), 3)
    q2["mo_t"] = round(e2.mean() / (e2.std(ddof=1) / np.sqrt(len(e2))), 2)
    rows.append(q2)
show(rows, "tradeable cell across the duration complex")

# ------------------------------------------------------------------- (D)
print("\n(D) GATE ATTRIBUTION -- parent FIRST, then today's actual state as gate")
X5 = dist.reindex(d5).values
show([summarize(r5, "PARENT: ME-5 ungated (gate OFF)"),
      summarize(r5[X5 <= 1.0], "GATE ON: TLT within 1.0% of 52w low"),
      summarize(r5[X5 <= 2.0], "GATE ON: within 2.0%"),
      summarize(r5[X5 <= 3.0], "GATE ON: within 3.0%"),
      summarize(r5[X5 > 3.0], "COMPLEMENT: >3% off low")],
     "the distance gate at ME-5")
sel = ~np.isnan(X5) & (X5 <= 1.0)
print("  the <=1%% observations: %s"
      % ", ".join("%s %+.2f%% (dist %.2f%%)" % (d.date(), 100 * v, x)
                  for d, v, x in zip(d5[sel], r5[sel], X5[sel])))
w = int((r5[sel] > 0).sum())
print("  record %d-%d; parent's own conditional up-rate %.3f -> sign p %.4f"
      % (w, int(sel.sum()) - w, (r5 > 0).mean(),
         sign_test(w, int(sel.sum()), float((r5 > 0).mean()))))

# ------------------------------------------------------------------- (E)
print("\n(E) JACKSON HOLE contamination of the AUGUST cell")
jh = load_events(["jackson_hole"])["date"]
cvv, ix, mee, lpp, poss = build("TLT")
posmap = pd.Series(range(len(ix)), index=ix)
rec = []
for dte in jh:
    if dte not in posmap.index:
        near = ix[ix <= dte]
        if len(near) == 0:
            continue
        dte2 = near[-1]
    else:
        dte2 = dte
    off = mee[posmap[dte2]]
    rec.append((dte.year, str(dte.date()), off))
print("  JH me_offset by year (0 = the month-end close itself):")
print("   ", ", ".join("%d:ME-%s" % (y, "%.0f" % o if o == o else "?") for y, s_, o in rec))
offs = np.array([o for _, _, o in rec if o == o])
print("  JH sits inside an ME-5 hold (offset 0..4) in %d of %d years"
      % (int(((offs >= 0) & (offs <= 4)).sum()), len(offs)))
aug = d5.month == 8
jhy = set(y for y, _, o in rec if o == o and 0 <= o <= 4)
inw = np.array([d.year in jhy for d in d5[aug]])
show([summarize(r5[aug], "August ME-5 all"),
      summarize(r5[aug][inw], "  August ME-5, JH INSIDE the hold"),
      summarize(r5[aug][~inw], "  August ME-5, JH outside")],
     "August ME-5 split on JH-in-window")

# ------------------------------------------------------------------- (F)
print("\n(F) CONCENTRATION / era / robustness")
print("  monthly anchors are ~21 td apart with h=5 -> already declustered, no overlap")
print("  " + cluster_note(d5, r5, k=2))
byyr = pd.Series(r5, index=d5).groupby(d5.year).mean()
print("  drop the best YEAR (%d, %+.3f%%): mean %+.4f%% t %+.2f"
      % (byyr.idxmax(), 100 * byyr.max(),
         100 * r5[d5.year != byyr.idxmax()].mean(),
         r5[d5.year != byyr.idxmax()].mean() /
         (r5[d5.year != byyr.idxmax()].std(ddof=1) / np.sqrt((d5.year != byyr.idxmax()).sum()))))
nf = d5.month != 2
print("  drop FEBRUARY (the +1.403%% month): mean %+.4f%% t %+.2f N=%d"
      % (100 * r5[nf].mean(),
         r5[nf].mean() / (r5[nf].std(ddof=1) / np.sqrt(nf.sum())), int(nf.sum())))
exnf = mo_matched("TLT", 5, d5[nf], r5[nf])
print("  drop FEB, month-matched excess: %+.4f%% t %+.2f"
      % (100 * exnf.mean(), exnf.mean() / (exnf.std(ddof=1) / np.sqrt(len(exnf)))))
rows = []
for lo, hi in [(2002, 2008), (2009, 2014), (2015, 2020), (2021, 2026), (2022, 2026), (2024, 2026)]:
    m = (d5.year >= lo) & (d5.year <= hi)
    rows.append(summarize(r5[m], f"{lo}-{hi}"))
show(rows, "ME-5 rolling eras")
rows = []
for k in (3, 4, 5, 6, 7):
    d, r = cell("TLT", k)
    e = mo_matched("TLT", k, d, r)
    q = summarize(r, f"ME-{k}")
    q["mo_matched_pct"] = round(100 * e.mean(), 3)
    q["mo_t"] = round(e.mean() / (e.std(ddof=1) / np.sqrt(len(e))), 2)
    q["sharpe_per_hold"] = round(q["mean_pct"] / q["sd_pct"], 3)
    rows.append(q)
show(rows, "definition neighbours: nudge the entry offset")
