"""C1 ROUND 3 (dev) -- only run because rounds 1 and 2 failed to kill it.

  (1) HORIZON CHOICE: hold from the ME-5 close for h = 1..10 and the exit
      placebo ladder (slide the exit past the true month-end close). The
      horizon is CHOSEN from this, not assumed from the calendar.
  (2) ENTRY FORM: MOC at the ME-5 close vs a close-anchored LIMIT at k ATR
      (Wilder-14), compared as WHOLE VARIANTS -- unfilled anchors book 0,
      never a fills-only subset (the repo's no-marginal-fill-decomposition
      rule).
  (3) HOLDOUT: fit 2002-2013, test 2014-2026 -- the parked entry's own debt
      ("re-derive it FORWARD rather than out of the 2,415-cell grid").
  (4) DECAY: is the month-end index-extension flow being arbitraged? ME-1 and
      ME-2 session excess by era across the whole duration complex.
  (5) WHAT KILLS IT: episode_paths on the losing episodes, intra-hold worst
      excursion, and the Jackson-Hole-at-ME-1 configuration that is live this
      year (1 of 24 historical Augusts).
  (6) the August x midterm subcell on TLT, scored against TLT's own August
      up-rate rather than a coin.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
COST_BPS = 3.0
raw = load_prices(["TLT", "IEF", "LQD", "AGG"])
d = raw["TLT"]
cl = d["Close"].dropna()
idx = cl.index
O, H_, L, C = (d["Open"].reindex(idx).values, d["High"].reindex(idx).values,
               d["Low"].reindex(idx).values, cl.values)
atr = wilder_atr(H_, L, C, 14)

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

E = pos[me == 5]                       # the ME-5 session = the ENTRY session
E = E[lp[E] < len(idx)]
ANCH = E - 1                           # the signal close (lag=1 convention)
r5 = C[lp[E]] / C[E] - 1.0
dts = idx[E]
print("ME-5 anchors: N=%d  %s .. %s" % (len(E), dts[0].date(), dts[-1].date()))
print("live: signal close 2026-08-21, entry MOC 2026-08-24 (ME-5), exit MOC "
      "2026-08-31 (ME-0), h=5 sessions")

# --------------------------------------------------------------------- (1)
print("\n(1) HORIZON CHOICE from the ME-5 entry close, and the EXIT placebo ladder")
rows = []
for h in range(1, 11):
    ok = E + h < len(idx)
    v = C[E[ok] + h] / C[E[ok]] - 1.0
    base = C[h:] / C[:-h] - 1.0
    s = summarize(v, f"h={h}%s" % ("  <-- lands ON the month-end close" if h == 5 else ""))
    s["excess_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    s["x_cost"] = round(100 * s["mean_pct"] / COST_BPS, 1)
    s["mean_over_sd"] = round(s["mean_pct"] / s["sd_pct"], 3)
    rows.append(s)
show(rows, "hold length from the ME-5 close")

rows = []
for j in range(-3, 11):
    tgt = lp[E] + j
    ok = (tgt >= 0) & (tgt < len(idx)) & (tgt > E)
    v = C[tgt[ok]] / C[E[ok]] - 1.0
    n = int(ok.sum())
    base = pd.Series([np.nan])
    lag = (tgt[ok] - E[ok])
    ctrl = np.array([np.nanmean(C[k:] / C[:-k] - 1.0) for k in np.unique(lag)])
    s = summarize(v, "exit at ME%+d%s" % (-j, "   <-- TRUE month-end close" if j == 0 else ""))
    s["excess_pct"] = round(s["mean_pct"] - 100 * float(np.mean(
        [np.nanmean(C[k:] / C[:-k] - 1.0) for k in lag])), 3)
    rows.append(s)
show(rows, "EXIT placebo ladder (entry fixed at the ME-5 close)")

# --------------------------------------------------------------------- (2)
print("\n(2) ENTRY FORM -- MOC vs a close-anchored LIMIT, WHOLE variants")
rows = []
moc = r5
rows.append({"variant": "MOC at the ME-5 close", "fills": len(moc),
             "fill_rate": 1.0, "mean_all_pct": round(100 * moc.mean(), 4),
             "t_all": round(moc.mean() / (moc.std(ddof=1) / np.sqrt(len(moc))), 2),
             "mean_filled_pct": round(100 * moc.mean(), 4),
             "x_cost": round(100 * 100 * moc.mean() / COST_BPS, 1)})
for k in (0.25, 0.5, 0.75):
    # limit = ANCHOR close (signal day) - k*ATR(anchor); live on the entry session
    lim = C[ANCH] - k * atr[ANCH]
    filled = L[E] <= lim
    v = np.where(filled, C[lp[E]] / lim - 1.0, 0.0)
    fv = v[filled]
    rows.append({"variant": f"LIMIT(close, -{k} ATR)", "fills": int(filled.sum()),
                 "fill_rate": round(filled.mean(), 3),
                 "mean_all_pct": round(100 * v.mean(), 4),
                 "t_all": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                 "mean_filled_pct": round(100 * fv.mean(), 4),
                 "x_cost": round(100 * 100 * v.mean() / COST_BPS, 1)})
    # and the UPSIDE limit for contrast (buy strength)
    lim2 = C[ANCH] + k * atr[ANCH]
    f2 = H_[E] >= lim2
    v2 = np.where(f2, C[lp[E]] / lim2 - 1.0, 0.0)
    rows.append({"variant": f"LIMIT(close, +{k} ATR) [stop-entry]", "fills": int(f2.sum()),
                 "fill_rate": round(f2.mean(), 3),
                 "mean_all_pct": round(100 * v2.mean(), 4),
                 "t_all": round(v2.mean() / (v2.std(ddof=1) / np.sqrt(len(v2))), 2),
                 "mean_filled_pct": round(100 * v2[f2].mean(), 4),
                 "x_cost": round(100 * 100 * v2.mean() / COST_BPS, 1)})
print(pd.DataFrame(rows).to_string(index=False))
print("  live ATR(14) on TLT = %.3f (%.2f%% of the %.2f close); 0.25 ATR = %.3f"
      % (atr[-1], 100 * atr[-1] / C[-1], C[-1], 0.25 * atr[-1]))

# --------------------------------------------------------------------- (3)
print("\n(3) HOLDOUT: fit 2002-2013, test 2014-2026")
fit = dts.year <= 2013
tst = dts.year >= 2014


def blk(v, lbl):
    b = C[5:] / C[:-5] - 1.0
    s = summarize(v, lbl)
    s["excess_pct"] = round(s["mean_pct"] - 100 * b.mean(), 3)
    s["x_cost"] = round(100 * s["mean_pct"] / COST_BPS, 1)
    return s


show([blk(r5[fit], "IN-SAMPLE 2002-2013"), blk(r5[tst], "HOLDOUT 2014-2026"),
      blk(r5[dts.year >= 2019], "  HOLDOUT 2019+"),
      blk(r5[dts.year >= 2023], "  HOLDOUT 2023+")], "holdout")
w = int((r5[tst] > 0).sum())
print("  holdout record %d-%d, sign p vs a coin %.4f; bootstrap P(mean<=0) %.3f"
      % (w, int(tst.sum()) - w, sign_test(w, int(tst.sum())), bootstrap_p_le0(r5[tst])))

# --------------------------------------------------------------------- (4)
print("\n(4) DECAY of the month-end index-extension flow, by era")
rows = []
for tkr in ("TLT", "IEF", "LQD", "AGG"):
    cc = raw[tkr]["Close"].dropna()
    ii = cc.index
    y2 = pd.Series(ii.year * 100 + ii.month, index=ii)
    il = y2.ne(y2.shift(-1)).values
    il[-1] = False
    p2 = np.arange(len(ii))
    l2 = np.full(len(ii), -1)
    cu = -1
    for i in range(len(ii) - 1, -1, -1):
        if il[i]:
            cu = i
        l2[i] = cu
    m2 = np.where(l2 >= 0, l2 - p2, np.nan).astype(float)
    v = cc.values
    rr = np.full(len(ii), np.nan)
    rr[:-1] = v[1:] / v[:-1] - 1.0
    for lo, hi in ((2002, 2012), (2013, 2019), (2020, 2026)):
        sel = (ii.year >= lo) & (ii.year <= hi)
        u = np.nanmean(rr[sel])
        rec = {"tkr": tkr, "era": f"{lo}-{hi}"}
        for k in (1, 2):
            mm = sel & (m2 == k) & ~np.isnan(rr)
            x = rr[mm]
            rec[f"ME-{k}_bp"] = round(100 * 100 * (x.mean() - u), 2)
            rec[f"n{k}"] = len(x)
            rec[f"t{k}"] = round((x.mean() - u) / (x.std(ddof=1) / np.sqrt(len(x))), 2)
        rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------------- (5)
print("\n(5) WHAT KILLS IT -- losing episodes and the intra-hold path")
lose = np.argsort(r5)[:8]
print("  8 worst ME-5 holds:")
for i in lose:
    print("    %s  %+.2f%%" % (dts[i].date(), 100 * r5[i]))
# intra-hold worst mark-to-market from the entry close
mae = []
for e, l_ in zip(E, lp[E]):
    lows = L[e + 1:l_ + 1]
    mae.append(lows.min() / C[e] - 1.0 if len(lows) else np.nan)
mae = np.array(mae)
print("  intra-hold worst LOW vs the entry close: mean %+.2f%%, median %+.2f%%, "
      "5th pctile %+.2f%%, worst %+.2f%%"
      % (100 * np.nanmean(mae), 100 * np.nanmedian(mae),
         100 * np.nanpercentile(mae, 5), 100 * np.nanmin(mae)))
print("  P(intra-hold low <= -1.0 ATR below entry) = %.1f%%; <= -1.5 ATR = %.1f%%"
      % (100 * np.nanmean(mae * C[E] <= -1.0 * atr[E]),
         100 * np.nanmean(mae * C[E] <= -1.5 * atr[E])))

print("\n  JACKSON HOLE at ME-1 -- 2026's configuration")
jh = load_events(["jackson_hole"])["date"]
pm = pd.Series(range(len(idx)), index=idx)
cfg = []
for x in jh:
    xx = x if x in pm.index else (idx[idx <= x][-1] if len(idx[idx <= x]) else None)
    if xx is None:
        continue
    cfg.append((x.year, me[pm[xx]]))
me1 = [y for y, o in cfg if o == 1]
print("   JH landed at ME-1 in: %s (%d of %d Augusts). 2026: JH 2026-08-28 is ME-1."
      % (me1, len(me1), len(cfg)))
aug = dts.month == 8
for y in me1:
    sel = aug & (dts.year == y)
    if sel.any():
        print("   %d ME-5 hold returned %+.2f%%" % (y, 100 * r5[sel][0]))
# TLT's own JH-day move distribution
jhmv = []
for x in jh:
    if x in pm.index:
        p = pm[x]
        if p + 1 < len(idx):
            jhmv.append(C[p] / C[p - 1] - 1.0)
jhmv = np.array(jhmv)
print("   TLT's own JH-SPEECH-DAY move: mean %+.2f%%, sd %.2f%%, worst %+.2f%%, "
      "best %+.2f%% (N=%d)" % (100 * jhmv.mean(), 100 * jhmv.std(ddof=1),
                               100 * jhmv.min(), 100 * jhmv.max(), len(jhmv)))

# --------------------------------------------------------------------- (6)
print("\n(6) the AUGUST x MIDTERM subcell on TLT")
b5 = C[5:] / C[:-5] - 1.0
augup = float((pd.Series(b5, index=idx[:-5])[idx[:-5].month == 8] > 0).mean())
mid = (dts.year % 4) == 2
show([summarize(r5[aug], "TLT ME-5 August (all)"),
      summarize(r5[aug & mid], "TLT ME-5 August x MIDTERM (live cell)"),
      summarize(r5[aug & ~mid], "TLT ME-5 August, non-midterm"),
      summarize(r5[mid], "TLT ME-5 all midterm months"),
      summarize(r5[aug & (dts.year >= 2013)], "TLT ME-5 August 2013+")])
w = int((r5[aug] > 0).sum())
print("  August record %d-%d; TLT's OWN August 5-session up-rate %.3f -> sign p %.4f"
      % (w, int(aug.sum()) - w, augup, sign_test(w, int(aug.sum()), augup)))
sel = aug & mid
print("  August-midterm years: %s"
      % ", ".join("%d %+.2f%%" % (y, 100 * v) for y, v in zip(dts[sel].year, r5[sel])))
print("  August ME-5 by year: %s"
      % ", ".join("%d %+.2f" % (y, 100 * v) for y, v in zip(dts[aug].year, r5[aug])))
