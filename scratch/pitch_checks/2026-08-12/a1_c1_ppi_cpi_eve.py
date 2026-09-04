"""C1 round 1 -- Long TLT across the PPI print session (entry MOC on the eve,
exit MOC on the print close). h=1, lag=1 from an anchor 2 sessions before.

THE DECISIVE QUESTION, flagged by the watchlist entry itself: this week's PPI
lands the session AFTER a CPI print, so tonight's entry close is a CPI-REACTION
close. Split the 286 historical PPI eves by whether a CPI printed on the eve
itself, and separately by whether a CPI printed within the prior 1-2 sessions.
Reverse ordering (PPI on a CPI eve) is the placebo.

Controls are tdom-matched FIRST (registry: an all-days control on a rates event
cell is invalid -- TLT's own tdom profile carries the whole thing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICKERS = ["TLT", "IEF", "LQD", "HYG", "SPY", "^TNX"]
px = close_panel(TICKERS)
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)

ev = load_events()
ppi = ev[ev.event == "ppi"]["date"]
cpi = ev[ev.event == "cpi"]["date"]


def sess_pos(dates):
    """Index position of the session ON or AFTER each event date."""
    out = []
    for x in dates:
        p = int(idx.searchsorted(x, "left"))
        if 0 <= p < N:
            out.append(p)
    return sorted(set(out))


ppi_p = [p for p in sess_pos(ppi) if 5 <= p < N - 2]
cpi_p = set(sess_pos(cpi))

# the cell: entry close at p-1 (the eve), exit close at p (the print)
r = np.array([c[p] / c[p - 1] - 1.0 for p in ppi_p])
eve_pos = np.array([p - 1 for p in ppi_p])
eve_dt = pd.DatetimeIndex([idx[p - 1] for p in ppi_p])
prn_dt = pd.DatetimeIndex([idx[p] for p in ppi_p])
yr = prn_dt.year.values

# --- conditioners -----------------------------------------------------------
cpi_on_eve = np.array([(p - 1) in cpi_p for p in ppi_p])          # today's state
cpi_prior2 = np.array([bool({p - 1, p - 2} & cpi_p) for p in ppi_p])
cpi_prior1_only = cpi_prior2 & ~cpi_on_eve
cpi_on_print = np.array([p in cpi_p for p in ppi_p])               # same session

print("=" * 100)
print("0. THE CELL AND ITS LIVE CONDITIONER")
print("=" * 100)
print(f"PPI prints with a tradeable eve: N={len(ppi_p)}  "
      f"{prn_dt[0].date()} .. {prn_dt[-1].date()}")
print(f"  CPI printed ON the eve      : {int(cpi_on_eve.sum())}  <-- TODAY'S STATE")
print(f"  CPI printed 1-2 sess before : {int(cpi_prior2.sum())}")
print(f"  CPI printed ON the print day: {int(cpi_on_print.sum())}")

# --- tdom control -----------------------------------------------------------
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ym.groupby(ym.values).cumcount().values + 1
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0

# every session touched by ANY ppi eve->print window is excluded from controls
win = set()
for p in ppi_p:
    win.add(p)
in_win = np.array([i in win for i in range(N)])


def tdom_bucket_mean(j):
    m = (tdom == j) & ~in_win & ~np.isnan(d1)
    return d1[m].mean() if m.sum() >= 10 else np.nan


BUCK = {j: tdom_bucket_mean(j) for j in range(1, 25)}


def excess_tdom(mask):
    """per-entry excess of the PRINT-session return over the tdom bucket mean
    of the print session's own trading-day-of-month."""
    out = []
    for i, p in enumerate(ppi_p):
        if not mask[i]:
            continue
        b = BUCK.get(int(tdom[p]), np.nan)
        if np.isnan(b):
            continue
        out.append(r[i] - b)
    return np.array(out)


def rep(v, lbl, base_rate=None):
    if len(v) == 0:
        return {"cell": lbl, "N": 0}
    w = int((v > 0).sum())
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    p = sign_test(w, len(v), base_rate if base_rate else 0.5)
    return {"cell": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 4),
            "hit": round(100 * w / len(v), 1),
            "t": round(v.mean() / (sd / np.sqrt(len(v))), 2) if sd else np.nan,
            "signp": round(p, 4),
            "worst": round(100 * v.min(), 2), "best": round(100 * v.max(), 2)}


base_hit = float((d1[~np.isnan(d1)] > 0).mean())
print(f"\nTLT unconditional daily: {100*np.nanmean(d1):+.4f}%  "
      f"base hit rate {100*base_hit:.2f}%")

print("\n" + "=" * 100)
print("1. RAW PRINT-SESSION RETURN, SPLIT BY THE LIVE CONDITIONER")
print("=" * 100)
rows = [rep(r, "PARENT: all PPI prints", base_hit),
        rep(r[cpi_on_eve], "*** CPI ON THE EVE (today) ***", base_hit),
        rep(r[~cpi_on_eve], "no CPI on the eve", base_hit),
        rep(r[cpi_prior1_only], "CPI 2 sessions before only", base_hit),
        rep(r[cpi_prior2], "CPI 1-2 sessions before (either)", base_hit),
        rep(r[~cpi_prior2], "no CPI in the prior 2 sessions", base_hit),
        rep(r[cpi_on_print], "PLACEBO: CPI on the PRINT day", base_hit)]
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("2. TDOM-MATCHED EXCESS (the only valid control for a rates event cell)")
print("=" * 100)
rows = []
for lbl, m in [("PARENT: all PPI prints", np.ones(len(r), bool)),
               ("*** CPI ON THE EVE (today) ***", cpi_on_eve),
               ("no CPI on the eve", ~cpi_on_eve),
               ("CPI 1-2 sess before", cpi_prior2),
               ("no CPI in prior 2 sess", ~cpi_prior2)]:
    v = excess_tdom(m)
    rows.append(rep(v, lbl, base_hit))
print(pd.DataFrame(rows).to_string(index=False))
pe = excess_tdom(np.ones(len(r), bool))
ce = excess_tdom(cpi_on_eve)
ne = excess_tdom(~cpi_on_eve)
if len(ce) > 1 and len(ne) > 1:
    se = np.sqrt(ce.var(ddof=1) / len(ce) + ne.var(ddof=1) / len(ne))
    print(f"\n  CPI-on-eve vs no-CPI-on-eve, tdom excess diff = "
          f"{100*(ce.mean()-ne.mean()):+.4f}pp   welch t {(ce.mean()-ne.mean())/se:+.2f}")
    print(f"  bootstrap P(mean<=0) CPI-on-eve cell = {bootstrap_p_le0(ce):.3f}   "
          f"parent = {bootstrap_p_le0(pe):.3f}")

print("\n" + "=" * 100)
print("3. ERA + EPISODE YEAR HISTOGRAM OF THE LIVE CELL")
print("=" * 100)
sub_y = yr[cpi_on_eve]
sub_r = r[cpi_on_eve]
hist = pd.Series(sub_r).groupby(sub_y).agg(["count", "mean"])
hist["mean_pct"] = (100 * hist["mean"]).round(3)
print("CPI-on-eve occurrences by year:")
print(hist[["count", "mean_pct"]].to_string())
for lbl, m in [("pre-2013", sub_y < 2013), ("2013-2017", (sub_y >= 2013) & (sub_y < 2018)),
               ("2018+", sub_y >= 2018), ("2023+", sub_y >= 2023)]:
    if m.sum():
        print(f"  {lbl:10s} {rep(sub_r[m], '', base_hit)}")

print("\nPARENT year histogram (for contrast):")
ph = pd.Series(r).groupby(yr).agg(["count", "mean"])
ph["mean_pct"] = (100 * ph["mean"]).round(3)
print(ph[["count", "mean_pct"]].T.to_string())
pos_yrs = int((ph["mean"] > 0).sum())
print(f"  years positive: {pos_yrs}/{len(ph)}")

print("\n" + "=" * 100)
print("4. MIDTERM SPLIT (the conditioner that has killed 3 ideas in this repo)")
print("=" * 100)
mid = (yr % 4) == 2
for lbl, m in [("PARENT midterm", mid), ("PARENT non-midterm", ~mid),
               ("CPI-on-eve midterm", mid & cpi_on_eve),
               ("CPI-on-eve non-midterm", (~mid) & cpi_on_eve)]:
    print(f"  {lbl:26s} {rep(r[m], '', base_hit)}")
print("\n  tdom-excess versions:")
for lbl, m in [("PARENT midterm", mid), ("CPI-on-eve midterm", mid & cpi_on_eve)]:
    print(f"  {lbl:26s} {rep(excess_tdom(m), '', base_hit)}")

print("\n" + "=" * 100)
print("5. AUGUST-SPECIFIC and MONTH PROFILE (is the print cell a month shape?)")
print("=" * 100)
mo = prn_dt.month.values
rows = []
for m in range(1, 13):
    s = r[mo == m]
    if len(s):
        rows.append({"month": m, "N": len(s), "mean_pct": round(100 * s.mean(), 3),
                     "hit": round(100 * (s > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("6. WORST PRINT-DAY MOVE (tomorrow's tail risk, the print is IN the hold)")
print("=" * 100)
ordr = np.argsort(r)[:8]
for i in ordr:
    print(f"  {prn_dt[i].date()}  {100*r[i]:+.2f}%   "
          f"(CPI on eve: {cpi_on_eve[i]})")
print("  best:")
for i in np.argsort(-r)[:4]:
    print(f"  {prn_dt[i].date()}  {100*r[i]:+.2f}%   (CPI on eve: {cpi_on_eve[i]})")
print(f"\n  sd of the print-session move = {100*r.std(ddof=1):.3f}%  "
      f"vs unconditional daily sd {100*np.nanstd(d1, ddof=1):.3f}%")

print("\n" + "=" * 100)
print("7. COST")
print("=" * 100)
for lbl, v in [("parent tdom-excess", pe), ("CPI-on-eve tdom-excess", ce)]:
    bps = 100 * 100 * v.mean()
    print(f"  {lbl:26s} {bps:+.2f} bps vs 2.5 bps TLT round trip "
          f"= {bps/2.5:.2f}x cost")

print("\n" + "=" * 100)
print("8. COHERENCE: same split in IEF / LQD / HYG / SPY / ^TNX")
print("=" * 100)
rows = []
for t in ["TLT", "IEF", "LQD", "HYG", "SPY", "^TNX"]:
    s = px[t].dropna()
    ii = s.index
    cc = s.values
    vals, flags = [], []
    for p in ppi_p:
        d_prev, d_prn = idx[p - 1], idx[p]
        if d_prev in ii and d_prn in ii:
            a = ii.get_loc(d_prev)
            b = ii.get_loc(d_prn)
            vals.append(cc[b] / cc[a] - 1.0)
            flags.append((p - 1) in cpi_p)
    vals, flags = np.array(vals), np.array(flags)
    if len(vals) < 10:
        continue
    dd = np.diff(cc) / cc[:-1]
    rows.append({"tkr": t, "N": len(vals),
                 "parent_pct": round(100 * vals.mean(), 4),
                 "own_drift": round(100 * dd.mean(), 4),
                 "parent_hit": round(100 * (vals > 0).mean(), 1),
                 "cpi_eve_N": int(flags.sum()),
                 "cpi_eve_pct": round(100 * vals[flags].mean(), 4),
                 "cpi_eve_hit": round(100 * (vals[flags] > 0).mean(), 1),
                 "no_cpi_pct": round(100 * vals[~flags].mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("9. REVERSE-ORDER PLACEBO: the CPI print session when PPI printed on ITS eve")
print("=" * 100)
cpi_list = [p for p in sess_pos(cpi) if 5 <= p < N - 2]
ppi_set = set(sess_pos(ppi))
rc = np.array([c[p] / c[p - 1] - 1.0 for p in cpi_list])
ppi_on_eve = np.array([(p - 1) in ppi_set for p in cpi_list])
print(f"  {rep(rc, 'CPI print session, all', base_hit)}")
print(f"  {rep(rc[ppi_on_eve], 'CPI print, PPI on ITS eve', base_hit)}")
print(f"  {rep(rc[~ppi_on_eve], 'CPI print, no PPI on eve', base_hit)}")
print("\n  If ordering carried information, one side of this pair would look")
print("  like the PPI cell and the other would not.")
