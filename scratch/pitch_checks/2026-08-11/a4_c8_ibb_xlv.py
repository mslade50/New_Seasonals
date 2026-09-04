"""C8 round 1: long IBB on XLV 63d leadership at a 52w high.

Same three attacks the surface map's own text invites, plus the one it does
not: the recon that produced the C8 numbers builds its trigger as
``rank(r(px["XLV"], 63), 63)``, i.e. pct_rank applied to an already-
differenced series, which is the 63-day percent CHANGE OF the 63-day return.
Same construction bug as the metals trigger. Check whether the stated
trigger ("XLV 63d return rank >= 98") reproduces the quoted cell at all.

Then: regress IBB on XLV and price the residual; year histogram and era split
for the 2013-2015 biotech bubble; and the horizon incoherence (h=1 48.6% and
h=3 55.0% against h=2 58.7% and h=10 58.7%).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,  # noqa: E402
                       pct_rank, bootstrap_p_le0, local_control, cluster_note,
                       era_split, show, battery)

warnings.filterwarnings("ignore")

px = close_panel(["XLV", "IBB", "XBI", "SPY", "QQQ"])
idx = px.index
correct = pct_rank(px["XLV"], 63)
buggy = pct_rank(px["XLV"].pct_change(63), 63)

print("=" * 96)
print("C8-1  THE TRIGGER AGAIN: recon statistic vs the stated one")
print("=" * 96)
ibb5 = fwd_lag(px["IBB"], 5, lag=1)
valid = ibb5.notna()
mc = (correct >= 98).fillna(False) & valid
mb = (buggy >= 98).fillna(False) & valid
dc, db = idx[mc.values], idx[mb.values]
print(f"  stated  XLV 63d return rank >= 98 : {len(dc)} days, "
      f"{len(declusters(dc, 5, idx))} episodes")
print(f"  recon   pct_change(63) of the 63d return, ranked >= 98 : {len(db)} days, "
      f"{len(declusters(db, 5, idx))} episodes   <-- the recon's N=109")
print(f"  overlap: {len(dc.intersection(db))} days")
last = px["XLV"].dropna().index[-1]
print(f"\n  TODAY ({last.date()}): real XLV 63d rank = {correct.loc[last]:.1f} "
      f"(63d ret {100*px['XLV'].pct_change(63).loc[last]:+.2f}%);  "
      f"recon statistic = {buggy.loc[last]:.1f}")

rows = []
for lbl, m in (("recon trigger (buggy)", mb), ("stated trigger (correct)", mc)):
    e = declusters(idx[m.values], 5, idx)
    for nm in ("IBB", "XLV", "XBI"):
        for h in (1, 2, 3, 5, 10):
            leg = fwd_lag(px[nm], h, lag=1)
            ee = declusters(idx[(m & leg.notna()).values], 5, idx)
            s = summarize(leg.loc[ee].values, f"{nm} h={h} | {lbl}")
            s["excess_pct"] = round(s["mean_pct"] - 100 * leg[leg.notna()].mean(), 3)
            s["signp"] = round(sign_test(int((leg.loc[ee].values > 0).sum()), len(ee)), 4)
            rows.append(s)
show([r for r in rows if r["label"].startswith("IBB")], "IBB by horizon, both definitions")
show([r for r in rows if r["label"].startswith("XLV")], "XLV (the leader) by horizon")
show([r for r in rows if r["label"].startswith("XBI")], "XBI by horizon")

print()
print("=" * 96)
print("C8-2  IS IBB ANYTHING BUT HIGH-BETA XLV?  regress the leg out.")
print("=" * 96)
for h in (2, 5, 10):
    e = declusters(idx[(mc & fwd_lag(px["IBB"], h, lag=1).notna()).values], 5, idx)
    y = fwd_lag(px["IBB"], h, lag=1).loc[e].values
    x1 = fwd_lag(px["XLV"], h, lag=1).loc[e].values
    x2 = fwd_lag(px["SPY"], h, lag=1).loc[e].values
    for cols, X in (("XLV", np.column_stack([np.ones(len(e)), x1])),
                    ("XLV+SPY", np.column_stack([np.ones(len(e)), x1, x2]))):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dof = len(y) - X.shape[1]
        se = np.sqrt(np.diag((res @ res / dof) * np.linalg.inv(X.T @ X)))
        r2 = 1 - res.var() / y.var()
        print(f"  h={h:<3} IBB ~ {cols:<8} N={len(e):<4} alpha {100*coef[0]:+7.3f}% "
              f"(t {coef[0]/se[0]:+5.2f})  beta_XLV {coef[1]:+.3f}  R2 {r2:.3f}")

print()
print("=" * 96)
print("C8-3  YEAR HISTOGRAM AND ERA SPLIT -- is this the 2013-2015 biotech bubble?")
print("=" * 96)
for h in (2, 10):
    leg = fwd_lag(px["IBB"], h, lag=1)
    e = declusters(idx[(mc & leg.notna()).values], 5, idx)
    v = leg.loc[e]
    by = v.groupby(v.index.year).agg(["count", "mean"])
    print(f"\n  IBB h={h} by year:")
    print("   " + "  ".join(f"{y}:{100*r['mean']:+.2f}({int(r['count'])})"
                            for y, r in by.iterrows()))
    show(era_split(e, v.values), f"  era split h={h}")
    for lo, hi in ((2013, 2016),):
        sel = (v.index.year >= lo) & (v.index.year < hi)
        a, b = v.values[sel], v.values[~sel]
        print(f"  {lo}-{hi-1} bubble years: N={len(a)} {100*a.mean():+.3f}%  |  "
              f"everything else: N={len(b)} {100*b.mean():+.3f}%  "
              f"(own drift {100*leg[leg.notna()].mean():+.3f}%)")
        print(f"    share of total return in {lo}-{hi-1}: "
              f"{100*a.sum()/v.values.sum():.0f}% of episodes' total from "
              f"{100*len(a)/len(v):.0f}% of the episodes")
    print(f"  concentration: {cluster_note(e, v.values)}")

print()
print("=" * 96)
print("C8-4  THRESHOLD SWEEP on the correct trigger")
print("=" * 96)
rows = []
for thr in (90, 95, 98, 99, 100):
    m = (correct >= thr).fillna(False)
    for h in (2, 10):
        leg = fwd_lag(px["IBB"], h, lag=1)
        e = declusters(idx[(m & leg.notna()).values], 5, idx)
        if len(e) < 4:
            rows.append({"label": f"h={h} rank>={thr}", "n": len(e)})
            continue
        s = summarize(leg.loc[e].values, f"h={h} rank>={thr}")
        s["excess_pct"] = round(s["mean_pct"] - 100 * leg[leg.notna()].mean(), 3)
        s["signp"] = round(sign_test(int((leg.loc[e].values > 0).sum()), len(e)), 4)
        rows.append(s)
show(rows, "IBB threshold sweep. TODAY XLV 63d rank = 100")

print()
print("=" * 96)
print("C8-5  FULL BATTERY at the two horizons the recon liked")
print("=" * 96)
variants = {f"rank>={t}": (correct >= t).fillna(False) for t in (90, 95, 98, 99)}
for h in (2, 10):
    battery(px, (correct >= 98).fillna(False), [("IBB", 1.0)], h,
            f"C8 long IBB on XLV 63d rank>=98", cost_bps=8.0,
            variants=variants, min_gap=5)
