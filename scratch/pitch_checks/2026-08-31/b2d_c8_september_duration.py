"""C8 -- short duration INTO September with the 10-year at a 52-week yield high.

The registry (2026-08-13) records TLT's 10td lag-1 forward return by month with
Sep -0.220%, second-worst of twelve. This script does NOT take that on trust.

Entry convention, stated exactly: the tradeable order given a signal on close D
is MOC at D+1. Today (2026-08-31) IS the last trading day of August and the
freshest bar is Friday 2026-08-28, so the historical analogue anchor is the
SECOND-to-last session of August (ME-1); lag=1 puts the entry MOC on the LAST
session of August (ME-0) and the exit h sessions into September. Anything else
would be measuring a trade that cannot be placed this morning.

Charges applied: 12-month rotation permutation (a month-of-year claim is a
12-way scan), era split incl. the bond-bull fossil objection (2026-08-17),
midterm split (required, 2026-08-26), gate attribution both ways, decluster,
concentration, local control, cost INCLUDING the short's negative carry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "TLT", "IEF", "SPY"])
idx = px.index

# ------------------------------------------------------------------ 0
print("=== 0. REPRODUCE the registry's month-of-year table from scratch ===")
print("   (TLT, ALL sessions, lag-1, h=10 -- the registry's stated construction)")
for veh in ("TLT", "IEF"):
    r = fwd_lag(px[veh], 10, 1)
    ok = r.notna()
    tab = r[ok].groupby(idx[ok].month).agg(["mean", "count"])
    tab["mean_pct"] = (100 * tab["mean"]).round(3)
    print(f"  {veh}: " + "  ".join(
        f"{pd.Timestamp(2020, m, 1).strftime('%b')} {tab.loc[m,'mean_pct']:+.3f}"
        for m in range(1, 13)))
    print(f"       rank of Sep (worst=1): "
          f"{int(tab['mean_pct'].rank().loc[9])} of 12")

# ------------------------------------------------------------------ 1
print("\n=== 1. THE ACTUAL TRADE: anchor = 2nd-to-last session of a month, "
      "lag-1 entry on the LAST session, hold h into the next month ===")


def month_end_anchors(index: pd.DatetimeIndex, month: int) -> pd.DatetimeIndex:
    """2nd-to-last trading session of `month` in each year."""
    s = pd.Series(index, index=index)
    key = index.to_period("M")
    out = []
    for p, g in s.groupby(key):
        if p.month != month or len(g) < 2:
            continue
        out.append(g.iloc[-2])
    return pd.DatetimeIndex(out)


HS = (3, 5, 10, 21)
print("  SHORT TLT (negative weight) -- the pitched side")
for h in HS:
    r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
    a = month_end_anchors(idx, 8)
    a = a[r.reindex(a).notna().values]
    v = r.loc[a].values
    upr = float((r.dropna() > 0).mean())
    print(f"   h={h:2d}: N={len(v)}  mean {100*np.nanmean(v):+.3f}%  "
          f"median {100*np.median(v):+.3f}%  hit {100*(v>0).mean():.1f}%  "
          f"t {np.nanmean(v)/(np.nanstd(v,ddof=1)/np.sqrt(len(v))):+.2f}  "
          f"sign p (vs short-TLT up-rate {100*upr:.1f}%) "
          f"{sign_test(int((v>0).sum()), len(v), p=upr):.4f}")

# ------------------------------------------------------------------ 2
print("\n=== 2. CHARGE THE 12-MONTH SCAN (rotation permutation) ===")
for veh, sgn in [("TLT", -1.0), ("IEF", -1.0)]:
    for h in (5, 10, 21):
        r = vehicle_ret(px, [(veh, sgn)], h, 1)
        means, ns = {}, {}
        for m in range(1, 13):
            a = month_end_anchors(idx, m)
            a = a[r.reindex(a).notna().values]
            means[m] = float(np.nanmean(r.loc[a].values))
            ns[m] = len(a)
        order = sorted(means, key=lambda m: -means[m])
        rank_sep = order.index(9) + 1
        obs = means[9]
        # rotation null: for each year, rotate WHICH month's anchor set the
        # returns come from -- i.e. compare Sep's mean against the max of the
        # twelve month means under a permutation of anchors across months.
        rng = np.random.default_rng(11)
        allA = np.concatenate([
            r.loc[month_end_anchors(idx, m)[
                r.reindex(month_end_anchors(idx, m)).notna().values]].values
            for m in range(1, 13)])
        sizes = [ns[m] for m in range(1, 13)]
        mx = []
        for _ in range(20000):
            p = rng.permutation(allA)
            i, best = 0, -np.inf
            for s in sizes:
                best = max(best, p[i:i + s].mean())
                i += s
            mx.append(best)
        mx = np.asarray(mx)
        print(f"  SHORT {veh} h={h:2d}: Sep {100*obs:+.3f}% ranks {rank_sep} of 12 "
              f"(best month {pd.Timestamp(2020,order[0],1):%b} "
              f"{100*means[order[0]]:+.3f}%);  P(max-of-12 >= Sep) = "
              f"{(mx >= obs).mean():.4f};  Bonferroni-style 12x on the raw "
              f"one-sided t: see below")
        print("     all months:", {pd.Timestamp(2020, m, 1).strftime('%b'):
                                   round(100*means[m], 3) for m in range(1, 13)})

# ------------------------------------------------------------------ 3
print("\n=== 3. GATE ATTRIBUTION: does the yield-at-52w-high state ADD to "
      "plain September? ===")
tnx = px["^TNX"]
tnx_dist = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
for h in (5, 10, 21):
    r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
    a = month_end_anchors(idx, 8)
    a = a[r.reindex(a).notna().values]
    dist = tnx_dist.reindex(a)
    for thr in (0.005, 0.01, 0.02, 0.05):
        hi = dist >= -thr
        vin, vout = r.loc[a[hi.values]].values, r.loc[a[~hi.values]].values
        print(f"  h={h:2d} gate TNX within {100*thr:4.1f}% of 252d high: "
              f"IN {100*np.nanmean(vin):+.3f}% (N={len(vin)}, "
              f"yrs {sorted(set(a[hi.values].year))}) | "
              f"OUT {100*np.nanmean(vout):+.3f}% (N={len(vout)}) | "
              f"gate {100*(np.nanmean(vin)-np.nanmean(vout)):+.3f}pp")
    # the other direction: yield-high WITHOUT September
    yh = (tnx_dist >= -0.01)
    rr = fwd_lag(px["TLT"], h, 1)
    ok = rr.notna()
    yh_days = idx[yh.values & ok.values]
    yh_sep = yh_days[yh_days.month == 9]
    yh_not = yh_days[yh_days.month != 9]
    print(f"       yield-high ALL months  {-100*float(rr.loc[yh_days].mean()):+.3f}% "
          f"(short, N={len(yh_days)}d) | yield-high in Sep only "
          f"{-100*float(rr.loc[yh_sep].mean()):+.3f}% (N={len(yh_sep)}d) | "
          f"yield-high ex-Sep {-100*float(rr.loc[yh_not].mean()):+.3f}% "
          f"(N={len(yh_not)}d)")

# ------------------------------------------------------------------ 4
print("\n=== 4. ERA + BOND-BULL FOSSIL + MIDTERM ===")
for h in (5, 10, 21):
    r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
    a = month_end_anchors(idx, 8)
    a = a[r.reindex(a).notna().values]
    v = r.loc[a].values
    yrs = a.year
    # rising- vs falling-yield regime: sign of TNX's own trailing 252d change
    tchg = rolling_on_valid(tnx, lambda x: x / x.shift(252) - 1.0).reindex(a)
    rise = (tchg > 0).values
    show([summarize(v[yrs < 2013], "2002-2012"),
          summarize(v[(yrs >= 2013) & (yrs < 2018)], "2013-2017"),
          summarize(v[yrs >= 2018], "2018+"),
          summarize(v[yrs >= 2021], "2021+"),
          summarize(v[rise], f"yields RISING yoy (N={int(rise.sum())})"),
          summarize(v[~rise], f"yields FALLING yoy (N={int((~rise).sum())})"),
          summarize(v[yrs % 4 == 2], "MIDTERM"),
          summarize(v[yrs % 4 != 2], "non-midterm")],
         f"SHORT TLT, Aug ME-1 anchor, h={h}")
    print("  per-year:", {int(y): round(100*x, 2) for y, x in zip(yrs, v)})

# ------------------------------------------------------------------ 5
print("\n=== 5. COST INCLUDING CARRY ON THE SHORT ===")
print("  TLT distribution yield ~4.5%/yr on ~252 sessions = ~1.79 bps/session")
for h in (5, 10, 21):
    r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
    a = month_end_anchors(idx, 8)
    a = a[r.reindex(a).notna().values]
    gross = 100 * float(np.nanmean(r.loc[a].values)) * 100
    carry = 1.79 * h
    net = gross - 3.0 - carry
    print(f"  h={h:2d}: gross {gross:+.1f} bps - 3.0 spread - {carry:.1f} carry "
          f"= NET {net:+.1f} bps -> {net/(3.0+carry):.2f}x the {3.0+carry:.1f} bp "
          f"cost bar (need >= 5x)")

# ------------------------------------------------------------------ 6
print("\n=== 6. LOCAL CONTROL + concentration (h=10) ===")
h = 10
r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
a = month_end_anchors(idx, 8)
a = a[r.reindex(a).notna().values]
loc = local_control(idx[r.notna().values], a, 126)
show([summarize(r.loc[a].values, f"Aug ME-1 anchors (N={len(a)})"),
      summarize(r.loc[loc].values, "local +/-126td ex-anchor"),
      summarize(r[r.notna()].values, "all days")], "SHORT TLT h=10")
print("  concentration:", cluster_note(a, r.loc[a].values, k=3))
