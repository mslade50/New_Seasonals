"""RED TEAM addendum for both inversions.

Fixes two things and adds the multiple-comparison accounting:
  A) horizon curves in r1/r2 used decluster gap = max(h,5), so the EPISODE SET
     changes at every h and the curve is not apples-to-apples. Redone here on a
     FIXED episode set (gap=10) so only the horizon varies.
  B) INV1: unconditional base rate of SPY-below-200d, to size the regime
     selection the trigger performs.
  C) INV2: 2021+2022 joint removal, the post-2022 (genuinely later) subsample,
     and era-cut sensitivity -- the 2018 cut was itself chosen post-hoc.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

# ============================== INVERSION 1 ==================================
px = load_prices(["SMH", "QQQ", "SPY"])
cal = px["SMH"]["Close"].dropna().index
for t in ("QQQ", "SPY"):
    cal = cal.intersection(px[t]["Close"].dropna().index)
smh, qqq, spy = (px[t]["Close"].reindex(cal) for t in ("SMH", "QQQ", "SPY"))
r63, r5 = pct_rank(smh, 63), pct_rank(smh, 5)
spy200 = spy.rolling(200).mean()
spy_above = spy > spy200


def fl(s, h, lag=1):
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def trade1(h):
    return fl(qqq, h) - fl(smh, h)


M1 = ((r63 <= 10) & (r5 >= 80)).fillna(False)
T1 = cal[M1.values & trade1(5).notna().reindex(cal).fillna(False).values]
EPI_FIX = declusters(T1, 10, cal)

print("=" * 80)
print("A/B) INVERSION 1 -- fixed episode set (gap=10, N=%d), horizon curve" % len(EPI_FIX))
print("=" * 80)
rows = []
for h in range(1, 22):
    v = trade1(h).reindex(EPI_FIX).to_numpy()
    v = v[~np.isnan(v)]
    r = summarize(v, "")
    rows.append({"h": h, "n_epi": r["n"], "mean_pct": r["mean_pct"], "t": r["t"],
                 "hit": r["hit"], "cum_note": ""})
show(rows, "INV1 short-SMH/long-QQQ, FIXED episode set")

valid200 = spy_above[spy200.notna()]
print(f"\nB) SPY below its 200d SMA -- base rates")
print(f"   unconditional: {100*(~valid200).mean():.1f}% of all sessions ({(~valid200).sum()}/{len(valid200)})")
ab = spy_above.reindex(T1).fillna(False)
print(f"   on INV1 trigger days: {100*(~ab).mean():.1f}% ({(~ab).sum()}/{len(ab)})")
print(f"   -> the trigger over-selects bear tape by "
      f"{100*(~ab).mean() - 100*(~valid200).mean():+.1f} pp")
# and what the trade earns split by regime, day level, for the record
for lbl, m in (("SPY BELOW 200d", ~spy_above), ("SPY ABOVE 200d", spy_above)):
    t = T1[m.reindex(T1).fillna(False).values]
    e = declusters(t, 10, cal)
    print(f"   {lbl}: n_day={len(t)} epi={len(e)} "
          f"epi_mean={summarize(trade1(5).reindex(e).to_numpy(),'')['mean_pct']:+.3f}% "
          f"t={summarize(trade1(5).reindex(e).to_numpy(),'')['t']:+.2f}")

# ============================== INVERSION 2 ==================================
P = close_panel(["TLT", "IEF", "^TNX", "SPY"]).dropna()
idx = P.index
off_lo = P["TLT"] / P["TLT"].rolling(252).min() - 1.0
tnx63 = pct_rank(P["^TNX"], 63)
M2 = ((off_lo <= 0.015) & (tnx63 >= 85)).fillna(False)


def short2(tkr, h):
    return -(P[tkr].shift(-(1 + h)) / P[tkr].shift(-1) - 1.0)


S_ALL = idx[M2.values & short2("TLT", 10).notna().values]
S18 = S_ALL[S_ALL >= pd.Timestamp("2018-01-01")]
E18 = declusters(S18, 10, idx)

print("\n" + "=" * 80)
print("A) INVERSION 2 -- fixed 2018+ episode set (gap=10, N=%d), horizon curve" % len(E18))
print("=" * 80)
rows = []
for h in range(1, 22):
    v = short2("TLT", h).reindex(E18).to_numpy()
    r = summarize(v[~np.isnan(v)], "")
    unc = short2("TLT", h)[idx >= pd.Timestamp("2018-01-01")].dropna()
    rows.append({"h": h, "n_epi": r["n"], "mean_pct": r["mean_pct"], "t": r["t"],
                 "hit": r["hit"], "uncond_pct": 100 * unc.mean(),
                 "excess_pct": r["mean_pct"] - 100 * unc.mean()})
show(rows, "INV2 short TLT, FIXED 2018+ episode set")

print("\n" + "=" * 80)
print("C) INVERSION 2 -- year removals and era-cut sensitivity")
print("=" * 80)
V18 = short2("TLT", 10).reindex(E18).to_numpy()
for lbl, drop in (("2018+ all", ()), ("ex-2022", (2022,)), ("ex-2021,2022", (2021, 2022)),
                  ("ex-2022,2023", (2022, 2023)), ("2023+ only (post-hike)", None)):
    if drop is None:
        m = E18.year >= 2023
    else:
        m = ~np.isin(E18.year, list(drop))
    v = V18[m]
    if len(v) < 2:
        print(f"  {lbl:<24s} N={len(v)}  (too few)")
        continue
    r = summarize(v, "")
    print(f"  {lbl:<24s} N={r['n']:>2d}  mean={r['mean_pct']:+7.3f}%  t={r['t']:+5.2f}  "
          f"hit={r['hit']:>5.1f}%  bootP(mean<=0)={bootstrap_p_le0(v):.3f}  "
          f"dates={[str(d.date()) for d in E18[m]]}")

print("\n  era-cut sensitivity (the 2018 cut was chosen post-hoc):")
for cut in ("2010-01-01", "2013-01-01", "2015-01-01", "2018-01-01", "2020-01-01", "2021-01-01"):
    s = S_ALL[S_ALL >= pd.Timestamp(cut)]
    if len(s) == 0:
        continue
    e = declusters(s, 10, idx)
    v = short2("TLT", 10).reindex(e).to_numpy()
    r = summarize(v, "")
    sp = S_ALL[S_ALL < pd.Timestamp(cut)]
    ep = declusters(sp, 10, idx)
    vp = short2("TLT", 10).reindex(ep).to_numpy() if len(ep) else np.array([])
    rp = summarize(vp, "") if len(vp) > 1 else {"n": len(vp), "mean_pct": np.nan, "t": np.nan}
    print(f"    cut {cut[:4]}: LATE N={r['n']:>2d} mean={r['mean_pct']:+6.3f}% t={r['t']:+5.2f}"
          f"   |  EARLY N={rp['n']:>2d} mean={rp['mean_pct']:+6.3f}% t={rp['t']:+5.2f}")

print("\n  MULTIPLE-COMPARISON ACCOUNTING for the 2018+ h=10 cell:")
print("    the cell was reached by choosing, AFTER seeing the data:")
print("      sign (long->short)            : 2 options")
print("      era (full / pre-2018 / 2018+) : 3 options")
print("      horizon (5 / 10 reported)     : 2 options")
print("      -> >=12 implicit cells before the threshold grid's 12 more.")
print("    a nominal t=2.18 (p~0.052 two-sided) at 12 looks: expected # of cells")
print("    with p<.052 under the null = 0.62. Sidak-adjusted p = "
      f"{1 - (1 - 0.052)**12:.3f}. Nothing survives that.")

# what would today's trade have to beat
print("\n  COST/HURDLE: TLT round trip ~2 bps x 1 leg = 2 bps; short borrow ~"
      "30 bps/yr = ~1.2 bps over 10 td. 5x hurdle ~ 0.16%.")
print(f"  2018+ conditional excess over unconditional at h=10: "
      f"{summarize(V18,'')['mean_pct'] - 100*short2('TLT',10)[idx>=pd.Timestamp('2018-01-01')].dropna().mean():+.3f}%")
