"""Posts check (2026-09-20): candidate cells for the Sunday queue.

Run date is Sunday 2026-09-20. The freshest bar is Friday 2026-09-18 and the
next session is Monday 2026-09-21, so every cell anchors on the 09-18 close
and the tradeable number is lag=1 (enter MOC Monday, exit h sessions later).

A. IDEA candidate. XLU at a new 252-session CLOSING LOW while SPY sits within
   3% of its own 252-session closing high. Both are true on 2026-09-18.
   Forward XLU and SPY at h=5/10/21, lag=1, novelty-declustered 21 sessions,
   against three controls (all sessions, +/-126td local window, and the
   SPY-near-high universe). Era split at 2018.
B. IDEA candidate. VNQ z10 <= -2 while the 10-year yield's 21-session return
   rank is >= 95. Both true on 2026-09-18. Same machinery.
C. STAT verification. The VIX's own same-session change on Mondays, September
   excluded, against Mondays in September and against all sessions.
D. STAT verification. ^MOVE 21-session return >= +10% while the VIX closes at
   least 10% below its own 200-session average. True on 2026-09-18.
   Forward MOVE / VIX / SPY.

Nothing here is declustered by pitch_lab.declusters alone: a NOVELTY filter is
used for the state cells (the state must have been ABSENT for the prior 21
sessions), which is stricter and is what "first time in a while" means.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    declusters, fwd_lag, load_prices, local_control, pct_rank,
    rolling_on_valid, sign_test, summarize, wilder_atr, zscore,
)

ASOF = pd.Timestamp("2026-09-18")
ERA = "2018-01-01"
TK = ["XLU", "SPY", "VNQ", "^TNX", "^VIX", "^MOVE"]

raw_px = load_prices(TK)
nyse = raw_px["SPY"]["Close"].dropna().index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C = {t: raw_px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

print(f"NYSE calendar (SPY): {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
for t in TK:
    v = C[t].dropna()
    print(f"  {t}: {v.index[0].date()} .. {v.index[-1].date()}  n={len(v)}  "
          f"last {v.iloc[-1]:.3f}")


def rec(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def novelty(trig, win=21):
    tset = set(pd.DatetimeIndex(trig))
    keep = []
    for d in sorted(tset):
        p = int(POS[d])
        if not any(x in tset for x in nyse[max(0, p - win):p]):
            keep.append(d)
    return pd.DatetimeIndex(keep)


def line(label, s, dates, ctrls=()):
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% "
          f"med {sm['median_pct']:+.3f}% hit {sm['hit']:.1f}% t {sm['t']:+.2f} "
          f"p_up {sign_test(up, n):.4f} p_dn {sign_test(dn, n):.4f} "
          f"worst {sm['worst_pct']:+.2f}% ({v.idxmin().date()}) "
          f"best {sm['best_pct']:+.2f}% ({v.idxmax().date()})")
    for cl, cs in ctrls:
        cv = cs.dropna()
        if len(cv) == 0:
            continue
        cu, cd, cn = rec(cv.values)
        print(f"      {cl}: n={cn} {cu}-{cd} mean {100*cv.mean():+.3f}% "
              f"med {100*float(np.median(cv.values)):+.3f}% "
              f"hit {100*(cv > 0).mean():.1f}%")
    return v


def era(label, v):
    a, b = v[v.index < ERA], v[v.index >= ERA]

    def part(x):
        if len(x) == 0:
            return "n=0"
        u, d, n = rec(x.values)
        return (f"n={n} {u}-{d} mean {100*x.mean():+.3f}% "
                f"med {100*float(np.median(x.values)):+.3f}%")
    print(f"   {label} era: pre-2018 [{part(a)}] | 2018+ [{part(b)}]")


def episodes(dates, series):
    cols = list(series)
    print("   episode      " + "".join(f"{c:>11}" for c in cols))
    for d in dates:
        cells = []
        for c in cols:
            x = series[c].get(d, np.nan)
            cells.append("       n/a" if (x is None or np.isnan(x))
                         else f"{100*x:+10.2f}")
        print(f"   {d.date()}  " + "".join(f"{c:>11}" for c in cells))


def state_cell(title, mask, subject, extra_subjects=(), horizons=(5, 10, 21),
               near_hi_universe=None):
    print("\n" + "=" * 78)
    print(f"=== {title} ===")
    valid = mask.notna()
    raw = pd.DatetimeIndex(nyse[mask.fillna(False).astype(bool) & valid])
    if len(raw) == 0:
        print("   no trigger sessions")
        return None, None
    print(f"   raw trigger sessions: {len(raw)} ({raw[0].date()} .. {raw[-1].date()})")
    trig = novelty(raw, 21)
    print(f"   after 21-session NOVELTY filter: {len(trig)} "
          f"(declusters(21) would keep {len(declusters(raw, 21, nyse))})")
    print("   trigger dates: " + ", ".join(str(d.date()) for d in trig))
    loc = local_control(nyse, trig, 126)
    out = {}
    for h in horizons:
        print(f"\n   --- h={h} (enter MOC the session after the signal) ---")
        fs = fwd_lag(C[subject], h, 1)
        ctrls = [("ctrl all days", fs.dropna()),
                 ("ctrl +/-126td", fs.reindex(loc).dropna())]
        if near_hi_universe is not None:
            ctrls.append(("ctrl near-high days",
                          fs.reindex(near_hi_universe).dropna()))
        v = line(f"{subject} h={h}", fs, trig, ctrls)
        out[h] = v
        if len(v):
            era(f"{subject} h={h}", v)
        for other in extra_subjects:
            fo = fwd_lag(C[other], h, 1)
            line(f"{other} h={h}", fo, trig, [("ctrl all days", fo.dropna())])
            sp = fs - fo
            line(f"{subject}-{other} spread h={h}", sp, trig,
                 [("ctrl all days", sp.dropna())])
    ser = {}
    for h in horizons:
        ser[f"{subject}_{h}"] = fwd_lag(C[subject], h, 1)
        for other in extra_subjects:
            ser[f"{other}_{h}"] = fwd_lag(C[other], h, 1)
    print("\n   episode table (%, lag=1):")
    episodes(trig, ser)
    return trig, out


# =========================================================================
# A. XLU at a 252-session closing low while SPY is near its 252-day high
# =========================================================================
xlu_lo = rolling_on_valid(C["XLU"], lambda x: x.rolling(252).min())
xlu_dist_lo = C["XLU"] / xlu_lo - 1.0
spy_hi = rolling_on_valid(C["SPY"], lambda x: x.rolling(252).max())
spy_dist_hi = C["SPY"] / spy_hi - 1.0
print(f"\nTODAY {nyse[-1].date()}: XLU {100*xlu_dist_lo.iloc[-1]:+.2f}% above its "
      f"252d closing low (state says 0.00) | SPY {100*spy_dist_hi.iloc[-1]:+.2f}% "
      f"from its 252d closing high (state says -2.08)")
maskA = (xlu_dist_lo <= 0.0) & (spy_dist_hi >= -0.03)
near_hi = pd.DatetimeIndex(nyse[(spy_dist_hi >= -0.03).fillna(False)])
print(f"   today qualifies: XLU at low {bool((xlu_dist_lo <= 0.0).iloc[-1])} | "
      f"SPY within 3% {bool((spy_dist_hi >= -0.03).iloc[-1])}")
print(f"   near-high universe: {len(near_hi)} sessions")
trigA, outA = state_cell(
    "A. XLU at a 252-session closing LOW while SPY is within 3% of its high",
    maskA, "XLU", extra_subjects=("SPY",), near_hi_universe=near_hi)

# =========================================================================
# B. VNQ washed out while the 10-year is ripping
# =========================================================================
vnq_z = zscore(C["VNQ"], 10)
tnx_rank21 = pct_rank(C["^TNX"], 21, 252)
print(f"\nTODAY {nyse[-1].date()}: VNQ z10 {vnq_z.iloc[-1]:+.2f} (state says -2.12) | "
      f"^TNX 21d rank {tnx_rank21.iloc[-1]:.1f} (state says 97.2)")
maskB = (vnq_z <= -2.0) & (tnx_rank21 >= 95.0)
trigB, outB = state_cell(
    "B. VNQ z10 <= -2 while the 10-year's 21d return rank is >= 95",
    maskB, "VNQ", extra_subjects=("SPY",))

# =========================================================================
# C. The VIX on Mondays
# =========================================================================
print("\n" + "=" * 78)
print("=== C. VIX same-session change on Mondays (September excluded) ===")
vix = C["^VIX"].dropna()
vch = vix.pct_change()
idx = vch.dropna().index
mon = idx[idx.weekday == 0]
mon_ex_sep = mon[mon.month != 9]
mon_sep = mon[mon.month == 9]
for label, dd in (("Mondays ex-September", mon_ex_sep),
                  ("Mondays in September", mon_sep),
                  ("all Mondays", mon),
                  ("all sessions", idx)):
    v = vch.reindex(dd).dropna()
    u, d, n = rec(v.values)
    sm = summarize(v.values)
    print(f"  {label:<24} n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% "
          f"med {sm['median_pct']:+.3f}% up {100*(v > 0).mean():.1f}% "
          f"t {sm['t']:+.2f}  ({dd[0].date()} .. {dd[-1].date()})")
non_mon = idx[idx.weekday != 0]
v = vch.reindex(non_mon).dropna()
u, d, n = rec(v.values)
sm = summarize(v.values)
print(f"  {'non-Mondays':<24} n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% "
      f"med {sm['median_pct']:+.3f}% up {100*(v > 0).mean():.1f}% t {sm['t']:+.2f}")
vm = vch.reindex(mon_ex_sep).dropna()
print("  era split, Mondays ex-September:")
era("    VIX Monday", vm)
for lo, hi in ((1990, 2000), (2000, 2010), (2010, 2020), (2020, 2027)):
    w = vm[(vm.index.year >= lo) & (vm.index.year < hi)]
    if len(w):
        u, d, n = rec(w.values)
        print(f"    {lo}-{hi-1}: n={n} {u}-{d} mean {100*w.mean():+.3f}% "
              f"up {100*(w > 0).mean():.1f}%")

# =========================================================================
# D. MOVE spiking into a calm VIX
# =========================================================================
mv = C["^MOVE"]
mv_r21 = (mv.dropna() / mv.dropna().shift(21) - 1.0).reindex(nyse)
vix_sma200 = rolling_on_valid(C["^VIX"], lambda x: x.rolling(200).mean())
vix_dist = C["^VIX"] / vix_sma200 - 1.0
print(f"\nTODAY {nyse[-1].date()}: MOVE 21d {100*mv_r21.iloc[-1]:+.2f}% "
      f"(state says +13.16) | VIX {100*vix_dist.iloc[-1]:+.2f}% vs its 200d SMA "
      f"(state says -18.16)")
maskD = (mv_r21 >= 0.10) & (vix_dist <= -0.10)
trigD, outD = state_cell(
    "D. MOVE 21d return >= +10% while the VIX is >= 10% below its 200d SMA",
    maskD, "^MOVE", extra_subjects=("^VIX", "SPY"))

# =========================================================================
# Frozen levels for any idea that ships
# =========================================================================
print("\n" + "=" * 78)
print("=== frozen levels (2026-09-18 bars) ===")
for t in ("XLU", "VNQ", "SPY"):
    f = raw_px[t]
    a = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    a = a[a.index <= ASOF]
    c = f["Close"][f["Close"].index <= ASOF]
    print(f"  {t}: close {c.iloc[-1]:.4f}  Wilder-14 ATR {a.iloc[-1]:.4f} "
          f"({100*a.iloc[-1]/c.iloc[-1]:.2f}% of price)  bar {c.index[-1].date()}")
