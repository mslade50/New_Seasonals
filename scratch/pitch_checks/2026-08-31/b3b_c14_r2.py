"""C14 round 2: the ONE definition that actually fires on 2026-08-28's tape.

Round 1 killed the briefed rank form and also showed it does NOT fire today
(DX r21 42.9 against the >=65 leg; ^TNX 21d change +5.7 bp = 6.9th pctile of
trigger days). The state that DOES exist today is a yield LEVEL near its
252-day high plus a 5-day dollar thrust, so restate the candidate that way
and kill THAT before a composer can. Registry precedent: the "^TNX at a
52-week high" LEVEL trigger coincides with the killed return-rank trigger on
91% of days and inherits its search charge (d2_c5_tnx_level_high.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from math import erf, sqrt

TK = ["DX-Y.NYB", "^TNX", "UUP", "SPY", "EURUSD=X", "JPY=X", "GLD", "TLT"]
panel = close_panel(TK)
dx = panel["DX-Y.NYB"].dropna()
tnx = panel["^TNX"].dropna()
IDX = dx.index
px = panel.reindex(IDX)
px["DX-Y.NYB"] = dx


def on_dx(s):
    return s.reindex(IDX).ffill(limit=3)


tnx_hi = on_dx(rolling_on_valid(tnx, lambda x: x.rolling(252).max()))
tnx_lvl = on_dx(tnx)
tnx_offhi = tnx_lvl / tnx_hi - 1.0                       # <=0, distance below 252d yield high
tnx_r21 = on_dx(pct_rank(tnx, 21))
tnx_bp21 = on_dx((tnx - tnx.shift(21)) * 100.0)
dx_r5 = pct_rank(dx, 5)
dx_r21 = pct_rank(dx, 21)
dx_ret5 = dx / dx.shift(5) - 1.0
spy = on_dx(panel["SPY"].dropna())
spy_200 = on_dx(rolling_on_valid(panel["SPY"].dropna(), lambda x: x.rolling(200).mean()))
tnx_252lo = on_dx(rolling_on_valid(tnx, lambda x: x.rolling(252).min()))
LEGS = [("DX-Y.NYB", 1.0)]
COST = 1.5

print("=" * 78)
print("ROUND 2  C14 restated to the LIVE state   asof", IDX[-1].date())
print("=" * 78)
print(f"  ^TNX off 252d high {100*tnx_offhi.iloc[-1]:+.3f}%   level {tnx_lvl.iloc[-1]:.3f}  "
      f"252d high {tnx_hi.iloc[-1]:.3f}  252d low {tnx_252lo.iloc[-1]:.3f}")
print(f"  ^TNX 21d change {tnx_bp21.iloc[-1]:+.1f} bp   r21 {tnx_r21.iloc[-1]:.1f}")
print(f"  DX r5 {dx_r5.iloc[-1]:.1f}  5d ret {100*dx_ret5.iloc[-1]:+.3f}%   r21 {dx_r21.iloc[-1]:.1f}")
print("  NOTE: the round-1 briefed rank form does NOT fire today (DX r21 42.9 < 65,")
print("        ^TNX r21 52.4 < 65). Everything below is the restated live form.")


def mask(off_rung, dxr5_rung):
    return ((tnx_offhi >= -off_rung / 100.0) & (dx_r5 >= dxr5_rung)).fillna(False)


LIVE = mask(1.0, 80)     # ^TNX within 1.0% of its 252d yield high AND DX r5 >= 80
print(f"\n  LIVE definition fires today: {bool(LIVE.iloc[-1])}")

# ---- occurrence counts first -------------------------------------------
rows = []
cells = {
    "^TNX within 0.25% of 252d hi & DX r5>=80": mask(0.25, 80),
    "^TNX within 0.50% of 252d hi & DX r5>=80": mask(0.50, 80),
    "^TNX within 1.00% of 252d hi & DX r5>=80 (LIVE)": LIVE,
    "^TNX within 2.00% of 252d hi & DX r5>=80": mask(2.00, 80),
    "^TNX within 1.00% of 252d hi ALONE": ((tnx_offhi >= -0.01)).fillna(False),
    "DX r5>=80 ALONE": (dx_r5 >= 80).fillna(False),
    "ALL DAYS": pd.Series(True, index=IDX),
}
for lbl, m in cells.items():
    d = IDX[m.reindex(IDX, fill_value=False).values]
    rows.append({"def": lbl, "days": len(d), "epi_gap5": len(declusters(d, 5, IDX)),
                 "first": str(d[0].date()) if len(d) else "-",
                 "last": str(d[-1].date()) if len(d) else "-",
                 "fires_today": bool(m.iloc[-1])})
show(rows, "occurrence counts (before any forward number)")

# overlap with the corpse: rank form vs level form
rankform = ((tnx_r21 >= 65) & (dx_r21 >= 65)).fillna(False)
lvl_only = ((tnx_offhi >= -0.01)).fillna(False)
rk_only = (tnx_r21 >= 65).fillna(False)
agree = (lvl_only == rk_only).mean()
print(f"\n  overlap check (registry d2_c5): ^TNX level-near-high vs ^TNX r21>=65 agree on "
      f"{100*agree:.1f}% of days; joint days {int((lvl_only & rk_only).sum())} of "
      f"{int(lvl_only.sum())} level-days")

# ---- horizon scan + battery on the LIVE form ---------------------------
d_live = IDX[LIVE.values]
show(horizon_scan(px, d_live, LEGS, hs=(3, 5, 10, 21), lag=1), "horizon scan, LIVE form")

variants = {}
for off in (0.25, 0.5, 1.0, 2.0):
    for r5 in (70, 80, 90):
        variants[f"off<={off}% & DXr5>={r5}"] = mask(off, r5)
for h in (5, 10):
    battery(px, LIVE, LEGS, h, f"C14-LIVE  ^TNX within 1% of 252d yield high & DX r5>=80",
            COST, variants=variants, lag=1)

# ---- gate attribution ---------------------------------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION on the LIVE form")
print("=" * 78)
for h in (3, 5, 10):
    ret = vehicle_ret(px, LEGS, h, 1)
    valid = ret.notna()
    base = ret[valid]
    rows = []
    for lbl, m in [("^TNX near 252d hi ALONE", lvl_only),
                   ("DX r5>=80 ALONE", (dx_r5 >= 80).fillna(False)),
                   ("JOINT (LIVE)", LIVE),
                   ("DX r5>=80 & ^TNX NOT near hi", ((dx_r5 >= 80) & (tnx_offhi < -0.01)).fillna(False)),
                   ("ALL DAYS", pd.Series(True, index=IDX))]:
        d = IDX[m.reindex(IDX, fill_value=False).values & valid.values]
        e = declusters(d, max(h, 5), IDX)
        r = summarize(ret.loc[e].values, lbl)
        r["n_days"] = len(d)
        r["excess_pp"] = round(r.get("mean_pct", np.nan) - 100 * base.mean(), 4)
        rows.append(r)
    show(rows, f"h={h}")
    je = declusters(IDX[LIVE.values & valid.values], max(h, 5), IDX)
    de = declusters(IDX[(dx_r5 >= 80).fillna(False).values & valid.values], max(h, 5), IDX)
    a, b = ret.loc[je].values, ret.loc[de].values
    se = np.sqrt(np.nanvar(a, ddof=1) / len(a) + np.nanvar(b, ddof=1) / len(b))
    print(f"   h={h}: joint - DX-r5-alone = {100*(np.nanmean(a)-np.nanmean(b)):+.4f}pp  "
          f"welch t {(np.nanmean(a)-np.nanmean(b))/se:+.2f}   episodes {len(b)} -> {len(a)}")

# ---- offset placebo -----------------------------------------------------
print("\n" + "=" * 78)
print("OFFSET PLACEBO on the LIVE form (is the trigger a LAGGING marker?)")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, LEGS, h, 1)
    e = declusters(IDX[LIVE.values & ret.notna().values], max(h, 5), IDX)
    rows = []
    for k in (-10, -5, -3, -1, 0, 1, 3, 5, 10):
        pos, kept = anchor_positions(IDX, e, offset=k)
        rows.append(summarize(ret.iloc[pos].values, f"k={k:+d}"))
    show(rows, f"offset placebo h={h}  (k=0 is the true anchor)")

# ---- midterm + regime splits -------------------------------------------
print("\n" + "=" * 78)
print("MIDTERM + REGIME SPLITS (today: MIDTERM, SPY above 200d, yields at a 252d high)")
print("=" * 78)
for h in (3, 5, 10):
    ret = vehicle_ret(px, LEGS, h, 1)
    e = declusters(IDX[LIVE.values & ret.notna().values], max(h, 5), IDX)
    v = ret.loc[e].values
    mid = np.array([(t.year % 4) == 2 for t in e])
    above = (spy.loc[e] > spy_200.loc[e]).values
    # rising-yield regime = TNX level above its 252d midpoint
    mid_lvl = ((tnx_hi + tnx_252lo) / 2.0)
    rising = (tnx_lvl.loc[e] > mid_lvl.loc[e]).values
    rows = [summarize(v, f"h={h} ALL"),
            summarize(v[mid], f"h={h} MIDTERM (live)"),
            summarize(v[~mid], f"h={h} non-midterm"),
            summarize(v[above], f"h={h} SPY>200d (live)"),
            summarize(v[~above], f"h={h} SPY<200d"),
            summarize(v[rising], f"h={h} yield-upper-half (live)"),
            summarize(v[~rising], f"h={h} yield-lower-half")]
    show(rows, f"splits h={h}")
    if mid.sum() >= 3:
        w = int((v[mid] > 0).sum())
        print(f"   midterm {w}-{int(mid.sum())-w}, sign p {sign_test(w, int(mid.sum())):.4f}, "
              f"bootstrap P(mean<=0) {bootstrap_p_le0(v[mid]):.3f}, "
              f"cost {100*np.nanmean(v[mid])*100:+.2f} bps = {100*np.nanmean(v[mid])*100/COST:.1f}x")
        print(f"   midterm dates: {', '.join(str(t.date()) for t in e[mid])}")
    if h == 5:
        print("   concentration:", cluster_note(e, v))
        show(era_split(e, v), "era split h=5")
        # honest worst window
        order = np.argsort(v)[:5]
        print("   worst 5 episodes:", ", ".join(
            f"{e[i].date()} {100*v[i]:+.2f}%" for i in order))
        by_yr = pd.Series(v).groupby(pd.DatetimeIndex(e).year.values).mean() * 100
        print("   by year (mean %):", {int(y): round(x, 3) for y, x in by_yr.items()})

# ---- reference class ----------------------------------------------------
print("\n" + "=" * 78)
print("REFERENCE CLASS: does the identical rule work on the crosses / on UUP?")
print("=" * 78)
for tkr, sign, lbl in [("EURUSD=X", -1.0, "short EURUSD (= long USD)"),
                       ("JPY=X", 1.0, "long USDJPY (= long USD)"),
                       ("UUP", 1.0, "long UUP (cross-check only)")]:
    s = panel[tkr].dropna()
    p2 = panel.reindex(s.index)
    p2[tkr] = s
    m = LIVE.reindex(s.index).fillna(False).infer_objects(copy=False).astype(bool)
    rows = []
    for h in (3, 5, 10):
        r = vehicle_ret(p2, [(tkr, sign)], h, 1)
        d = s.index[m.values & r.notna().values]
        e = declusters(d, max(h, 5), s.index)
        rr = summarize(r.loc[e].values, f"{lbl} h={h}")
        rr["ctl_all_pct"] = round(100 * r.dropna().mean(), 4)
        rows.append(rr)
    show(rows, f"{tkr} ({s.index[0].date()}+)")

# ---- grid charge --------------------------------------------------------
print("\n" + "=" * 78)
print("GRID CHARGE for round 2 (the cells actually walked here)")
print("=" * 78)
OFFS = (0.25, 0.5, 1.0, 2.0)
R5 = (70, 80, 90)
HS = (3, 5, 10, 21)
rets = {h: vehicle_ret(px, LEGS, h, 1) for h in HS}
pos_map = {}
for off in OFFS:
    for r5 in R5:
        m = mask(off, r5).reindex(IDX, fill_value=False).values
        for h in HS:
            d = IDX[m & rets[h].notna().values]
            pos_map[(off, r5, h)] = (IDX.get_indexer(declusters(d, max(h, 5), IDX)), h)
arr = {h: rets[h].values for h in HS}
print(f"  round-2 cells = {len(pos_map)}; round-1 cells = 320; family total = {len(pos_map)+320}")


def grid_max_t(shift_by=0):
    best, key = -np.inf, None
    for k, (ipos, h) in pos_map.items():
        a = arr[h]
        if shift_by:
            a = np.roll(a, shift_by)
        v = a[ipos]
        v = v[~np.isnan(v)]
        if len(v) < 5:
            continue
        sd = v.std(ddof=1)
        if sd <= 0:
            continue
        t = v.mean() / (sd / np.sqrt(len(v)))
        if t > best:
            best, key = t, (k, len(v), 100 * v.mean())
    return best, key


obs, key = grid_max_t(0)
rng = np.random.default_rng(42)
offs = rng.integers(250, len(IDX) - 250, size=400)
null = np.array([grid_max_t(int(o))[0] for o in offs])
print(f"  observed round-2 grid max t = {obs:.3f} at {key[0]} n={key[1]} mean={key[2]:+.3f}%")
print(f"  rotation P(grid max t >= {obs:.3f}) = {(null >= obs).mean():.3f}  null median {np.median(null):.2f}")
ipos, h = pos_map[(1.0, 80, 5)]
v = arr[5][ipos]; v = v[~np.isnan(v)]
t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
p1 = 0.5 * (1 - erf(t / sqrt(2)))
K = len(pos_map) + 320
print(f"  LIVE cell (1.0%,80,h=5): t={t:+.3f} n={len(v)} raw p={p1:.4f}  "
      f"Sidak over the whole {K}-cell family p_fw={1-(1-p1)**K:.4f}")
print("\nDONE round 2")
