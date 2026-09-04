"""C14 round 1: long the dollar when a rate rise IS confirmed by dollar strength.

Inversion of the parked watchlist entry "Short the dollar on a rate rise the
currency does NOT confirm" (TNX 21d rank >= 65 while DX 21d rank <= 20).
Vehicle = DX-Y.NYB spot, cost charged as DX futures 1.5 bps round trip.
UUP is a cross-check only (registry: dead as a vehicle).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["DX-Y.NYB", "^TNX", "UUP", "SPY", "TLT", "IEF", "GLD"]
LOOKBACKS = (10, 13, 15, 18, 21)
COST_BPS = 1.5

panel = close_panel(TK)
dx = panel["DX-Y.NYB"].dropna()
tnx = panel["^TNX"].dropna()
uup = panel["UUP"].dropna()
IDX = dx.index                      # the vehicle's own calendar
px = panel.reindex(IDX)             # so shift(-h) is exactly h DX sessions
px["DX-Y.NYB"] = dx                 # no holes in the traded column

def on_dx(s):
    return s.reindex(IDX).ffill(limit=3)

# ---- state series -------------------------------------------------------
tnx_rank = {n: on_dx(pct_rank(tnx, n)) for n in LOOKBACKS}
tnx_bp = {n: on_dx((tnx - tnx.shift(n)) * 100.0) for n in LOOKBACKS}   # basis points
dx_rank = {n: pct_rank(dx, n) for n in LOOKBACKS + (5,)}
dx_ret = {n: (dx / dx.shift(n) - 1.0) for n in LOOKBACKS + (5,)}
tnx_hi252 = on_dx(rolling_on_valid(tnx, lambda x: x.rolling(252).max()))
dx_hi252 = rolling_on_valid(dx, lambda x: x.rolling(252).max())

TODAY = IDX[-1]
print("=" * 78)
print(f"STEP 0  MAGNITUDE FIRST   asof bar {TODAY.date()}   (today = 2026-08-31, Monday, MIDTERM)")
print("=" * 78)
print(f"DX-Y.NYB close {dx.iloc[-1]:.3f}   off 252d high {100*(dx.iloc[-1]/dx_hi252.iloc[-1]-1):+.2f}%")
print(f"  DX  5d ret {100*dx_ret[5].iloc[-1]:+.3f}%  rank5  {dx_rank[5].iloc[-1]:.1f}   (live state says +0.91%, rank 87.7)")
print(f"  DX 21d ret {100*dx_ret[21].iloc[-1]:+.3f}%  rank21 {dx_rank[21].iloc[-1]:.1f}")
print(f"^TNX level {tnx.iloc[-1]:.3f}   off 252d high {100*(tnx.iloc[-1]/tnx_hi252.iloc[-1]-1):+.2f}%")
for n in LOOKBACKS:
    print(f"  ^TNX {n:>2}-session change {tnx_bp[n].iloc[-1]:+7.1f} bp   rank{n} {tnx_rank[n].iloc[-1]:5.1f}")

# ---- STEP 1: define confirmed, COUNT FIRST -----------------------------
def mask_conf(L, tnx_thr, dx_thr, dx_lb=None, bp_floor=None):
    dxlb = dx_lb if dx_lb is not None else L
    m = (tnx_rank[L] >= tnx_thr) & (dx_rank[dxlb] >= dx_thr)
    if bp_floor is not None:
        m = m & (tnx_bp[L] >= bp_floor)
    return m.fillna(False)

BP_X = float(tnx_bp[21].iloc[-1])
print("\n" + "=" * 78)
print("STEP 1  COUNT OCCURRENCES BEFORE READING ANY FORWARD NUMBER")
print("=" * 78)
defs = {
    "(a) pure rank  TNX r21>=65 & DX r21>=65": mask_conf(21, 65, 65),
    "(a2) live-match TNX r21>=65 & DX r5>=80": mask_conf(21, 65, 80, dx_lb=5),
    f"(b) rank + magnitude floor  TNX 21d >= {BP_X:.1f} bp (today's own value)":
        mask_conf(21, 65, 65, bp_floor=BP_X),
    "(b10) rank + TNX 21d >= +10 bp": mask_conf(21, 65, 65, bp_floor=10.0),
    "(b20) rank + TNX 21d >= +20 bp": mask_conf(21, 65, 65, bp_floor=20.0),
    "TNX leg alone r21>=65": (tnx_rank[21] >= 65).fillna(False),
    "DX leg alone r21>=65": (dx_rank[21] >= 65).fillna(False),
    "DX leg alone r5>=80": (dx_rank[5] >= 80).fillna(False),
}
rows = []
for lbl, m in defs.items():
    d = IDX[m.reindex(IDX, fill_value=False).values]
    e5 = declusters(d, 5, IDX)
    rows.append({"def": lbl, "days": len(d), "epi_gap5": len(e5),
                 "first": str(d[0].date()) if len(d) else "-",
                 "last": str(d[-1].date()) if len(d) else "-",
                 "fires_today": bool(m.iloc[-1])})
show(rows, "1. occurrence counts (no forward numbers read yet)")

# where does today's magnitude sit inside the trigger distribution?
print("\n  today's magnitude vs the trigger-day distribution (pure-rank def (a)):")
trig_a = IDX[defs["(a) pure rank  TNX r21>=65 & DX r21>=65"].reindex(IDX, fill_value=False).values]
for n in LOOKBACKS:
    v = tnx_bp[n].loc[trig_a].dropna()
    today = tnx_bp[n].iloc[-1]
    pctile = 100.0 * (v <= today).mean()
    print(f"   ^TNX {n:>2}d bp: today {today:+7.1f}  trigger-set median {v.median():+7.1f} "
          f"mean {v.mean():+7.1f}  -> today is the {pctile:5.1f}th pctile of triggers (n={len(v)})")
v = dx_ret[5].loc[trig_a].dropna() * 100
print(f"   DX 5d ret : today {100*dx_ret[5].iloc[-1]:+7.3f}%  trigger-set median {v.median():+7.3f}% "
      f"-> {100.0*(v <= 100*dx_ret[5].iloc[-1]).mean():5.1f}th pctile")
v = dx_ret[21].loc[trig_a].dropna() * 100
print(f"   DX 21d ret: today {100*dx_ret[21].iloc[-1]:+7.3f}%  trigger-set median {v.median():+7.3f}% "
      f"-> {100.0*(v <= 100*dx_ret[21].iloc[-1]).mean():5.1f}th pctile")

# ---- STEP 2: horizon scan + battery -------------------------------------
LEGS = [("DX-Y.NYB", 1.0)]
print("\n" + "=" * 78)
print("STEP 2  HORIZON SCAN + BATTERY  (legs=DX-Y.NYB long, lag=1, cost 1.5 bp)")
print("=" * 78)
for lbl in ["(a) pure rank  TNX r21>=65 & DX r21>=65",
            "(a2) live-match TNX r21>=65 & DX r5>=80"]:
    d = IDX[defs[lbl].reindex(IDX, fill_value=False).values]
    show(horizon_scan(px, d, LEGS, hs=(3, 5, 10, 21), lag=1), f"horizon scan: {lbl}")

variants = {}
for L in LOOKBACKS:
    variants[f"lookback {L} (both legs, 65/65)"] = mask_conf(L, 65, 65)
for tt in (55, 65, 75, 85):
    for dt in (55, 65, 75, 85):
        variants[f"L21 TNX>={tt} DX>={dt}"] = mask_conf(21, tt, dt)
variants["L21 TNX>=65 DX5>=80 (live)"] = mask_conf(21, 65, 80, dx_lb=5)
variants[f"L21 65/65 + bp>={BP_X:.0f}"] = mask_conf(21, 65, 65, bp_floor=BP_X)
variants["L21 65/65 + bp>=20"] = mask_conf(21, 65, 65, bp_floor=20.0)

for h in (5, 10):
    battery(px, defs["(a) pure rank  TNX r21>=65 & DX r21>=65"], LEGS, h,
            f"C14 CONFIRMED DOLLAR  (a) TNX r21>=65 & DX r21>=65", COST_BPS,
            variants=variants, lag=1)

battery(px, defs["(a2) live-match TNX r21>=65 & DX r5>=80"], LEGS, 5,
        "C14 LIVE-MATCHED  TNX r21>=65 & DX r5>=80", COST_BPS, lag=1)

# ---- STEP 3: gate attribution both ways ---------------------------------
print("\n" + "=" * 78)
print("STEP 3  GATE ATTRIBUTION  (does confirmation ADD over either leg alone?)")
print("=" * 78)
for h in (3, 5, 10, 21):
    ret = vehicle_ret(px, LEGS, h, 1)
    valid = ret.notna()
    base = ret[valid]
    rows = []
    cells = {
        "TNX r21>=65 ALONE": (tnx_rank[21] >= 65).fillna(False),
        "DX  r21>=65 ALONE": (dx_rank[21] >= 65).fillna(False),
        "JOINT (confirmed)": mask_conf(21, 65, 65),
        "TNX>=65 & DX<65 (the parked NON-confirmed half)":
            ((tnx_rank[21] >= 65) & (dx_rank[21] < 65)).fillna(False),
        "TNX>=65 & DX r21<=20 (parked sibling exactly)":
            ((tnx_rank[21] >= 65) & (dx_rank[21] <= 20)).fillna(False),
        "DX>=65 & TNX<65 (dollar strong, no rate rise)":
            ((dx_rank[21] >= 65) & (tnx_rank[21] < 65)).fillna(False),
        "ALL DAYS": pd.Series(True, index=IDX),
    }
    for lbl, m in cells.items():
        d = IDX[m.reindex(IDX, fill_value=False).values & valid.values]
        e = declusters(d, max(h, 5), IDX)
        r = summarize(ret.loc[e].values, lbl)
        r["n_days"] = len(d)
        r["excess_pp"] = round(r.get("mean_pct", np.nan) - 100 * base.mean(), 4)
        rows.append(r)
    show(rows, f"gate attribution, h={h} (episode level, gap={max(h,5)})")
    # explicit add-over-leg test
    joint_d = IDX[cells["JOINT (confirmed)"].reindex(IDX, fill_value=False).values & valid.values]
    tnx_only = IDX[cells["TNX r21>=65 ALONE"].reindex(IDX, fill_value=False).values & valid.values]
    dxo = IDX[cells["DX  r21>=65 ALONE"].reindex(IDX, fill_value=False).values & valid.values]
    je = declusters(joint_d, max(h, 5), IDX)
    te = declusters(tnx_only, max(h, 5), IDX)
    de = declusters(dxo, max(h, 5), IDX)
    for lbl, par in (("TNX-alone", te), ("DX-alone", de)):
        a, b = ret.loc[je].values, ret.loc[par].values
        se = np.sqrt(np.nanvar(a, ddof=1) / len(a) + np.nanvar(b, ddof=1) / len(b))
        print(f"   h={h}: joint - {lbl} = {100*(np.nanmean(a)-np.nanmean(b)):+.4f}pp "
              f"welch t {(np.nanmean(a)-np.nanmean(b))/se:+.2f}   "
              f"(episodes discarded by the 2nd gate: {len(par)} -> {len(je)})")

# ---- STEP 4: definition ladder / placebo --------------------------------
print("\n" + "=" * 78)
print("STEP 4  DEFINITION LADDER (lookback 10/13/15/18/21) + OFFSET PLACEBO")
print("=" * 78)
for h in (3, 5, 10):
    rows = []
    for L in LOOKBACKS:
        m = mask_conf(L, 65, 65)
        ret = vehicle_ret(px, LEGS, h, 1)
        d = IDX[m.reindex(IDX, fill_value=False).values & ret.notna().values]
        e = declusters(d, max(h, 5), IDX)
        r = summarize(ret.loc[e].values, f"L={L}")
        r["n_days"] = len(d)
        rows.append(r)
    show(rows, f"lookback ladder, h={h}")

print("\n  offset placebo: anchor = first day of each declustered L21 65/65 episode,")
print("  shifted k sessions (k=0 is the true anchor).")
ret5 = vehicle_ret(px, LEGS, 5, 1)
true_e = declusters(IDX[mask_conf(21, 65, 65).reindex(IDX, fill_value=False).values
                        & ret5.notna().values], 5, IDX)
rows = []
for k in (-10, -5, -3, -1, 0, 1, 3, 5, 10):
    pos, kept = anchor_positions(IDX, true_e, offset=k)
    vals = ret5.iloc[pos].values
    rows.append(summarize(vals, f"k={k:+d}"))
show(rows, "offset placebo (h=5)")

# ---- STEP 5: midterm, era, concentration, local, cost -------------------
print("\n" + "=" * 78)
print("STEP 5  MIDTERM SPLIT (required, today is midterm) + era + concentration")
print("=" * 78)
for h in (3, 5, 10, 21):
    ret = vehicle_ret(px, LEGS, h, 1)
    d = IDX[mask_conf(21, 65, 65).reindex(IDX, fill_value=False).values & ret.notna().values]
    e = declusters(d, max(h, 5), IDX)
    v = ret.loc[e].values
    mid = np.array([(t.year % 4) == 2 for t in e])
    rows = [summarize(v, f"h={h} ALL episodes"),
            summarize(v[mid], f"h={h} MIDTERM (live cycle)"),
            summarize(v[~mid], f"h={h} non-midterm")]
    show(rows, f"midterm split h={h}")
    if mid.sum() >= 3:
        w = int((v[mid] > 0).sum())
        print(f"   midterm record {w}-{mid.sum()-w}, sign p {sign_test(w, int(mid.sum())):.4f}, "
              f"bootstrap P(mean<=0) {bootstrap_p_le0(v[mid]):.3f}")
        print(f"   midterm episode dates: {', '.join(str(t.date()) for t in e[mid])}")
    if h == 5:
        show(era_split(e, v), "era split h=5")
        print("  concentration:", cluster_note(e, v))
        edge_bps = 100 * np.nanmean(v) * 100
        print(f"  cost: episode mean {edge_bps:+.2f} bps vs 1.5 bp RT -> {edge_bps/1.5:.1f}x "
              f"(need >= 5x = 7.5 bps)")
        mid_bps = 100 * np.nanmean(v[mid]) * 100
        print(f"  cost MIDTERM: {mid_bps:+.2f} bps -> {mid_bps/1.5:.1f}x")

# UUP cross-check (never the vehicle)
print("\n  UUP cross-check (NOT the vehicle, registry: UUP dead on drag):")
uup_px = panel.reindex(uup.index)
uup_px["UUP"] = uup
for h in (5, 10):
    r = vehicle_ret(uup_px, [("UUP", 1.0)], h, 1)
    m = mask_conf(21, 65, 65).reindex(uup.index).fillna(False)
    d = uup.index[m.values & r.notna().values]
    e = declusters(d, max(h, 5), uup.index)
    s = summarize(r.loc[e].values, f"UUP h={h} (2007+)")
    # matched DX cell over the same span
    rd = vehicle_ret(px, LEGS, h, 1)
    de = declusters(IDX[mask_conf(21, 65, 65).reindex(IDX, fill_value=False).values
                        & rd.notna().values & (IDX >= uup.index[0])], max(h, 5), IDX)
    show([s, summarize(rd.loc[de].values, f"DX  h={h} (2007+ matched span)")], "")

# ---- STEP 6: grid charge ------------------------------------------------
print("\n" + "=" * 78)
print("STEP 6  GRID CHARGE (rotation permutation + Sidak over the cells walked)")
print("=" * 78)
HS = (3, 5, 10, 21)
TT = (55, 65, 75, 85)
DT = (55, 65, 75, 85)
cells = []
for L in LOOKBACKS:
    for tt in TT:
        for dt in DT:
            m = mask_conf(L, tt, dt).reindex(IDX, fill_value=False).values
            for h in HS:
                cells.append((L, tt, dt, h, m))
print(f"  cells walked = {len(cells)}  ({len(LOOKBACKS)} lookbacks x {len(TT)} TNX x "
      f"{len(DT)} DX x {len(HS)} horizons)")

rets = {h: vehicle_ret(px, LEGS, h, 1) for h in HS}
pos_map = {}
for (L, tt, dt, h, m) in cells:
    r = rets[h]
    d = IDX[m & r.notna().values]
    e = declusters(d, max(h, 5), IDX)
    pos_map[(L, tt, dt, h)] = (IDX.get_indexer(e), h)

arr = {h: rets[h].values for h in HS}

def grid_max_t(shift_by=0):
    best, bestkey = -np.inf, None
    for key, (ipos, h) in pos_map.items():
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
            best, bestkey = t, (key, len(v), 100 * v.mean())
    return best, bestkey

obs_t, obs_key = grid_max_t(0)
print(f"  observed grid max t = {obs_t:.3f} at {obs_key[0]} n={obs_key[1]} mean={obs_key[2]:+.3f}%")
rng = np.random.default_rng(42)
n_rot = 400
offs = rng.integers(250, len(IDX) - 250, size=n_rot)
null = np.array([grid_max_t(int(o))[0] for o in offs])
print(f"  rotation permutation ({n_rot} rotations): P(grid max t >= {obs_t:.3f}) = "
      f"{(null >= obs_t).mean():.3f}   null median max t {np.median(null):.2f}")

# the PITCHED cell's own t and its Sidak charge
key = (21, 65, 65, 5)
ipos, h = pos_map[key]
v = arr[5][ipos]; v = v[~np.isnan(v)]
t_pitch = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
from math import erf, sqrt
p_one = 0.5 * (1 - erf(t_pitch / sqrt(2)))
print(f"  pitched cell L21 65/65 h=5: t={t_pitch:+.3f} n={len(v)} raw one-sided p={p_one:.4f}")
print(f"  Sidak over {len(cells)} cells: p_fw = {1-(1-p_one)**len(cells):.4f}")
print(f"  rotation P(max t >= pitched t) = {(null >= t_pitch).mean():.3f}")
print("\nDONE round 1")
