"""D2 round 1 -- watchlist #18, long IEF vs short 0.523 TLT (duration-neutral)
with ^TNX at a trailing-252 maximum AND a 252-session yield change >= +78 bp.

Both binding legs armed on the 2026-09-09 bar for the first time. Kill attempt.

 0. Reproduce the parked construction (a1_r2c_verdict.py) and the ARMED half.
 1. THE DENOMINATOR ROLL. How much of the arm is the trailing reference falling
    away rather than the live yield rising? Historical episodes that cleared the
    +78 bp bar by < 5 bp vs genuine thrusts.
 2. MULTIPLICITY, charged over the FULL disclosed walk against the DEFENDED
    statistic (the armed h=8 cell), uncharged reported beside it.
 3. GATE ATTRIBUTION -- each leg alone, and the DISCARDED COMPLEMENT of each.
 4. LEG ATTRIBUTION -- IEF alone, short TLT alone, the curve.
 5. IS IT STILL A DIRECTIONAL RATES BET? beta of the pair on the yield change
    over the hold.
 6. Era split, concentration, decluster order.
 7. horizon_scan h=1..10.
 8. Cost.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, vehicle_ret, summarize, sign_test,
                       rolling_on_valid, show, bootstrap_p_le0, cluster_note,
                       horizon_scan, declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025                      # a1_r2c's implementation
STRICT = tnx >= hi252 - 1e-12                  # definition neighbour: exactly AT
chg252 = (tnx - tnx.shift(252)) * 100.0
THRUST = chg252 >= 78.0

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
W = 1.0 / BETA
FLAT = [("IEF", 1.0), ("TLT", -W)]
print(f"duration-neutral weight: TLT beta on IEF = {BETA:.4f} -> short {W:.4f} TLT")
print(f"panel {idx[0].date()} .. {idx[-1].date()}  n={len(idx)}")

COST_C = 3.594           # a1_r2b conv C, half-spread MOC both legs
COST_BORROW = 4.423      # + 0.50%/yr borrow on the short leg for 8 td
COST_BRIEF = 5.0

POS = {dd: i for i, dd in enumerate(idx)}


def fast_decluster(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


def cell(mask, h=8, gap=10, legs=None):
    legs = legs or FLAT
    ret = vehicle_ret(px, legs, h, 1)
    sig = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    ep = fast_decluster(sig, max(h, gap))
    return ep, ret.loc[ep].values


def bl(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"label": label, "n": 0}
    s = summarize(v, label)
    s["bps"] = round(100 * 100 * v.mean(), 1)
    s["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    s["x_cost"] = round(100 * 100 * v.mean() / COST_BORROW, 2)
    return {k: s[k] for k in ("label", "n", "bps", "hit", "t", "rec", "signp",
                              "x_cost", "worst_pct", "best_pct")}


print("\n" + "=" * 122)
print("0. THE PARKED CELL AND THE ARMED HALF")
print("=" * 122)
ARMED = LEVEL & THRUST
ep_lvl, v_lvl = cell(LEVEL)
ep_arm, v_arm = cell(ARMED)
show([bl(v_lvl, "LEVEL only (within 0.25% of 252d max), h=8  [the parked cell]"),
      bl(v_arm, "ARMED: LEVEL and 252d chg >= +78 bp, h=8   [DEFENDED]"),
      bl(cell(LEVEL & ~THRUST)[1], "DISCARDED COMPLEMENT: LEVEL and chg < +78 bp")],
     "reproduce")
print(f"  armed episode dates: {[str(x.date()) for x in ep_arm]}")
print(f"  armed boot P(mean<=0) = {bootstrap_p_le0(v_arm):.4f}")
print("  live state 2026-09-09: ^TNX %.4f, 252d max %.4f (off-high %+.4f%%), "
      "252d change %+.1f bp"
      % (tnx.iloc[-1], hi252.iloc[-1], 100 * off_hi.iloc[-1], chg252.iloc[-1]))
print("  strict-max definition neighbour:")
show([bl(cell(STRICT & THRUST)[1], "STRICT max AND chg >= +78 bp"),
      bl(cell(STRICT)[1], "STRICT max only")], "definition neighbour")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("1. THE DENOMINATOR ROLL -- did it arm on repricing or on the reference")
print("   falling out of the window?")
print("=" * 122)
ref = tnx.shift(252)
margin = chg252 - 78.0            # bp of clearance over the arm
d_live = (tnx - tnx.shift(6)) * 100.0
d_ref = (ref - ref.shift(6)) * 100.0
print("  live last 8 sessions: yield %+.1f bp over 6 sessions, year-ago reference "
      "%+.1f bp -> %.1f%% of the crossing is the reference rolling off"
      % (d_live.iloc[-1], d_ref.iloc[-1],
         100 * abs(d_ref.iloc[-1]) / (abs(d_ref.iloc[-1]) + abs(d_live.iloc[-1]))))
print("  clearance today: %+.1f bp" % margin.iloc[-1])
mg = margin.reindex(ep_arm)
print(f"  historical armed episodes and their clearance (bp over the +78 bar):")
tab = pd.DataFrame({"clearance_bp": mg.round(1),
                    "ret_bps": (100 * 100 * pd.Series(v_arm, index=ep_arm)).round(1),
                    "d_ref_6d_bp": d_ref.reindex(ep_arm).round(1),
                    "d_live_6d_bp": d_live.reindex(ep_arm).round(1)})
print(tab.to_string())
thin = mg <= 5.0
print(f"\n  episodes clearing by <= 5 bp: {int(thin.sum())} of {len(mg)}")
show([bl(pd.Series(v_arm, index=ep_arm)[thin.values].values, "THIN crossings (<=5 bp clearance)"),
      bl(pd.Series(v_arm, index=ep_arm)[~thin.values].values, "GENUINE thrusts (>5 bp)")],
     "today is a THIN crossing at +1.1 bp")
# roll-driven vs price-driven crossings
rolldriven = (d_ref.reindex(ep_arm) < 0) & (d_ref.reindex(ep_arm).abs() > d_live.reindex(ep_arm).abs())
show([bl(pd.Series(v_arm, index=ep_arm)[rolldriven.fillna(False).values].values,
         "ROLL-driven (|d_ref| > |d_live|, ref falling) -- today's kind"),
      bl(pd.Series(v_arm, index=ep_arm)[~rolldriven.fillna(False).values].values,
         "PRICE-driven")], "crossing provenance")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("2. MULTIPLICITY, charged over the FULL disclosed walk")
print("=" * 122)
print("  DEFENDED statistic: MEAN 8-session lag-1 return of the duration-neutral")
print("  pair over declustered episodes with ^TNX within 0.25% of its trailing-252")
print("  max AND a 252-session yield change >= +78 bp.")
VEHICLES = {"curve": FLAT, "IEF": [("IEF", 1.0)], "TLT": [("TLT", 1.0)]}
HS = list(range(1, 11))
PROX = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]
LOOKBACKS = [63, 126, 189, 252, 378, 504]
print(f"  walk = {len(VEHICLES)} vehicles x 2 signs x {len(HS)} horizons x "
      f"{len(PROX)} proximity rungs x {len(LOOKBACKS)} lookbacks = "
      f"{len(VEHICLES)*2*len(HS)*len(PROX)*len(LOOKBACKS)} cells")
obs = float(np.mean(v_arm))
rets = {}
for vn, legs in VEHICLES.items():
    for h in HS:
        rets[(vn, h)] = vehicle_ret(px, legs, h, 1)
masks = {}
for lb in LOOKBACKS:
    hi = rolling_on_valid(tnx, lambda x, L=lb: x.rolling(L).max())
    oh = tnx / hi - 1.0
    for p in PROX:
        masks[(lb, p)] = (oh >= -p if p > 0 else tnx >= hi - 1e-12) & THRUST
rng = np.random.default_rng(11)
NB = 3000
n_all = len(idx)
arr = {k: np.nan_to_num(v.values, nan=0.0) for k, v in rets.items()}
valid = {k: (~np.isnan(v.values)) for k, v in rets.items()}
mask_arr = {k: m.reindex(idx, fill_value=False).values for k, m in masks.items()}
pos_arr = np.arange(n_all)
# decluster STRUCTURE is fixed by (mask, gap): precompute once so the permutation
# loop is pure array indexing. Gap = max(h, 10) -> only three distinct gaps.
GAPS = sorted({max(h, 10) for h in HS})
keeps = {}
for mk, m in mask_arr.items():
    sp = pos_arr[m]
    for g in GAPS:
        kp, last = [], -10 ** 9
        for pp in sp:
            if pp - last >= g:
                kp.append(pp)
                last = pp
        keeps[(mk, g)] = np.array(kp, dtype=int)
print("  declustered episode counts per (lookback, prox) at gap 10: "
      + ", ".join(f"{k[0]}:{len(v)}" for k, v in keeps.items() if k[1] == 10))
unch = 0
charged = 0
for b in range(NB):
    sh = int(rng.integers(21, n_all - 21))
    best = -9.9
    for mk in mask_arr:
        for h in HS:
            kp = keeps[(mk, max(h, 10))]
            if len(kp) < 8:
                continue
            src = (kp + sh) % n_all
            for vn in VEHICLES:
                vd = valid[(vn, h)][src]
                if vd.sum() < 8:
                    continue
                mu = float(arr[(vn, h)][src][vd].mean())
                if mu > best:
                    best = mu
                if -mu > best:
                    best = -mu
                if (vn == "curve" and h == 8 and mk == (252, 0.0025) and mu >= obs):
                    unch += 1
    if best >= obs:
        charged += 1
print(f"  observed DEFENDED mean = {100*100*obs:+.1f} bps on N={len(v_arm)}")
print(f"  UNCHARGED p (same cell under the null)  = {unch/NB:.4f}  ({NB} draws)")
print(f"  CHARGED   p (walk max >= defended stat) = {charged/NB:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("3. GATE ATTRIBUTION")
print("=" * 122)
show([bl(cell(LEVEL & THRUST)[1], "BOTH legs (defended)"),
      bl(cell(THRUST)[1], "THRUST leg alone (chg252 >= +78)"),
      bl(cell(~THRUST)[1], "  discarded complement: chg252 < +78"),
      bl(cell(LEVEL)[1], "LEVEL leg alone (within 0.25% of 252d max)"),
      bl(cell(~LEVEL)[1], "  discarded complement: NOT near the max"),
      bl(cell(THRUST & ~LEVEL)[1], "THRUST without the max touch"),
      bl(cell(LEVEL & ~THRUST)[1], "max touch without the thrust")],
     "each leg, and what each gate throws away")
allret = vehicle_ret(px, FLAT, 8, 1).dropna()
print(f"  UNCONDITIONAL all-days h=8 pair: {100*100*allret.mean():+.1f} bps "
      f"on n={len(allret)}, hit {100*(allret>0).mean():.1f}%")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("4. LEG ATTRIBUTION -- long IEF, short TLT, or genuinely the curve?")
print("=" * 122)
rows = []
for lab, legs in (("curve (IEF - %.3f TLT)" % W, FLAT), ("IEF alone", [("IEF", 1.0)]),
                  ("TLT alone (long)", [("TLT", 1.0)]),
                  ("short TLT alone", [("TLT", -1.0)]),
                  ("short 0.523 TLT alone", [("TLT", -W)])):
    rows.append(bl(cell(ARMED, legs=legs)[1], lab))
show(rows, "armed episodes, h=8")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("5. IS THE 'DURATION-NEUTRAL' PAIR STILL A DIRECTIONAL RATES BET?")
print("=" * 122)
h = 8
dy = (tnx.shift(-(1 + h)) - tnx.shift(-1)) * 100.0     # yield change over the hold
pr = vehicle_ret(px, FLAT, h, 1)
j = pd.concat([dy.rename("dy"), pr.rename("pair")], axis=1).dropna()
b, a = np.polyfit(j["dy"].values, j["pair"].values, 1)
print(f"  ALL DAYS: pair = {100*100*a:+.2f} bps + {100*100*b:+.3f} bps per bp of "
      f"yield change   corr {j['dy'].corr(j['pair']):+.3f}  n={len(j)}")
ja = j.reindex(ep_arm).dropna()
if len(ja) > 3:
    b2, a2 = np.polyfit(ja["dy"].values, ja["pair"].values, 1)
    print(f"  ARMED EPISODES: pair = {100*100*a2:+.2f} bps + {100*100*b2:+.3f} bps/bp "
          f"corr {ja['dy'].corr(ja['pair']):+.3f}  n={len(ja)}")
    print(f"  mean yield change over the armed holds: {ja['dy'].mean():+.1f} bp "
          f"(all days {j['dy'].mean():+.1f} bp)")
    print(f"  residual after removing the ALL-DAYS rate beta: "
          f"{100*100*(ja['pair'] - (a + b*ja['dy'])).mean():+.1f} bps")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("6. ERA, CONCENTRATION, DECLUSTER ORDER")
print("=" * 122)
va = pd.Series(v_arm, index=ep_arm)
print("  " + cluster_note(va.index, va.values, k=2))
order = np.argsort(-np.abs(va.values))[:2]
keep = np.ones(len(va), bool)
keep[order] = False
print(f"  drop-best-2: {100*100*va.values[keep].mean():+.1f} bps on n={keep.sum()}")
byyr = pd.Series(va.values).groupby(va.index.year.values).agg(["sum", "count"])
print(f"  by year: {dict((y, (round(100*100*r['sum'],0), int(r['count']))) for y, r in byyr.iterrows())}")
best_yr = (byyr["sum"]).idxmax()
kv = va[va.index.year != best_yr]
print(f"  drop-best-year ({best_yr}): {100*100*kv.mean():+.1f} bps on n={len(kv)} "
      f"record {int((kv>0).sum())}-{int((kv<0).sum())}")
for cut in ("2010-01-01", "2018-01-01", "2022-01-01"):
    m = va.index < pd.Timestamp(cut)
    show([bl(va.values[m], f"pre-{cut[:4]}"), bl(va.values[~m], f"{cut[:4]}+")],
         f"era split at {cut[:4]}")
print("  decluster gap sensitivity:")
for gap in (1, 5, 10, 21, 42, 63):
    e, v = cell(ARMED, gap=gap)
    print(f"    gap {gap:2d}: N={len(e):3d}  {100*100*v.mean():+6.1f} bps  "
          f"|t| {abs(v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))):.2f}  "
          f"{100*100*v.mean()/COST_BORROW:.2f}x")
print("  ORDER TEST (rule 4): filter-then-decluster vs decluster-then-filter")
ret8 = vehicle_ret(px, FLAT, 8, 1)
sig = idx[ARMED.reindex(idx, fill_value=False).values & ret8.notna().values]
ftd = fast_decluster(sig, 10)
lvl_sig = idx[LEVEL.reindex(idx, fill_value=False).values & ret8.notna().values]
dtf = pd.DatetimeIndex([d for d in fast_decluster(lvl_sig, 10)
                        if bool(THRUST.get(d, False))])
print(f"    filter-then-decluster: N={len(ftd)}  {100*100*ret8.loc[ftd].mean():+.1f} bps")
print(f"    decluster-then-filter: N={len(dtf)}  "
      f"{100*100*ret8.loc[dtf].mean() if len(dtf) else float('nan'):+.1f} bps")
print(f"    dates only in FTD: {[str(x.date()) for x in ftd.difference(dtf)]}")
print(f"    dates only in DTF: {[str(x.date()) for x in dtf.difference(ftd)]}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("7. HORIZON SCAN h=1..10 on the ARMED gate")
print("=" * 122)
rows = []
for h in HS:
    e, v = cell(ARMED, h=h)
    r = bl(v, f"h={h}")
    base = vehicle_ret(px, FLAT, h, 1).dropna()
    r["ctl_bps"] = round(100 * 100 * base.mean(), 1)
    r["edge_bps"] = round(r["bps"] - r["ctl_bps"], 1) if r["n"] else np.nan
    rows.append(r)
show(rows, "armed gate, duration-neutral pair, episode level")

print("\n" + "=" * 122)
print("8. COST")
print("=" * 122)
print(f"  armed h=8 {100*100*np.mean(v_arm):+.1f} bps -> "
      f"{100*100*np.mean(v_arm)/COST_C:.2f}x (conv C {COST_C} bps), "
      f"{100*100*np.mean(v_arm)/COST_BORROW:.2f}x (with borrow {COST_BORROW} bps), "
      f"{100*100*np.mean(v_arm)/COST_BRIEF:.2f}x (brief's 5 bp)")
