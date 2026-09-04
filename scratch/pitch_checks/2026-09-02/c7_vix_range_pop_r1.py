"""C7 round 1 -- Long SPY on a violent VIX pop out of a COMPRESSED 21-day
range, with the index barely down.

Live 2026-09-01: ^VIX +9.52% to 16.34, its 21-day high/low range at the 8.3rd
percentile of its trailing year, SPY -0.69%. 40 declustered episodes over 20
years at the bottom-15% range rung (the bottom-5% rung has 11 and is NOT live).

The decisive question is NOT whether the cell beats zero. Watchlist 12 is the
same family reached by a different road -- "VIX 21-day RETURN rank <= 25 AND
VIX up >= 5% AND SPY down less than 0.75%" -- and it is parked precisely
because its increment over its own regime control is only +0.395pp at Welch
t +1.09 (arm: t >= 2.0 at h=10). C7 swaps a calm LEVEL/return-rank conditioner
for a compressed RANGE conditioner. If the two trigger sets overlap heavily,
C7 inherits watchlist 12's unmet arm. Everything below is scored as an
INCREMENT, never against zero.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from _rc import dial_series, jaccard, welch  # noqa

pd.set_option("display.width", 250)

px = close_panel(["SPY", "^VIX", "^VIX3M", "SVXY"])
core = px[["SPY", "^VIX"]].dropna()
D = core.index
r_spy = core["SPY"].pct_change()
r_vix = core["^VIX"].pct_change()
rng21 = rolling_on_valid(core["^VIX"], lambda x: x.rolling(21).max() / x.rolling(21).min() - 1)
rng_pct = rolling_on_valid(rng21.dropna(), lambda x: x.rolling(252).rank(pct=True) * 100).reindex(D)
w12_rank = pct_rank(core["^VIX"], 21)

POP = (r_vix >= 0.08) & (r_spy > -0.0125) & (r_spy < 0)
COMP = rng_pct <= 15
MAIN = (POP & COMP).fillna(False)
W12 = ((w12_rank <= 25) & (r_vix >= 0.05) & (r_spy > -0.0075)).fillna(False)

print(f"LIVE 2026-09-01: VIX {core['^VIX'].iloc[-1]:.2f} ({100*r_vix.iloc[-1]:+.2f}%), "
      f"21d range pctile {rng_pct.iloc[-1]:.1f}, SPY {100*r_spy.iloc[-1]:+.2f}%, "
      f"VIX 21d return rank {w12_rank.iloc[-1]:.1f}")
print(f"MAIN (C7) fires today: {bool(MAIN.iloc[-1])} | watchlist-12 fires today: "
      f"{bool(W12.iloc[-1])}")

# ------------------------------------------ 1. IS THIS WATCHLIST 12 RE-SKINNED
print("\n########## 1. OVERLAP WITH WATCHLIST 12 ##########")
a, b = D[MAIN.values], D[W12.values]
i, u, j = jaccard(a, b)
print(f"  C7 days {len(a)}, w12 days {len(b)}, intersection {i}, union {u}, "
      f"Jaccard {j:.3f}")
print(f"  share of C7 days that are ALSO w12 days: {100*i/len(a):.1f}%")
print(f"  share of w12 days that are ALSO C7 days: {100*i/len(b):.1f}%")
# the shared parent both cells sit inside
POP5 = ((r_vix >= 0.05) & (r_spy > -0.0075)).fillna(False)
print(f"  both are subsets of 'VIX pops while SPY barely moves': "
      f"POP5 parent has {int(POP5.sum())} days")
print(f"  correlation of the two conditioners: rng_pct vs w12 21d return rank = "
      f"{rng_pct.corr(w12_rank):.3f}")

# --------------------------------------------------------------- 2. HORIZONS
print("\n########## 2. HORIZON SCAN ##########")
show(horizon_scan(px, D[MAIN.values], [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "long SPY, episode level, lag=1")
print("  NOTE: 6-point horizon grid walked -> x6 multiplicity on any best-h claim.")

H, GAP = 10, 10

# ------------------------------------------------------------- 3. BATTERY
battery(px, MAIN, [("SPY", 1.0)], H,
        "C7 long SPY on a VIX pop out of a compressed 21d range",
        cost_bps=3.0, min_gap=GAP,
        variants={
            "range pctile <= 5 (tighter)": (POP & (rng_pct <= 5)).fillna(False),
            "range pctile <= 30 (looser)": (POP & (rng_pct <= 30)).fillna(False),
            "pop >= 5% instead of 8%": ((r_vix >= 0.05) & (r_spy > -0.0125) & (r_spy < 0) & COMP).fillna(False),
            "pop >= 12%": ((r_vix >= 0.12) & (r_spy > -0.0125) & (r_spy < 0) & COMP).fillna(False),
            "SPY down < 0.75% (w12's clause)": (POP & COMP & (r_spy > -0.0075)).fillna(False),
            "SPY UP on the day (sign flip)": ((r_vix >= 0.08) & (r_spy >= 0) & COMP).fillna(False),
        }, event_kinds=("nfp",))

ret = vehicle_ret(px, [("SPY", 1.0)], H)
valid = ret.dropna().index


def ep(m, gap=GAP):
    t = D[pd.Series(m).reindex(D, fill_value=False).values].intersection(valid)
    return declusters(t, gap, valid)


EPI = ep(MAIN)
VALS = ret.loc[EPI].values

# ------------------------------------- 4. THE INCREMENT TEST (watchlist 12's)
print("\n########## 4. INCREMENT TESTS -- Welch t of the DIFFERENCE ##########")
print("  (scoring against zero is what made watchlist 12 look strong; every row")
print("   below is C7 MINUS a control that already contains part of the story)")
for h in (3, 5, 10):
    r_ = vehicle_ret(px, [("SPY", 1.0)], h)
    v_ = r_.dropna().index
    g = max(h, 5)

    def e_(m):
        return declusters(D[pd.Series(m).reindex(D, fill_value=False).values]
                          .intersection(v_), g, v_)
    d = r_.loc[e_(MAIN)].values
    comp_only = r_.loc[e_(COMP.fillna(False))].values           # regime control
    pop_only = r_.loc[e_(POP.fillna(False))].values             # pop, no range cond
    w12v = r_.loc[e_(W12)].values
    loc = r_.loc[local_control(v_, D[MAIN.values].intersection(v_))].values
    allv = r_.loc[v_].values
    print(f"\n  h={h}  C7 N={len(d)} mean {100*d.mean():+.3f}% "
          f"(t vs 0 = {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f})")
    for lbl, c in [("compressed RANGE alone (its own regime control)", comp_only),
                   ("the POP alone, no range condition", pop_only),
                   ("local +/-126td ex-trigger", loc),
                   ("all days", allv),
                   ("watchlist-12 cell", w12v)]:
        print(f"     vs {lbl:46s} N={len(c):5d}  diff "
              f"{100*(d.mean()-c.mean()):+7.3f}pp  welch t {welch(d, c):+6.2f}")

# ------------------------------------------------- 5. DOSE ON RANGE PERCENTILE
print("\n########## 5. DOSE RESPONSE on the range percentile (today = 8.3) ##########")
rows = []
for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 50), (50, 101)]:
    m = (POP & (rng_pct >= lo) & (rng_pct < hi)).fillna(False)
    e = ep(m)
    r = summarize(ret.loc[e].values, f"range pctile [{lo},{hi})") if len(e) else {"label": f"[{lo},{hi})", "n": 0}
    if r.get("n"):
        r["n_days"] = int(m.sum())
        r["edge_pct"] = round(r["mean_pct"] - 100 * ret.loc[valid].mean(), 3)
    rows.append(r)
show(rows, f"POP conditioned on range-percentile bucket, h={H}")
print("  today sits in [5,15) -- the INTERIOR bucket, not the extreme one. "
      "A ladder that is flat or non-monotone with the live value interior is "
      "the 'mid-range wearing an extremity label' shape.")

print("\n########## 6. DOSE on pop size and on SPY's own move (today +9.52% / -0.69%) ##########")
rows = []
for lo, hi, lbl in [(0.05, 0.08, "VIX pop [5,8)%"), (0.08, 0.12, "VIX pop [8,12)%"),
                    (0.12, 9.9, "VIX pop 12%+")]:
    m = ((r_vix >= lo) & (r_vix < hi) & (r_spy > -0.0125) & (r_spy < 0) & COMP).fillna(False)
    e = ep(m)
    r = summarize(ret.loc[e].values, lbl) if len(e) else {"label": lbl, "n": 0}
    if r.get("n"):
        r["n_days"] = int(m.sum())
    rows.append(r)
show(rows, "pop-size ladder (range compressed)")
rows = []
for lo, hi, lbl in [(-0.0125, -0.0075, "SPY [-1.25,-0.75)%"),
                    (-0.0075, -0.0035, "SPY [-0.75,-0.35)%  <-- today -0.69"),
                    (-0.0035, 0.0, "SPY [-0.35,0)%"),
                    (0.0, 0.02, "SPY UP (sign flip)")]:
    m = ((r_vix >= 0.08) & (r_spy >= lo) & (r_spy < hi) & COMP).fillna(False)
    e = ep(m)
    r = summarize(ret.loc[e].values, lbl) if len(e) else {"label": lbl, "n": 0}
    if r.get("n"):
        r["n_days"] = int(m.sum())
    rows.append(r)
show(rows, "SPY-move ladder (VIX +8%+, range compressed)")

# ------------------------------------------------------- 7. ERA / 2020 / CONC
print("\n########## 7. ERA, 2020 SENSITIVITY, CONCENTRATION ##########")
print("  " + cluster_note(EPI, VALS, k=2))
yrs = pd.DatetimeIndex(EPI).year
by = pd.Series(VALS).groupby(yrs.values).agg(["size", "mean", "sum"])
print((by * pd.Series({"size": 1, "mean": 100, "sum": 100})).round(2).to_string())
for lbl, m in [("ex-2020", yrs != 2020), ("ex-2008/09", ~np.isin(yrs, [2008, 2009])),
               ("ex-2020 and ex-2008/09", (yrs != 2020) & ~np.isin(yrs, [2008, 2009]))]:
    v = VALS[m]
    w = int((v > 0).sum())
    print(f"  {lbl:24s} N={len(v):3d} mean {100*v.mean():+.3f}%  record {w}-{len(v)-w}  "
          f"sign p {sign_test(w, len(v)):.4f}  edge vs all-days "
          f"{100*(v.mean()-ret.loc[valid].mean()):+.3f}pp  "
          f"bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
show(era_split(EPI, VALS), "era split (episodes)")

# ------------------------------------------------------------- 8. DIAL / NFP
print("\n########## 8. FRAGILITY DIAL ##########")
dl = dial_series()
dv = dl.reindex(EPI).dropna()
print(f"  live ma10-63d dial {dl.iloc[-1]:.1f}; episodes with a reading {len(dv)} "
      f"of {len(EPI)} (series starts 2016)")
if len(dv):
    print("  " + ", ".join(f"{str(k.date())}={v:.0f}" for k, v in dv.items()))
    print(f"  MAX episode dial {dv.max():.1f} vs today {dl.iloc[-1]:.1f} -> today "
          f"{'INSIDE' if dl.iloc[-1] <= dv.max() else 'OUTSIDE'} the population")
    hi = dv[dv >= 60]
    if len(hi):
        hv = ret.loc[hi.index].values
        print(f"  episodes at dial >= 60: N={len(hv)} mean {100*hv.mean():+.3f}% "
              f"({', '.join(str(d.date()) for d in hi.index)})")

print("\n########## 9. NFP INSIDE THE HOLD (today NFP is +2 td) ##########")
for h in (3, 5, 10):
    r_ = vehicle_ret(px, [("SPY", 1.0)], h)
    v_ = r_.dropna().index
    e_ = declusters(D[MAIN.values].intersection(v_), max(h, 5), v_)
    fl = event_in_window(e_, D, h, 1, ("nfp",))
    vv = r_.loc[e_].values
    print(f"  h={h}: NFP in hold N={int(fl.sum())} mean "
          f"{100*vv[fl].mean() if fl.sum() else float('nan'):+.3f}%  |  "
          f"NFP out N={int((~fl).sum())} mean {100*vv[~fl].mean():+.3f}%  "
          f"welch t of the diff {welch(vv[fl], vv[~fl]):+.2f}")
# today's specific geometry: NFP lands exactly 1 session after the entry close
pos = pd.Series(range(len(D)), index=D)
gap_rows = []
nfp = load_events(["nfp"])["date"]
for dte in EPI:
    p = pos[dte]
    nxt = nfp[nfp > dte]
    if len(nxt) == 0:
        continue
    q = int(D.searchsorted(nxt.iloc[0]))
    if q < len(D):
        gap_rows.append((dte, q - p))
g2 = [d for d, k in gap_rows if k == 2]
print(f"  episodes whose NEXT NFP is exactly +2 td (today's geometry): "
      f"{len(g2)} of {len(gap_rows)}: {[str(d.date()) for d in g2]}")
if g2:
    vv = ret.loc[pd.DatetimeIndex(g2)].values
    print(f"    their h={H} mean {100*vv.mean():+.3f}% "
          f"record {int((vv>0).sum())}-{int((vv<=0).sum())}")

# ----------------------------------------------------------- 10. VEHICLES
print("\n########## 10. VEHICLES as WHOLE variants (SPY vs SVXY) ##########")
sv = px["SVXY"].dropna()
print(f"  SVXY history starts {sv.index[0].date()} -- pre-2011 episodes are "
      "unmeasurable in it, so this is a DIFFERENT (shorter) sample, not a "
      "translation of the same one.")
rows = []
for lbl, legs, cost in [("SPY long", [("SPY", 1.0)], 3.0),
                        ("SVXY long (short vol)", [("SVXY", 1.0)], 20.0)]:
    r_ = vehicle_ret(px, legs, H)
    v_ = r_.dropna().index
    e_ = declusters(D[MAIN.values].intersection(v_), GAP, v_)
    r = summarize(r_.loc[e_].values, lbl)
    if r["n"]:
        r["drift_pct"] = round(100 * r_.loc[v_].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * r_.loc[v_].mean(), 3)
        r["x_cost"] = round(100 * r["mean_pct"] / cost, 1)
    rows.append(r)
# SPY over the SVXY-era subsample so the comparison is like-for-like
r_ = vehicle_ret(px, [("SPY", 1.0)], H)
v_ = r_.dropna().index
e_ = declusters(D[MAIN.values].intersection(v_).intersection(sv.index), GAP, v_)
r = summarize(r_.loc[e_].values, "SPY long, SVXY-era subsample")
r["drift_pct"] = round(100 * r_.loc[v_.intersection(sv.index)].mean(), 3)
rows.append(r)
show(rows, f"vehicle comparison, h={H}")
print("  SVXY decay: its own all-days drift over the same era is printed as "
      "drift_pct above; the edge column already nets it out.")
