"""A10 KILL CHECK - EWZ against EEM on a 21-day rank thrust.

Pre-specified: EWZ 21-day PIT rank >= 90 AND EEM 21-day PIT rank < 60 on the
same close. Legs EWZ +1.0 / EEM -1.0 (plus the beta-neutral form). lag=1 MOC.
Direction and horizon FROM THE MEASUREMENT -- report the sign honestly.

LIVE 2026-09-10: EWZ +13.48%/21d at rank 90.1, z10 +1.84; EEM +2.40%/21d at
rank 44.8.

Registry adjacency the candidate names: "EWZ against FXI is one leg, and it is
not the named one" (2026-09-09) and "EWZ is EEM with a Brazil label on print
days" (63% of daily variance is EEM at beta 1.056). So:
  1. BOTH LEGS SEPARATELY, first thing
  2. measured beta + R^2 + the beta-neutral residual
  3. a 9-vehicle country reference class with each as the LONG leg
  4. gate attribution: the EEM<60 half is the "decoupling" claim -- run the
     idea WITHOUT it and price the discarded complement
  5. the usual battery, era, concentration, cost
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

TK = ["EWZ", "EEM", "EFA", "FXI", "EWW", "EWY", "EWT", "EWJ", "INDA", "KWEB",
      "SPY"]
px = close_panel(TK)
cal = px["EWZ"].dropna().index          # EWZ's own calendar drives the cell
px = px.reindex(cal)
ASOF = pd.Timestamp("2026-09-10")
COST = 6.0   # EWZ + EEM MOC half-spreads + ~1%/yr EEM borrow on a week

r21_z = pct_rank(px["EWZ"], 21)
r21_e = pct_rank(px["EEM"], 21)

print("=" * 78)
print("0. LIVE STATE + BOTH z10 CONVENTIONS (registry 2026-09-03 trap)")
print(f"  EWZ 21d rank {r21_z.loc[ASOF]:.1f}   EEM 21d rank {r21_e.loc[ASOF]:.1f}")
print(f"  EWZ 21d ret {100*(px['EWZ'].loc[ASOF]/px['EWZ'].shift(21).loc[ASOF]-1):+.2f}%"
      f"   EEM {100*(px['EEM'].loc[ASOF]/px['EEM'].shift(21).loc[ASOF]-1):+.2f}%")
print(f"  pitch_lab.zscore EWZ z10 = {zscore(px['EWZ'], 10).loc[ASOF]:+.2f} "
      f"(the tape's _metrics_for convention reads +1.84)")
print(f"  EWZ first bar {px['EWZ'].dropna().index[0].date()}, "
      f"EEM first bar {px['EEM'].dropna().index[0].date()} -> sample is 2003+")

mask = (r21_z >= 90) & (r21_e < 60)
sig = cal[mask.reindex(cal, fill_value=False).values]
print(f"  trigger days: {len(sig)}  "
      f"{sig[0].date() if len(sig) else '-'} .. "
      f"{sig[-1].date() if len(sig) else '-'}")
print(f"  live 2026-09-10 fires: {bool(mask.loc[ASOF])}")

# --------------------------------------------------------------- 1. legs
print("\n" + "=" * 78)
print("1. BOTH LEGS SEPARATELY, BEFORE THE SPREAD")
rows = []
for h in (1, 2, 3, 5, 10):
    rZ = vehicle_ret(px, [("EWZ", 1.0)], h, 1)
    rE = vehicle_ret(px, [("EEM", 1.0)], h, 1)
    rP = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -1.0)], h, 1)
    epi = declusters(sig.intersection(rP.dropna().index), h, cal)
    rows.append({
        "h": h, "N_epi": len(epi),
        "EWZ_long": round(100 * rZ.loc[epi].mean(), 3),
        "EWZ_drift": round(100 * rZ.dropna().mean(), 3),
        "EWZ_edge": round(100 * (rZ.loc[epi].mean() - rZ.dropna().mean()), 3),
        "EEM_long": round(100 * rE.loc[epi].mean(), 3),
        "EEM_drift": round(100 * rE.dropna().mean(), 3),
        "EEM_edge": round(100 * (rE.loc[epi].mean() - rE.dropna().mean()), 3),
        "PAIR": round(100 * rP.loc[epi].mean(), 3),
        "PAIR_drift": round(100 * rP.dropna().mean(), 3),
        "PAIR_edge": round(100 * (rP.loc[epi].mean() - rP.dropna().mean()), 3),
        "pair_hit": round(100 * (rP.loc[epi] > 0).mean(), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------- 2. beta
print("\n" + "=" * 78)
print("2. BETA, R^2 AND THE BETA-NEUTRAL RESIDUAL")
dZ, dE = px["EWZ"].pct_change(), px["EEM"].pct_change()
ok = dZ.notna() & dE.notna()
b, a = np.polyfit(dE[ok], dZ[ok], 1)
resid = dZ[ok] - (a + b * dE[ok])
r2 = 1 - resid.var() / dZ[ok].var()
print(f"  beta(EWZ on EEM) = {b:.3f}   R^2 = {r2:.3f}   "
      f"(registry 2026-09-03: beta 1.056, 63% of variance)")
print(f"  EWZ daily sd {100*dZ[ok].std():.3f}% vs EEM {100*dE[ok].std():.3f}% "
      f"= {dZ[ok].std()/dE[ok].std():.2f}x  -> the 1:1 pair is "
      f"{100*(dZ[ok].std()/dE[ok].std()-1):.0f}% net long EWZ vol")
rows = []
for h in (1, 2, 3, 5, 10):
    rP = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -1.0)], h, 1)
    rN = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -b)], h, 1)
    epi = declusters(sig.intersection(rP.dropna().index), h, cal)
    vN = rN.loc[epi].values
    w = int((vN > 0).sum())
    rows.append({"h": h, "N": len(epi),
                 "pair_1to1": round(100 * rP.loc[epi].mean(), 3),
                 "beta_neutral": round(100 * vN.mean(), 3),
                 "bn_drift": round(100 * rN.dropna().mean(), 3),
                 "bn_edge": round(100 * (vN.mean() - rN.dropna().mean()), 3),
                 "bn_hit": round(100 * (vN > 0).mean(), 1),
                 "bn_signp_long": round(sign_test(w, len(vN)), 4),
                 "bn_signp_short": round(sign_test(len(vN) - w, len(vN)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------- 3. reference class
print("\n" + "=" * 78)
print("3. REFERENCE CLASS: swap the LONG leg for each other country/region")
print("   (same EEM<60 gate, same rank>=90 thrust on the swapped name)")
H = 5
CLASS = ["EWZ", "EWW", "EWY", "EWT", "EWJ", "FXI", "EFA", "INDA", "KWEB"]
rows = []
for t in CLASS:
    rk = pct_rank(px[t], 21)
    m = (rk >= 90) & (r21_e < 60)
    d = cal[m.reindex(cal, fill_value=False).values]
    rP = vehicle_ret(px, [(t, 1.0), ("EEM", -1.0)], H, 1)
    epi = declusters(d.intersection(rP.dropna().index), H, cal)
    if len(epi) < 3:
        rows.append({"long_leg": t, "N": len(epi), "pair_pct": np.nan})
        continue
    v = rP.loc[epi].values
    rows.append({"long_leg": t, "N": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "first_bar": str(px[t].dropna().index[0].date())})
R = pd.DataFrame(rows).sort_values("pair_pct", ascending=False)
print(R.to_string(index=False))
if not R["pair_pct"].isna().all():
    rk_z = int(list(R["long_leg"]).index("EWZ")) + 1
    print(f"  EWZ RANKS {rk_z} OF {R['pair_pct'].notna().sum()} scored members "
          f"as the long leg (max-of-K charge applies).")

# ------------------------------------------------- 4. gate attribution
print("\n" + "=" * 78)
print("4. GATE ATTRIBUTION: is the EEM<60 'decoupling' half doing anything?")
for h in (3, 5, 10):
    rP = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -1.0)], h, 1)
    rZ = vehicle_ret(px, [("EWZ", 1.0)], h, 1)
    out = []
    for lbl, m in [("ungated EWZ r21>=90", (r21_z >= 90)),
                   ("GATED (EEM<60)", (r21_z >= 90) & (r21_e < 60)),
                   ("COMPLEMENT (EEM>=60)", (r21_z >= 90) & (r21_e >= 60)),
                   ("EEM<60 alone", (r21_e < 60))]:
        d = cal[m.reindex(cal, fill_value=False).values]
        epi = declusters(d.intersection(rP.dropna().index), h, cal)
        out.append({"cell": lbl, "N": len(epi),
                    "pair_pct": round(100 * rP.loc[epi].mean(), 3),
                    "EWZ_outright_pct": round(100 * rZ.loc[epi].mean(), 3),
                    "hit": round(100 * (rP.loc[epi] > 0).mean(), 1)})
    print(f"\n  --- h={h} ---")
    print(pd.DataFrame(out).to_string(index=False))
print("\n  READ: if the COMPLEMENT pays as much as the GATED cell, the gate is")
print("  an anti-filter and the 'Brazil leading EM' story is decoration.")

# ------------------------------------------------- 5. threshold neighbours
print("\n" + "=" * 78)
print("5. DEFINITION NEIGHBOURS (rank thresholds, ranking window)")
rows = []
for zt in (85, 90, 95):
    for et in (50, 60, 70):
        m = (r21_z >= zt) & (r21_e < et)
        d = cal[m.reindex(cal, fill_value=False).values]
        rP = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -1.0)], H, 1)
        epi = declusters(d.intersection(rP.dropna().index), H, cal)
        if len(epi) < 3:
            continue
        v = rP.loc[epi].values
        rows.append({"EWZ>=": zt, "EEM<": et, "N": len(epi),
                     "pair_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "LIVE": "<==" if (zt, et) == (90, 60) else ""})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  ranking-window neighbours (EWZ rank window n, EEM same n):")
rows = []
for n in (10, 21, 42, 63):
    m = (pct_rank(px["EWZ"], n) >= 90) & (pct_rank(px["EEM"], n) < 60)
    d = cal[m.reindex(cal, fill_value=False).values]
    rP = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -1.0)], H, 1)
    epi = declusters(d.intersection(rP.dropna().index), H, cal)
    if len(epi) < 3:
        continue
    v = rP.loc[epi].values
    rows.append({"rank_window": n, "N": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "LIVE": "<==" if n == 21 else ""})
print(pd.DataFrame(rows).to_string(index=False))

# ----------------------------------------------------------- 6. battery
print("\n" + "=" * 78)
print("6. FULL BATTERY on the 1:1 pair at h=5 and on the EWZ leg alone")
battery(px, mask, [("EWZ", 1.0), ("EEM", -1.0)], 5,
        "A10 EWZ +1 / EEM -1 on the thrust gate", cost_bps=3.0,
        variants={"EWZ>=95 / EEM<60": (r21_z >= 95) & (r21_e < 60),
                  "EWZ>=90 ungated": (r21_z >= 90),
                  "EWZ>=90 / EEM<40": (r21_z >= 90) & (r21_e < 40)},
        event_kinds=("cpi",))
battery(px, mask, [("EWZ", 1.0)], 5,
        "A10 EWZ OUTRIGHT LONG on the same gate", cost_bps=3.0,
        event_kinds=("cpi",))

print("\nDONE")
