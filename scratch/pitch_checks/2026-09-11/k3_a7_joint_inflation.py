"""A7 KILL CHECK - the joint inflation repricing.

Pre-specified state: DBC within 0.5% of its trailing-252 HIGH **and** IEF
within 1.0% of its trailing-252 LOW **and** LQD within 1.0% of its trailing-252
LOW, on the SAME close. Direction and vehicle are NOT pre-specified: measure
SPY, IWM, XLE, GLD, TLT and IEF forward and see which carries content.

LIVE 2026-09-10: DBC 0.000% off its high, USO 0.000% off, TLT/IEF/LQD all
0.000% above their 252-day lows.

Mandatory attacks (the registry names all three in advance):
  1. the EX-INFLATION-SHOCK-YEARS split (2007/2008/2021/2022) -- the registry
     says those four years held 26 of 53 episodes and MORE than 100% of the
     total on the commodity-high parent
  2. BOTH single-gate baselines: the commodity gate ALONE and the IG gate
     ALONE. Does the JOINT gate add anything over either?
  3. the discarded complements of each gate
  4. does the event leg sneak back in through the episode dates?
  5. concentration, era, cost, local control
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

TK = ["SPY", "IWM", "XLE", "GLD", "TLT", "IEF", "LQD", "DBC", "USO", "HYG"]
px = close_panel(TK)
cal = px["SPY"].dropna().index
px = px.reindex(cal)
ASOF = pd.Timestamp("2026-09-10")
SHOCK = {2007, 2008, 2021, 2022}


def near_high(t, pct):
    s = px[t]
    mx = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return ((mx - s) / mx * 100 <= pct) & mx.notna()


def near_low(t, pct):
    s = px[t]
    mn = rolling_on_valid(s, lambda x: x.rolling(252).min())
    return ((s - mn) / mn * 100 <= pct) & mn.notna()


dbc_hi = near_high("DBC", 0.5)
ief_lo = near_low("IEF", 1.0)
lqd_lo = near_low("LQD", 1.0)
ig_lo = ief_lo & lqd_lo
joint = dbc_hi & ig_lo

print("=" * 78)
print("0. SANITY + LIVE STATE")
for t in ["DBC", "IEF", "LQD", "USO", "TLT"]:
    s = px[t]
    mx = rolling_on_valid(s, lambda x: x.rolling(252).max())
    mn = rolling_on_valid(s, lambda x: x.rolling(252).min())
    print(f"  {t:<5} close {s.loc[ASOF]:8.3f}  "
          f"{100*(mx.loc[ASOF]-s.loc[ASOF])/mx.loc[ASOF]:6.3f}% below 252d high"
          f"   {100*(s.loc[ASOF]-mn.loc[ASOF])/mn.loc[ASOF]:6.3f}% above low")
print(f"  DBC first bar {px['DBC'].dropna().index[0].date()} -> the sample is "
      f"2006+, not 2000+")
print(f"  live state: dbc_hi={bool(dbc_hi.loc[ASOF])} "
      f"ief_lo={bool(ief_lo.loc[ASOF])} lqd_lo={bool(lqd_lo.loc[ASOF])} "
      f"JOINT={bool(joint.loc[ASOF])}")
for lbl, m in [("DBC hi alone", dbc_hi), ("IG lo alone (IEF&LQD)", ig_lo),
               ("JOINT", joint)]:
    d = cal[m.reindex(cal, fill_value=False).values]
    print(f"  {lbl:<24} {len(d):>5} days  "
          f"{d[0].date() if len(d) else '-'} .. "
          f"{d[-1].date() if len(d) else '-'}   years "
          f"{sorted(set(d.year)) if len(d) else []}")

VEH = ["SPY", "IWM", "XLE", "GLD", "TLT", "IEF"]
HS = (1, 2, 3, 5, 10)

# ------------------------------------------------------------ 1. the sweep
print("\n" + "=" * 78)
print("1. WHICH VEHICLE, IF ANY, CARRIES CONTENT ON THE JOINT STATE?")
print("   (LONG side shown; the short is the negative. edge = vs own drift)")
sig = cal[joint.reindex(cal, fill_value=False).values]
rows = []
for t in VEH:
    for h in HS:
        r = vehicle_ret(px, [(t, 1.0)], h, 1)
        b = r.dropna()
        epi = declusters(sig.intersection(b.index), h, cal)
        if len(epi) == 0:
            continue
        v = r.loc[epi].values
        w = int((v > 0).sum())
        rows.append({"veh": t, "h": h, "N_epi": len(epi),
                     "long_pct": round(100 * v.mean(), 3),
                     "drift_pct": round(100 * b.mean(), 3),
                     "edge_pp": round(100 * (v.mean() - b.mean()), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "signp_long": round(sign_test(w, len(v)), 4),
                     "signp_short": round(sign_test(len(v) - w, len(v)), 4)})
S = pd.DataFrame(rows)
print(S.to_string(index=False))
print("\n  biggest |edge_pp| cells (the whole 6x5 = 30 cell grid was walked, so")
print("  family-wise |t| for p=0.05 at 30 cells is ~3.2):")
print(S.reindex(S["edge_pp"].abs().sort_values(ascending=False).index)
      .head(6).to_string(index=False))

# -------------------------------------------- 2. single-gate baselines
print("\n" + "=" * 78)
print("2. BOTH SINGLE-GATE BASELINES + THE DISCARDED COMPLEMENTS")
print("   Does the JOINT gate add anything over EITHER half alone?")
for h in (3, 5, 10):
    print(f"\n  --- h={h} ---")
    rows = []
    for t in VEH:
        r = vehicle_ret(px, [(t, 1.0)], h, 1)
        b = r.dropna()
        out = {"veh": t, "drift": round(100 * b.mean(), 3)}
        for lbl, m in [("DBChi", dbc_hi), ("IGlo", ig_lo), ("JOINT", joint),
                       ("DBChi_notIGlo", dbc_hi & ~ig_lo),
                       ("IGlo_notDBChi", ig_lo & ~dbc_hi)]:
            d = cal[m.reindex(cal, fill_value=False).values]
            epi = declusters(d.intersection(b.index), h, cal)
            out[lbl] = round(100 * r.loc[epi].mean(), 3) if len(epi) else np.nan
            out[lbl + "_n"] = len(epi)
        rows.append(out)
    print(pd.DataFrame(rows).to_string(index=False))
print("\n  READ: JOINT must beat BOTH DBChi and IGlo, and the two COMPLEMENTS")
print("  must be the worse halves, or the conjunction is a label.")

# ------------------------------------------- 3. ex-shock-years (the kill)
print("\n" + "=" * 78)
print("3. THE EX-INFLATION-SHOCK-YEARS SPLIT (drop 2007, 2008, 2021, 2022)")
for h in (3, 5, 10):
    rows = []
    for t in VEH:
        r = vehicle_ret(px, [(t, 1.0)], h, 1)
        b = r.dropna()
        epi = declusters(sig.intersection(b.index), h, cal)
        if len(epi) == 0:
            continue
        v = r.loc[epi].values
        sm = np.asarray([y in SHOCK for y in epi.year])
        rows.append({
            "veh": t, "h": h, "N_all": len(v),
            "all_pct": round(100 * v.mean(), 3),
            "N_shock": int(sm.sum()),
            "shock_pct": round(100 * v[sm].mean(), 3) if sm.any() else np.nan,
            "N_exshock": int((~sm).sum()),
            "exshock_pct": (round(100 * v[~sm].mean(), 3)
                            if (~sm).any() else np.nan),
            "exshock_hit": (round(100 * (v[~sm] > 0).mean(), 1)
                            if (~sm).any() else np.nan)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
print("  episode year histogram (h=5):")
r5 = vehicle_ret(px, [("SPY", 1.0)], 5, 1)
epi5 = declusters(sig.intersection(r5.dropna().index), 5, cal)
print("   ", dict(pd.Series(epi5.year).value_counts().sort_index()))
print(f"    shock-year share of episodes: "
      f"{100*np.mean([y in SHOCK for y in epi5.year]):.1f}%")

# ------------------------------------------------- 4. the event leg check
print("\n" + "=" * 78)
print("4. DOES THE EVENT LEG SNEAK BACK IN? (no event leg by design)")
for kinds in [("cpi",), ("ppi",), ("cpi", "ppi"), ("fomc_decision",)]:
    fl = event_in_window(epi5, cal, 5, 1, kinds)
    print(f"  {'+'.join(kinds):<16} in-window on "
          f"{int(fl.sum())}/{len(fl)} h=5 episodes "
          f"({100*fl.mean():.1f}%)")
print("  baseline: an arbitrary 5-session window contains a CPI or PPI print")
allep = declusters(cal[r5.notna().values], 5, cal)
flb = event_in_window(allep, cal, 5, 1, ("cpi", "ppi"))
print(f"  on {100*flb.mean():.1f}% of all 5-session windows "
      f"(N={len(flb)}), so the joint state's "
      f"{100*event_in_window(epi5, cal, 5, 1, ('cpi','ppi')).mean():.1f}% is "
      f"the comparison.")

# --------------------------------------------------------- 5. the battery
print("\n" + "=" * 78)
print("5. FULL BATTERY on the two best-looking cells from step 1")
best = S.reindex(S["edge_pp"].abs().sort_values(ascending=False).index).head(2)
for _, row in best.iterrows():
    t, h = row["veh"], int(row["h"])
    sgn = 1.0 if row["edge_pp"] > 0 else -1.0
    battery(px, joint, [(t, sgn)], h,
            f"A7 {'LONG' if sgn > 0 else 'SHORT'} {t} on the joint state",
            cost_bps=3.0,
            variants={
                "DBC hi ALONE": dbc_hi,
                "IG lo ALONE": ig_lo,
                "loose: DBC<=1.5%, IG<=2.0%":
                    near_high("DBC", 1.5) & near_low("IEF", 2.0)
                    & near_low("LQD", 2.0),
                "tight: DBC<=0.1%, IG<=0.25%":
                    near_high("DBC", 0.1) & near_low("IEF", 0.25)
                    & near_low("LQD", 0.25),
            },
            event_kinds=("cpi", "ppi"))

print("\nDONE")
