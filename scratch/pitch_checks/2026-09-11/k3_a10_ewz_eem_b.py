"""A10 round 2 - the pair survived round 1 (h=5 +0.954% over 20 episodes,
70% hit, 14-6, sign p 0.0577, 15.9x cost, top-2 episodes only 4% of total).
Round 2 is mandatory. Attacks:

  1. decluster + concentration BY YEAR (episode-level concentration was low;
     year-level was not -- cluster_note flagged 2022 +12.6 and 2025 +8.3 of a
     +19.09pp total) + leave-one-year-out
  2. definition neighbours. Round 1 already found the ranking window is a
     KNIFE EDGE (n=10 +0.100, n=21 +0.954, n=42 +0.049, n=63 +0.136). Press
     it: window ladder on the pair AND on the beta-neutral residual, rank vs
     raw-return form, and an entry-offset persistence ladder
  3. era / cycle / regime: the cell has NO pre-2015 instances at all
  4. gate attribution across the EEM rank continuum (is it EWZ thrusting or
     EEM not thrusting?)
  5. dose: today's EWZ 21d return vs the trigger distribution (the
     "rank gates in a quiet tape buy a fraction of the force" trap)
  6. tomorrow-specific: FOMC 2026-09-16 (+3 td) and quad witching 09-18 (+5)
     both sit INSIDE an h=5 hold
  7. reference-class max-of-K permutation (EWZ ranked 1 of 9)
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
cal = px["EWZ"].dropna().index
px = px.reindex(cal)
ASOF = pd.Timestamp("2026-09-10")
H = 5
LEGS = [("EWZ", 1.0), ("EEM", -1.0)]
r21_z, r21_e = pct_rank(px["EWZ"], 21), pct_rank(px["EEM"], 21)
mask = (r21_z >= 90) & (r21_e < 60)
sig = cal[mask.reindex(cal, fill_value=False).values]
rP = vehicle_ret(px, LEGS, H, 1)
epi = declusters(sig.intersection(rP.dropna().index), H, cal)
v = rP.loc[epi].values

print("=" * 78)
print("1. CONCENTRATION BY YEAR AND LEAVE-ONE-YEAR-OUT")
tab = pd.DataFrame({"date": [str(d.date()) for d in epi],
                    "year": epi.year, "pct": np.round(100 * v, 2)})
print(tab.to_string(index=False))
byy = tab.groupby("year")["pct"].agg(["count", "sum", "mean"])
print("\n  by year:")
print(byy.round(2).to_string())
print(f"  TOTAL {tab['pct'].sum():+.2f}pp over {len(tab)} episodes")
print("\n  leave-one-YEAR-out (mean of the remainder):")
for y in sorted(set(epi.year)):
    m = epi.year != y
    w = int((v[m] > 0).sum())
    print(f"    drop {y}: N={int(m.sum()):>3}  mean {100*v[m].mean():+.3f}%  "
          f"record {w}-{int(m.sum())-w}  sign p "
          f"{sign_test(w, int(m.sum())):.4f}")
o = np.sort(v)[::-1]
print(f"  drop-best-1 {100*o[1:].mean():+.3f}%   "
      f"drop-best-2 {100*o[2:].mean():+.3f}%   "
      f"drop-best-3 {100*o[3:].mean():+.3f}%")
for gap in (5, 21, 42, 63, 126):
    e = declusters(sig.intersection(rP.dropna().index), gap, cal)
    vv = rP.loc[e].values
    w = int((vv > 0).sum())
    print(f"  gap={gap:>4} td: N={len(vv):>3}  {100*vv.mean():+.3f}%  "
          f"record {w}-{len(vv)-w}  sign p {sign_test(w, len(vv)):.4f}")

print("\n" + "=" * 78)
print("2. DEFINITION NEIGHBOURS -- PRESS THE RANKING WINDOW")
dZ, dE = px["EWZ"].pct_change(), px["EEM"].pct_change()
ok = dZ.notna() & dE.notna()
beta = float(np.polyfit(dE[ok], dZ[ok], 1)[0])
rN = vehicle_ret(px, [("EWZ", 1.0), ("EEM", -beta)], H, 1)
rows = []
for n in (5, 10, 15, 21, 26, 31, 42, 63):
    m = (pct_rank(px["EWZ"], n) >= 90) & (pct_rank(px["EEM"], n) < 60)
    d = cal[m.reindex(cal, fill_value=False).values]
    e = declusters(d.intersection(rP.dropna().index), H, cal)
    if len(e) < 3:
        rows.append({"window": n, "N": len(e)})
        continue
    vv, vn = rP.loc[e].values, rN.loc[e].values
    w = int((vv > 0).sum())
    rows.append({"window": n, "N": len(e),
                 "pair_pct": round(100 * vv.mean(), 3),
                 "betaN_pct": round(100 * vn.mean(), 3),
                 "hit": round(100 * (vv > 0).mean(), 1),
                 "signp": round(sign_test(w, len(vv)), 4),
                 "LIVE": "<==" if n == 21 else ""})
print(pd.DataFrame(rows).to_string(index=False))
print("  If only n=21 pays, the 21-day window is a fitted parameter and the")
print("  'Brazil leading EM' mechanism is not what the number is measuring.")

print("\n  RAW-RETURN form instead of ranks (EWZ 21d ret >= X, EEM 21d < Y):")
z21 = px["EWZ"] / px["EWZ"].shift(21) - 1
e21 = px["EEM"] / px["EEM"].shift(21) - 1
rows = []
for zt, et in [(0.08, 0.05), (0.10, 0.05), (0.13, 0.05), (0.10, 0.03),
               (0.13, 0.03)]:
    m = (z21 >= zt) & (e21 < et)
    d = cal[m.reindex(cal, fill_value=False).values]
    e = declusters(d.intersection(rP.dropna().index), H, cal)
    if len(e) < 3:
        continue
    vv = rP.loc[e].values
    w = int((vv > 0).sum())
    rows.append({"EWZ21>=": zt, "EEM21<": et, "N": len(e),
                 "pair_pct": round(100 * vv.mean(), 3),
                 "hit": round(100 * (vv > 0).mean(), 1),
                 "signp": round(sign_test(w, len(vv)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n  ENTRY-OFFSET PERSISTENCE LADDER (enter j sessions later, same h=5):")
pos = pd.Series(range(len(cal)), index=cal)
rows = []
for j in range(0, 6):
    rj = vehicle_ret(px, LEGS, H, 1 + j)
    e = declusters(sig.intersection(rj.dropna().index), H, cal)
    vv = rj.loc[e].values
    rows.append({"entry_lag": 1 + j, "N": len(vv),
                 "pair_pct": round(100 * vv.mean(), 3),
                 "hit": round(100 * (vv > 0).mean(), 1),
                 "LIVE": "<==" if j == 0 else ""})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("3. ERA / CYCLE / REGIME")
show(era_split(epi, v), "pre-2018 vs 2018+")
mid = (epi.year % 4 == 2)
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "cycle")
print(f"  FIRST trigger day in 23 years of EWZ/EEM overlap: {sig[0].date()}")
print("  -> the cell has NO pre-2015 instances. It is not that it died after")
print("     2018; it did not EXIST before 2015, because EWZ and EEM used to")
print("     thrust together. Check that directly:")
co = dZ[ok].rolling(252).corr(dE[ok])
for a, b_ in [("2003-2014", ("2003-01-01", "2014-12-31")),
              ("2015-2026", ("2015-01-01", "2026-12-31"))]:
    s = co.loc[b_[0]:b_[1]].mean()
    n_hi = int(((r21_z >= 90) & mask.notna()).loc[b_[0]:b_[1]].sum())
    n_fire = int(mask.loc[b_[0]:b_[1]].sum())
    print(f"    {a}: mean trailing-252 corr(EWZ,EEM) {s:.3f}   "
          f"EWZ r21>=90 days {n_hi}   of which the gate fires {n_fire}")

print("\n" + "=" * 78)
print("4. GATE ATTRIBUTION ACROSS THE EEM RANK CONTINUUM")
rows = []
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]:
    m = (r21_z >= 90) & (r21_e >= lo) & (r21_e < hi)
    d = cal[m.reindex(cal, fill_value=False).values]
    e = declusters(d.intersection(rP.dropna().index), H, cal)
    if len(e) < 2:
        continue
    vv = rP.loc[e].values
    rZ = vehicle_ret(px, [("EWZ", 1.0)], H, 1)
    rE = vehicle_ret(px, [("EEM", 1.0)], H, 1)
    rows.append({"EEM_r21_band": f"[{lo},{hi})", "N": len(e),
                 "pair_pct": round(100 * vv.mean(), 3),
                 "EWZ_leg": round(100 * rZ.loc[e].mean(), 3),
                 "EEM_leg": round(100 * rE.loc[e].mean(), 3),
                 "hit": round(100 * (vv > 0).mean(), 1),
                 "LIVE": "<== 44.8" if lo == 40 else ""})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("5. DOSE: today's thrust vs the trigger distribution")
print(f"  live EWZ 21d return {100*z21.loc[ASOF]:+.2f}%, rank "
      f"{r21_z.loc[ASOF]:.1f} (pitched rung is >= 90, so today is ON the edge)")
tr = z21.loc[sig]
print(f"  trigger-day EWZ 21d return: median {100*tr.median():+.2f}%, "
      f"mean {100*tr.mean():+.2f}%, "
      f"pctile of today = {100*(tr < z21.loc[ASOF]).mean():.0f}")
trk = r21_z.loc[sig]
print(f"  trigger-day EWZ rank: median {trk.median():.1f}; today "
      f"{r21_z.loc[ASOF]:.1f} is the {100*(trk < r21_z.loc[ASOF]).mean():.0f}th "
      f"pctile of the trigger distribution")
lo_half = sig[(z21.loc[sig] <= tr.median()).values]
hi_half = sig[(z21.loc[sig] > tr.median()).values]
for lbl, d in [("weak-dose half (today's half)", lo_half),
               ("strong-dose half", hi_half)]:
    e = declusters(d.intersection(rP.dropna().index), H, cal)
    vv = rP.loc[e].values
    w = int((vv > 0).sum())
    print(f"  {lbl:<30} N={len(vv):>3}  {100*vv.mean():+.3f}%  "
          f"record {w}-{len(vv)-w}  sign p {sign_test(w, len(vv)):.4f}")

print("\n" + "=" * 78)
print("6. TOMORROW-SPECIFIC: FOMC +3 td AND QUAD WITCHING +5 td ARE BOTH")
print("   INSIDE AN h=5 HOLD ENTERED 2026-09-11")
for kinds in [("fomc_decision",), ("quad_witching",), ("cpi",),
              ("fomc_decision", "quad_witching")]:
    fl = event_in_window(epi, cal, H, 1, kinds)
    if fl.sum() == 0:
        print(f"  {'+'.join(kinds):<28} 0 of {len(fl)} episodes -- the live "
              f"configuration has NO historical instance")
        continue
    show([summarize(v[fl], f"{'+'.join(kinds)} IN (N={int(fl.sum())})"),
          summarize(v[~fl], f"OUT (N={int((~fl).sum())})")], "")

print("\n" + "=" * 78)
print("7. REFERENCE-CLASS MAX-OF-K PERMUTATION (EWZ ranked 1 of 9)")
CLASS = ["EWZ", "EWW", "EWY", "EWT", "EWJ", "FXI", "EFA", "INDA", "KWEB"]
obs, pool = {}, {}
for t in CLASS:
    m = (pct_rank(px[t], 21) >= 90) & (r21_e < 60)
    d = cal[m.reindex(cal, fill_value=False).values]
    r = vehicle_ret(px, [(t, 1.0), ("EEM", -1.0)], H, 1).dropna()
    e = declusters(d.intersection(r.index), H, cal)
    if len(e) >= 3:
        obs[t] = 100 * r.loc[e].mean()
        pool[t] = (r.values, len(e))
print("  observed:", {k: round(x, 3) for k, x in
                      sorted(obs.items(), key=lambda kv: -kv[1])})
rng = np.random.default_rng(42)
NB = 5000
draws = np.empty((NB, len(pool)))
for j, (t, (vals, n)) in enumerate(pool.items()):
    draws[:, j] = 100 * rng.choice(vals, size=(NB, n)).mean(axis=1)
mx = draws.max(axis=1)
print(f"  max-of-{len(obs)} random-date null ({NB} draws): mean "
      f"{mx.mean():.3f}%, 95th pctile {np.percentile(mx, 95):.3f}%")
print(f"  P(max-of-K null >= EWZ's {obs['EWZ']:.3f}%) = "
      f"{(mx >= obs['EWZ']).mean():.4f}")
j_ewz = list(pool).index("EWZ")
print(f"  single-name null P(EWZ random-date >= {obs['EWZ']:.3f}%) = "
      f"{(draws[:, j_ewz] >= obs['EWZ']).mean():.4f}")
print("\nDONE")
