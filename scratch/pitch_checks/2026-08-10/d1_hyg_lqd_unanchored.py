"""D1 -- credit-quality divergence at joint 52w extremes, UNANCHORED.

Today: HYG closed AT its 52w high (dist 0.00%) while LQD is -3.16% from its
high and only +1.16% off its 52w LOW.

Killed twice already in EVENT-anchored forms on counts (NFP N=2, CPI N=4 with
three inside one 2018 quarter). This is the price-state version.

Order of operations is fixed by the registry:
  0. COUNT before measuring anything. Under ~8 declustered episodes at today's
     tolerances this is unmeasurable and dies there.
  1. Price BOTH LEGS SEPARATELY before the spread.
  2. Beta-neutralise. If the equal-dollar spread is pure beta the credit story
     is decoration.
  3. Ask whether the LQD leg is distinguishable from IEF/TLT. If not, this is
     D5 in disguise and correlated with the other candidate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["HYG", "LQD", "IEF", "TLT", "SPY"]).dropna(subset=["HYG", "LQD"])
idx = px.index
hyg, lqd = px["HYG"], px["LQD"]
hyg_hi = hyg / hyg.rolling(252).max() - 1.0     # 0 = at the high
lqd_lo = lqd / lqd.rolling(252).min() - 1.0     # 0 = at the low
lqd_hi = lqd / lqd.rolling(252).max() - 1.0

print("=" * 96)
print("0. COUNT FIRST")
print("=" * 96)
print(f"HYG history {idx[0].date()} .. {idx[-1].date()}  N={len(idx)} sessions")
print(f"TODAY (2026-08-07 close): HYG {100*hyg_hi.iloc[-1]:+.2f}% from 52w high, "
      f"LQD {100*lqd_lo.iloc[-1]:+.2f}% above 52w low, "
      f"{100*lqd_hi.iloc[-1]:+.2f}% from its 52w high")

tols = [(0.005, 0.02, "TODAY'S ACTUAL TOLERANCES"),
        (0.005, 0.03, ""), (0.01, 0.03, ""), (0.01, 0.05, ""),
        (0.02, 0.05, ""), (0.02, 0.10, ""), (0.03, 0.10, "")]
counts = {}
for hi_t, lo_t, note in tols:
    st = (hyg_hi >= -hi_t) & (lqd_lo <= lo_t)
    days = idx[st.fillna(False).values]
    epi = declusters(days, 21, idx)
    counts[(hi_t, lo_t)] = (days, epi)
    yrs = sorted(set(d.year for d in epi))
    print(f"  HYG within {100*hi_t:.1f}% of high & LQD within {100*lo_t:.0f}% of "
          f"low: {len(days):4d} days, {len(epi):3d} episodes (21td gap), "
          f"years {yrs}  {note}")

DAYS, EPI = counts[(0.005, 0.02)]
print(f"\n  At today's tolerances: {len(DAYS)} days / {len(EPI)} episodes.")
print("  Episode dates:", ", ".join(str(d.date()) for d in EPI) or "(none)")
if len(EPI) < 8:
    print("\n  >>> UNDER 8 EPISODES. UNMEASURABLE, not merely small. <<<")

# Use the loosest tolerance that has a real population, and say so.
USE = None
for hi_t, lo_t, _ in tols:
    d, e = counts[(hi_t, lo_t)]
    if len(e) >= 8:
        USE = (hi_t, lo_t, d, e)
        break
if USE is None:
    USE = (0.03, 0.10, *counts[(0.03, 0.10)])
HI_T, LO_T, DAYS_U, EPI_U = USE
print(f"\n  Loosest-first tolerance with >=8 episodes: HYG {100*HI_T:.1f}% / "
      f"LQD {100*LO_T:.0f}%  ->  {len(DAYS_U)} days, {len(EPI_U)} episodes")
print("  (measuring here is ALREADY a widening of today's stated state; every")
print("   number below is an upper bound on what today's tighter cell is worth)")

# ------------------------------------------------------------------ legs
print("\n" + "=" * 96)
print("1. LEGS SEPARATELY, BEFORE THE SPREAD  (entry lag=1 MOC, episode level)")
print("=" * 96)


def rep(v, lbl, ctrl=None):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"cell": lbl, "N": 0}
    w = int((v > 0).sum())
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    r = {"cell": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 3),
         "hit": round(100 * w / len(v), 1),
         "t": round(v.mean() / (sd / np.sqrt(len(v))), 2) if sd else np.nan,
         "sign_p": round(sign_test(w, len(v)), 4),
         "boot": round(bootstrap_p_le0(v), 3) if len(v) >= 3 else np.nan,
         "worst_pct": round(100 * v.min(), 2)}
    if ctrl is not None:
        r["edge_pp"] = round(100 * (v.mean() - ctrl), 3)
    return r


for H in (3, 5, 10, 21):
    rows = []
    for lbl, legs in [("LONG LQD", [("LQD", 1.0)]),
                      ("LONG HYG", [("HYG", 1.0)]),
                      ("LONG IEF", [("IEF", 1.0)]),
                      ("LONG TLT", [("TLT", 1.0)]),
                      ("LONG SPY", [("SPY", 1.0)]),
                      ("SPREAD +LQD -HYG (equal $)", [("LQD", 1.0), ("HYG", -1.0)])]:
        r = vehicle_ret(px, legs, H, 1)
        base = r.dropna()
        e = declusters(DAYS_U, max(H, 21), base.index)
        rows.append(rep(r.loc[e].values, lbl, base.mean()))
        rows[-1]["ctrl_pct"] = round(100 * base.mean(), 3)
    print(f"\n--- h={H} td ---")
    print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ beta
print("\n" + "=" * 96)
print("2. BETA-NEUTRALITY. Is the spread anything but a duration bet?")
print("=" * 96)
for H in (3, 5, 10, 21):
    rl = fwd_lag(lqd, H, 1)
    rh = fwd_lag(hyg, H, 1)
    ok = rl.notna() & rh.notna()
    b = np.polyfit(rh[ok].values, rl[ok].values, 1)[0]
    corr = np.corrcoef(rh[ok].values, rl[ok].values)[0, 1]
    eq = (rl - rh)
    bn = (rl - b * rh)
    base_eq = eq.dropna()
    base_bn = bn.dropna()
    e = declusters(DAYS_U, max(H, 21), rl.dropna().index)
    r1 = rep(eq.loc[e].values, f"h={H} equal-$ +LQD -HYG", base_eq.mean())
    r2 = rep(bn.loc[e].values, f"h={H} BETA-NEUTRAL +LQD -{b:.2f}xHYG",
             base_bn.mean())
    print(f"\n  h={H}: beta(LQD~HYG) = {b:.3f}, corr {corr:.3f}")
    print(pd.DataFrame([r1, r2]).to_string(index=False))

# ------------------------------------------------- 3. is the LQD leg = rates?
print("\n" + "=" * 96)
print("3. IS THE LQD LEG DISTINGUISHABLE FROM PLAIN DURATION?")
print("   if long LQD on this trigger == long IEF, the credit story is")
print("   decoration and this is the same bet as D5")
print("=" * 96)
for H in (5, 10, 21):
    rows = []
    for a, b_ in [("LQD", "IEF"), ("LQD", "TLT")]:
        ra, rb = fwd_lag(px[a], H, 1), fwd_lag(px[b_], H, 1)
        ok = ra.notna() & rb.notna()
        beta = np.polyfit(rb[ok].values, ra[ok].values, 1)[0]
        corr = np.corrcoef(ra[ok].values, rb[ok].values)[0, 1]
        resid = ra - beta * rb
        e = declusters(DAYS_U, max(H, 21), ra.dropna().index)
        r = rep(resid.loc[e].values, f"h={H} {a} residual vs {b_} "
                                     f"(beta {beta:.2f}, corr {corr:.2f})",
                resid.dropna().mean())
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- 4. era / midterm / concentration
print("\n" + "=" * 96)
print("4. ERA, MIDTERM, CONCENTRATION on the best-looking leg/spread, h=21")
print("=" * 96)
H = 21
for lbl, legs in [("LONG LQD", [("LQD", 1.0)]),
                  ("SPREAD +LQD -HYG", [("LQD", 1.0), ("HYG", -1.0)])]:
    r = vehicle_ret(px, legs, H, 1)
    base = r.dropna()
    e = declusters(DAYS_U, H, base.index)
    v = r.loc[e].values
    print(f"\n--- {lbl} ---")
    print(f"  episodes: {', '.join(str(d.date()) for d in e)}")
    yr = e.year
    mid = (yr % 4) == 2
    rows = [rep(v, "ALL", base.mean()),
            rep(v[yr < 2018], "pre-2018", base.mean()),
            rep(v[yr >= 2018], "2018+", base.mean()),
            rep(v[~((yr >= 2020) & (yr <= 2022))], "ex 2020-2022", base.mean()),
            rep(v[mid], "MIDTERM", base.mean()),
            rep(v[~mid], "non-midterm", base.mean())]
    print(pd.DataFrame(rows).to_string(index=False))
    print("  " + cluster_note(e, v))
    byy = pd.Series(v).groupby(yr.values).agg(["count", "mean"])
    byy["mean_pct"] = (100 * byy["mean"]).round(3)
    print(byy[["count", "mean_pct"]].to_string())

# ------------------------------------------------- 5. today's state location
print("\n" + "=" * 96)
print("5. WHERE TODAY SITS, and what the trigger population actually is")
print("=" * 96)
st = (hyg_hi >= -HI_T) & (lqd_lo <= LO_T)
d = idx[st.fillna(False).values]
print(f"  trigger days by year: "
      f"{dict(pd.Series(d.year).value_counts().sort_index())}")
print(f"  today's HYG dist-from-high {100*hyg_hi.iloc[-1]:+.3f}% vs trigger-day "
      f"median {100*hyg_hi.reindex(d).median():+.3f}%")
print(f"  today's LQD dist-from-low  {100*lqd_lo.iloc[-1]:+.3f}% vs trigger-day "
      f"median {100*lqd_lo.reindex(d).median():+.3f}%")

# ------------------------------------------------- 6. event risk in window
print("\n" + "=" * 96)
print("6. EVENT RISK INSIDE THE WINDOW (CPI +2 td, PPI +3 td from today)")
print("=" * 96)
for H in (5, 10, 21):
    r = vehicle_ret(px, [("LQD", 1.0), ("HYG", -1.0)], H, 1)
    e = declusters(DAYS_U, max(H, 21), r.dropna().index)
    fl = event_in_window(e, idx, H, 1, ("cpi",))
    v = r.loc[e].values
    print(f"  h={H}: CPI in window N={int(fl.sum())} "
          f"{100*np.nanmean(v[fl]):+.3f}%  |  no CPI N={int((~fl).sum())} "
          f"{100*np.nanmean(v[~fl]):+.3f}%")

print("\n" + "=" * 96)
print("7. COST")
print("=" * 96)
print("  LQD ~2 bps + HYG ~2 bps round trip = ~4 bps for the pair, ~2 bps for a")
print("  single leg. Multiples are computed against the edge_pp column above,")
print("  not the raw mean.")
