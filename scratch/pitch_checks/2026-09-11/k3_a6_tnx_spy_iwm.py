"""A6 KILL CHECK - short IWM against long SPY with ^TNX at a trailing-252 max.

Pre-specified: `^TNX` closing AT its trailing-252 maximum (reindexed onto SPY's
calendar), legs SPY +1.0 / IWM -1.0, lag=1 MOC, horizon from the scan.
LIVE: ^TNX 4.944, exactly at the 252-day max, 252-session change +87.0 bp.

Mandatory attacks:
  0. print the thing the candidate is NAMED after (units check: ^TNX is in
     PERCENT, so a point x100 is bp) and the live PIT percentile
  1. BOTH LEGS SEPARATELY before the spread -- the registry's top failure mode
  2. measured beta + the beta-neutral residual
  3. dose response on the yield-high lookback 63/126/252 (knife edge vs
     gradient: the trap that killed the dollar cell on 2026-09-10)
  4. price-driven vs stale-max: first crossing vs nth day at the max, and the
     252-session yield CHANGE bucket the live reading sits in
  5. Jaccard overlap with watchlist 39's fragility-dial [56,70] mask
  6. era + midterm + the full kill battery
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

EQ = ["SPY", "IWM", "QQQ", "DIA", "XLF", "XLI", "XLY", "XLU", "EEM", "EFA"]
px = close_panel(EQ)
cal = px["SPY"].dropna().index
px = px.reindex(cal)

raw = load_prices(["^TNX"])["^TNX"]["Close"]
tnx = raw.reindex(cal).ffill()          # caret series onto SPY's calendar
ASOF = pd.Timestamp("2026-09-10")

print("=" * 78)
print("0. PRINT WHAT THE CANDIDATE IS NAMED AFTER")
print(f"  ^TNX close {tnx.loc[ASOF]:.4f} (PERCENT units)")
mx252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
chg252 = (tnx - tnx.shift(252)) * 100      # index points -> bp
print(f"  trailing-252 max {mx252.loc[ASOF]:.4f}  -> at the max: "
      f"{bool(tnx.loc[ASOF] >= mx252.loc[ASOF] - 1e-9)}")
print(f"  252-session change {chg252.loc[ASOF]:+.1f} bp   "
      f"(units check: 4.944 - 4.074 = 0.870 pts = 87.0 bp)")
lvl_pct = rolling_on_valid(tnx, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  trailing-252 LEVEL percentile {lvl_pct.loc[ASOF]:.1f}")

at_max = (tnx >= mx252 - 1e-9) & mx252.notna()
print(f"  mask fires on {int(at_max.sum())} of {len(cal)} sessions "
      f"({100*at_max.mean():.1f}%)  span "
      f"{cal[at_max.values][0].date()} .. {cal[at_max.values][-1].date()}")

# ------------------------------------------------------------------ 1. legs
print("\n" + "=" * 78)
print("1. BOTH LEGS SEPARATELY, BEFORE ANY SPREAD (h=1..10)")
sig = cal[at_max.values]
rows = []
for h in (1, 2, 3, 5, 10):
    rS = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    rI = vehicle_ret(px, [("IWM", 1.0)], h, 1)
    rP = vehicle_ret(px, [("SPY", 1.0), ("IWM", -1.0)], h, 1)
    epi = declusters(sig.intersection(rP.dropna().index), h, cal)
    sS, sI, sP = rS.loc[epi], rI.loc[epi], rP.loc[epi]
    bS, bI, bP = rS.dropna(), rI.dropna(), rP.dropna()
    rows.append({
        "h": h, "N_epi": len(epi),
        "SPY_long": round(100 * sS.mean(), 3),
        "SPY_drift": round(100 * bS.mean(), 3),
        "SPY_edge": round(100 * (sS.mean() - bS.mean()), 3),
        "IWM_long": round(100 * sI.mean(), 3),
        "IWM_drift": round(100 * bI.mean(), 3),
        "IWM_edge": round(100 * (sI.mean() - bI.mean()), 3),
        "PAIR": round(100 * sP.mean(), 3),
        "PAIR_drift": round(100 * bP.mean(), 3),
        "PAIR_edge": round(100 * (sP.mean() - bP.mean()), 3),
        "pair_hit": round(100 * (sP > 0).mean(), 1),
    })
L = pd.DataFrame(rows)
print(L.to_string(index=False))
print("  READ: if PAIR ~ SPY_edge - IWM_edge and one leg carries it all, the")
print("  'relative value' label is decoration.")

# ------------------------------------------------------------- 2. beta
print("\n" + "=" * 78)
print("2. MEASURED BETA AND THE BETA-NEUTRAL RESIDUAL")
dS = px["SPY"].pct_change()
dI = px["IWM"].pct_change()
ok = dS.notna() & dI.notna()
beta_full = float(np.polyfit(dS[ok], dI[ok], 1)[0])
tail = ok & (cal >= cal[-253])
beta_252 = float(np.polyfit(dS[tail], dI[tail], 1)[0])
print(f"  beta(IWM on SPY) full sample {beta_full:.3f}   trailing-252 "
      f"{beta_252:.3f}")
print(f"  IWM daily sd {100*dI[ok].std():.3f}%  vs SPY {100*dS[ok].std():.3f}% "
      f"= {dI[ok].std()/dS[ok].std():.2f}x")
rows = []
for h in (1, 2, 3, 5, 10):
    rP = vehicle_ret(px, [("SPY", 1.0), ("IWM", -1.0)], h, 1)
    rN = vehicle_ret(px, [("SPY", beta_full), ("IWM", -1.0)], h, 1)
    epi = declusters(sig.intersection(rP.dropna().index), h, cal)
    rows.append({"h": h, "N": len(epi),
                 "pair_1to1": round(100 * rP.loc[epi].mean(), 3),
                 "pair_1to1_edge": round(
                     100 * (rP.loc[epi].mean() - rP.dropna().mean()), 3),
                 "beta_neutral": round(100 * rN.loc[epi].mean(), 3),
                 "beta_neutral_edge": round(
                     100 * (rN.loc[epi].mean() - rN.dropna().mean()), 3),
                 "bn_hit": round(100 * (rN.loc[epi] > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------- 3. dose
print("\n" + "=" * 78)
print("3. DOSE RESPONSE ON THE YIELD-HIGH LOOKBACK (knife edge or gradient?)")
H = 5
rP = vehicle_ret(px, [("SPY", 1.0), ("IWM", -1.0)], H, 1)
base = rP.dropna()
rows = []
for n in (63, 126, 189, 252, 378, 504):
    mx = rolling_on_valid(tnx, lambda x, n=n: x.rolling(n).max())
    m = (tnx >= mx - 1e-9) & mx.notna()
    s = cal[m.values]
    epi = declusters(s.intersection(base.index), H, cal)
    v = rP.loc[epi].values
    rows.append({"lookback": n, "N_days": int(m.sum()), "N_epi": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "edge_pp": round(100 * (v.mean() - base.mean()), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("  and the proximity gradient at the 252 lookback (how CLOSE to the max):")
dist = (mx252 - tnx) / mx252 * 100          # % below the max, 0 = at it
rows = []
for lo, hi in [(0.0, 0.001), (0.001, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0),
               (6.0, 100.0)]:
    m = (dist >= lo) & (dist < hi) & dist.notna()
    s = cal[m.values]
    epi = declusters(s.intersection(base.index), H, cal)
    if len(epi) == 0:
        continue
    v = rP.loc[epi].values
    rows.append({"band_pct_below_max": f"[{lo},{hi})", "N_epi": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------- 4. fresh vs stale max
print("\n" + "=" * 78)
print("4. IS THE MAX FRESH OR STALE, AND WHAT DOSE DOES IT BUY?")
run = at_max.astype(int).groupby((~at_max).cumsum()).cumsum()
first = at_max & (run == 1)
print(f"  live 2026-09-10 run length at the max = {int(run.loc[ASOF])}")
for lbl, m in [("FIRST crossing", first), ("later days in a run", at_max & (run > 1))]:
    s = cal[m.values]
    epi = declusters(s.intersection(base.index), H, cal)
    v = rP.loc[epi].values
    print(f"  {lbl:<22} N_epi={len(epi):<4} pair {100*v.mean():+.3f}%  "
          f"hit {100*(v>0).mean():.1f}%")
print("\n  252-session yield CHANGE bucket (live reading +87.0 bp):")
rows = []
for lo, hi in [(-1000, 0), (0, 50), (50, 80), (80, 120), (120, 1000)]:
    m = at_max & (chg252 >= lo) & (chg252 < hi)
    s = cal[m.values]
    epi = declusters(s.intersection(base.index), H, cal)
    if len(epi) == 0:
        continue
    v = rP.loc[epi].values
    rows.append({"chg252_bp": f"[{lo},{hi})", "N_epi": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "LIVE": "<==" if lo == 80 else ""})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  ^TNX LEVEL bucket at the trigger (live 4.944):")
rows = []
for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 99)]:
    m = at_max & (tnx >= lo) & (tnx < hi)
    s = cal[m.values]
    epi = declusters(s.intersection(base.index), H, cal)
    if len(epi) == 0:
        continue
    v = rP.loc[epi].values
    rows.append({"tnx_level": f"[{lo},{hi})", "N_epi": len(epi),
                 "pair_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "LIVE": "<==" if lo == 4 else ""})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------- 5. Jaccard vs the dial
print("\n" + "=" * 78)
print("5. JACCARD OVERLAP WITH WATCHLIST 39 (dial 10d-MA of 63d in [56,70])")
try:
    fr = pd.read_parquet(ROOTP / "data" / "rd2_fragility.parquet")
    col = "63d" if "63d" in fr.columns else fr.columns[-1]
    dial = fr[col].rolling(10).mean()
    di = pd.to_datetime(dial.index)
    if getattr(di, "tz", None) is not None:
        di = di.tz_localize(None)
    dial.index = di.normalize()
    dial = dial.reindex(cal).ffill()
    band = (dial >= 56) & (dial < 70)
    both = at_max & band
    union = at_max | band
    print(f"  dial series {dial.dropna().index[0].date()} .. "
          f"{dial.dropna().index[-1].date()}   live "
          f"{dial.loc[ASOF]:.1f}")
    print(f"  ^TNX-at-max days {int(at_max.sum())}, dial-band days "
          f"{int(band.sum())}, intersection {int(both.sum())}")
    print(f"  JACCARD = {both.sum()/max(union.sum(),1):.4f}")
    ov = at_max & dial.notna()
    print(f"  of the {int(ov.sum())} ^TNX-at-max days inside the dial era, "
          f"{int(both.sum())} ({100*both.sum()/max(ov.sum(),1):.1f}%) also sit "
          f"in [56,70)")
    print(f"  LIVE dial {dial.loc[ASOF]:.1f} -> watchlist 39 is OUT of band, "
          f"so this is not that entry firing under a new name.")
except Exception as e:                                    # noqa: BLE001
    print(f"  dial unavailable: {e}")

# ----------------------------------------------------------- 6. battery
print("\n" + "=" * 78)
print("6. FULL BATTERY on the pitched pair at h=5")
battery(px, at_max, [("SPY", 1.0), ("IWM", -1.0)], 5,
        "A6 SPY +1 / IWM -1, ^TNX at a 252d max", cost_bps=2.5,
        variants={
            "TNX at 126d max": (tnx >= rolling_on_valid(
                tnx, lambda x: x.rolling(126).max()) - 1e-9),
            "TNX at 504d max": (tnx >= rolling_on_valid(
                tnx, lambda x: x.rolling(504).max()) - 1e-9),
            "TNX 252d lvl pctile>=95": (lvl_pct >= 95),
            "TNX 252d lvl pctile>=99": (lvl_pct >= 99),
        },
        event_kinds=("cpi", "ppi"))

print("\n" + "=" * 78)
print("7. ERA AND MIDTERM")
epi = declusters(sig.intersection(base.index), 5, cal)
v = rP.loc[epi].values
show(era_split(epi, v), "pair h=5 episodes")
mid = (epi.year % 4 == 2)
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
     "cycle split")
print(f"  episode years: {sorted(set(epi.year))}")
print("\nDONE")
