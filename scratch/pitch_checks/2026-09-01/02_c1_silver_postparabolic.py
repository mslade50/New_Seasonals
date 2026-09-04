"""C1 KILL CHECK — long a high-beta commodity vehicle deep inside a
post-parabolic drawdown while still hugely positive on the trailing year.

Live: SLV -43.06% below its trailing-252 high while +69.28% over 252 sessions.

Order of operations is the 2026-08-07 count-first rule, then the two things
that actually decide it:
  1. COUNT the state on SLV over a (depth x year-return) grid before measuring
     anything. A cell that only exists in one episode is not a cell.
  2. GATE ATTRIBUTION both ways: does the "still up on the year" leg filter, or
     does it relabel a plain deep-drawdown buy? Does the DEPTH leg filter, or
     is it a plain "still up on the year" momentum buy?
  3. REFERENCE CLASS of 10 high-beta commodity/metal/miner vehicles plus an
     equity-index null. The registry precedent is explicit: a DEPTH BAND is
     instrument-specific and cannot be transplanted, and the 205-name washout
     reference class returned family-wise p 1.0000. If the pooled family has a
     common excess and SLV is not distinguishable, that is the kill.
  4. What does the trigger SELECT? SPY forward on SLV trigger days tells us
     whether the drawdown gate is just a crash-tape sampler.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
FAM = ["SLV", "GLD", "GDX", "GDXJ", "NEM", "XME", "FCX", "USO", "UNG", "DBC",
       "CEF", "COPX", "PPLT", "PALL", "URA"]
NULL = ["SPY", "QQQ", "IWM"]
PX = load_prices(FAM + NULL)
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
SPY = PX["SPY"]["Close"]


def state(s):
    """(drawdown from trailing-252 max, trailing-252 return), own index."""
    hi = s.rolling(252, min_periods=252).max()
    dd = s / hi - 1.0
    r252 = s / s.shift(252) - 1.0
    return dd, r252


def mask_for(s, dd_max, r_min):
    dd, r252 = state(s)
    return (dd <= dd_max) & (r252 >= r_min)


# ---------------------------------------------------------------------------
# 1. COUNT FIRST on SLV
# ---------------------------------------------------------------------------
slv = PX["SLV"]["Close"]
dd, r252 = state(slv)
print("== LIVE SLV state: dd %.2f%%  r252 %+.2f%%  (bar %s)"
      % (100 * dd.iloc[-1], 100 * r252.iloc[-1], slv.index[-1].date()))
print("\n===== 1. COUNT FIRST — SLV days (and DECLUSTERED episodes at gap 21) "
      "in each (depth, year-return) cell =====")
rows = []
allidx = slv.dropna().index
for d_th in (-0.20, -0.25, -0.30, -0.35, -0.40):
    row = {"dd<=": f"{100*d_th:.0f}%"}
    for r_th in (0.0, 0.20, 0.30, 0.50):
        m = mask_for(slv, d_th, r_th).fillna(False)
        t = allidx[m.reindex(allidx, fill_value=False).values]
        e = declusters(t, 21, allidx)
        row[f"r252>={100*r_th:.0f}%"] = f"{len(t)}d/{len(e)}ep"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

BASE_DD, BASE_R = -0.30, 0.30
base_mask = mask_for(slv, BASE_DD, BASE_R).fillna(False)
tr = allidx[base_mask.reindex(allidx, fill_value=False).values]
print(f"\n  BASE CELL dd<={100*BASE_DD:.0f}% & r252>={100*BASE_R:.0f}%: "
      f"{len(tr)} days")
print("  calendar episodes (gap 21 td):",
      [str(d.date()) for d in declusters(tr, 21, allidx)])
print("  year histogram of trigger DAYS:",
      pd.Series(tr.year).value_counts().sort_index().to_dict())

# ---------------------------------------------------------------------------
# 2. horizon scan then the battery on SLV alone
# ---------------------------------------------------------------------------
px_slv = pd.DataFrame({"SLV": slv})
print("\n===== 2. HORIZON SCAN, SLV long, base cell, lag=1 =====")
show(horizon_scan(px_slv, tr, [("SLV", 1.0)], hs=(1, 2, 3, 5, 10, 21), min_gap=21),
     "horizon scan (episodes, min_gap=21)")

variants = {}
for d_th in (-0.20, -0.25, -0.30, -0.35, -0.40):
    variants[f"dd<={100*d_th:.0f}% & r252>=30%"] = mask_for(slv, d_th, 0.30).fillna(False)
variants["dd<=-30% ALONE (no year leg)"] = (state(slv)[0] <= -0.30).fillna(False)
variants["r252>=+30% ALONE (no depth leg)"] = (state(slv)[1] >= 0.30).fillna(False)
variants["dd<=-30% & r252 in [0,30%)"] = (
    (state(slv)[0] <= -0.30) & (state(slv)[1] >= 0.0) & (state(slv)[1] < 0.30)).fillna(False)
variants["dd<=-30% & r252 < 0 (the OPPOSITE year leg)"] = (
    (state(slv)[0] <= -0.30) & (state(slv)[1] < 0.0)).fillna(False)

for H in (10, 21):
    battery(px_slv, base_mask, [("SLV", 1.0)], H,
            f"C1 SLV long, dd<=-30% & r252>=+30%", cost_bps=8.0,
            variants=variants, min_gap=21)

# ---------------------------------------------------------------------------
# 3. GATE ATTRIBUTION, stated as a clean table
# ---------------------------------------------------------------------------
print("\n===== 3. GATE ATTRIBUTION (SLV, h=10 and h=21, lag=1, episodes gap 21) =====")
for H in (10, 21):
    r = fwd_lag(slv, H, 1)
    rows = []
    for lbl, m in list(variants.items()) + [("BASE dd<=-30 & r252>=30", base_mask)]:
        t = allidx[m.reindex(allidx, fill_value=False).values]
        t = t.intersection(r.dropna().index)
        e = declusters(t, 21, allidx)
        s = summarize(r.reindex(e).values, lbl)
        s["n_days"] = len(t)
        rows.append(s)
    rows.append(summarize(r.dropna().values, "ALL DAYS (SLV drift)"))
    show(rows, f"gate attribution h={H}")

# ---------------------------------------------------------------------------
# 4. REFERENCE CLASS — the registry's standing kill for depth bands
# ---------------------------------------------------------------------------
print("\n===== 4. REFERENCE CLASS: same cell on 15 commodity/metal/miner "
      "vehicles + 3 equity-index nulls =====")
for H in (10, 21):
    rows = []
    pooled_ep, pooled_dates = [], []
    for t in FAM + NULL:
        s = PX[t]["Close"].dropna()
        if len(s) < 600:
            continue
        m = mask_for(s, BASE_DD, BASE_R).fillna(False)
        r = fwd_lag(s, H, 1)
        idx = s.index[m.reindex(s.index, fill_value=False).values]
        idx = idx.intersection(r.dropna().index)
        e = declusters(idx, 21, s.index)
        drift = 100 * r.dropna().mean()
        rec = summarize(r.reindex(e).values, t)
        rec["n_days"] = len(idx)
        rec["own_drift_pct"] = round(drift, 3)
        rec["excess_pct"] = round(rec.get("mean_pct", np.nan) - drift, 3) if rec["n"] else np.nan
        if rec["n"]:
            wins = int((r.reindex(e).values > 0).sum())
            rec["sign_p"] = round(sign_test(wins, rec["n"]), 4)
        rows.append(rec)
        if t in FAM and rec["n"]:
            pooled_ep.extend(list(r.reindex(e).values - r.dropna().mean()))
            pooled_dates.extend(list(e))
    show(rows, f"per-vehicle, h={H} (excess = episode mean minus that "
              f"vehicle's own all-days drift)")
    pe = np.asarray(pooled_ep, float)
    pe = pe[~np.isnan(pe)]
    if len(pe) > 2:
        wins = int((pe > 0).sum())
        print(f"  POOLED family EXCESS across commodity vehicles: N={len(pe)} "
              f"mean {100*pe.mean():+.3f}pp  t={pe.mean()/(pe.std(ddof=1)/np.sqrt(len(pe))):+.2f}"
              f"  record {wins}-{len(pe)-wins} sign p={sign_test(wins, len(pe)):.4f}"
              f"  bootstrap P(<=0)={bootstrap_p_le0(pe):.3f}")
        # is SLV distinguishable from the family?
        s = PX["SLV"]["Close"].dropna()
        r = fwd_lag(s, H, 1)
        m = mask_for(s, BASE_DD, BASE_R).fillna(False)
        idx = s.index[m.reindex(s.index, fill_value=False).values].intersection(r.dropna().index)
        e = declusters(idx, 21, s.index)
        slv_ex = r.reindex(e).values - r.dropna().mean()
        slv_ex = slv_ex[~np.isnan(slv_ex)]
        if len(slv_ex) > 1:
            others = pe  # includes SLV; fine as a family reference
            se = np.sqrt(slv_ex.var(ddof=1) / len(slv_ex) + others.var(ddof=1) / len(others))
            print(f"  SLV excess {100*slv_ex.mean():+.3f}pp (N={len(slv_ex)}) vs family "
                  f"{100*others.mean():+.3f}pp -> welch t = "
                  f"{(slv_ex.mean()-others.mean())/se:+.2f}  "
                  f"(is SLV distinguishable from its own family?)")

# ---------------------------------------------------------------------------
# 5. WHAT DOES THE TRIGGER SELECT? base-rate shift on the tape
# ---------------------------------------------------------------------------
print("\n===== 5. WHAT THE TRIGGER SELECTS (SPY forward on SLV trigger days) =====")
for H in (10, 21):
    sr = fwd_lag(SPY, H, 1)
    t = tr.intersection(sr.dropna().index)
    e = declusters(t, 21, SPY.index)
    show([summarize(sr.reindex(e).values, f"SPY on SLV-trigger episodes h={H}"),
          summarize(sr.dropna().values, f"SPY all days h={H}")], f"base-rate shift h={H}")
vix_note = pd.Series(tr.year).value_counts().sort_index()
print("  trigger-day year histogram again:", vix_note.to_dict())

# ---------------------------------------------------------------------------
# 6. LAG PROFILE + entry-day-move split (registry traps)
# ---------------------------------------------------------------------------
print("\n===== 6. LAG PROFILE (SLV base cell) =====")
for H in (10, 21):
    for lag in (0, 1, 2, 3):
        r = fwd_lag(slv, H, lag)
        e = declusters(tr.intersection(r.dropna().index), 21, allidx)
        s = summarize(r.reindex(e).values, f"h={H} lag={lag}")
        print("  h=%2d lag=%d  N=%2d  mean %+.3f%%  hit %.1f%%  t %s"
              % (H, lag, s["n"], s["mean_pct"], s["hit"],
                 f"{s['t']:+.2f}" if s["n"] > 1 else "na"))

print("\n===== 6b. ENTRY-DAY-MOVE split (does the mechanism survive its own data?) =====")
d1 = slv.pct_change().shift(-1)  # the entry day's own move (D+1)
for H in (10, 21):
    r = fwd_lag(slv, H, 1)
    e = declusters(tr.intersection(r.dropna().index), 21, allidx)
    up = e[d1.reindex(e).values > 0]
    dn = e[d1.reindex(e).values <= 0]
    show([summarize(r.reindex(up).values, f"entry day UP (N={len(up)})"),
          summarize(r.reindex(dn).values, f"entry day DOWN (N={len(dn)})")],
         f"entry-day split h={H}")
    j = pd.DataFrame({"e": d1.reindex(e), "f": r.reindex(e)}).dropna()
    print(f"  corr(entry-day move, forward) = {j['e'].corr(j['f']):+.3f}  n={len(j)}")

# ---------------------------------------------------------------------------
# 7. percentile / definition-convention note
# ---------------------------------------------------------------------------
print("\n===== 7. CONVENTIONS RECORDED =====")
print("  drawdown uses rolling(252, min_periods=252).max() on the ticker's OWN "
      "valid sessions (inclusive of today's bar).")
print("  r252 = s / s.shift(252) - 1 on the ticker's own index (no pad).")
print("  declustering min_gap = 21 td throughout; battery re-runs gap sensitivity.")
