"""S2b -- attacking the one S2 cell with a pulse: long duration when the IG /
duration complex is at its 252d floor and HY is at its 252d ceiling.

The headline from 03_pricestate_s2_rates_credit.py was TLT at h=10 in the strict
LQD-floor & HYG-ceiling cell: 7 episodes, +1.003%, 7-0, sign p 0.0078, worst
episode +0.146%. This script exists to try to break it, because that cell has an
obvious defect the first script already printed: only TWO calendar years appear
(2018 and 2026), and the 2026 half is the LIVE cluster whose forward windows
overlap each other.

Attacks run here:
  1. REGIME declustering (63 td, not 10) -- how many independent episodes are
     really there?
  2. Drop the live 2026 cluster entirely. Does 2018 alone carry it?
  3. Drop 2018. Does 2026 alone carry it?
  4. The BROADER cell (TLT within 2% of its 252d low AND HYG within 1% of its
     252d high), which spans more regimes -- is the pulse still there when the
     sample is not two clusters?
  5. The TLT-floor parent WITHOUT the HY gate, to see whether the credit side
     conditions anything at all.
  6. Rate-side placebo: does the same cell work on IEF (less duration) and on
     the TLT-minus-IEF curve trade?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, cluster_note, declusters, era_split, fwd_lag, hscan,
    load_prices, local_control, np, pd, roll_max, roll_min, show, sign_test,
    summarize, bootstrap_p_le0,
)

PX = load_prices(["TLT", "IEF", "LQD", "HYG", "SPY", "AGG"])
IDX = PX["HYG"].index
C = {t: PX[t]["Close"] for t in PX}

lqd_floor = align(C["LQD"] <= 1.01 * roll_min(C["LQD"], 252), IDX).fillna(0).astype(bool)
hyg_ceil = align(C["HYG"] >= 0.99 * roll_max(C["HYG"], 252), IDX).fillna(0).astype(bool)
tlt_floor2 = align(C["TLT"] <= 1.02 * roll_min(C["TLT"], 252), IDX).fillna(0).astype(bool)
tlt_floor5 = align(C["TLT"] <= 1.05 * roll_min(C["TLT"], 252), IDX).fillna(0).astype(bool)

STRICT = IDX[(lqd_floor & hyg_ceil).values]
BROAD = IDX[(tlt_floor2 & hyg_ceil).values]
BROAD5 = IDX[(tlt_floor5 & hyg_ceil).values]
TLT_ONLY = IDX[tlt_floor2.values]

print("=" * 78)
print("S2b  ATTACKING THE TLT-AT-THE-FLOOR CELL")
print("=" * 78)
for nm, t in (("STRICT  LQD<=1%>252low & HYG>=1%<252high", STRICT),
              ("BROAD   TLT<=2%>252low & HYG>=1%<252high", BROAD),
              ("BROAD5  TLT<=5%>252low & HYG>=1%<252high", BROAD5),
              ("PARENT  TLT<=2%>252low (no credit gate)", TLT_ONLY)):
    print(f"  {nm}: {len(t)} days, {t[0].date()} .. {t[-1].date()}, "
          f"years {sorted(set(t.year))}")


def tlt(h):
    return align(fwd_lag(C["TLT"], h, 1), IDX)


def ief(h):
    return align(fwd_lag(C["IEF"], h, 1), IDX)


def curve(h):
    return tlt(h) - ief(h)


print("\n" + "=" * 78)
print("1. REGIME DECLUSTERING: how many independent episodes are really there?")
print("=" * 78)
for gap in (5, 10, 21, 63, 126):
    r = tlt(10)
    valid = r.dropna().index
    for nm, t0 in (("STRICT", STRICT), ("BROAD", BROAD)):
        t = pd.DatetimeIndex(t0).intersection(valid)
        epi = declusters(t, gap, valid)
        ep = r.loc[epi].values
        w = int((ep > 0).sum())
        print(f"  gap={gap:3d} td  {nm:6s} n={len(epi):3d}  TLT h=10 mean "
              f"{100 * ep.mean():+.3f}%  rec {w}-{len(epi) - w}  "
              f"sign p {sign_test(w, len(epi)):.4f}  "
              f"dates {[str(d.date()) for d in epi][:8]}")

print("\n" + "=" * 78)
print("2/3. LEAVE-ONE-CLUSTER-OUT on the STRICT cell (TLT, h=5 and h=10)")
print("=" * 78)
for h in (5, 10):
    r = tlt(h)
    valid = r.dropna().index
    t = pd.DatetimeIndex(STRICT).intersection(valid)
    epi = declusters(t, h, valid)
    rows = []
    for lbl, keep in (("ALL episodes", epi),
                      ("drop 2026 (the live cluster)", epi[epi.year != 2026]),
                      ("drop 2018", epi[epi.year != 2018])):
        v = r.loc[keep].values
        s = summarize(v, f"h={h} {lbl}")
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v) - w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
        rows.append(s)
    show(rows, f"STRICT cell, TLT h={h}, leave-one-cluster-out")

print("\n" + "=" * 78)
print("4. THE BROADER CELL: TLT within 2% of its 252d low AND HYG at its ceiling")
print("=" * 78)
hscan(tlt, BROAD, "TLT long | BROAD cell")
hscan(ief, BROAD, "IEF long | BROAD cell")
hscan(curve, BROAD, "TLT minus IEF (curve) | BROAD cell")
for h in (5, 10):
    cell(tlt(h), BROAD, h, "BROAD -> TLT long")

print("\n=== BROAD cell, era + leave-one-year-out (TLT h=5 episodes) ===")
r = tlt(5)
valid = r.dropna().index
epi = declusters(pd.DatetimeIndex(BROAD).intersection(valid), 5, valid)
ep = r.loc[epi].values
show(era_split(epi, ep), "BROAD TLT h=5")
rows = []
for y in sorted(set(epi.year)):
    keep = epi.year != y
    v = ep[keep]
    w = int((v > 0).sum())
    rows.append({"drop_year": y, "n_dropped": int((~keep).sum()), "n_left": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "rec": f"{w}-{len(v) - w}",
                 "sign_p": round(sign_test(w, len(v)), 4)})
print("\nLeave-one-year-out on the BROAD cell (TLT h=5):")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  concentration: {cluster_note(epi, ep, k=3)}")

print("\n" + "=" * 78)
print("5. GATE ATTRIBUTION: does the HY-at-ceiling condition add anything?")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    r = tlt(h)
    valid = r.dropna().index
    base = float(r.loc[valid].mean())
    for nm, t0 in (("TLT floor + HY ceiling (BROAD)", BROAD),
                   ("TLT floor ONLY (no credit gate)", TLT_ONLY),
                   ("TLT floor + HY NOT at ceiling",
                    IDX[(tlt_floor2 & ~hyg_ceil).values])):
        t = pd.DatetimeIndex(t0).intersection(valid)
        if len(t) == 0:
            continue
        epi = declusters(t, h, valid)
        ep = r.loc[epi].values
        w = int((ep > 0).sum())
        rows.append({"h": h, "cell": nm, "n_days": len(t), "n": len(epi),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "edge_all_pct": round(100 * (ep.mean() - base), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("6. PLACEBO / VEHICLE CHOICE at h=10 (BROAD cell)")
print("=" * 78)
rows = []
for nm, f in (("TLT", tlt), ("IEF", ief), ("TLT-IEF curve", curve),
              ("LQD", lambda h: align(fwd_lag(C["LQD"], h, 1), IDX)),
              ("AGG", lambda h: align(fwd_lag(C["AGG"], h, 1), IDX))):
    for h in (5, 10):
        r = f(h)
        valid = r.dropna().index
        t = pd.DatetimeIndex(BROAD).intersection(valid)
        if len(t) == 0:
            continue
        epi = declusters(t, h, valid)
        ep = r.loc[epi].values
        base = float(r.loc[valid].mean())
        w = int((ep > 0).sum())
        rows.append({"veh": nm, "h": h, "n": len(epi),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "edge_all_pct": round(100 * (ep.mean() - base), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== S2b COST NOTE ===")
print("  TLT round trip ~3-5 bps -> the 3x bar is ~+0.12% per episode.")
