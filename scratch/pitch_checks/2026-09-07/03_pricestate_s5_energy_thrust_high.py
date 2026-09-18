"""S5 -- crude thrust WITH the energy equity ETF at its own 252d high.

Live 2026-09-04: USO +19.42% over 21d (5d rank 87.7), DBC 0.19% off its 52w
high, XOP rank21 91.7, XLE 1.60% off its 52w high and +17.58% over its 200d.

STANDING SUSPICION, stated up front: this repo has already killed several
energy-thrust cells. Anything that reads positive here is presumed to be the
same finding wearing a hat until the script shows what is DIFFERENT. Two things
could be: (i) the conditioner is the SECTOR at a 252d high, not the commodity's
move alone, and (ii) the vehicle is the equity, not the commodity. Both are
measured against the commodity-only parent so the increment is visible.

Cells (lag=1, declustered at h):
  A. XLE long      B. XOP long      C. XLE minus SPY      D. USO long
Masks: crude thrust alone / sector at 252d high alone / the conjunction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, declusters, era_split, fwd_lag, hscan, load_prices, np,
    pct_rank, pd, roll_max, show, sign_test, sma, summarize,
)

TICKERS = ["XLE", "XOP", "USO", "DBC", "SPY", "OIH", "XES"]
PX = load_prices(TICKERS)
IDX = PX["XLE"].index
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S5  CRUDE THRUST + ENERGY ETF AT ITS 252d HIGH  (asof 2026-09-04)")
print("=" * 78)

print("\nLIVE STATE VERIFICATION:")
for t in [x for x in ["USO", "DBC", "XLE", "XOP", "OIH", "XES"] if x in C]:
    s = C[t]
    print(f"  {t:5s} last {s.iloc[-1]:8.2f}  21d {100 * (s.iloc[-1] / s.iloc[-22] - 1):+7.2f}%  "
          f"rank21 {pct_rank(s, 21).iloc[-1]:5.1f}  rank5 {pct_rank(s, 5).iloc[-1]:5.1f}  "
          f"off52wh {100 * (s.iloc[-1] / roll_max(s, 252).iloc[-1] - 1):+6.2f}%  "
          f"vs200d {100 * (s.iloc[-1] / sma(s, 200).iloc[-1] - 1):+6.2f}%")
print(f"  freshest bar: {IDX[-1].date()}")

uso_rank21 = pct_rank(C["USO"], 21)
xle_hi = C["XLE"] >= 0.98 * roll_max(C["XLE"], 252)
xop_hi = C["XOP"] >= 0.98 * roll_max(C["XOP"], 252)

m_thrust = align(uso_rank21 >= 85, IDX).fillna(0).astype(bool)
m_xle_hi = align(xle_hi, IDX).fillna(0).astype(bool)
m_conj = m_thrust & m_xle_hi

MASKS = {
    "crude rank21>=85 ALONE": m_thrust,
    "XLE within 2% of 252d high ALONE": m_xle_hi,
    "CONJUNCTION": m_conj,
    "thrust ex-conj (crude hot, XLE NOT at high)": m_thrust & ~m_xle_hi,
    "high ex-conj (XLE at high, crude NOT hot)": m_xle_hi & ~m_thrust,
}
print("\nMASK COUNTS:")
for k, m in MASKS.items():
    d = IDX[m.values]
    print(f"  {k:<46s} {len(d):5d} days"
          + (f"   {d[0].date()} .. {d[-1].date()}   yrs {len(set(d.year))}" if len(d) else ""))
print(f"  live day in conjunction: {bool(m_conj.iloc[-1])}"
      f"   (USO rank21 {uso_rank21.iloc[-1]:.1f}, XLE at high {bool(m_xle_hi.iloc[-1])})")


def xle(h):
    return fwd_lag(C["XLE"], h, 1)


def xop(h):
    return align(fwd_lag(C["XOP"], h, 1), IDX)


def uso(h):
    return align(fwd_lag(C["USO"], h, 1), IDX)


def xle_rel(h):
    return fwd_lag(C["XLE"], h, 1) - align(fwd_lag(C["SPY"], h, 1), IDX)


for vname, vf in (("A. XLE long", xle), ("B. XOP long", xop),
                  ("C. XLE minus SPY", xle_rel), ("D. USO long", uso)):
    for mname, m in MASKS.items():
        t = IDX[m.values]
        if len(t) == 0:
            print(f"\n{vname} x {mname}: NO TRIGGERS")
            continue
        hscan(vf, t, f"{vname}  |  {mname}")

print("\n" + "=" * 78)
print("ATTRIBUTION: conjunction minus each parent-ex-conjunction (episode means, %)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    for vname, vf in (("XLE", xle), ("XOP", xop), ("XLE-SPY", xle_rel), ("USO", uso)):
        r = vf(h)
        valid = r.dropna().index
        base = float(r.loc[valid].mean())
        means = {}
        for mname, m in MASKS.items():
            t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
            if len(t) == 0:
                means[mname] = np.nan
                continue
            epi = declusters(t, h, valid)
            ep = r.loc[epi].values
            means[mname] = 100 * ep.mean()
            w = int((ep > 0).sum())
            rows.append({"h": h, "veh": vname, "mask": mname, "n_days": len(t),
                         "n": len(epi), "mean_pct": round(100 * ep.mean(), 3),
                         "edge_all_pct": round(100 * (ep.mean() - base), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
        rows.append({"h": h, "veh": vname, "mask": ">>> conj - thrust_ex",
                     "mean_pct": round(means["CONJUNCTION"]
                                       - means["thrust ex-conj (crude hot, XLE NOT at high)"], 3)})
        rows.append({"h": h, "veh": vname, "mask": ">>> conj - high_ex",
                     "mean_pct": round(means["CONJUNCTION"]
                                       - means["high ex-conj (XLE at high, crude NOT hot)"], 3)})
print(pd.DataFrame(rows).to_string(index=False))

for h in (5, 10):
    t = IDX[m_conj.values]
    cell(xle(h), t, h, "CONJUNCTION -> XLE long")
    cell(xop(h), t, h, "CONJUNCTION -> XOP long")
    cell(xle_rel(h), t, h, "CONJUNCTION -> XLE minus SPY")

print("\n=== S5 ERA SPLIT on the conjunction (h=5 episodes) ===")
for vname, vf in (("XLE", xle), ("XOP", xop), ("XLE-SPY", xle_rel)):
    r = vf(5)
    t = pd.DatetimeIndex(IDX[m_conj.values]).intersection(r.dropna().index)
    if len(t) == 0:
        continue
    epi = declusters(t, 5, r.dropna().index)
    show(era_split(epi, r.loc[epi].values), vname)

print("\n=== S5 SENSITIVITY (h=5, XLE and XOP episodes) ===")
rows = []
for thr in (75, 85, 90, 95):
    for band in (0.01, 0.02, 0.05):
        m = (align(uso_rank21 >= thr, IDX).fillna(0).astype(bool)
             & align(C["XLE"] >= (1 - band) * roll_max(C["XLE"], 252), IDX).fillna(0).astype(bool))
        for vname, vf in (("XLE", xle), ("XOP", xop)):
            r = vf(5)
            t = pd.DatetimeIndex(IDX[m.values]).intersection(r.dropna().index)
            if len(t) == 0:
                rows.append({"crude_rank>=": thr, "band": band, "veh": vname, "n": 0})
                continue
            epi = declusters(t, 5, r.dropna().index)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            rows.append({"crude_rank>=": thr, "band": band, "veh": vname,
                         "n_days": len(t), "n": len(epi),
                         "mean_pct": round(100 * ep.mean(), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== S5 CELL COUNT AND COST ===")
print("  grid here: 4 vehicles x 5 masks x 5 horizons = 100 cells, plus a")
print("  4x3x2 = 24-cell threshold sweep. Any pulse is UNCHARGED for the grid.")
print("  XLE/XOP round trip ~4-6 bps -> 3x bar ~+0.18% per episode.")
