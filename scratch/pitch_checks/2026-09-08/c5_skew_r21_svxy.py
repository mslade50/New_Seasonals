"""C5 -- Long SVXY on the same ^SKEW 21-day return-rank state.

The instrument-translation form of C4. The registry's 2026-09-07 control leg
put it at +1.374% RAW at h=5 (59-36, sign p 0.0117); this asks what the EXCESS
is against SVXY's own violent positive drift, and whether the cell survives the
three legs that killed the SPY parent (threshold, midterm, dip).

INSTRUMENT CAVEAT carried throughout: SVXY was a -1.0x product until
2018-02-28 and -0.5x after. The pre-2018 leg is a DIFFERENT INSTRUMENT with
roughly twice the exposure, so an era split cut at the rebalance is
load-bearing rather than decoration, and any raw mean pooled across it is a
blend of two vehicles.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    align, cell, cluster_note, declusters, dial_series, era_split, fwd_lag,
    horizon_scan, jaccard, level_pct_trailing, level_pct_expanding, load_prices,
    local_control, np, pct_rank, pd, roll_max, row, show, sign_test, summarize,
)

TICKERS = ["SPY", "SVXY", "^SKEW", "^VIX"]
PX = load_prices(TICKERS)
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}
SKEW = C["^SKEW"]
SPY = C["SPY"]
SVXY = align(C["SVXY"], IDX)
REBAL = pd.Timestamp("2018-02-28")

print("=" * 78)
print("C5  LONG SVXY on ^SKEW 21-day RETURN rank   (asof 2026-09-04, lag=1)")
print("=" * 78)
print(f"  SVXY first bar {C['SVXY'].dropna().index[0].date()}  "
      f"last {C['SVXY'].dropna().index[-1].date()}   "
      f"leverage break {REBAL.date()} (-1.0x -> -0.5x)")

r21 = pct_rank(SKEW, 21)
M90 = align(r21 >= 90, IDX).fillna(0).astype(bool)
M95 = align(r21 >= 95, IDX).fillna(0).astype(bool)
M98 = align(r21 >= 98, IDX).fillna(0).astype(bool)
print(f"  live r21 {r21.iloc[-1]:.1f}   SVXY {C['SVXY'].iloc[-1]:.2f}  "
      f"off 52w high {100 * (SVXY.iloc[-1] / roll_max(SVXY).iloc[-1] - 1):+.2f}%")


def svxy(h):
    return align(fwd_lag(C["SVXY"], h, 1), IDX)


# ------------------------------------------------------ 1. cell vs controls
for h in (3, 5, 10):
    cell(svxy(h), IDX[M95.values], h, "C5 r21>=95 -> LONG SVXY")

print("\n=== C5 horizon scan (episodes) ===")
for nm, m in (("r21>=90", M90), ("r21>=95", M95), ("r21>=98 (LIVE band)", M98)):
    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        rows.append(row(svxy(h), IDX[m.values], h, nm, extra={"h": h}))
    print(f"\n  {nm}")
    print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------- 2. dose response
print("\n" + "=" * 78)
print("1. DOSE RESPONSE by band. Live reading is 98.0.")
print("=" * 78)
rows = []
for h in (3, 5, 10):
    for lo, hi in [(85, 90), (90, 95), (95, 98), (98, 101), (0, 85)]:
        m = align((r21 >= lo) & (r21 < hi), IDX).fillna(0).astype(bool)
        rows.append(row(svxy(h), IDX[m.values], h, f"band [{lo},{hi})", extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- 3. ERA: the leverage break
print("\n" + "=" * 78)
print("2. ERA SPLIT AT THE LEVERAGE BREAK (2018-02-28), not the calendar year")
print("   pre-break SVXY is -1.0x: a different instrument, twice the exposure")
print("=" * 78)
rows = []
for h in (5, 10):
    for nm, m in (("r21>=95", M95), ("r21>=98", M98)):
        ret = svxy(h)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = ret.loc[epi]
        for lbl, sel in (("-1.0x era (pre 2018-02-28)", epi < REBAL),
                         ("-0.5x era (LIVE instrument)", epi >= REBAL)):
            v = ep.values[sel]
            if len(v) == 0:
                continue
            sub = valid[(valid < REBAL) if "pre" in lbl else (valid >= REBAL)]
            base = float(ret.loc[sub].mean())
            w = int((v > 0).sum())
            rows.append({"h": h, "mask": nm, "era": lbl, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "era_base_pct": round(100 * base, 3),
                         "excess_pct": round(100 * (v.mean() - base), 3),
                         "med_pct": round(100 * float(np.median(v)), 3),
                         "hit": round(100 * float((v > 0).mean()), 1),
                         "worst_pct": round(100 * v.min(), 2),
                         "rec": f"{w}-{len(v) - w}",
                         "sign_p": round(sign_test(w, len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------- 4. midterm split
print("\n" + "=" * 78)
print("3. MIDTERM SPLIT (2026 is midterm). SVXY history covers 2014/2018/2022/2026.")
print("=" * 78)
rows = []
for h in (5, 10):
    for nm, m in (("r21>=90", M90), ("r21>=95", M95), ("r21>=98", M98)):
        ret = svxy(h)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = ret.loc[epi]
        base = float(ret.loc[valid].mean())
        mid = epi.year % 4 == 2
        for lbl, sel in (("MIDTERM", mid), ("non-midterm", ~mid)):
            v = ep.values[sel]
            if len(v) == 0:
                continue
            w = int((v > 0).sum())
            rows.append({"h": h, "mask": nm, "cycle": lbl, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "excess_pct": round(100 * (v.mean() - base), 3),
                         "med_pct": round(100 * float(np.median(v)), 3),
                         "hit": round(100 * float((v > 0).mean()), 1),
                         "worst_pct": round(100 * v.min(), 2),
                         "rec": f"{w}-{len(v) - w}",
                         "sign_p": round(sign_test(w, len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- 5. dip + near-high buckets
print("\n" + "=" * 78)
print("4. THE SAME GATE ATTRIBUTION THAT KILLED C4: dip / distance from high")
print("=" * 78)
spy5d = SPY / SPY.shift(5) - 1.0
dip = align(spy5d <= -0.01, IDX).fillna(0).astype(bool)
offhi = SPY / roll_max(SPY) - 1.0
below1 = align(offhi <= -0.01, IDX).fillna(0).astype(bool)
rows = []
for h in (5, 10):
    ret = svxy(h)
    valid = ret.dropna().index
    for nm, m in (("A skew>=95 ALONE", M95),
                  ("B dip ALONE, no skew", dip),
                  ("C skew>=95 AND dip", M95 & dip),
                  ("D skew>=95 NOT dip (LIVE bucket)", M95 & ~dip),
                  ("E skew>=95 & SPY >1% off high", M95 & below1),
                  ("F skew>=95 & SPY within 1% of high (LIVE)", M95 & ~below1)):
        rows.append(row(ret, IDX[m.values], h, nm, extra={"h": h}))
    rows.append({"h": h, "label": "CTRL-b SVXY all days", "n": len(valid),
                 "mean_pct": round(100 * float(ret.loc[valid].mean()), 3),
                 "excess_pct": 0.0})
print(pd.DataFrame(rows).to_string(index=False))
print(f"  live: SPY 5d {100 * (SPY.iloc[-1] / SPY.iloc[-6] - 1):+.2f}% "
      f"({'DIP' if bool(dip.iloc[-1]) else 'NON-DIP'}), "
      f"SPY off high {100 * float(offhi.iloc[-1]):+.2f}% "
      f"({'>1% off' if bool(below1.iloc[-1]) else 'WITHIN 1%'})")

# ------------------------------------ 6. is it just the SPY leg? beta teardown
print("\n" + "=" * 78)
print("5. IS THERE A VOL-SPECIFIC RESIDUAL, or is it the SPY move levered?")
print("   regress SVXY h-forward on SPY h-forward over ALL days, then score the")
print("   cell's RESIDUAL. (the 2026-09-02 b4_c12 method)")
print("=" * 78)
for h in (5, 10):
    rs, rv = SPY.pipe(fwd_lag, h, 1), svxy(h)
    d = pd.DataFrame({"spy": align(rs, IDX), "svxy": rv}).dropna()
    beta = float(np.polyfit(d["spy"], d["svxy"], 1)[0])
    alpha = float(np.polyfit(d["spy"], d["svxy"], 1)[1])
    resid = d["svxy"] - (alpha + beta * d["spy"])
    r2 = float(np.corrcoef(d["spy"], d["svxy"])[0, 1] ** 2)
    print(f"\n  h={h}: SVXY = {alpha * 100:+.3f}% + {beta:.2f} x SPY   "
          f"R2 {r2:.3f}  (n={len(d)})")
    for nm, m in (("r21>=95", M95), ("r21>=98", M98)):
        t = pd.DatetimeIndex(IDX[m.values]).intersection(d.index)
        epi = declusters(t, h, d.index)
        v = resid.loc[epi].values
        w = int((v > 0).sum())
        print(f"    {nm:<10s} RESIDUAL mean {100 * v.mean():+.3f}%  med "
              f"{100 * float(np.median(v)):+.3f}%  n={len(v)}  {w}-{len(v) - w}  "
              f"sign p {sign_test(w, len(v)):.4f}")

# ------------------------------------------------------------ 7. concentration
print("\n" + "=" * 78)
print("6. CONCENTRATION + WORST WINDOW on the pitched cell (r21>=95, h=5)")
print("=" * 78)
ret5 = svxy(5)
valid5 = ret5.dropna().index
t95 = pd.DatetimeIndex(IDX[M95.values]).intersection(valid5)
epi95 = declusters(t95, 5, valid5)
ep95 = ret5.loc[epi95]
print("  " + cluster_note(epi95, ep95.values, k=2))
byyr = ep95.groupby(epi95.year).sum().sort_values(ascending=False)
print(f"  years: {dict((int(y), round(100 * v, 1)) for y, v in byyr.items())}")
keep = ~np.isin(epi95.year, byyr.head(2).index)
base = float(ret5.loc[valid5].mean())
print(f"  drop the best 2 years -> mean {100 * ep95.values[keep].mean():+.3f}% "
      f"(excess {100 * (ep95.values[keep].mean() - base):+.3f}pp, n={int(keep.sum())})")
for g in (5, 10, 21, 42):
    e = declusters(t95, g, valid5)
    print(f"  min_gap {g:>2d} td -> n={len(e):3d}  mean {100 * ret5.loc[e].mean():+.3f}%"
          f"  excess {100 * (ret5.loc[e].mean() - base):+.3f}pp")

print("\n7. COST: SVXY round trip ~8-12 bps (thin book). 3x bar = ~24-36 bps.")
print("\nDONE C5")
