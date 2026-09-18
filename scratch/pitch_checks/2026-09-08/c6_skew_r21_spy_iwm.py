"""C6 -- Long SPY / short IWM on the ^SKEW 21-day return-rank spike.

Rationale to TEST, not assume: an index-level tail bid should show up as
large-over-small if it is hedging demand rather than a directional view.

The registry's live blocker is explicit and must be settled with a number:
2026-09-07's 10_dial_spy_iwm.py found SPY-over-IWM's ONLY dial content in the
[56,70) band (49 episodes, +0.594%, 35-14, sign p 0.0019) with [70,80) at
+0.071% on 6-8 and a return-on-dial slope of -0.0053pp/point. The dial is 88.0
today. So the first question is whether the skew mask and the dial mask are the
SAME mask -- Jaccard, reported before anything else -- and the second is
whether SPY-over-IWM on skew has any content once the dial band is controlled.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    align, cell, cluster_note, declusters, dial_series, era_split, fwd_lag,
    jaccard, load_prices, local_control, np, pct_rank, pd, roll_max, row, show,
    sign_test, summarize, vehicle_ret,
)

TICKERS = ["SPY", "IWM", "^SKEW", "^VIX"]
PX = load_prices(TICKERS)
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}
SKEW, SPY, IWM = C["^SKEW"], C["SPY"], C["IWM"]
PANEL = pd.DataFrame({"SPY": SPY, "IWM": IWM}).dropna()

print("=" * 78)
print("C6  LONG SPY / SHORT IWM on ^SKEW 21d rank   (asof 2026-09-04, lag=1)")
print("=" * 78)

r21 = pct_rank(SKEW, 21)
M90 = align(r21 >= 90, IDX).fillna(0).astype(bool)
M95 = align(r21 >= 95, IDX).fillna(0).astype(bool)
M98 = align(r21 >= 98, IDX).fillna(0).astype(bool)
dial = dial_series(IDX)
print(f"  live r21 {r21.iloc[-1]:.1f}   dial(10dMA 63d) {dial.iloc[-1]:.1f}   "
      f"SPY off high {100 * (SPY.iloc[-1] / roll_max(SPY).iloc[-1] - 1):+.2f}%  "
      f"IWM off high {100 * (IWM.iloc[-1] / roll_max(IWM).iloc[-1] - 1):+.2f}%")

# --------------------------------------------- 0. THE MASK-IDENTITY QUESTION
print("\n" + "=" * 78)
print("0. IS THE SKEW MASK THE DIAL MASK? (the registry's standing blocker)")
print("=" * 78)
dial_days = dial.dropna().index
for lo, hi, lbl in [(56, 70, "dial [56,70) -- the ONLY band with content"),
                    (70, 80, "dial [70,80)"),
                    (80, 999, "dial >=80 (LIVE 88.0)"),
                    (50, 999, "dial >=50")]:
    dm = align((dial >= lo) & (dial < hi), IDX).fillna(0).astype(bool)
    for nm, m in (("skew>=90", M90), ("skew>=95", M95)):
        sub = m & pd.Series(IDX.isin(dial_days), index=IDX)
        j, inter, na, nb = jaccard(sub, dm)
        print(f"  Jaccard({nm} n2016+={na:4d}, {lbl} n={nb:4d}) = {j:.3f}"
              f"   intersection {inter} days")
print("\n  (the fragility series starts 2016-07-05, so every dial-conditioned")
print("   count above is out of ~2,550 sessions, not the 6,700 the skew mask has)")

# ------------------------------------------------------ 1. the pair, unconditional
def pair(h):
    return vehicle_ret(PANEL, [("SPY", 1.0), ("IWM", -1.0)], h, 1)


print("\n" + "=" * 78)
print("1. THE PAIR vs its own drift. SPY-over-IWM has a large secular drift;")
print("   the control is what the pair does unconditionally, never zero.")
print("=" * 78)
for h in (3, 5, 10):
    cell(pair(h), IDX[M95.values], h, "C6 skew r21>=95 -> LONG SPY / SHORT IWM")

print("\n=== C6 horizon scan (episodes) ===")
rows = []
for nm, m in (("skew>=90", M90), ("skew>=95", M95), ("skew>=98 LIVE band", M98)):
    for h in (1, 2, 3, 5, 7, 10):
        rows.append(row(pair(h), IDX[m.values], h, nm, extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------ 2. dose response
print("\n" + "=" * 78)
print("2. DOSE RESPONSE by band. Live = 98.0.")
print("=" * 78)
rows = []
for h in (3, 5, 10):
    for lo, hi in [(85, 90), (90, 95), (95, 98), (98, 101), (0, 85)]:
        m = align((r21 >= lo) & (r21 < hi), IDX).fillna(0).astype(bool)
        rows.append(row(pair(h), IDX[m.values], h, f"band [{lo},{hi})", extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ----------------------------------------------- 3. dial-band conditioned cells
print("\n" + "=" * 78)
print("3. THE SKEW CELL INSIDE EACH DIAL BAND (2016+ only, n is small by")
print("   construction -- report it, do not hide behind it)")
print("=" * 78)
rows = []
for h in (5, 10):
    for lo, hi, lbl in [(0, 56, "dial <56"), (56, 70, "dial [56,70)"),
                        (70, 80, "dial [70,80)"), (80, 999, "dial >=80 LIVE")]:
        dm = align((dial >= lo) & (dial < hi), IDX).fillna(0).astype(bool)
        rows.append(row(pair(h), IDX[(M90 & dm).values], h,
                        f"skew>=90 & {lbl}", extra={"h": h}))
        rows.append(row(pair(h), IDX[dm.values], h, f"{lbl} ALONE (no skew)",
                        extra={"h": h}))
    rows.append(row(pair(h), IDX[(M90 & pd.Series(IDX.isin(dial_days), index=IDX)).values],
                    h, "skew>=90, 2016+ only", extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------- 4. midterm + era
print("\n" + "=" * 78)
print("4. MIDTERM SPLIT and ERA SPLIT on the pair")
print("=" * 78)
rows = []
for h in (5, 10):
    for nm, m in (("skew>=90", M90), ("skew>=95", M95), ("skew>=98", M98)):
        ret = pair(h)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = ret.loc[epi]
        base = float(ret.loc[valid].mean())
        for lbl, sel in (("MIDTERM", epi.year % 4 == 2),
                         ("non-midterm", epi.year % 4 != 2),
                         ("pre-2018", epi < pd.Timestamp("2018-01-01")),
                         ("2018+", epi >= pd.Timestamp("2018-01-01"))):
            v = ep.values[sel]
            if len(v) == 0:
                continue
            w = int((v > 0).sum())
            rows.append({"h": h, "mask": nm, "split": lbl, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "excess_pct": round(100 * (v.mean() - base), 3),
                         "med_pct": round(100 * float(np.median(v)), 3),
                         "hit": round(100 * float((v > 0).mean()), 1),
                         "worst_pct": round(100 * v.min(), 2),
                         "rec": f"{w}-{len(v) - w}",
                         "sign_p": round(sign_test(w, len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------- 5. is the pair just the SPY leg again?
print("\n" + "=" * 78)
print("5. LEG DECOMPOSITION: does the SHORT IWM leg add anything, or is this")
print("   C4 wearing a hedge?")
print("=" * 78)
rows = []
for h in (5, 10):
    for nm, m in (("skew>=95", M95), ("skew>=98", M98)):
        t = IDX[m.values]
        rows.append(row(vehicle_ret(PANEL, [("SPY", 1.0)], h, 1), t, h,
                        f"{nm}: SPY leg only", extra={"h": h}))
        rows.append(row(vehicle_ret(PANEL, [("IWM", 1.0)], h, 1), t, h,
                        f"{nm}: IWM leg only", extra={"h": h}))
        rows.append(row(pair(h), t, h, f"{nm}: SPY - IWM", extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------ 6. concentration
print("\n" + "=" * 78)
print("6. CONCENTRATION + DECLUSTER STABILITY (skew>=95, h=5)")
print("=" * 78)
ret5 = pair(5)
valid5 = ret5.dropna().index
t95 = pd.DatetimeIndex(IDX[M95.values]).intersection(valid5)
epi95 = declusters(t95, 5, valid5)
ep95 = ret5.loc[epi95]
base = float(ret5.loc[valid5].mean())
print("  " + cluster_note(epi95, ep95.values, k=2))
byyr = ep95.groupby(epi95.year).sum().sort_values(ascending=False)
print(f"  best years {dict((int(y), round(100 * v, 1)) for y, v in byyr.head(3).items())}")
keep = ~np.isin(epi95.year, byyr.head(3).index)
print(f"  drop the best 3 years -> mean {100 * ep95.values[keep].mean():+.3f}% "
      f"excess {100 * (ep95.values[keep].mean() - base):+.3f}pp n={int(keep.sum())}")
for g in (5, 10, 21, 42):
    e = declusters(t95, g, valid5)
    print(f"  min_gap {g:>2d} td -> n={len(e):3d}  mean {100 * ret5.loc[e].mean():+.3f}%"
          f"  excess {100 * (ret5.loc[e].mean() - base):+.3f}pp")

print("\n7. COST: 2 legs, SPY ~1.5-2 bps + IWM ~2-3 bps round trip. 3x bar")
print("   on the pair = ~11-15 bps. Compare against EXCESS, not raw mean.")
print("\nDONE C6")
