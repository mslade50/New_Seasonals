"""C4 -- Long SPY on ^SKEW's 21-DAY return rank >= 90.

The never-checked-on-its-own-terms PARENT of watchlist 6's 5-day form, and the
control leg of the conjunction killed on 2026-09-07.

Live 2026-09-04: ^SKEW 151.58, 21d return rank 98.0 (+12.51% over 21 sessions),
trailing-252 LEVEL percentile 49.2 as of 09-03. BOTH conventions are reported
here because the registry's 2026-08-14 trap is exactly this series.

Order of business, all of it adversarial:
 0. reproduce 2026-09-07's numbers and separate RAW from EXCESS
 1. dose response by BAND -- [90,95) and [95,100] quoted separately, because a
    left-open >= claim is a monotonicity claim and today reads 98.0
 2. the MIDTERM split on the PARENT (never run; 2026-09-03 split the JOINT)
 3. the DIP decomposition -- how much of the cell is plain dip-buying
 4. definition neighbours: lookback x threshold x rank-basis grid, and the
    multiplicity count that grid implies
 5. era split, concentration, cost, event-in-window
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    align, cell, cluster_note, declusters, dial_series, era_split, fwd_lag,
    horizon_scan, jaccard, level_pct_expanding, level_pct_trailing, load_prices,
    local_control, np, pct_rank, pd, roll_max, row, show, sign_test, summarize,
)

TICKERS = ["SPY", "^SKEW", "^VIX", "IWM"]
PX = load_prices(TICKERS)
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}
SKEW = C["^SKEW"]
SPY = C["SPY"]

print("=" * 78)
print("C4  LONG SPY on ^SKEW 21-day RETURN rank >= 90   (asof 2026-09-04, lag=1)")
print("=" * 78)

# ---------------------------------------------------------------- 0. live state
r21 = pct_rank(SKEW, 21)
r5 = pct_rank(SKEW, 5)
lvl252 = level_pct_trailing(SKEW, 252)
lvlfull = level_pct_expanding(SKEW)
print("\n0. LIVE STATE, BOTH CONVENTIONS (the 2026-08-14 trap)")
print(f"  ^SKEW close                    {SKEW.iloc[-1]:.2f}   ({SKEW.index[-1].date()})")
print(f"  21-day RETURN rank (252)       {r21.iloc[-1]:.1f}   <- THE CELL'S OBJECT")
print(f"   5-day RETURN rank (252)       {r5.iloc[-1]:.1f}   (watchlist 6's object)")
print(f"  LEVEL pctile trailing-252      {lvl252.iloc[-1]:.1f}")
print(f"  LEVEL pctile FULL history      {lvlfull.iloc[-1]:.1f}")
post18 = SKEW[SKEW.index >= "2018-01-01"].dropna()
print(f"  LEVEL pctile 2018+ era         "
      f"{100 * float((post18 <= SKEW.iloc[-1]).mean()):.1f}")
print(f"  ^SKEW 21d return               "
      f"{100 * (SKEW.iloc[-1] / SKEW.dropna().iloc[-22] - 1):+.2f}%")
print(f"  medians: full {SKEW.median():.2f}  last252 {SKEW.dropna().iloc[-252:].median():.2f}"
      f"  first252 {SKEW.dropna().iloc[:252].median():.2f}")
print(f"  SPY {SPY.iloc[-1]:.2f}  off 52w high "
      f"{100 * (SPY.iloc[-1] / roll_max(SPY).iloc[-1] - 1):+.2f}%   "
      f"5d ret {100 * (SPY.iloc[-1] / SPY.iloc[-6] - 1):+.2f}%")

M90 = align(r21 >= 90, IDX).fillna(0).astype(bool)
M95 = align(r21 >= 95, IDX).fillna(0).astype(bool)
M98 = align(r21 >= 98, IDX).fillna(0).astype(bool)
B9095 = align((r21 >= 90) & (r21 < 95), IDX).fillna(0).astype(bool)
B95100 = M95.copy()
print(f"\n  mask day counts: r21>=90 {int(M90.sum())}  >=95 {int(M95.sum())}"
      f"  >=98 {int(M98.sum())}  band[90,95) {int(B9095.sum())}")
print(f"  live day in >=90 {bool(M90.iloc[-1])}   >=95 {bool(M95.iloc[-1])}"
      f"   >=98 {bool(M98.iloc[-1])}")


def spy(h):
    return fwd_lag(SPY, h, 1)


# ---------------------------------------------- 1. the pitched cell + controls
for h in (3, 5, 10):
    cell(spy(h), IDX[M90.values], h, "C4 r21>=90 -> LONG SPY", show_dates=0)

print("\n=== C4 horizon scan, r21>=90 and r21>=95 (episodes) ===")
for nm, m in (("r21>=90", M90), ("r21>=95", M95)):
    df = pd.DataFrame(horizon_scan(pd.DataFrame({"SPY": SPY}), IDX[m.values],
                                   [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10)))
    print(f"\n  {nm}")
    print(df.round(3).to_string(index=False))

# ------------------------------------------- 2. DOSE RESPONSE, band by band
print("\n" + "=" * 78)
print("1. DOSE RESPONSE -- a left-open >= X is a monotonicity claim. Today = 98.0")
print("=" * 78)
rows = []
bands = [(85, 90), (90, 95), (95, 98), (98, 101), (0, 85)]
for h in (3, 5, 10):
    for lo, hi in bands:
        m = align((r21 >= lo) & (r21 < hi), IDX).fillna(0).astype(bool)
        rows.append(row(spy(h), IDX[m.values], h, f"band [{lo},{hi})",
                        extra={"h": h}))
    for th in (85, 90, 95, 98):
        m = align(r21 >= th, IDX).fillna(0).astype(bool)
        rows.append(row(spy(h), IDX[m.values], h, f"cum  >= {th}", extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------ 3. MIDTERM SPLIT ON THE PARENT
print("\n" + "=" * 78)
print("2. MIDTERM SPLIT ON THE PARENT (2026-09-03 split the JOINT, not this)")
print("   2026 IS a midterm year. This is the leg that blocks the 5-day sibling.")
print("=" * 78)
rows = []
for h in (3, 5, 10):
    for nm, m in (("r21>=90", M90), ("r21>=95", M95), ("r21>=98", M98)):
        ret = spy(h)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = ret.loc[epi]
        mid = epi.year % 4 == 2
        for lbl, sel in (("MIDTERM", mid), ("non-midterm", ~mid)):
            v = ep.values[sel]
            w = int((v > 0).sum())
            base = float(ret.loc[valid].mean())
            rows.append({"h": h, "mask": nm, "cycle": lbl, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                         "excess_pct": round(100 * (v.mean() - base), 3) if len(v) else np.nan,
                         "med_pct": round(100 * float(np.median(v)), 3) if len(v) else np.nan,
                         "hit": round(100 * float((v > 0).mean()), 1) if len(v) else np.nan,
                         "worst_pct": round(100 * v.min(), 2) if len(v) else np.nan,
                         "rec": f"{w}-{len(v) - w}",
                         "sign_p": round(sign_test(w, len(v)), 4) if len(v) else np.nan})
print(pd.DataFrame(rows).to_string(index=False))

ret5 = spy(5)
valid5 = ret5.dropna().index
t90 = pd.DatetimeIndex(IDX[M90.values]).intersection(valid5)
epi90 = declusters(t90, 5, valid5)
mid90 = epi90[epi90.year % 4 == 2]
print(f"\n  midterm episode dates at r21>=90, h=5 (N={len(mid90)}):")
print("   " + ", ".join(str(d.date()) for d in mid90))
print("  per-midterm-year mean, r21>=90 h=5:")
ep90 = ret5.loc[epi90]
g = ep90.groupby(epi90.year)
mm = pd.DataFrame({"n": g.size(), "mean_pct": 100 * g.mean()})
print(mm[mm.index % 4 == 2].round(3).to_string())

# ------------------------------------------------- 4. THE DIP DECOMPOSITION
print("\n" + "=" * 78)
print("3. DIP DECOMPOSITION -- watchlist 6's own caveat, owed by the 21d form")
print("   plain dip-buying with no skew condition is the control that matters")
print("=" * 78)
spy5d = SPY / SPY.shift(5) - 1.0
dip = align(spy5d <= -0.01, IDX).fillna(0).astype(bool)
rows = []
for h in (5, 10):
    ret = spy(h)
    valid = ret.dropna().index
    base = float(ret.loc[valid].mean())
    for nm, m in (("A skew r21>=90 ALONE", M90),
                  ("B skew r21>=95 ALONE", M95),
                  ("C dip (SPY 5d<=-1%) ALONE, no skew", dip),
                  ("D skew>=90 AND dip", M90 & dip),
                  ("E skew>=90 and NOT dip (live-ish)", M90 & ~dip),
                  ("F dip and NOT skew>=90", dip & ~M90)):
        rows.append(row(ret, IDX[m.values], h, nm, extra={"h": h}))
    rows.append({"h": h, "label": "CTRL-b all days", "n": len(valid),
                 "mean_pct": round(100 * base, 3), "excess_pct": 0.0})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  live SPY 5d return {100 * (SPY.iloc[-1] / SPY.iloc[-6] - 1):+.2f}% "
      f"-> live day is in the {'DIP' if bool(dip.iloc[-1]) else 'NON-DIP'} bucket")

# distance-from-high leg (watchlist 6's other arm)
offhi = SPY / roll_max(SPY) - 1.0
below1 = align(offhi <= -0.01, IDX).fillna(0).astype(bool)
print("\n  watchlist 6's OTHER arm (SPY more than 1% below its 52w high):")
rows = []
for h in (5, 10):
    for nm, m in (("skew>=90 & SPY >1% off high", M90 & below1),
                  ("skew>=90 & SPY within 1% of high (LIVE, -0.99%)", M90 & ~below1)):
        rows.append(row(spy(h), IDX[m.values], h, nm, extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))
print(f"  live SPY off 52w high {100 * float(offhi.iloc[-1]):+.2f}% -> "
      f"{'>1% off' if bool(below1.iloc[-1]) else 'WITHIN 1% (the weaker bucket)'}")

# ----------------------------------------- 5. DEFINITION NEIGHBOURS + MULTIPLICITY
print("\n" + "=" * 78)
print("4. DEFINITION NEIGHBOURS: lookback x threshold x basis. If it exists")
print("   under one definition only, the definition IS the finding.")
print("=" * 78)
rows = []
for h in (5,):
    for lb in (5, 10, 15, 21, 42, 63):
        rk = pct_rank(SKEW, lb)
        for th in (85, 90, 95):
            m = align(rk >= th, IDX).fillna(0).astype(bool)
            rows.append(row(spy(h), IDX[m.values], h, f"r{lb} >= {th}",
                            extra={"h": h, "basis": "return rank", "lb": lb, "th": th}))
    for th in (85, 90, 95):
        rows.append(row(spy(h), IDX[align(lvl252 >= th, IDX).fillna(0).astype(bool).values],
                        h, f"LEVEL pctile(252) >= {th}",
                        extra={"h": h, "basis": "level-252", "lb": 252, "th": th}))
        rows.append(row(spy(h), IDX[align(lvlfull >= th, IDX).fillna(0).astype(bool).values],
                        h, f"LEVEL pctile(FULL) >= {th}",
                        extra={"h": h, "basis": "level-full", "lb": 0, "th": th}))
df = pd.DataFrame(rows)
print(df[["h", "basis", "lb", "th", "n_days", "n", "mean_pct", "excess_pct",
          "hit", "worst_pct", "rec", "sign_p"]].to_string(index=False))
pos = int((df["excess_pct"] > 0).sum())
print(f"\n  cells with POSITIVE excess: {pos} of {len(df)}   "
      f"median excess {df['excess_pct'].median():+.3f}pp   "
      f"max {df['excess_pct'].max():+.3f}pp on "
      f"'{df.loc[df['excess_pct'].idxmax(), 'label']}'")
print("  MULTIPLICITY: 6 lookbacks x 3 thresholds x 2 rank bases (+2 level")
print("  conventions) x 6 horizons = 6*3*6 + 2*3*6 = 144 cells reachable by a")
print("  walk of this grid, before any conditioner.")

# --------------------------------------------------------- 6. era + concentration
print("\n" + "=" * 78)
print("5. ERA SPLIT + CONCENTRATION on the pitched cell (r21>=90, h=5)")
print("=" * 78)
show(era_split(epi90, ep90.values), "r21>=90 h=5 episodes")
print("  " + cluster_note(epi90, ep90.values, k=2))
byyr = ep90.groupby(epi90.year).sum().sort_values(ascending=False)
print(f"  top 3 years by summed return: "
      f"{dict((int(y), round(100 * v, 2)) for y, v in byyr.head(3).items())}")
print(f"  total {100 * ep90.sum():+.2f}pp over {len(ep90)} episodes; "
      f"drop the best 3 years -> "
      f"{100 * ep90[~np.isin(epi90.year, byyr.head(3).index)].mean():+.3f}% mean "
      f"(n={int((~np.isin(epi90.year, byyr.head(3).index)).sum())})")

# ------------------------------------------------------- 7. gate attribution
print("\n" + "=" * 78)
print("6. GATE ATTRIBUTION: VIX level, the fragility dial, and mask overlap")
print("=" * 78)
vix = align(C["^VIX"], IDX)
dial = dial_series(IDX)
print(f"  live VIX {vix.iloc[-1]:.2f}   live dial(10dMA 63d) {dial.iloc[-1]:.1f}")
rows = []
for h in (5,):
    for nm, m in (("skew>=90 & VIX < 16 (live 15.3)",
                   M90 & align(vix < 16, IDX).fillna(0).astype(bool)),
                  ("skew>=90 & VIX >= 16", M90 & align(vix >= 16, IDX).fillna(0).astype(bool)),
                  ("skew>=90 & dial >= 80 (live 88.0)",
                   M90 & align(dial >= 80, IDX).fillna(0).astype(bool)),
                  ("skew>=90 & dial < 80 (2016+ only)",
                   M90 & align(dial < 80, IDX).fillna(0).astype(bool)),
                  ("skew>=90, 2016+ only (dial era)",
                   M90 & pd.Series(IDX >= pd.Timestamp("2016-07-05"), index=IDX))):
        rows.append(row(spy(h), IDX[m.values], h, nm, extra={"h": h}))
print(pd.DataFrame(rows).to_string(index=False))

dialhi = align(dial >= 80, IDX).fillna(0).astype(bool)
j, inter, na, nb = jaccard(M90, dialhi)
print(f"\n  Jaccard(skew r21>=90, dial>=80) = {j:.3f}  "
      f"(intersection {inter} days; skew {na}, dial {nb})")

print("\n7. COST: SPY 1 leg, ~1.5-2 bps round trip, 3x bar = 4.5-6 bps.")
print("   Compare against the EXCESS, not the raw mean.")
print("\nDONE C4")
