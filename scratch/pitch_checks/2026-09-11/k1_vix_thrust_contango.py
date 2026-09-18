"""A5 ROUND 1 -- long SPY on a five-session ^VIX thrust that leaves the term
structure in CONTANGO.

Live (2026-09-10, ON SPY's CALENDAR, which is the whole point -- the tape's
+24.58% differences ^VIX on ^VIX's own calendar and picks up the Labor Day bar):
    ^VIX 5d  +17.37%       ^VIX 17.84   ^VIX3M 19.73   ratio 0.9042
    ^VIX 21-day LEVEL rank 100.0

STEP 0 IS MANDATORY AND COMES FIRST: state the trailing-252 PIT percentile of
the live 5-session ^VIX return BEFORE choosing the rank threshold. Three weeks
running a cross-sectional / spread statistic has been proposed here without it.

Then the attacks:
  1. battery on the pre-specified cell
  2. GATE ATTRIBUTION on the contango leg -- what do the INVERTED-curve thrusts
     pay? If the complement pays as much, the curve gate is decoration and this
     is dip-buying wearing a volatility label (explicit registry trap).
  3. The registry baseline the label must beat: plain SPY 5d <= -1% pays
     +0.220pp on N=512. Measured here from scratch, not quoted.
  4. Is the whole thing SPY's own five-day drawdown? 2x2 of thrust x dip.
  5. Any vol-vehicle content must clear SVXY = a + b*SPY. Run it so the
     temptation is closed rather than left open.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
TK = ["SPY", "^VIX", "^VIX3M", "SVXY"]
raw = load_prices(TK)
SP = raw["SPY"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(SP).ffill() for t in TK})
PX = PX.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
# SVXY must NOT be forward-filled before its inception; restore the NaNs
PX["SVXY"] = raw["SVXY"]["Close"].reindex(SP)
PX["SVXY"] = PX["SVXY"].where(SP >= raw["SVXY"].index[0])

vix, v3m = PX["VIX"], PX["VIX3M"]
ratio = vix / v3m
v5 = vix / vix.shift(5) - 1.0
spy5 = PX["SPY"] / PX["SPY"].shift(5) - 1.0

print("=" * 78)
print("0. THE PIT PERCENTILE OF THE LIVE READING, STATED BEFORE THE THRESHOLD")
print("=" * 78)
r5_rank = rolling_on_valid(v5, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  ^VIX 5-session return (SPY calendar) = {100*v5.loc[ASOF]:+.2f}%")
print(f"  trailing-252 PIT percentile          = {r5_rank.loc[ASOF]:.1f}")
print(f"  full-history percentile              = "
      f"{100*(v5.dropna() < v5.loc[ASOF]).mean():.1f}")
print(f"  ^VIX/^VIX3M = {ratio.loc[ASOF]:.4f}   "
      f"(trailing-252 pctile of the RATIO = "
      f"{rolling_on_valid(ratio, lambda x: x.rolling(252).rank(pct=True)*100).loc[ASOF]:.1f})")
print(f"  SPY 5-session return                 = {100*spy5.loc[ASOF]:+.2f}%")
LIVE_RANK = float(r5_rank.loc[ASOF])
THR = 90.0 if LIVE_RANK >= 90 else round(np.floor(LIVE_RANK / 5) * 5, 0)
print(f"\n  --> the live reading sits at pctile {LIVE_RANK:.1f}; the "
      f"pre-specified rank threshold is >= {THR:.0f} (the live reading must "
      f"itself clear it)")

CONT = (ratio <= 0.95)
THRUST = (r5_rank >= THR)
CELL = (THRUST & CONT).fillna(False)
print(f"  cell fires today? {bool(CELL.loc[ASOF])}")

print("\n" + "=" * 78)
print("1. THE BATTERY, h=1,3,5,10")
print("=" * 78)
for h in (1, 3, 5, 10):
    battery(PX, CELL, [("SPY", 1.0)], h=h,
            title=f"A5 long SPY | VIX 5d rank>={THR:.0f} AND VIX/VIX3M<=0.95",
            cost_bps=2.0, min_gap=10,
            event_kinds=("cpi", "fomc_decision"))

print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION ON THE CONTANGO LEG")
print("=" * 78)
INV = (THRUST & ~CONT).fillna(False)
for h in (1, 3, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    for lbl, m in (("THRUST & contango (retained)", CELL),
                   ("THRUST & INVERTED (discarded)", INV),
                   ("THRUST parent, any curve", THRUST.fillna(False)),
                   ("contango alone, no thrust", (CONT & ~THRUST).fillna(False))):
        d = pd.DatetimeIndex(SP[m.reindex(SP, fill_value=False).values]).intersection(valid)
        e = declusters(d, 10, valid)
        v = r.loc[e].values
        if len(v) == 0:
            continue
        w = int((v > 0).sum())
        print(f"  h={h:2d} {lbl:<32s} epi n={len(v):4d} mean {100*np.mean(v):+.3f}% "
              f"hit {100*(v>0).mean():.1f}%  rec {w}-{len(v)-w}  "
              f"sign p {sign_test(w, len(v)):.4f}")
    base = r.loc[valid].mean()
    print(f"       [all-days control {100*base:+.3f}%]")

print("\n" + "=" * 78)
print("3. THE REGISTRY BASELINE, RE-DERIVED: plain SPY 5d <= -1%")
print("=" * 78)
DIP = (spy5 <= -0.01).fillna(False)
for h in (1, 3, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    d = pd.DatetimeIndex(SP[DIP.values]).intersection(valid)
    print(f"  h={h:2d} SPY 5d<=-1%: day-level n={len(d)} {100*r.loc[d].mean():+.3f}% "
          f"edge {100*(r.loc[d].mean()-base):+.3f}pp   |  cell "
          f"{100*r.loc[pd.DatetimeIndex(SP[CELL.values]).intersection(valid)].mean():+.3f}% "
          f"edge {100*(r.loc[pd.DatetimeIndex(SP[CELL.values]).intersection(valid)].mean()-base):+.3f}pp")

print("\n" + "=" * 78)
print("4. IS IT JUST THE DIP? 2x2 of thrust x SPY's own 5-day drawdown")
print("=" * 78)
for h in (1, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    rows = []
    for tl, tm in (("thrust", CELL), ("no thrust", (~CELL).fillna(False))):
        for dl, dm in (("dip", DIP), ("no dip", ~DIP)):
            d = pd.DatetimeIndex(SP[(tm & dm).reindex(SP, fill_value=False).values]).intersection(valid)
            e = declusters(d, 10, valid)
            s = summarize(r.loc[e].values, f"{tl} x {dl}")
            s["n_days"] = len(d)
            rows.append(s)
    show(rows, f"h={h} (episode level, 10td decluster)")
    # the residual question: does the thrust add anything INSIDE the dip cell?
    dd = pd.DatetimeIndex(SP[(DIP & CELL).reindex(SP, fill_value=False).values]).intersection(valid)
    dn = pd.DatetimeIndex(SP[(DIP & ~CELL).reindex(SP, fill_value=False).values]).intersection(valid)
    print(f"    within SPY 5d<=-1%: WITH thrust n={len(dd)} "
          f"{100*r.loc[dd].mean():+.3f}%  vs WITHOUT thrust n={len(dn)} "
          f"{100*r.loc[dn].mean():+.3f}%   -> thrust contributes "
          f"{100*(r.loc[dd].mean()-r.loc[dn].mean()):+.3f}pp")

print("\n" + "=" * 78)
print("5. MANDATORY SVXY RESIDUAL (closing the vol-vehicle temptation)")
print("=" * 78)
for h in (1, 5, 10):
    rs = fwd_lag(PX["SPY"], h, 1)
    rv = fwd_lag(PX["SVXY"], h, 1)
    d = pd.concat([rs, rv], axis=1, keys=["spy", "svxy"]).dropna()
    b, a = np.polyfit(d["spy"], d["svxy"], 1)
    resid = d["svxy"] - (a + b * d["spy"])
    cell_d = pd.DatetimeIndex(SP[CELL.values]).intersection(d.index)
    e = declusters(cell_d, 10, d.index)
    rr = resid.loc[e]
    print(f"  h={h:2d}: SVXY = {100*a:+.3f}% + {b:.2f}*SPY (R2 "
          f"{np.corrcoef(d['spy'], d['svxy'])[0,1]**2:.3f}); cell residual "
          f"n={len(rr)} {100*rr.mean():+.3f}% hit {100*(rr>0).mean():.1f}% "
          f"t {rr.mean()/(rr.std(ddof=1)/np.sqrt(len(rr))):+.2f}")

print("\n" + "=" * 78)
print("6. THRESHOLD NEIGHBOURS on BOTH legs")
print("=" * 78)
for h in (1, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    print(f"  h={h}  (all-days {100*base:+.3f}%)")
    for thr in (80, 85, 90, 95, 98):
        for cg in (0.90, 0.95, 1.00, 99.0):
            m = ((r5_rank >= thr) & (ratio <= cg)).fillna(False)
            d = pd.DatetimeIndex(SP[m.values]).intersection(valid)
            e = declusters(d, 10, valid)
            if len(e) == 0:
                continue
            v = r.loc[e].values
            tag = "any curve" if cg > 10 else f"<= {cg:.2f}"
            print(f"    rank>={thr:2d} ratio {tag:<9s} epi n={len(e):3d} "
                  f"{100*np.mean(v):+.3f}% edge {100*(np.mean(v)-base):+.3f}pp "
                  f"hit {100*(v>0).mean():.0f}%")
