"""C4b -- the LIVE intersection, and the event exposure the hold actually carries.

C4's parent is flat and its midterm leg is negative. This asks the only
question that decides a trade today: what has the state McKinley would
actually be entering paid? Today is, jointly:

    ^SKEW r21 = 98.0   (band [98,101), not the left-open >= 90)
    midterm year
    SPY 5d +0.11%      (NON-dip)
    SPY -0.99% off its 52w high  (WITHIN 1%, the weaker bucket)

plus PPI at +2 td, CPI at +3 td and FOMC at +6 td inside any hold of 5 or 10
sessions. Both the intersection and the event split are reported, on SPY and on
the SPY/IWM pair, with the counts stated rather than hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c456_common import (  # noqa: E402
    align, declusters, fwd_lag, load_prices, np, pct_rank, pd, roll_max, row,
    show, sign_test, summarize, vehicle_ret,
)
from pitch_lab import event_in_window  # noqa: E402

PX = load_prices(["SPY", "IWM", "^SKEW"])
IDX = PX["SPY"].index
SPY, IWM, SKEW = PX["SPY"]["Close"], PX["IWM"]["Close"], PX["^SKEW"]["Close"]
PANEL = pd.DataFrame({"SPY": SPY, "IWM": IWM}).dropna()

r21 = pct_rank(SKEW, 21)
M90 = align(r21 >= 90, IDX).fillna(0).astype(bool)
M98 = align((r21 >= 98), IDX).fillna(0).astype(bool)
mid = pd.Series(IDX.year % 4 == 2, index=IDX)
dip = align(SPY / SPY.shift(5) <= 0.99, IDX).fillna(0).astype(bool)
nearhi = align(SPY / roll_max(SPY) > 0.99, IDX).fillna(0).astype(bool)

print("=" * 78)
print("C4b  THE LIVE INTERSECTION (skew band 98+, midterm, non-dip, near high)")
print("=" * 78)
print(f"  live: r21 {r21.iloc[-1]:.1f} | midterm {bool(mid.iloc[-1])} | "
      f"dip {bool(dip.iloc[-1])} | within 1% of 52w high {bool(nearhi.iloc[-1])}")

LAYERS = [
    ("L0 skew r21>=90 (the pitched mask)", M90),
    ("L1 + band [98,101] (the live reading)", M98),
    ("L2 + midterm year", M98 & mid),
    ("L3 + NON-dip (SPY 5d > -1%)", M98 & mid & ~dip),
    ("L4 + SPY within 1% of 52w high", M98 & mid & ~dip & nearhi),
]
for veh, legs in (("LONG SPY", [("SPY", 1.0)]),
                  ("LONG SPY / SHORT IWM", [("SPY", 1.0), ("IWM", -1.0)])):
    rows = []
    for h in (5, 10):
        for lbl, m in LAYERS:
            rows.append(row(vehicle_ret(PANEL, legs, h, 1), IDX[m.values], h,
                            lbl, extra={"h": h}))
    print(f"\n--- {veh}: the conditioning ladder ---")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("EVENT EXPOSURE: PPI +2td, CPI +3td, FOMC +6td all sit inside the hold")
print("=" * 78)
for h in (5, 10):
    ret = fwd_lag(SPY, h, 1)
    valid = ret.dropna().index
    for lbl, m in (("skew>=90", M90), ("skew band 98+", M98)):
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = ret.loc[epi].values
        fl = event_in_window(epi, IDX, h, 1, ("cpi", "ppi", "fomc_decision"))
        base = float(ret.loc[valid].mean())
        for nm, sel in ((f"print/FOMC IN hold (N={int(fl.sum())})", fl),
                        (f"OUT (N={int((~fl).sum())})", ~fl)):
            v = ep[sel]
            if not len(v):
                continue
            w = int((v > 0).sum())
            print(f"  h={h:>2d} {lbl:<14s} {nm:<28s} mean {100 * v.mean():+.3f}% "
                  f"excess {100 * (v.mean() - base):+.3f}pp med "
                  f"{100 * float(np.median(v)):+.3f}% {w}-{len(v) - w} "
                  f"sign p {sign_test(w, len(v)):.4f}")

print("\n" + "=" * 78)
print("BOOK OVERLAP: the event sleeve's T2 stages SHORT SPY 10% NAV on")
print("2026-09-10 (midterm pre-FOMC, gate SPY 21d rank < 50 -> reads 31.3).")
print("Any long-SPY hold of >= 3 sessions from today is opposite the house")
print("sleeve for most of its life. Quantified: how did the skew cell do when")
print("its hold contained the T2 window (4 sessions before an FOMC decision)?")
print("=" * 78)
ev = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "macro_events.csv")
ev["date"] = pd.to_datetime(ev["date"])
fomc = ev[ev["event"] == "fomc_decision"]["date"]
pos = pd.Series(range(len(IDX)), index=IDX)
t2_days = set()
for d in fomc:
    p = pos.get(d)
    if p is None or p - 4 < 0:
        continue
    t2_days.add(IDX[p - 4])
t2 = pd.Series(IDX.isin(sorted(t2_days)), index=IDX)
for h in (5, 10):
    ret = fwd_lag(SPY, h, 1)
    valid = ret.dropna().index
    base = float(ret.loc[valid].mean())
    for lbl, m in (("skew>=90 & midterm", M90 & mid),
                   ("skew band 98+ & midterm", M98 & mid)):
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        if not len(epi):
            continue
        v = ret.loc[epi].values
        w = int((v > 0).sum())
        print(f"  h={h:>2d} {lbl:<26s} n={len(epi):3d} mean {100 * v.mean():+.3f}% "
              f"excess {100 * (v.mean() - base):+.3f}pp {w}-{len(v) - w} "
              f"sign p {sign_test(w, len(v)):.4f}")

print("\nDONE C4b")
