"""C1 round 2 — the entry-convention decomposition that decides it.

The thesis is that the position is HELD ACROSS the closure and "collects the
gap". Under lag=1 an anchor ON the eve enters at the FIRST CLOSE AFTER the
closure, so a1's headline cell measures a trade that never owns the gap. The
tradeable form of the pitched idea is anchor = eve-1, lag=1, entry MOC on the
eve. That is the k=-1 rung of a1's ladder.

This script separates:
  GAP      eve close -> first post-closure close        (what the thesis claims)
  POST     first post-closure close -> +h               (what a1 measured)
  PITCHED  eve close -> +h  (GAP + POST, the real order)
and puts each against the ordinary-weekend twin. Then it asks whether the
mark-down happens on the eve SESSION itself, i.e. before an order at the close
can own it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
PRINT_KINDS = ("nfp", "cpi", "ppi", "fomc_decision")

prices = load_prices(["SVXY", "SPY", "^VIX", "^VIX3M"])
cal = prices["SPY"].index
px = pd.DataFrame({t: prices[t]["Close"].reindex(cal) for t in prices})
all_dates = px.index
posmap = pd.Series(range(len(all_dates)), index=all_dates)

gap_days = pd.Series(np.append((all_dates[1:] - all_dates[:-1]).days, np.nan),
                     index=all_dates)
closure = gap_days - 1.0

ev = load_events(list(PRINT_KINDS))["date"]
ev_pos = np.array(sorted({int(all_dates.searchsorted(d)) for d in ev
                          if all_dates[0] <= d <= all_dates[-1]}))
runway = np.full(len(all_dates), 999.0)
for i in range(len(all_dates)):
    j = int(np.searchsorted(ev_pos, i, side="right"))
    if j < len(ev_pos):
        runway[i] = ev_pos[j] - i
runway = pd.Series(runway, index=all_dates)
CLEAR = runway >= 3

eves3 = all_dates[(closure >= 3).values & CLEAR.values]
eves2 = all_dates[(closure == 2).values & CLEAR.values]
eves1 = all_dates[(closure == 1).values & CLEAR.values]


def seg(tkr, dates, a, b, sign=1.0):
    """sign * (C[p+b]/C[p+a] - 1) for each eve at position p."""
    c = px[tkr].values
    out, keep = [], []
    for d in dates:
        p = posmap.get(d)
        if p is None or p + a < 0 or p + b >= len(all_dates):
            continue
        x, y = c[p + a], c[p + b]
        if not (np.isfinite(x) and np.isfinite(y)) or x == 0:
            continue
        out.append(sign * (y / x - 1.0))
        keep.append(d)
    return pd.DatetimeIndex(keep), np.array(out)


def line(tag, dates, vals):
    if len(vals) == 0:
        print(f"  {tag:42s} n=0")
        return
    w = int((vals > 0).sum())
    s = summarize(vals)
    print(f"  {tag:42s} n={len(vals):4d} mean {s['mean_pct']:+7.3f}% "
          f"med {s['median_pct']:+7.3f}% hit {s['hit']:5.1f}% t {s['t']:+6.2f} "
          f"rec {w}-{len(vals)-w} signp {sign_test(w, len(vals)):.4f} "
          f"worst {s['worst_pct']:+7.2f}%")


print("=" * 96)
print("PART A — SVXY: the GAP the thesis says it collects, by closure length (clear calendar)")
print("=" * 96)
for tag, dts in [("3-day closure (the pitch)", eves3),
                 ("2-day weekend", eves2),
                 ("1-day midweek holiday", eves1)]:
    d, v = seg("SVXY", dts, 0, 1)
    line(f"GAP  {tag}", d, v)
# all ordinary sessions with no closure at all: the plain overnight
plain = all_dates[(closure == 0).values & CLEAR.values]
d, v = seg("SVXY", plain, 0, 1)
line("GAP  no closure (plain 1-session)", d, v)

print("\n  --- same, in ^VIX POINTS (short side; positive = VIX fell) ---")
for tag, dts in [("3-day closure", eves3), ("2-day weekend", eves2),
                 ("1-day midweek holiday", eves1), ("no closure", plain)]:
    d, v = seg("^VIX", dts, 0, 1, sign=-1.0)
    line(f"GAP short^VIX {tag}", d, v)

print("\n" + "=" * 96)
print("PART B — decomposition of the PITCHED order (entry MOC on the eve, lag=1 from eve-1)")
print("=" * 96)
for h in (1, 2, 3, 4, 5):
    dP, vP = seg("SVXY", eves3, 0, h)          # eve close -> +h sessions
    dG, vG = seg("SVXY", eves3, 0, 1)          # the gap
    dO, vO = seg("SVXY", eves3, 1, h)          # post-closure remainder
    print(f"\n h={h} sessions held from the eve close:")
    line("PITCHED  eve close -> +h", dP, vP)
    line("  of which GAP  eve -> first post close", dG, vG)
    line("  of which POST first post close -> +h", dO, vO)
    dW, vW = seg("SVXY", eves2, 0, h)
    line("TWIN     ordinary weekend, same form", dW, vW)
    # plain control: every clear-calendar session, same holding length
    dC, vC = seg("SVXY", plain, 0, h)
    line("CTRL     any clear session, same length", dC, vC)

print("\n" + "=" * 96)
print("PART C — placebo ladder on the TRADEABLE anchor (entry always MOC on eve+k)")
print("=" * 96)
for h in (1, 3):
    rows = []
    for k in range(-8, 9):
        d, v = seg("SVXY", eves3, k, k + h)
        if len(v) < 20:
            continue
        s = summarize(v)
        rows.append({"entry_offset_k": k, "n": len(v), "mean_pct": s["mean_pct"],
                     "hit": s["hit"], "t": s["t"]})
    L = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    L["rank"] = range(1, len(L) + 1)
    print(f"\n h={h}: entry at the close k sessions from the eve (k=0 IS the pitch)")
    print(L.round(3).to_string(index=False))
    r = int(L.loc[L.entry_offset_k == 0, "rank"].iloc[0])
    print(f"  PITCHED ENTRY k=0 rank = {r} of {len(L)}")

print("\n" + "=" * 96)
print("PART D — is the mark-down already IN the eve session? (so an eve MOC is late)")
print("=" * 96)
for tag, dts in [("3-day closure eve", eves3), ("2-day weekend eve", eves2),
                 ("no-closure session", plain)]:
    d, v = seg("^VIX", dts, -1, 0, sign=-1.0)
    line(f"short^VIX ON the eve session {tag}", d, v)
    d, v = seg("SVXY", dts, -1, 0)
    line(f"SVXY      ON the eve session {tag}", d, v)

print("\n" + "=" * 96)
print("PART E — the pitched form: leverage split, runway control, midterm, Labor Day")
print("=" * 96)
for h in (1, 3):
    d, v = seg("SVXY", eves3, 0, h)
    m = pd.DatetimeIndex(d) < pd.Timestamp("2018-02-28")
    print(f"\n h={h}")
    line("pre-2018-02-28 (-1x)", d[m], v[m])
    line("post-2018-02-28 (-0.5x, TRADEABLE)", d[~m], v[~m])
    yr = pd.DatetimeIndex(d).year
    line("midterm years", d[(yr % 4) == 2], v[(yr % 4) == 2])
    line("non-midterm years", d[(yr % 4) != 2], v[(yr % 4) != 2])
    rw = runway.reindex(d).values
    line("runway>=4", d[rw >= 4], v[rw >= 4])
    line("runway==3 (TODAY)", d[rw == 3], v[rw == 3])
    # Labor Day only
    lab = [x for x in d if (all_dates[posmap[x] + 1].month == 9
                            and all_dates[posmap[x] + 1].day <= 8)]
    lab = pd.DatetimeIndex(lab)
    mm = np.isin(d.values, lab.values)
    line("LABOR DAY eve only", d[mm], v[mm])
    if mm.sum():
        print("     ", ", ".join(f"{x.date()}:{100*y:+.2f}"
                                 for x, y in zip(d[mm], v[mm])))

print("\n" + "=" * 96)
print("PART F — cost")
print("=" * 96)
for h in (1, 3):
    d, v = seg("SVXY", eves3, 0, h)
    m = pd.DatetimeIndex(d) >= pd.Timestamp("2018-02-28")
    print(f" h={h} pitched, post-leverage-change: mean {100*v[m].mean():+.3f}% "
          f"= {100*v[m].mean()*100:+.1f} bps vs 8 bps round trip -> "
          f"{100*v[m].mean()*100/8:.1f}x")
