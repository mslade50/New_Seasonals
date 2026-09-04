"""C2 round 2 — gate attribution, drop-the-episode, and the one sign flip.

Three questions round 1 raised and did not close:
  1. The post-Labor-Day short: does the LABOR DAY gate do work, or is it just the
     September calendar position? Compare the boundary anchor against a FIXED
     September trading-day anchor over the same years.
  2. Drop the single biggest episode (2001 = the 9/11 week) from side B.
  3. Round 1's era split flipped SIGN: 2018+ long-into-the-eve is 0-for-8 on
     both SPY and IWM. Is that Labor Day, or the general post-2018 long-closure
     gap that C1 found in vol? Price it, and state the number that turns it on.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 230)

prices = load_prices(["SPY", "IWM", "QQQ", "SVXY", "^VIX"])
cal = prices["SPY"].index
px = pd.DataFrame({t: prices[t]["Close"].reindex(cal) for t in prices})
all_dates = px.index
posmap = pd.Series(range(len(all_dates)), index=all_dates)
closure = pd.Series(np.append((all_dates[1:] - all_dates[:-1]).days, np.nan),
                    index=all_dates) - 1.0
eves3 = all_dates[(closure >= 3).values]
labor = pd.DatetimeIndex([d for d in eves3
                          if all_dates[posmap[d] + 1].month == 9
                          and all_dates[posmap[d] + 1].day <= 8])


def seg(tkr, dates, a, b, sign=1.0):
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


def line(tag, vals, n_note=""):
    if len(vals) == 0:
        print(f"  {tag:52s} n=0")
        return
    w = int((vals > 0).sum())
    s = summarize(vals)
    print(f"  {tag:52s} n={len(vals):4d} mean {s['mean_pct']:+7.3f}% "
          f"med {s['median_pct']:+7.3f}% hit {s['hit']:5.1f}% t {s['t']:+6.2f} "
          f"rec {w}-{len(vals)-w} signp {sign_test(w, len(vals)):.4f} "
          f"worst {s['worst_pct']:+7.2f}% {n_note}")


# ---------------------------------------------------------------- Q1
print("=" * 108)
print("Q1 — post-Labor-Day short: the boundary anchor vs a FIXED September td anchor")
print("=" * 108)
sept_first = {}
for d in all_dates:
    if d.month == 9:
        sept_first.setdefault(d.year, []).append(d)
for tkr in ("SPY", "IWM"):
    print(f"\n--- short {tkr} ---")
    for h in (5, 7, 10):
        d, v = seg(tkr, labor, 1, 1 + h, sign=-1.0)
        line(f"h={h} boundary anchor (first post-LD close)", v)
        for td in (2, 3, 4, 5):
            anchors = pd.DatetimeIndex(
                [dd[td - 1] for y, dd in sorted(sept_first.items())
                 if len(dd) >= td and y >= 2000])
            _, vv = seg(tkr, anchors, 0, h, sign=-1.0)
            line(f"h={h}   FIXED Sept td{td} close, same years", vv)

# ---------------------------------------------------------------- Q2
print("\n" + "=" * 108)
print("Q2 — drop the biggest episode from side B")
print("=" * 108)
for tkr in ("SPY", "IWM"):
    for h in (7, 10):
        d, v = seg(tkr, labor, 1, 1 + h, sign=-1.0)
        i = int(np.argmax(v))
        j = int(np.argmax(np.abs(v)))
        keep = np.ones(len(v), bool); keep[i] = False
        print(f"\n {tkr} short h={h}: best episode {d[i].year} {100*v[i]:+.2f}%, "
              f"largest-abs {d[j].year} {100*v[j]:+.2f}%")
        line("   all episodes", v)
        line(f"   drop best ({d[i].year})", v[keep])
        keep2 = keep.copy()
        k2 = int(np.argmax(np.where(keep, v, -9)))
        keep2[k2] = False
        line(f"   drop best two ({d[i].year}, {d[k2].year})", v[keep2])

# ---------------------------------------------------------------- Q3
print("\n" + "=" * 108)
print("Q3 — the sign flip: the >=3-calendar-day CLOSURE GAP as a cross-instrument")
print("     risk premium (SHORT equity / LONG vol across the closure), by era")
print("=" * 108)
for tkr, sgn, nm in [("SPY", -1.0, "SHORT SPY"), ("IWM", -1.0, "SHORT IWM"),
                     ("QQQ", -1.0, "SHORT QQQ"), ("SVXY", -1.0, "SHORT SVXY"),
                     ("^VIX", 1.0, "LONG ^VIX (pts)")]:
    print(f"\n--- {nm} across the closure gap (eve close -> first post close) ---")
    d, v = seg(tkr, eves3, 0, 1, sign=sgn)
    yr = pd.DatetimeIndex(d).year
    line("all >=3d closures", v)
    line("  pre-2013", v[yr < 2013])
    line("  2013-2017", v[(yr >= 2013) & (yr < 2018)])
    line("  2018+", v[yr >= 2018])
    dw, vw = seg(tkr, all_dates[(closure == 2).values], 0, 1, sign=sgn)
    yw = pd.DatetimeIndex(dw).year
    line("CTRL ordinary weekend gap, 2018+", vw[yw >= 2018])
    dp, vp = seg(tkr, all_dates[(closure == 0).values], 0, 1, sign=sgn)
    yp = pd.DatetimeIndex(dp).year
    line("CTRL plain overnight, 2018+", vp[yp >= 2018])

print("\n" + "=" * 108)
print("Q3b — Labor Day gate attribution on the 2018+ inversion")
print("=" * 108)
for tkr in ("SPY", "IWM"):
    d, v = seg(tkr, eves3, 0, 1, sign=-1.0)
    yr = pd.DatetimeIndex(d).year
    is_lab = np.isin(d.values, labor.values)
    print(f"\n--- SHORT {tkr} across the closure gap, 2018+ ---")
    line("Labor Day eves only", v[(yr >= 2018) & is_lab])
    line("all OTHER >=3d closures", v[(yr >= 2018) & ~is_lab])
    line("gate OFF: every >=3d closure", v[yr >= 2018])
    edge = v[(yr >= 2018)].mean() * 100 * 100
    print(f"    gate-off edge {edge:+.1f} bps vs "
          f"{2.0 if tkr == 'SPY' else 3.0} bps round trip = "
          f"{edge / (2.0 if tkr == 'SPY' else 3.0):+.1f}x")

print("\n" + "=" * 108)
print("Q3c — the same closure-gap premium priced on SVXY, 2018+ (the tradeable -0.5x era)")
print("=" * 108)
d, v = seg("SVXY", eves3, 0, 1, sign=-1.0)
yr = pd.DatetimeIndex(d).year
line("SHORT SVXY across >=3d closure, 2018+", v[yr >= 2018])
print(f"    = {100*v[yr>=2018].mean()*100:+.1f} bps vs 8 bps round trip = "
      f"{100*v[yr>=2018].mean()*100/8:+.1f}x")
print("    per-episode 2018+:",
      ", ".join(f"{x.date()}:{100*y:+.2f}" for x, y in zip(d[yr >= 2018], v[yr >= 2018])))
