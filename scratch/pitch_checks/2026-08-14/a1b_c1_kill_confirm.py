"""C1 confirmatory kills.

Three things round 1 flagged, each nailed to a number:
 (1) LEVEL DRIFT — "1.6th percentile of the trailing year" is a RELATIVE
     statement. Against full history, where does ^SKEW 134.37 actually sit?
     This is the 2026-08-10 ^MOVE trap one layer deeper: the map quoted a
     LEVEL percentile, but only against 252 days.
 (2) GATE ATTRIBUTION — run the cell WITHOUT the skew leg. If the skew leg
     does not move the result, nothing may be attributed to it.
 (3) SIGN across the definition neighbourhood — a grid, not two points.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (close_panel, declusters, sign_test, show,  # noqa: E402
                       summarize, vehicle_ret)

px = close_panel(["SPY", "^SKEW", "^VIX"]).dropna()
sk, vx, sp = px["^SKEW"], px["^VIX"], px["SPY"]


def lvl_pctile(s, lb=252):
    return s.rolling(lb).rank(pct=True) * 100.0


sk_p, vx_p = lvl_pctile(sk), lvl_pctile(vx)
sp_hi = sp / sp.rolling(252).max() - 1.0

print("=" * 78)
print("1. LEVEL DRIFT — is 134.37 a LOW skew level, or only low vs 252 days?")
print("=" * 78)
last = sk.iloc[-1]
print(f"  ^SKEW 2026-08-13 close = {last:.2f}")
print(f"  pctile vs trailing 252d : {lvl_pctile(sk).iloc[-1]:5.1f}")
print(f"  pctile vs FULL history  : {100*(sk < last).mean():5.1f}  "
      f"(n={len(sk)}, 2000-2026)")
print(f"  pctile vs 2000-2013     : {100*(sk[sk.index < '2014'] < last).mean():5.1f}")
print(f"  pctile vs 2018+         : {100*(sk[sk.index >= '2018'] < last).mean():5.1f}")
for yr0, yr1 in [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020),
                 (2020, 2026), (2026, 2027)]:
    seg = sk[(sk.index.year >= yr0) & (sk.index.year < yr1)]
    print(f"    {yr0}-{yr1-1}: median {seg.median():7.2f}  mean {seg.mean():7.2f}"
          f"  min {seg.min():7.2f}  max {seg.max():7.2f}")

print("\n  the same question for the historical trigger days:")
CORE = (sk_p <= 5) & (vx_p <= 10) & (sp_hi >= -0.005)
epi = declusters(px.index[CORE.values], 21, px.index)
for d in epi:
    print(f"    {d.date()}  ^SKEW level {sk.loc[d]:7.2f}  "
          f"(pctile252 {sk_p.loc[d]:4.1f}, full-hist pctile "
          f"{100*(sk < sk.loc[d]).mean():5.1f})")

print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION — what does the SKEW leg actually add?")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    rows = []
    cells = {
        "WITH skew: SKEW<=5 & VIX<=10 & hi>=-0.5%": CORE,
        "WITHOUT skew: VIX<=10 & hi>=-0.5%": (vx_p <= 10) & (sp_hi >= -0.005),
        "WITHOUT skew or vix: hi>=-0.5% alone": (sp_hi >= -0.005),
        "skew leg ALONE: SKEW<=5": (sk_p <= 5),
        "skew leg ALONE: SKEW<=2": (sk_p <= 2),
        "skew leg ALONE: SKEW<=10": (sk_p <= 10),
    }
    for lbl, m in cells.items():
        d = px.index[m.values & ret.notna().values]
        e = declusters(d, 21, px.index)
        r = summarize(ret.loc[e].values, lbl)
        r["n_days"] = len(d)
        rows.append(r)
    base = ret.dropna()
    rows.append(summarize(base.values, "unconditional all days"))
    show(rows, f"h={h} td, entry lag=1, episode level (min_gap 21td)")

print("\n" + "=" * 78)
print("3. DEFINITION NEIGHBOURHOOD — sign across a grid, not two points")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    print(f"\n  h={h}: episode mean_pct [n_episodes] by (skew pctile cut, "
          f"vix pctile cut), SPY within 0.5% of 52wh")
    hdr = "    skew\\vix  " + "".join(f"{v:>16d}" for v in (5, 10, 15, 20))
    print(hdr)
    for skt in (2, 3, 5, 8, 10, 15, 20):
        line = f"    <={skt:2d}      "
        for vxt in (5, 10, 15, 20):
            m = (sk_p <= skt) & (vx_p <= vxt) & (sp_hi >= -0.005)
            d = px.index[m.values & ret.notna().values]
            e = declusters(d, 21, px.index)
            if len(e) == 0:
                line += f"{'--':>16s}"
            else:
                line += f"{100*ret.loc[e].mean():>+11.3f}[{len(e):>2d}]"
        print(line)

print("\n" + "=" * 78)
print("4. the state's OCCURRENCE era (can this cell be measured at all today?)")
print("=" * 78)
for lbl, m in [("STRICT SKEW<=2 VIX<=5 hi>=0", (sk_p <= 2) & (vx_p <= 5) & (sp_hi >= 0)),
               ("CORE SKEW<=5 VIX<=10 hi>=-0.5%", CORE),
               ("LOOSE SKEW<=10 VIX<=20 hi>=-1%", (sk_p <= 10) & (vx_p <= 20) & (sp_hi >= -0.01))]:
    d = px.index[m.values]
    e = declusters(d, 21, px.index)
    pre = [x for x in e if x.year < 2018]
    post = [x for x in e if 2018 <= x.year < 2026]
    print(f"  {lbl:34s} episodes: {len(e):2d}  pre-2018 {len(pre):2d}  "
          f"2018-2025 {len(post):2d}  2026 {len([x for x in e if x.year==2026])}"
          f"   last non-2026 = {max([x for x in e if x.year<2026]).date() if any(x.year<2026 for x in e) else '-'}")
