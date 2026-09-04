"""Corn closes at a 252-day high on a 5-day run of 10%+. What follows, and how
does it compare to corn's own drift and to its local neighbourhood?
Also the wider grain-complex version: corn AND wheat both up 2%+ on the session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, summarize, sign_test, fwd_ret, declusters,
                       local_control, era_split, cluster_note)  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["ZC=F", "ZW=F", "ZS=F"])
c = px["ZC=F"]["Close"]
alld = px["ZC=F"].index


def report(dts, label, series=c, horizons=(1, 5, 10, 21)):
    dts = pd.DatetimeIndex([d for d in dts if d <= ASOF])
    dc = declusters(dts, 10, series.index)
    print(f"-- {label}: raw n={len(dts)}, declustered@10td n={len(dc)}")
    print("   dates:", [str(x.date()) for x in dc])
    ctrl_dates = local_control(series.index, dc, 126)
    for h in horizons:
        f = fwd_ret(series, h)
        v = f.reindex(dc).dropna()
        if len(v) < 3:
            print(f"   h{h}: n={len(v)} too few")
            continue
        s = summarize(v.values, "")
        up = int((v.values > 0).sum())
        cv = f.reindex(ctrl_dates).dropna().values
        cs = summarize(cv, "")
        allv = f.dropna().values
        a = summarize(allv, "")
        print(f"   h{h:<3} n={s['n']:<3} mean {s['mean_pct']:>7.2f}%  med {s['median_pct']:>7.2f}%  "
              f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  "
              f"t {s['t']:>5.2f} | local ctrl {cs['mean_pct']:>6.2f}% (n={cs['n']}) | "
              f"all days {a['mean_pct']:>6.2f}%")
        if h in (5, 21):
            print("      era:", [(e['label'], e['n'], round(e['mean_pct'], 2)) for e in era_split(v.index, v.values)])
            print("      ", cluster_note(v.index, v.values))


hi252 = c.rolling(252).max()
r5 = c.pct_change(5)
m1 = (c >= hi252 * 0.9999) & (r5 >= 0.10)
report(alld[m1.reindex(alld).fillna(False)], "corn at a 252d high with a 5d run of 10%+")

print()
m2 = (c >= hi252 * 0.9999) & (c.pct_change() >= 0.04)
report(alld[m2.reindex(alld).fillna(False)], "corn at a 252d high on a single 4%+ session")

print()
# grain complex joint thrust
w = px["ZW=F"]["Close"]
pan = pd.concat({"ZC": c, "ZW": w}, axis=1).dropna()
joint = (pan["ZC"].pct_change() >= 0.04) & (pan["ZW"].pct_change() >= 0.02)
print("=== corn +4% and wheat +2% on the same session ===")
report(pan.index[joint], "corn+4% & wheat+2% joint session", series=c)
print()
print("   ... and what WHEAT did after the same joint session:")
report(pan.index[joint], "same anchors, wheat forward", series=w)
