"""C3 -- the ONE way today could be materially different from the registry cell.

v1c showed today's SVXY trailing-21d return is +4.59%, BELOW the +9.16%
median on historical rank<=2 triggers though still above the +3.76% all-days
median. So today is a MILDER lagging marker than the typical trigger. If the
low-trailing-carry subset of the trigger set paid better -- i.e. if the
"contango extreme that is NOT preceded by a vol crush" is a different animal
-- that would be a genuine new cell and a near-miss. This tests it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SVXY_BREAK = pd.Timestamp("2018-03-01")
px = close_panel(["^VIX", "^VIX3M", "SVXY", "SPY"])
ratio = (px["^VIX"] / px["^VIX3M"]).dropna()
rr = ratio.rolling(252).rank(pct=True) * 100
post = px.loc[px.index >= SVXY_BREAK]
t21 = post["SVXY"].pct_change(21)
TODAY = float(t21.iloc[-1])
print(f"  today's SVXY trailing-21d carry: {100*TODAY:+.2f}%")

trig = pd.DatetimeIndex(post.index[(rr <= 2.0).reindex(post.index, fill_value=False).values])
for h in (5, 10):
    ret = vehicle_ret(post, [("SVXY", 1.0)], h, 1)
    valid = ret.dropna().index
    d = trig.intersection(valid).intersection(t21.dropna().index)
    med = float(t21.loc[d].median())
    rows = []
    for lbl, sub in (("carry BELOW trigger median (today's bucket)",
                      d[t21.loc[d].values <= med]),
                     ("carry ABOVE trigger median", d[t21.loc[d].values > med]),
                     (f"carry <= today's {100*TODAY:+.1f}%",
                      d[t21.loc[d].values <= TODAY]),
                     ("all triggers", d)):
        e = declusters(sub, max(h, 5), valid)
        r = summarize(ret.loc[e].values, lbl)
        if r["n"]:
            r["local_ctl"] = round(100 * ret.loc[local_control(valid, sub)].mean(), 3)
            r["edge_pp"] = round(r["mean_pct"] - r["local_ctl"], 3)
        rows.append(r)
    rows.append(summarize(ret.loc[valid].values, "ALL DAYS"))
    show(rows, f"long SVXY h={h}, split by trailing carry (median {100*med:+.2f}%)")

print("\nVERDICT: if the low-carry bucket does not beat the high-carry one, "
      "today is the registry cell with a slightly milder trailing print,\n"
      "not a new state, and the 2026-08-13 kill applies verbatim.")
