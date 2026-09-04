"""E:nfp|NG=F|k2 is the strongest t in the whole payrolls group and the only
one that is era-stable with a real magnitude: n=309, h1 -0.424%, t -2.16,
136-172 down, sign p 0.0265, h5 -0.516%. Nat gas has no payrolls mechanism, so
the job here is to find out whether it is a real cell or a coincidence: does it
hold in both eras, is it concentrated, and does it survive a matched control on
the same weekday?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["NG=F", "CL=F"])
ng = px["NG=F"]["Close"].dropna()
ev = load_events(["nfp"])["date"]
pos, kept = anchor_positions(ng.index, ev, offset=-2)
anch = ng.index[pos]
anch = anch[anch < ng.index[-1]]
r1 = fwd_ret(ng, 1)


def rec(v, lab):
    d = summarize(v.values, lab)
    u = int((v > 0).sum())
    d["up"], d["down"] = u, len(v) - u
    d["sign_p"] = round(sign_test(len(v) - u, len(v)), 4) if len(v) else None
    return d


v = r1.reindex(anch).dropna()
rows = [rec(v, "pre-payrolls anchor -> next close")]
for lab, m in [("pre-2018", v.index < pd.Timestamp("2018-01-01")),
               ("2018+", v.index >= pd.Timestamp("2018-01-01")),
               ("September anchors", v.index.month == 9),
               ("midterm years", (v.index.year % 4) == 2)]:
    rows.append(rec(v[np.asarray(m)], lab))
rows.append(rec(r1.dropna(), "ctl: all NG sessions"))
# matched weekday control: same weekday, no payrolls two sessions out
wd = sorted(set(anch.weekday))
same_wd = ng.index[np.isin(ng.index.weekday, wd)].difference(anch)
rows.append(rec(r1.reindex(same_wd).dropna(), f"ctl: same weekday {wd}, non-anchor"))
lc = local_control(ng.index, anch, 126)
rows.append(rec(r1.reindex(lc).dropna(), "ctl: local +/-126td"))
show(rows, "NG=F, the session before the session before payrolls")
print(cluster_note(v.index, v.values, k=2))

# horizon: where does it sit
rows = []
for h in (1, 2, 3, 5, 10):
    r = fwd_ret(ng, h)
    vv = r.reindex(anch).dropna()
    d = rec(vv, f"h={h}")
    d["ctl_pct"] = round(100 * r.dropna().mean(), 3)
    d["edge_pct"] = round(d["mean_pct"] - 100 * r.dropna().mean(), 3)
    rows.append(d)
show(rows, "horizon scan from the anchor")

# is it seasonal contamination? payrolls fall in the first week of every month
print("\n=== is this just 'first week of the month' for nat gas? ===")
dom = pd.Series(ng.index.day, index=ng.index)
first_week = ng.index[(dom <= 7).values].difference(anch)
show([rec(r1.reindex(first_week).dropna(), "ctl: day-of-month <= 7, non-anchor"),
      rec(v, "the anchor cell")])
