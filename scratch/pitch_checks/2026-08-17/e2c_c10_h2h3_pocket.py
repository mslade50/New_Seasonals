"""C10 - the one pocket round 1/2 had not torn down: h=2 and h=3.

The horizon scan left h=2 (+1.171pp excess, 13-5) and h=3 (+0.796pp, 13-5)
looking better than the h=10 cell that round 2 killed. Before parking anything
on the watchlist, price those two the same way: concentration, drop-top-2,
LOYO, era, and the live magnitude/drawdown cells.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["GDX", "GLD", "SPY"]).dropna(subset=["GDX"])
idx = px.index
rk21 = pct_rank(px["GDX"], 21)
r21 = px["GDX"].pct_change(21)
dd = px["GDX"] / px["GDX"].rolling(252).max() - 1.0
trig = rk21 >= 100.0

for H in (2, 3):
    ret = fwd_lag(px["GDX"], H, 1)
    epi = declusters(idx[trig.values & ret.notna().values], 21, idx)
    ev = ret.loc[epi].values
    base = 100 * ret.dropna().mean()
    print(f"\n===== h={H}: N={len(epi)}, mean {100*ev.mean():+.3f}%, "
          f"all-days {base:+.3f}%, excess {100*ev.mean()-base:+.3f}pp, "
          f"record {(ev>0).sum()}-{(ev<=0).sum()}, "
          f"sign p {sign_test(int((ev>0).sum()), len(ev)):.4f} =====")
    print("  ", cluster_note(epi, ev, k=2))
    order = np.argsort(-ev)
    for k in (1, 2, 3):
        keep = np.ones(len(ev), bool)
        keep[order[:k]] = False
        print(f"   drop top-{k}: {100*ev[keep].mean():+.3f}% "
              f"(excess {100*ev[keep].mean()-base:+.3f}pp) on "
              f"{(ev[keep]>0).sum()}-{(ev[keep]<=0).sum()}")
    yrs = pd.DatetimeIndex(epi).year
    loyo = [(y, 100 * ev[yrs != y].mean()) for y in sorted(set(yrs))]
    print(f"   LOYO floor {min(v for _, v in loyo):+.3f}% "
          f"(drop {min(loyo, key=lambda x: x[1])[0]})")
    show(era_split(epi, ev), f"  h={H} era split")
    # local control
    loc = local_control(idx[ret.notna().values],
                        idx[trig.values & ret.notna().values])
    print(f"   CTRL-c local +/-126td ex-trigger: "
          f"{100*ret.loc[loc].mean():+.3f}%  -> local excess "
          f"{100*ev.mean()-100*ret.loc[loc].mean():+.3f}pp")
    # live cells
    rows = []
    for lbl, m in [("LIVE dd<=-20%", trig & (dd <= -0.20)),
                   ("near high dd>-10%", trig & (dd > -0.10)),
                   ("LIVE 21d ret>=26%", trig & (r21 >= 0.26)),
                   ("21d ret>=20%", trig & (r21 >= 0.20)),
                   ("rank>=99 (nudge)", rk21 >= 99.0),
                   ("rank>=97 (nudge)", rk21 >= 97.0)]:
        e = declusters(idx[m.values & ret.notna().values], 21, idx)
        r = summarize(ret.loc[e].values, lbl)
        if r["n"]:
            r["excess_pp"] = round(r["mean_pct"] - base, 3)
        rows.append(r)
    show(rows, f"  h={H} definition neighbours + live cells")
