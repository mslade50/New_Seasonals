"""C9 correction -- which opex window is TODAY actually in?

v3b's headline cell used signal-day-in-[opex-4, opex]. Today's SIGNAL day is
the freshest bar, Friday 2026-08-14, and opex is Friday 2026-08-21: that is
opex MINUS 5 sessions, with the lag-1 entry landing on Monday = opex-4.
So the live cell is signal-day-in-[opex-5, opex-1], NOT [opex-4, opex].
This script pins the arithmetic and reports the LIVE cell's numbers, so the
kill is quoted against the window today is actually in.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["IWM", "SPY"])
ALL = px.index
opex = pd.to_datetime(load_events(["opex"])["date"].unique())
sig = ALL[-1]
nxt = opex[opex > sig][0]
# NOT ALL.searchsorted(nxt): the next opex is in the FUTURE, past the end of
# the price index, so searchsorted returns len(ALL) and mints a fake offset of
# +1. This is the 2026-08-11 registry trap ("a future event date silently
# manufacturing a fake anchor") and it fired here on the first run.
off = len(pd.bdate_range(sig, nxt)) - 1
print(f"  signal day (freshest bar) {sig.date()};  next opex {nxt.date()};  "
      f"signal day sits at opex{-off:+d} td;  lag-1 entry sits at opex{-(off-1):+d}")

oh = {t: 1.0 - px[t] / px[t].rolling(252).max() for t in ("IWM", "SPY")}
state = (100 * oh["IWM"] <= 0.10) & (100 * oh["SPY"] > 0.10)


def week(lo, hi):
    m = pd.Series(False, index=ALL)
    for d in opex:
        p = ALL.searchsorted(d)
        if p >= len(ALL):
            continue
        m.iloc[max(0, p + lo):max(0, p + hi) + 1] = True
    return m


live = state & week(-off, -off + 4)
print(f"  live window = signal day in [opex{-off:+d}, opex{-off+4:+d}]; "
      f"trigger days {int(live.sum())}")

for legs, cost, nm in (([("IWM", 1.0), ("SPY", -1.0)], 8.0, "IWM/SPY pair"),
                       ([("IWM", 1.0)], 2.0, "IWM outright")):
    rows = []
    for h in (1, 3, 4, 5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        d = pd.DatetimeIndex(ALL[live.values]).intersection(valid)
        e = declusters(d, max(h, 5), valid)
        v = ret.loc[e].values
        r = summarize(v, f"h={h}")
        r["ctl_local_pct"] = round(100 * ret.loc[local_control(valid, d)].mean(), 3)
        r["edge_pp"] = round(r["mean_pct"] - r["ctl_local_pct"], 3)
        r["x_cost"] = round(100 * v.mean() * 100 / cost, 2)
        rows.append(r)
    show(rows, f"LIVE window [opex{-off:+d}..opex{-off+4:+d}], {nm}, "
               f"cost {cost} bps")

# concentration of the live-window pair cell at its best horizon
ret = vehicle_ret(px, [("IWM", 1.0), ("SPY", -1.0)], 4, 1)
valid = ret.dropna().index
d = pd.DatetimeIndex(ALL[live.values]).intersection(valid)
e = declusters(d, 5, valid)
ep = ret.loc[e].values
print(f"\n  live-window pair h=4: N_ep={len(ep)} record "
      f"{int((ep>0).sum())}-{int((ep<=0).sum())}, sign p "
      f"{sign_test(int((ep>0).sum()), len(ep)):.4f}")
print(f"  {cluster_note(e, ep)}")
