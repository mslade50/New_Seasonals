"""C8 "Deep-drawdown momentum restart": LONG GDX outright, 10 td, entry MOC D+1.

Trigger on close D: GDX 21d ret >= +12% AND 63d-return trailing-252d rank < 30.
Cost: GDX ~2.5 bps/side => ~5 bps round trip.
This is the OPPOSITE side of C5 on the same instrument -- the overlap is measured
explicitly at the bottom.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

H = 10
LAG = 1
COST_BPS = 5.0
P = close_panel(["GDX", "GLD"]).dropna()
g = P["GDX"]
ASOF = g.index[-1]
r21 = g.pct_change(21) * 100
rk63 = pct_rank(g, 63)
print(f"sample {g.index.min().date()} .. {ASOF.date()}  n={len(g)}")
print(f"FIRES TODAY: r21={r21.loc[ASOF]:+.2f}% (>=12), rank63={rk63.loc[ASOF]:.1f} (<30) "
      f"-> {(r21.loc[ASOF] >= 12.0) and (rk63.loc[ASOF] < 30)}")


def trigger(th=12.0, rkmax=30.0):
    return r21.index[((r21 >= th) & (rk63 < rkmax)).fillna(False)]


fw = fwd_lag(g, H, LAG)
trig = trigger()
vt = pd.DatetimeIndex(trig).intersection(fw.dropna().index)
print(f"\n[1] day-level triggers: {len(vt)}")
show(report(fw, trig, "C8", H), "1. conditional vs 2 controls (long GDX outright)")
show([summarize(fwd_lag(P["GLD"], H, LAG).loc[vt].values, "GLD same days"),
      summarize((fwd_lag(g, H, LAG) - fwd_lag(P["GLD"], H, LAG)).loc[vt].values, "GDX-minus-GLD")],
     "1b. is it GDX or just gold beta?")

show(era_split(vt, fw.loc[vt].values), "2. era split (day-level)")

ep, ev_ = episodes(fw, trig, H)
print(f"\n[3] declustered episodes (min gap {H} td): {len(ep)} (day-level {len(vt)})")
show([summarize(ev_, f"C8 episodes h{H}")], "3. episode-level")
show(era_split(ep, ev_), "3b. episode era split")
print("   boot P(mean<=0) =", round(bootstrap_p_le0(ev_), 4))
print("   concentration:", cluster_note(ep, ev_))
print("   episode dates:", [str(d.date()) for d in ep])

rows = []
for th in (10.0, 12.0, 15.0):
    for rk in (20.0, 30.0, 40.0):
        for hh in (5, 10, 21):
            fwh = fwd_lag(g, hh, LAG)
            e, v = episodes(fwh, trigger(th, rk), hh)
            r = summarize(v, f"r21>={th} rk63<{rk} h={hh}")
            r["boot_p"] = bootstrap_p_le0(v)
            rows.append(r)
show(rows, "4. sensitivity (episode level)")

print(f"\n[5] cost: ~{COST_BPS:.0f} bps rt; 5x = {5*COST_BPS/100:.2f}%. "
      f"episode mean = {100*np.nanmean(ev_):+.3f}% -> net {100*np.nanmean(ev_) - COST_BPS/100:+.3f}%, "
      f"edge/cost = {100*np.nanmean(ev_)/(COST_BPS/100):.1f}x; "
      f"excess over unconditional GDX drift = {100*np.nanmean(ev_) - 100*fw.dropna().mean():+.3f}pp")

m = cpi_in_window(ep, fw.dropna().index, H, LAG)
show([summarize(ev_[m], "CPI inside"), summarize(ev_[~m], "no CPI")], "6. CPI split (episodes)")

fw0 = fwd_lag(g, H, 0)
e0, v0 = episodes(fw0, trig, H)
show([summarize(v0, "T+0 entry episodes")], "7. entry-lag sensitivity")

# ---- C5 vs C8 contradiction audit ----
sp = (P["GDX"].pct_change(21) - P["GLD"].pct_change(21)) * 100
c5 = sp.index[(sp >= 8.0).fillna(False)]
both = pd.DatetimeIndex(trig).intersection(c5)
print(f"\n[8] C5/C8 OVERLAP: C5 day-triggers={len(c5)}, C8={len(trig)}, BOTH={len(both)} "
      f"({100*len(both)/max(len(trig),1):.0f}% of C8 days also fire C5, incl. today)")
b = both.intersection(fw.dropna().index)
show([summarize(fw.loc[b].values, "GDX long on BOTH-fire days"),
      summarize((fwd_lag(P["GLD"], H, LAG) - fw).loc[b].values, "C5 pair (GLD-GDX) on BOTH days")],
     "8b. same days, both candidates' P&L")
