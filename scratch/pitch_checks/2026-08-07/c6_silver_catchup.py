"""C6 "Silver catch-up": LONG SLV / SHORT GLD, equal dollar, 10 td.

Trigger on close D: (SLV 63d ret - GLD 63d ret) <= -8pp AND SLV 5d ret > 0.
Entry MOC D+1. Pair return = r_SLV - r_GLD.
Cost: SLV ~2bp + GLD ~1bp per side => ~6 bps round trip for the pair.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

H = 10
LAG = 1
COST_BPS = 6.0
P = close_panel(["SLV", "GLD"]).dropna()
ASOF = P.index[-1]
sp63 = (P["SLV"].pct_change(63) - P["GLD"].pct_change(63)) * 100
r5 = P["SLV"].pct_change(5) * 100

print(f"sample {P.index.min().date()} .. {ASOF.date()}  n={len(P)}")
print(f"FIRES TODAY: sp63={sp63.loc[ASOF]:+.2f}pp (<=-8), SLV r5={r5.loc[ASOF]:+.2f}% (>0) "
      f"-> {(sp63.loc[ASOF] <= -8.0) and (r5.loc[ASOF] > 0)}")


def trigger(th=-8.0, need_bounce=True):
    m = sp63 <= th
    if need_bounce:
        m &= r5 > 0
    return sp63.index[m.fillna(False)]


fw = pair_fwd(P, "SLV", "GLD", H, LAG)
trig = trigger()
print(f"\n[1] day-level triggers: {len(trig)}")
show(report(fw, trig, "C6", H), "1. conditional vs 2 controls (pair = long SLV / short GLD)")
vt = pd.DatetimeIndex(trig).intersection(fw.dropna().index)
show([summarize(fwd_lag(P["SLV"], H, LAG).loc[vt].values, "SLV leg fwd"),
      summarize(fwd_lag(P["GLD"], H, LAG).loc[vt].values, "GLD leg fwd")],
     "1b. leg decomposition on trigger days")

show(era_split(vt, fw.loc[vt].values), "2. era split (day-level)")

ep, ev_ = episodes(fw, trig, H)
print(f"\n[3] declustered episodes (min gap {H} td): {len(ep)} (day-level {len(vt)})")
show([summarize(ev_, f"C6 episodes h{H}")], "3. episode-level")
show(era_split(ep, ev_), "3b. episode era split")
print("   boot P(mean<=0) =", round(bootstrap_p_le0(ev_), 4))
print("   concentration:", cluster_note(ep, ev_))
print("   episodes by year:", pd.Series(ev_).groupby(ep.year.values).agg(['count', 'mean']).round(4).to_dict('index'))

rows = []
for th in (-6.0, -8.0, -10.0, -12.0):
    for hh in (5, 10, 21):
        fwh = pair_fwd(P, "SLV", "GLD", hh, LAG)
        e, v = episodes(fwh, trigger(th), hh)
        r = summarize(v, f"th={th}pp h={hh}")
        r["boot_p"] = bootstrap_p_le0(v)
        rows.append(r)
show(rows, "4. threshold/horizon sensitivity (episode level)")

rows = []
for nb in (True, False):
    e, v = episodes(fw, trigger(-8.0, nb), H)
    r = summarize(v, f"bounce_filter={nb}")
    r["boot_p"] = bootstrap_p_le0(v)
    rows.append(r)
show(rows, "4b. is the 'SLV 5d > 0' bounce filter load-bearing?")

print(f"\n[5] cost: pair round trip ~{COST_BPS:.1f} bps; 5x = {5*COST_BPS/100:.2f}%. "
      f"episode mean = {100*np.nanmean(ev_):+.3f}% -> net {100*np.nanmean(ev_) - COST_BPS/100:+.3f}%, "
      f"edge/cost = {100*np.nanmean(ev_)/(COST_BPS/100):.1f}x")

m = cpi_in_window(ep, fw.dropna().index, H, LAG)
show([summarize(ev_[m], "CPI inside window"), summarize(ev_[~m], "no CPI in window")],
     "6. CPI-in-window split (episodes)")

fw0 = pair_fwd(P, "SLV", "GLD", H, 0)
e0, v0 = episodes(fw0, trig, H)
show([summarize(v0, "T+0 entry episodes")], "7. entry-lag sensitivity")
