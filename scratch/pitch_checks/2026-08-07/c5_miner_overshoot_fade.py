"""C5 "Miner overshoot fade": SHORT GDX / LONG GLD, equal dollar, 5 td.

Trigger on close D: (GDX 21d ret - GLD 21d ret) >= +8pp. Entry MOC D+1.
Pair return booked as (GLD - GDX) since GLD is the long leg.
Cost: GLD ~1bp + GDX ~2.5bp per side => ~7 bps round trip for the pair.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

H = 5
LAG = 1
COST_BPS = 7.0
P = close_panel(["GLD", "GDX"]).dropna()
ASOF = P.index[-1]
spread = (P["GDX"].pct_change(21) - P["GLD"].pct_change(21)) * 100

print(f"sample {P.index.min().date()} .. {ASOF.date()}  n={len(P)}")
print(f"FIRES TODAY: spread={spread.loc[ASOF]:+.2f}pp  thresh=+8.00  -> {spread.loc[ASOF] >= 8.0}")

fw = pair_fwd(P, "GLD", "GDX", H, LAG)
trig = spread.index[spread >= 8.0]
print(f"\n[1] day-level triggers: {len(trig)}")
show(report(fw, trig, "C5", H), "1. conditional vs 2 controls (pair = long GLD / short GDX)")

# per-leg decomposition: which leg carries it?
show([summarize(fwd_lag(P["GLD"], H, LAG).loc[trig.intersection(fw.dropna().index)].values, "GLD leg fwd"),
      summarize(fwd_lag(P["GDX"], H, LAG).loc[trig.intersection(fw.dropna().index)].values, "GDX leg fwd")],
     "1b. leg decomposition on trigger days")

vt = pd.DatetimeIndex(trig).intersection(fw.dropna().index)
show(era_split(vt, fw.loc[vt].values), "2. era split (day-level)")

ep, ev_ = episodes(fw, trig, H)
print(f"\n[3] declustered episodes (min gap {H} td): {len(ep)} (day-level {len(vt)})")
show([summarize(ev_, f"C5 episodes h{H}")], "3. episode-level")
show(era_split(ep, ev_), "3b. episode era split")
print("   boot P(mean<=0) =", round(bootstrap_p_le0(ev_), 4))
print("   concentration:", cluster_note(ep, ev_))
print("   episode dates:", [str(d.date()) for d in ep])

rows = []
for th in (6.0, 8.0, 10.0, 12.0):
    for hh in (3, 5, 10):
        fwh = pair_fwd(P, "GLD", "GDX", hh, LAG)
        tg = spread.index[spread >= th]
        e, v = episodes(fwh, tg, hh)
        r = summarize(v, f"th={th}pp h={hh}")
        r["boot_p"] = bootstrap_p_le0(v)
        rows.append(r)
show(rows, "4. sensitivity (episode level)")

print(f"\n[5] cost: pair round trip ~{COST_BPS:.1f} bps; 5x = {5*COST_BPS/100:.2f}%. "
      f"episode mean = {100*np.nanmean(ev_):+.3f}% -> net {100*np.nanmean(ev_) - COST_BPS/100:+.3f}%, "
      f"edge/cost = {100*np.nanmean(ev_)/(COST_BPS/100):.2f}x")

m = cpi_in_window(ep, fw.dropna().index, H, LAG)
show([summarize(ev_[m], "CPI inside window"), summarize(ev_[~m], "no CPI in window")],
     "6. CPI-in-window split (episodes)")
mf = cpi_in_window(ep, fw.dropna().index, H, LAG, kinds=("fomc_decision",))
show([summarize(ev_[mf], "FOMC inside"), summarize(ev_[~mf], "no FOMC")], "6b. FOMC split")

# T+0 entry variant (idealised, no MOC-tomorrow lag) for comparison
fw0 = pair_fwd(P, "GLD", "GDX", H, 0)
e0, v0 = episodes(fw0, trig, H)
show([summarize(v0, "T+0 entry episodes")], "7. entry-lag sensitivity")
