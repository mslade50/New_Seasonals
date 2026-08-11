"""Evening check, 2026-08-10: the four-metal thrust and what silver does next.

Tonight's tape: GC=F +5%+, SI=F +8%+, PL=F +5%+, PA=F +5%+ over the same
five sessions (the context brief flagged this cell lag-0; this is the
pitch-doctrine version: lag-1 entry, declustered, controlled).

Question for a POST (and maybe an idea): after a simultaneous 5d thrust in
all four precious metals, what does silver do over the next 5 sessions,
entering at the NEXT close (lag=1)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

import pitch_lab as pl

METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]
px = pl.close_panel(METALS + ["SLV"])
px = px.dropna(subset=METALS)

r5 = {t: px[t].pct_change(5) for t in METALS}
mask = (r5["GC=F"] >= 0.05) & (r5["SI=F"] >= 0.08) & \
       (r5["PL=F"] >= 0.05) & (r5["PA=F"] >= 0.05)
trig_all = px.index[mask.fillna(False)]
trig = pl.declusters(trig_all, 10, px.index)
print(f"trigger days: {len(trig_all)} raw, {len(trig)} declustered (10td)")
print("declustered dates:", [str(d.date()) for d in trig])
print("fires tonight (2026-08-10)?", pd.Timestamp("2026-08-10") in trig_all)

H, LAG = 5, 1
rows = []
# the tradeable read: SI=F full history (2000+), lag-1
fwd_si = pl.fwd_lag(px["SI=F"], H, LAG)
v = fwd_si.reindex(trig).dropna()
rows.append(pl.summarize(v.values, f"SI=F fwd{H} lag1 on thrust"))
wins_dn = int((v < 0).sum())
print(f"sign test (down): {wins_dn}/{len(v)} negative, "
      f"p={pl.sign_test(wins_dn, len(v)):.4f}")

# controls
ctrl = pl.local_control(px.index, trig, win=126)
rows.append(pl.summarize(fwd_si.reindex(ctrl).dropna().values,
                         "local control (+/-126td)"))
rows.append(pl.summarize(fwd_si.dropna().values, "all days"))

# vehicle check: SLV where it exists (2006+)
slv = pl.fwd_lag(px["SLV"], H, LAG).reindex(trig).dropna()
rows.append(pl.summarize(slv.values, "SLV fwd5 lag1 (2006+)"))

# looser variant: silver-only thrust (is the 4-metal AND doing work?)
si_only = px.index[(r5["SI=F"] >= 0.08).fillna(False)]
si_only = pl.declusters(si_only, 10, px.index)
rows.append(pl.summarize(fwd_si.reindex(si_only).dropna().values,
                         f"silver-only 8% thrust (n={len(si_only)})"))

pl.show(rows, "four-metal 5d thrust -> silver next 5, lag-1")
v2 = fwd_si.reindex(trig).dropna()
print(pl.cluster_note(v2.index, v2.values))
for row in pl.era_split(v2.index, v2.values):
    print(row)

# horizon scan for the time stop
for h in (2, 3, 5, 10):
    vv = pl.fwd_lag(px["SI=F"], h, LAG).reindex(trig).dropna()
    s = pl.summarize(vv.values, f"h={h}")
    print(f"h={h:2d}: n={s['n']} mean={s['mean_pct']:+.2f}% "
          f"hit={s['hit']:.0f}% worst={s['worst_pct']:+.2f}%")

# --- entry-timing honesty (tonight is 2 sessions into the 08-07 cluster) ---
# 1) anchored on LATE-in-cluster trigger days, lag-1: FLAT. The edge is
#    cluster-first only; do not chase a thrust that has been running.
late = trig_all.difference(trig)
v_late = fwd_si.reindex(late).dropna()
print("\nlate-in-cluster lag-1:",
      {k: round(x, 2) if isinstance(x, float) else x
       for k, x in pl.summarize(v_late.values, "late").items()})

# 2) anchored on cluster-FIRST, sliding the entry lag. lag=2 (tomorrow's
#    close, given Friday's first trigger) still works; lag=3 decays. So the
#    idea's MOC-tomorrow entry quotes the lag-2 row and tomorrow is the last
#    supported entry.
for lg in (1, 2, 3):
    vv = pl.fwd_lag(px["SI=F"], H, lg).reindex(trig).dropna()
    s = pl.summarize(vv.values, f"lag={lg}")
    dn = int((vv < 0).sum())
    print(f"cluster-first lag={lg}: n={s['n']} mean={s['mean_pct']:+.2f}% "
          f"hit={s['hit']:.0f}% t={s['t']:.2f} worst={s['worst_pct']:+.1f}% "
          f"down {dn}/{s['n']} sign_p={pl.sign_test(dn, s['n']):.4f}")
