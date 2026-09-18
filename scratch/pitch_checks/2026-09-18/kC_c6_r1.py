"""c6 round 1: LONG SLV after a complex-wide metals UP day (GLD, SLV, GDX all
>= +1.5% on session D, dollar DX-Y.NYB not up). Entry MOC D+1 (lag=1).
Signal 2026-09-17: GLD +1.69, SLV +3.37, GDX +3.36, DX -0.09 -> entry 09-18 close.

1. battery at h=1,3,5 (controls own drift / all days / local +/-126td, era, cost)
2. LAG PROFILE lag 0/1/2 at h=1..5 (registry debt, line 2645)
3. record vs SLV's OWN up-rate, signed concentration (cluster_note netting debt)
4. rows: GLD and GDX as vehicles; deep-drawdown row (SLV >= 20% under 52w high,
   and below its 200d); midterm split
5. ENTRY-DAY split (registry line 2660): split by SLV's move on the entry session
   D+1, return measured from the D+1 close
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

GAP = 5
COST = 6.0
px = panel(["GLD", "SLV", "GDX", "DX-Y.NYB", "UUP", "SPY"], "GLD", ffill=("DX-Y.NYB", "UUP"))
px = px.loc["2006-05-22":]  # GDX inception
r1 = {t: dret(px[t]) for t in px.columns}
slv = px["SLV"]
dd52 = slv / slv.rolling(252, min_periods=200).max() - 1.0
v200 = slv / slv.rolling(200).mean() - 1.0

trig = ((r1["GLD"] >= 0.015) & (r1["SLV"] >= 0.015) & (r1["GDX"] >= 0.015)
        & (r1["DX-Y.NYB"] <= 0)).fillna(False)
last = px.index[-1]
print(f"panel {px.index[0].date()}..{last.date()}  day-level triggers {int(trig.sum())}")
print(f"LIVE {last.date()}: GLD {100*r1['GLD'].iloc[-1]:+.2f} SLV {100*r1['SLV'].iloc[-1]:+.2f} "
      f"GDX {100*r1['GDX'].iloc[-1]:+.2f} DX {100*r1['DX-Y.NYB'].iloc[-1]:+.3f}  fired={bool(trig.iloc[-1])}"
      f"   SLV dd52 {100*dd52.iloc[-1]:+.1f}%  vs200d {100*v200.iloc[-1]:+.1f}%")

L = [("SLV", 1.0)]
for h in (1, 3, 5):
    battery(px, trig, L, h, f"c6 LONG SLV complex-up h={h}", COST, min_gap=GAP,
            variants={
                "thr 1.0%": ((r1["GLD"] >= .01) & (r1["SLV"] >= .01) & (r1["GDX"] >= .01) & (r1["DX-Y.NYB"] <= 0)).fillna(False),
                "thr 2.0%": ((r1["GLD"] >= .02) & (r1["SLV"] >= .02) & (r1["GDX"] >= .02) & (r1["DX-Y.NYB"] <= 0)).fillna(False),
                "no dollar gate": ((r1["GLD"] >= .015) & (r1["SLV"] >= .015) & (r1["GDX"] >= .015)).fillna(False),
                "dollar UP (anti)": ((r1["GLD"] >= .015) & (r1["SLV"] >= .015) & (r1["GDX"] >= .015) & (r1["DX-Y.NYB"] > 0)).fillna(False),
            })

print("\n" + "=" * 100)
print("2. LAG PROFILE (episode-level, gap 5): lag 0 / 1 / 2, long SLV")
rows = []
for h in (1, 2, 3, 5):
    for lag in (0, 1, 2):
        s, _, _ = cellstats(px, trig, L, h, f"h={h} lag={lag}", GAP, lag)
        rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("3. record vs own up-rate + SIGNED concentration (lag=1)")
for h in (1, 2, 3, 5):
    s, epi, vals = cellstats(px, trig, L, h, f"h={h}", GAP)
    print(f"  h={h}: {s['rec']} p_coin {s['p_coin']} p_base {s['p_base']} mean {s['mean_pct']:+.3f}% "
          f"ctl {s['ctl_pct']:+.3f}% edge {s['edge_pp']:+.3f}pp")
    print("   ", signed_conc(epi, vals, f"h={h}"))

print("\n" + "=" * 100)
print("4. ROWS: vehicles GLD/GDX, deep drawdown, below 200d, midterm")
rows = []
for h in (1, 3, 5):
    for tk in ("SLV", "GLD", "GDX"):
        s, _, _ = cellstats(px, trig, [(tk, 1.0)], h, f"h={h} long {tk}", GAP, cost_bps=COST)
        rows.append(s)
show(rows, "vehicles")
rows = []
for h in (1, 3, 5):
    for lbl, m in [("all", trig),
                   ("SLV dd52 <= -20%", trig & (dd52 <= -0.20)),
                   ("SLV dd52 > -20%", trig & (dd52 > -0.20)),
                   ("SLV dd52 <= -35%", trig & (dd52 <= -0.35)),
                   ("SLV below 200d", trig & (v200 < 0)),
                   ("SLV above 200d", trig & (v200 >= 0)),
                   ("dd52<=-20 & below200", trig & (dd52 <= -0.20) & (v200 < 0)),
                   ("midterm", trig & midterm_mask(px.index)),
                   ("non-midterm", trig & ~midterm_mask(px.index))]:
        s, _, _ = cellstats(px, m.fillna(False), L, h, f"h={h} {lbl}", GAP)
        rows.append(s)
show(rows, "state rows (long SLV)")

print("\n" + "=" * 100)
print("5. ENTRY-DAY SPLIT: SLV return on D+1 (the entry session); fwd measured from D+1 close")
e1 = r1["SLV"].shift(-1)  # SLV move on D+1, aligned to D
rows = []
for h in (1, 3, 5):
    for lbl, m in [("entry day SLV > +1%", trig & (e1 > 0.01)),
                   ("entry day in [-1%,+1%]", trig & (e1.abs() <= 0.01)),
                   ("entry day SLV < -1%", trig & (e1 < -0.01)),
                   ("entry day up (>0)", trig & (e1 > 0)),
                   ("entry day down (<=0)", trig & (e1 <= 0))]:
        s, _, _ = cellstats(px, m.fillna(False), L, h, f"h={h} {lbl}", GAP)
        rows.append(s)
show(rows)
s, epi, _ = cellstats(px, trig, L, 1, "", GAP)
e1v = e1.loc[epi].dropna()
print(f"  D+1 (entry-session) SLV move on trigger episodes: mean {100*e1v.mean():+.3f}% "
      f"hit {100*(e1v>0).mean():.1f}% N={len(e1v)}  (SLV all-days 1d {100*r1['SLV'].mean():+.3f}%)")
