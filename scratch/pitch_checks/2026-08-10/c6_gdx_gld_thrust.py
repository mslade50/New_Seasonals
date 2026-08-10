"""C6 round 1 -- Miner-over-metal: GDX vs GLD after an extreme 5d thrust spread.

Trigger on close D: (GDX 5d ret - GLD 5d ret) >= +8pp.  Today = +14.06pp.
Entry MOC D+1 (lag=1). Both directions priced, BOTH LEGS PRICED SEPARATELY
first, because the registry kill on sector-vs-index pairs is exactly "the
trigger selects tape that BOTH legs share".

Also splits by WHERE the thrust happens (GDX deep in a drawdown vs near a high),
which is the candidate's actual novelty claim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
TH = 8.0
P = close_panel(["GDX", "GLD"]).dropna()
ASOF = P.index[-1]
g, gl = P["GDX"], P["GLD"]
sp5 = (g.pct_change(5) - gl.pct_change(5)) * 100
dd = 100 * (g / g.rolling(252).max() - 1)

print(f"sample {P.index.min().date()} .. {ASOF.date()}  n={len(P)}")
print(f"TODAY: 5d spread {sp5.loc[ASOF]:+.2f}pp (th {TH:+.1f}), GDX dd {dd.loc[ASOF]:+.2f}% "
      f"-> fires {bool(sp5.loc[ASOF] >= TH)}")

mask = (sp5 >= TH).fillna(False)
print(f"day-level trigger days: {int(mask.sum())}")

# ---------------------------------------------------------------- legs first
for H in (5, 10):
    trig = P.index[mask.values & fwd_lag(g, H, LAG).notna().values]
    rows = [
        summarize(fwd_lag(g, H, LAG).loc[trig].values, f"GDX long, trigger days (h={H})"),
        summarize(fwd_lag(g, H, LAG).dropna().values, "GDX long, ALL days"),
        summarize(fwd_lag(gl, H, LAG).loc[trig].values, f"GLD long, trigger days (h={H})"),
        summarize(fwd_lag(gl, H, LAG).dropna().values, "GLD long, ALL days"),
    ]
    show(rows, f"0. LEGS PRICED SEPARATELY, day-level (h={H})")

# ------------------------------------------------------- the two spread sides
variants = {
    "spread>=6pp": (sp5 >= 6.0).fillna(False),
    "spread>=8pp": mask,
    "spread>=10pp": (sp5 >= 10.0).fillna(False),
    "spread>=12pp": (sp5 >= 12.0).fillna(False),
}

for H in (5, 10):
    battery(P, mask, [("GDX", 1.0), ("GLD", -1.0)], H,
            f"C6-CONT long GDX / short GLD, 5d-thrust>= {TH}pp", cost_bps=3.5,
            variants=variants, lag=LAG, event_kinds=("cpi", "ppi"))
    battery(P, mask, [("GLD", 1.0), ("GDX", -1.0)], H,
            f"C6-REV long GLD / short GDX, 5d-thrust>= {TH}pp", cost_bps=3.5,
            variants=variants, lag=LAG, event_kinds=("cpi", "ppi"))

# ------------------------------------------- is the spread just gold beta?
# regress GDX fwd on GLD fwd over ALL days -> beta; then ask whether the
# trigger's spread survives a BETA-NEUTRAL construction.
H = 10
fg, fl = fwd_lag(g, H, LAG), fwd_lag(gl, H, LAG)
ok = fg.notna() & fl.notna()
beta = np.polyfit(fl[ok].values, fg[ok].values, 1)[0]
print(f"\n### beta of GDX h={H} fwd on GLD h={H} fwd (all days) = {beta:.2f}")
trig = P.index[mask.values & ok.values]
epi = declusters(trig, H, P.index[ok.values])
raw = (fg - fl)
bn = (fg - beta * fl)
show([summarize(raw.loc[epi].values, f"equal-$ GDX-GLD episodes (N={len(epi)})"),
      summarize(bn.loc[epi].values, f"beta-neutral GDX-{beta:.2f}*GLD episodes"),
      summarize(raw[ok].values, "equal-$ ALL days"),
      summarize(bn[ok].values, "beta-neutral ALL days")],
     "BETA TEST: does the spread survive hedging out gold?")

# ------------------------------------------- WHERE the thrust happens
print("\n### drawdown-position split (the candidate's novelty claim) ###")
for H in (5, 10):
    fgh, flh = fwd_lag(g, H, LAG), fwd_lag(gl, H, LAG)
    okh = fgh.notna() & flh.notna()
    sprd = fgh - flh
    rows = []
    for lbl, sub in (("deep dd (<=-20%)", dd <= -20), ("mid dd (-20..-8%)", (dd > -20) & (dd <= -8)),
                     ("near high (>-8%)", dd > -8)):
        m = mask & sub.fillna(False)
        t = P.index[m.values & okh.values]
        if len(t) == 0:
            rows.append({"label": f"{lbl} h={H}", "n": 0})
            continue
        e = declusters(t, H, P.index[okh.values])
        r = summarize(sprd.loc[e].values, f"{lbl} h={H} SPREAD")
        r["n_days"] = len(t)
        rows.append(r)
        rows.append(summarize(fgh.loc[e].values, f"{lbl} h={H} GDX leg"))
        rows.append(summarize(flh.loc[e].values, f"{lbl} h={H} GLD leg"))
    show(rows, f"drawdown split, episode level (h={H})")

# ------------------------------------------- midterm split
print("\n### midterm split (2026 is midterm) ###")
for H in (5, 10):
    fgh, flh = fwd_lag(g, H, LAG), fwd_lag(gl, H, LAG)
    okh = fgh.notna() & flh.notna()
    sprd = fgh - flh
    t = P.index[mask.values & okh.values]
    e = declusters(t, H, P.index[okh.values])
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(sprd.loc[e].values[mt], f"MIDTERM spread h={H} (N={int(mt.sum())})"),
          summarize(sprd.loc[e].values[~mt], f"non-midterm spread h={H}"),
          summarize(fgh.loc[e].values[mt], f"MIDTERM GDX leg h={H}"),
          summarize(fgh.loc[e].values[~mt], f"non-midterm GDX leg h={H}")],
         f"midterm split h={H}")
