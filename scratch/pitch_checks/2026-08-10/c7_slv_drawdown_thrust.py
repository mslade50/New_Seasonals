"""C7 round 1 -- Silver thrust from deep inside a drawdown.

Claim: a violent 5d thrust while STILL deep below the 52w high is a materially
different state from the same thrust near a high, and the repo has never
separated them.

Trigger on close D: SLV 5d ret >= +8% AND dist-from-52w-high <= -25%.
Today: +9.82% / 5d, -45.55% from the high, -9.82% below the 200d.
Entry MOC D+1. THE TEST THAT DECIDES IT: does the drawdown gate SEPARATE the
populations, or is this a plain momentum-thrust cell with a story stapled on?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
P = close_panel(["SLV", "GLD"]).dropna(subset=["SLV"])
ASOF = P.index[-1]
s = P["SLV"]
r5 = s.pct_change(5) * 100
dd = 100 * (s / s.rolling(252).max() - 1)
sma200 = s.rolling(200).mean()
d200 = 100 * (s / sma200 - 1)

print(f"sample {s.index.min().date()} .. {ASOF.date()}  n={len(s)}")
print(f"TODAY: r5={r5.loc[ASOF]:+.2f}% dd={dd.loc[ASOF]:+.2f}% d200={d200.loc[ASOF]:+.2f}%")

THRUST = (r5 >= 8.0).fillna(False)
DEEP = (dd <= -25.0).fillna(False)
mask = THRUST & DEEP
print(f"thrust days {int(THRUST.sum())}, deep-dd days {int(DEEP.sum())}, JOINT {int(mask.sum())}")

variants = {
    "thrust>=6 & dd<=-25": ((r5 >= 6.0) & DEEP).fillna(False),
    "thrust>=8 & dd<=-25": mask,
    "thrust>=10 & dd<=-25": ((r5 >= 10.0) & DEEP).fillna(False),
    "thrust>=8 & dd<=-35": ((r5 >= 8.0) & (dd <= -35.0)).fillna(False),
    "thrust>=8 & dd<=-15": ((r5 >= 8.0) & (dd <= -15.0)).fillna(False),
    "thrust>=8, NO dd gate": THRUST,
    "thrust>=8 & below 200d": (THRUST & (d200 < 0)).fillna(False),
}

for H in (5, 10):
    battery(P, mask, [("SLV", 1.0)], H,
            f"C7 long SLV, 5d thrust >=8% deep in drawdown", cost_bps=3.0,
            variants=variants, lag=LAG, event_kinds=("cpi", "ppi"))

# ---------------------------------------------------- DOES THE GATE FILTER?
print("\n### THE DECIDING TEST: does the drawdown gate separate the populations? ###")
for H in (2, 5, 10):
    fw = fwd_lag(s, H, LAG)
    ok = fw.notna()
    rows = []
    for lbl, sub in (("thrust & DEEP dd<=-25", DEEP),
                     ("thrust & MID dd -25..-10", ((dd > -25) & (dd <= -10))),
                     ("thrust & NEAR HIGH dd>-10", (dd > -10)),
                     ("thrust, ANY dd", pd.Series(True, index=s.index))):
        m = (THRUST & sub.reindex(s.index).fillna(False))
        t = s.index[m.values & ok.values]
        if len(t) == 0:
            rows.append({"label": f"{lbl} h={H}", "n": 0})
            continue
        e = declusters(t, H, s.index[ok.values])
        r = summarize(fw.loc[e].values, f"{lbl} h={H}")
        r["n_days"] = len(t)
        r["boot_p"] = round(bootstrap_p_le0(fw.loc[e].values), 3)
        rows.append(r)
    rows.append(summarize(fw[ok].values, f"CTRL SLV all days h={H}"))
    show(rows, f"drawdown-position separation, episode level (h={H})")

# ---------------------------------------------------- below/above 200d cut
print("\n### alternate conditioner: below vs above the 200d ###")
for H in (5, 10):
    fw = fwd_lag(s, H, LAG)
    ok = fw.notna()
    rows = []
    for lbl, sub in (("thrust & BELOW 200d", d200 < 0), ("thrust & ABOVE 200d", d200 >= 0)):
        m = THRUST & sub.fillna(False)
        t = s.index[m.values & ok.values]
        e = declusters(t, H, s.index[ok.values])
        r = summarize(fw.loc[e].values, f"{lbl} h={H}")
        r["n_days"] = len(t)
        rows.append(r)
    show(rows, f"200d cut (h={H})")

# ---------------------------------------------------- midterm + year hist
print("\n### midterm split + year histogram (deep-dd cell) ###")
for H in (5, 10):
    fw = fwd_lag(s, H, LAG)
    ok = fw.notna()
    t = s.index[mask.values & ok.values]
    e = declusters(t, H, s.index[ok.values])
    v = fw.loc[e].values
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(v[mt], f"MIDTERM h={H} (N={int(mt.sum())})"),
          summarize(v[~mt], f"non-midterm h={H}")], f"midterm split h={H}")
    print("  episode years:", dict(pd.Series(v).groupby(pd.DatetimeIndex(e).year.values)
                                   .agg(['count', 'mean']).round(4).to_dict('index')))
    print("  episode dates:", [str(d.date()) for d in e])

# ---------------------------------------------------- vs GLD (is it silver?)
print("\n### is it SILVER or the whole metals complex? ###")
H = 10
fs, fg = fwd_lag(s, H, LAG), fwd_lag(P["GLD"], H, LAG)
ok = fs.notna() & fg.notna()
t = s.index[mask.values & ok.values]
e = declusters(t, H, s.index[ok.values])
show([summarize(fs.loc[e].values, "SLV leg"),
      summarize(fg.loc[e].values, "GLD same days"),
      summarize((fs - fg).loc[e].values, "SLV-GLD spread"),
      summarize(fs[ok].values, "SLV all days"),
      summarize(fg[ok].values, "GLD all days")],
     f"complex decomposition h={H}")
