import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["HYG", "IEF", "LQD", "SPY"])
px = px[px["HYG"].notna() & px["IEF"].notna()]
hyg, ief = px["HYG"], px["IEF"]
ratio = hyg / ief
px["RATIO"] = ratio

hyg_r5 = pct_rank(hyg, 5)
ief_r5 = pct_rank(ief, 5)
rat_r5 = pct_rank(ratio, 5)
hyg_z10 = zscore(hyg, 10)
hi252 = rolling_on_valid(hyg, lambda x: x.rolling(252).max())
print("LIVE 2026-09-11: HYG r5 %.1f z10 %.2f off-high %.2f%% | ratio r5 %.1f | IEF r5 %.1f z10 %.2f"
      % (hyg_r5.iloc[-1], hyg_z10.iloc[-1], 100 * (hyg.iloc[-1] / hi252.iloc[-1] - 1),
         rat_r5.iloc[-1], ief_r5.iloc[-1], zscore(ief, 10).iloc[-1]), px.index[-1].date())

parent = hyg_r5 <= 10
child = parent & (rat_r5 >= 80)
spread = parent & (rat_r5 <= 20)
mid = parent & (rat_r5 > 20) & (rat_r5 < 80)
print("day counts parent %d child %d spread-driven %d mid %d" % (parent.sum(), child.sum(), spread.sum(), mid.sum()))

# 5d-return beta of HYG on IEF (non-overlapping 5d)
r5h = hyg.pct_change(5).iloc[::5]; r5i = ief.pct_change(5).iloc[::5]
d = pd.concat([r5h, r5i], axis=1).dropna()
beta = np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var()
d18 = d[d.index >= "2018-01-01"]
beta18 = np.cov(d18.iloc[:, 0], d18.iloc[:, 1])[0, 1] / d18.iloc[:, 1].var()
print(f"beta HYG on IEF (5d non-overlap): full {beta:.3f}  2018+ {beta18:.3f}")

variants = {"r5<=5 & rat>=80": (hyg_r5 <= 5) & (rat_r5 >= 80),
            "r5<=15 & rat>=80": (hyg_r5 <= 15) & (rat_r5 >= 80),
            "r5<=10 & rat>=70": parent & (rat_r5 >= 70),
            "r5<=10 & rat>=90": parent & (rat_r5 >= 90),
            "PARENT r5<=10": parent,
            "SPREAD r5<=10 & rat<=20": spread,
            "MID r5<=10 & 20<rat<80": mid}

H = 5
battery(px, child, [("HYG", 1.0)], H, "C4 child long HYG", 3.5, variants=variants,
        event_kinds=("fomc_decision",))
battery(px, child, [("IEF", 1.0)], H, "C4 SAME DATES long IEF", 3.0, variants=variants,
        event_kinds=("fomc_decision",))
battery(px, parent, [("HYG", 1.0)], H, "PARENT HYG r5<=10 long HYG", 3.5,
        event_kinds=("fomc_decision",))
battery(px, child, [("HYG", 1.0), ("IEF", -beta)], H, f"C4 spread-hedged HYG - {beta:.2f} IEF", 3.5,
        event_kinds=("fomc_decision",))

# compact side-by-side: vehicles x splits, episode-level, excess vs own all-days
rows = []
for lbl, m in [("child", child), ("parent", parent), ("spread", spread), ("mid", mid)]:
    for veh, legs in [("HYG", [("HYG", 1.0)]), ("IEF", [("IEF", 1.0)]), ("LQD", [("LQD", 1.0)]),
                      ("hedged", [("HYG", 1.0), ("IEF", -beta)]), ("SPY", [("SPY", 1.0)])]:
        ret = vehicle_ret(px, legs, H)
        v = ret.notna()
        s = px.index[m.reindex(px.index, fill_value=False).values & v.values]
        e = declusters(s, H, px.index)
        r = summarize(ret.loc[e].values, f"{lbl}/{veh}")
        r["ctl_pct"] = 100 * ret[v & (px.index >= s[0])].mean()
        r["excess_pct"] = r["mean_pct"] - r["ctl_pct"]
        r["pre18"] = 100 * np.nanmean(ret.loc[e[e < "2018"]].values)
        r["post18"] = 100 * np.nanmean(ret.loc[e[e >= "2018"]].values)
        rows.append(r)
show(rows, "SIDE BY SIDE h=5 episodes")
