"""K3R c5 quick parent read (candidate NOT LIVE per k3r_live_out.txt:
spread rank 17.06 vs the 252-low rung 0.40).

Long GDX / short beta x GLD (rolling-252 OLS beta on daily returns, lagged 1
session) after the GDX-minus-GLD 5d spread sits at (a) its trailing-252 low,
(b) rank <= 5, (c) rank <= 17.1 (today's actual rung). Episodes = filter then
decluster at gap 10. Rows: pair, outright GDX excess over own all-days drift,
short-GLD leg, and the placebo (GDX 5d rank <= same level, spread rank > 50).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
px = close_panel(["GDX", "GLD"]).dropna()
px = px[px.index <= BAR]
D = px.index
r1 = px.pct_change()
beta = (r1["GDX"].rolling(252).cov(r1["GLD"]) / r1["GLD"].rolling(252).var()).shift(1)
s5 = px["GDX"].pct_change(5) - px["GLD"].pct_change(5)
srank = s5.rolling(252).rank(pct=True) * 100
grank = pct_rank(px["GDX"], 5)
print(f"span {D[0].date()}..{D[-1].date()}  live beta {beta.iloc[-1]:.3f}")

rows = []
for h in (1, 3, 5, 10):
    g = fwd_lag(px["GDX"], h)
    l = fwd_lag(px["GLD"], h)
    pair = g - beta * l
    valid = pair.dropna().index
    for lbl, m in [("252-low", srank <= 100 / 252 + 1e-9),
                   ("rank<=5", srank <= 5),
                   ("rank<=17.1 (live rung)", srank <= 17.1),
                   ("PLACEBO GDXr5<=5 & spread rank>25", (grank <= 5) & (srank > 25)),
                   ("PLACEBO GDXr5<=17 & spread rank>25", (grank <= 17) & (srank > 25))]:
        t = D[m.reindex(D, fill_value=False).values].intersection(valid)
        e = declusters(t, 10, D)
        if len(e) == 0:
            rows.append({"label": f"h={h} {lbl}", "n": 0})
            continue
        v = pair.loc[e].values
        r = summarize(v, f"h={h} {lbl} PAIR")
        r["ctl_pct"] = 100 * pair.loc[valid].mean()
        r["edge_pp"] = r["mean_pct"] - r["ctl_pct"]
        r["gdx_leg_excess_pp"] = 100 * (g.loc[e].mean() - g.loc[valid].mean())
        r["short_gld_leg_excess_pp"] = 100 * (-(beta * l).loc[e].mean() + (beta * l).loc[valid].mean())
        w = int((v > 0).sum())
        r["sign_p"] = sign_test(w, len(v))
        pre = summarize(v[e < pd.Timestamp("2018-01-01")])
        post = summarize(v[e >= pd.Timestamp("2018-01-01")])
        r["pre18_pct"] = pre.get("mean_pct", np.nan)
        r["post18_pct"] = post.get("mean_pct", np.nan)
        rows.append(r)
show(rows, "c5 parent read (episodes, gap 10)")
print(f"\ncost: two legs x 3 bps = 6 bps round trip")
