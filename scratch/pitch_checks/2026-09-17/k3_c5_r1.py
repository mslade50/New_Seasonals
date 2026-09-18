"""K3 c5 round 1 - Long GDX against beta x GLD after a DOWNSIDE miner/metal
5d ratio break (GDX 5d minus GLD 5d at a trailing-252 low). Signal 2026-09-16.

Step 0 (gate): is the live spread actually at a trailing-252 low?
Then (measured regardless, so the verdict is not only 'not live'):
  - cell at rank <= 0.4 (the 252 min), <= 2, <= 5
  - beta-neutral pair (rolling 252 OLS beta on daily returns, as of D)
  - outright GDX vs own drift
  - placebo: GDX 5d rank equally extreme regardless of GLD
  - decluster, era, midterm
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
s5 = px["GDX"].pct_change(5) - px["GLD"].pct_change(5)
rank = s5.rolling(252).rank(pct=True) * 100
mn = s5.rolling(252).min()
live = s5.iloc[-1]
print(f"live GDX5 {100*px['GDX'].pct_change(5).iloc[-1]:+.2f}%  GLD5 "
      f"{100*px['GLD'].pct_change(5).iloc[-1]:+.2f}%  spread {100*live:+.2f}pp")
print(f"trailing-252 pctile of live spread {rank.iloc[-1]:.2f}  "
      f"trailing-252 min {100*mn.iloc[-1]:+.2f}pp  "
      f"(min set on {s5.iloc[-252:].idxmin().date()})")
print(f"spread needed to be AT the 252 low: <= {100*s5.iloc[-252:-1].min():+.2f}pp "
      f"(gap {100*(live - s5.iloc[-252:-1].min()):+.2f}pp)")
# how many of last 252 spreads are below today
print(f"sessions in trailing 252 with a lower spread: "
      f"{int((s5.iloc[-252:-1] < live).sum())}")
gdx_rank5 = pct_rank(px["GDX"], 5)
print(f"GDX 5d rank {gdx_rank5.iloc[-1]:.1f}")
for d in s5.index[-8:]:
    print(f"  {d.date()} spread {100*s5[d]:+.2f}pp rank {rank[d]:.1f}")

# rolling beta, as of D (uses returns through D only)
cov = r1["GDX"].rolling(252).cov(r1["GLD"])
var = r1["GLD"].rolling(252).var()
beta = (cov / var)
print(f"live beta GDX on GLD {beta.iloc[-1]:.3f}")

COST = 6.0  # bps two-leg round trip


def pair_ret(h, lag=1):
    return fwd_lag(px["GDX"], h, lag) - beta * fwd_lag(px["GLD"], h, lag)


def row(mask, ret, h, lab, gap=None):
    valid = ret.notna()
    sig = D[(mask.reindex(D, fill_value=False) & valid).values]
    epi = declusters(sig, gap or max(h, 5), D)
    r = summarize(ret.loc[epi].values, lab)
    base = ret[valid].mean()
    if r["n"]:
        r["ctl_all_pct"] = round(100 * base, 3)
        r["edge_pp"] = round(r["mean_pct"] - 100 * base, 3)
        w = int((ret.loc[epi] > 0).sum())
        r["rec"] = f"{w}-{r['n']-w}"
        r["sign_p"] = round(sign_test(w, r["n"]), 4)
        r["cost_x"] = round(1e4 * ret.loc[epi].mean() / COST, 2)
    return r, epi


for thr in (100 / 252 + 1e-9, 2.0, 5.0):
    cell = rank <= thr
    rows = []
    for h in (1, 2, 3, 5, 10):
        rows.append(row(cell, pair_ret(h), h, f"PAIR thr{thr:.1f} h={h}")[0])
        rows.append(row(cell, fwd_lag(px["GDX"], h), h, f"GDX outright h={h}")[0])
    show(rows, f"cell spread rank <= {thr:.2f}  days={int(cell.sum())}")

# placebo: GDX 5d rank equally bad regardless of GLD
cell = rank <= 2.0
plac = gdx_rank5 <= 2.0
anti = plac & ~cell
rows = []
for h in (3, 5, 10):
    rows.append(row(cell, pair_ret(h), h, f"cell PAIR h={h}")[0])
    rows.append(row(plac, pair_ret(h), h, f"GDX5 rank<=2 PAIR h={h}")[0])
    rows.append(row(anti, pair_ret(h), h, f"GDX flush NOT ratio PAIR h={h}")[0])
    rows.append(row(plac, fwd_lag(px['GDX'], h), h, f"GDX5 rank<=2 GDX h={h}")[0])
show(rows, "placebo: is it the ratio or a GDX flush?")

# era / midterm on h=5 cell rank<=2
for h in (3, 5):
    r, epi = row(cell, pair_ret(h), h, "x")
    v = pair_ret(h).loc[epi].values
    show(era_split(epi, v), f"era h={h}")
    mid = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(v[mid], "midterm"), summarize(v[~mid], "non-midterm")],
         f"midterm h={h}")
    print(cluster_note(epi, v))
