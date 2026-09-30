"""C2 round 1: short SLV against long beta-GLD after the FIRST complex-wide metals
break (GLD, SLV, GDX each <= -2%, no faithful break in the prior 5 sessions),
watchlist 28's pair translation.

Scored against the PAIR's own drift (registry 5160-5170). Beta: point-in-time
trailing-252 daily OLS beta of SLV on GLD known at the signal close; fixed 1.5
and equal-dollar forms for contrast. Lag 0/1/2 profile (registry 2646: the W28
outright cell was one session wide and started a session late).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-28")
px = close_panel(["GLD", "SLV", "GDX"]).dropna().loc[:BAR]
r1 = px.pct_change(fill_method=None)
faith = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
prior = faith.shift(1).rolling(5).max().fillna(0).astype(bool)
first = faith & ~prior
follow = faith & prior

cov = r1["SLV"].rolling(252).cov(r1["GLD"])
var = r1["GLD"].rolling(252).var()
beta = (cov / var)
print(f"TODAY {BAR.date()}: faithful={bool(faith.loc[BAR])} first={bool(first.loc[BAR])} "
      f"PIT beta={beta.loc[BAR]:.3f}  full-sample beta="
      f"{r1['SLV'].cov(r1['GLD'])/r1['GLD'].var():.3f}")
print(f"GLD {100*r1.loc[BAR,'GLD']:+.2f}%  SLV {100*r1.loc[BAR,'SLV']:+.2f}%  "
      f"GDX {100*r1.loc[BAR,'GDX']:+.2f}%  SLV-beta*GLD residual "
      f"{100*(r1.loc[BAR,'SLV']-beta.loc[BAR]*r1.loc[BAR,'GLD']):+.2f}pp")


def pair_ret(h: int, lag: int, form: str) -> pd.Series:
    s = fwd_lag(px["SLV"], h, lag)
    g = fwd_lag(px["GLD"], h, lag)
    b = {"pit": beta, "b1.5": 1.5, "eqdollar": 1.0}[form]
    return -s + b * g


def cell(mask, h, lag, form, label):
    r = pair_ret(h, lag, form)
    ok = r.notna()
    d = px.index[mask.values & ok.values]
    e = declusters(d, max(h, 1), px.index)
    v = r.loc[e].values
    base = r[ok]
    w = int((v > 0).sum())
    o = summarize(v, label)
    o["rec"] = f"{w}-{len(v)-w}"
    o["drift_pct"] = round(100 * base.mean(), 3)
    o["edge_pp"] = round(o.get("mean_pct", np.nan) - 100 * base.mean(), 3)
    o["p_coin"] = round(sign_test(w, len(v)), 4)
    o["p_vs_uprate"] = round(sign_test(w, len(v), float((base > 0).mean())), 4)
    return o, e, v


for form in ("pit", "b1.5", "eqdollar"):
    for lag in (0, 1, 2):
        rows = []
        for h in (1, 2, 3, 5):
            for lbl, m in (("FIRST", first), ("ALL faithful", faith), ("FOLLOW", follow)):
                rows.append(cell(m, h, lag, form, f"{lbl} h={h}")[0])
        show(rows, f"pair short SLV / long {form} GLD, lag={lag}")

# leg attribution at the headline lag 1 (PIT beta)
print("\n=== leg attribution, FIRST breaks, lag 1, PIT beta ===")
for h in (1, 2, 3, 5):
    s = -fwd_lag(px["SLV"], h, 1)
    g = beta * fwd_lag(px["GLD"], h, 1)
    ok = (s + g).notna()
    e = declusters(px.index[first.values & ok.values], h, px.index)
    print(f"h={h}: n={len(e)} short-SLV leg {100*s.loc[e].mean():+.3f}%  "
          f"long beta*GLD leg {100*g.loc[e].mean():+.3f}%  pair {100*(s+g).loc[e].mean():+.3f}%  "
          f"| unconditional: SLV-short {100*s[ok].mean():+.3f}%  bGLD {100*g[ok].mean():+.3f}%")

# era + concentration by VALUE on the traded side, h=1 lag 1 PIT
for h in (1, 3):
    o, e, v = cell(first, h, 1, "pit", f"FIRST h={h}")
    show(era_split(e, v), f"FIRST breaks era split, h={h} lag 1 PIT")
    order = np.argsort(-v)
    tot = v.sum()
    print(f"  total {100*tot:+.2f}pp; best-2 by value {100*v[order[:2]].sum():+.2f}pp "
          f"({100*v[order[:2]].sum()/tot:.0f}% of total); drop-best-2 mean "
          f"{100*np.delete(v, order[:2]).mean():+.3f}%; drop-best-3 mean "
          f"{100*np.delete(v, order[:3]).mean():+.3f}%")
    yrs = pd.Series(v, index=e.year).groupby(level=0).sum().sort_values(ascending=False)
    print(f"  best years {dict((k, round(100*x, 2)) for k, x in yrs.head(3).items())}; "
          f"2025-26 share {100*yrs.loc[[y for y in yrs.index if y >= 2025]].sum()/tot:.0f}%")
    lc = local_control(pair_ret(h, 1, "pit").dropna().index, e)
    print(f"  local +/-126td pair drift {100*pair_ret(h, 1, 'pit').loc[lc].mean():+.3f}%")

# battery for the structural checks, fixed beta 1.5 (closest to recent PIT)
for h in (1, 3):
    battery(px, first, [("SLV", -1.0), ("GLD", 1.5)], h,
            "short SLV / long 1.5x GLD after a FIRST complex break", cost_bps=3,
            variants={"ALL faithful": faith, "FOLLOW": follow,
                      "first & SLV<=-4%": first & (r1["SLV"] <= -0.04),
                      "first & GLD<=-3.5%": first & (r1["GLD"] <= -0.035)},
            event_kinds=("nfp",))
