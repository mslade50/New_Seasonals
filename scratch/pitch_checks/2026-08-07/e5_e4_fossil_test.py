"""E5: the decisive test for E4 -- is the "Friday at the high" short a pre-2013 fossil?

E4 SPEC-A h=1 showed episode mean -0.115% (t=-1.28), and the Friday condition appeared to
flip the sign of the price cell (+0.121% not-Fri vs -0.115% Fri). This script asks whether
that interaction survives into the modern era, and whether it survives at all once you
account for the fact that the raw weekend effect itself has reversed since 2022.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY"])
spy = px["SPY"]["Close"]
idx = spy.index
rk5 = pct_rank(spy, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100
nxt_dow = pd.Series(idx, index=idx).shift(-1).dt.dayofweek


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


def welch(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))


price = (rk5 >= 90) & (dist >= -0.5)

for H in (1, 3):
    f = fwd_entry_next(spy, H)
    valid = rk5.notna() & dist.notna() & f.notna()
    print(f"\n########## h={H} Friday x price-cell interaction by era ##########")
    rows = []
    for lab, lo, hi in [("FULL 2000-2026", 2000, 2027), ("pre-2013", 2000, 2013),
                        ("2013+", 2013, 2027), ("2018+", 2018, 2027),
                        ("2022+", 2022, 2027)]:
        w = valid & (idx >= pd.Timestamp(f"{lo}-01-01")) & (idx < pd.Timestamp(f"{hi}-01-01"))
        fri = f[w & price & (nxt_dow == 4)].dropna().values
        nfr = f[w & price & (nxt_dow != 4)].dropna().values
        allfri = f[w & (nxt_dow == 4)].dropna().values
        alld = f[w].dropna().values
        rows.append(dict(era=lab, n_fri=len(fri),
                         fri_mean=round(100*fri.mean(), 4) if len(fri) else np.nan,
                         n_notfri=len(nfr),
                         notfri_mean=round(100*nfr.mean(), 4) if len(nfr) else np.nan,
                         interaction=round(100*(fri.mean()-nfr.mean()), 4) if len(fri) and len(nfr) else np.nan,
                         t_inter=round(welch(fri, nfr), 2),
                         raw_weekend=round(100*(allfri.mean()-alld.mean()), 4)))
    print(pd.DataFrame(rows).to_string(index=False))

# ---------- bootstrap the modern-era short directly ----------
H = 1
f = fwd_entry_next(spy, H)
valid = rk5.notna() & dist.notna() & f.notna()
cond = price & (nxt_dow == 4) & valid
for lab, lo in [("FULL", 2000), ("2013+", 2013), ("2018+", 2018)]:
    m = cond & (idx >= pd.Timestamp(f"{lo}-01-01"))
    ep = declusters(idx[m], H + 1, idx)
    v = f[ep].dropna().values
    s = summarize(-v, f"SHORT {lab} h=1")
    print(f"\n{lab:6s} SHORT h=1: N={s['n']} mean={s['mean_pct']:+.4f}% t={s['t']:+.2f} "
          f"hit={s['hit']:.0f}% worst={s['worst_pct']:+.3f}% "
          f"bootP(<=0)={bootstrap_p_le0(-v):.4f}")
    print(f"        after 1bp round-trip cost: {s['mean_pct']-0.01:+.4f}% "
          f"({(s['mean_pct']-0.01)/0.01:.1f}x cost)")

# ---------- what actually drives the pre-2013 cell? list the episodes ----------
print("\n########## the pre-2013 episodes that carry E4 (h=1, SHORT sign) ##########")
ep = declusters(idx[cond], H + 1, idx)
v = f[ep].dropna().values
for d, r in zip(ep, v):
    tag = "PRE-2013" if d < pd.Timestamp("2013-01-01") else ""
    print(f"  trigger {d.date()} ({d.day_name()[:3]})  entry Fri {idx[idx.get_loc(d)+1].date()}  "
          f"long_ret={100*r:+.3f}%  short_pnl={-100*r:+.3f}%  {tag}")

# ---------- placebo: same price cell, each entry weekday ----------
print("\n########## placebo: price cell by ENTRY weekday (h=1, episodes) ##########")
names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
rows = []
for dw, nm in names.items():
    m = price & (nxt_dow == dw) & valid
    ep_ = declusters(idx[m], H + 1, idx)
    vv = f[ep_].dropna().values
    if len(vv) < 3:
        continue
    s = summarize(vv, f"entry {nm}")
    rows.append(dict(entry_dow=nm, n=s["n"], mean=round(s["mean_pct"], 4),
                     t=round(s["t"], 2), hit=round(s["hit"], 0)))
print(pd.DataFrame(rows).to_string(index=False))
print("\n(if several weekdays look as extreme as Friday, the Friday cell is a 5-way "
      "multiple-comparison winner, not a weekend effect)")

# ---------- 2026 regime check: is the weekend effect currently even negative? ----------
print("\n########## trailing weekend effect, rolling 252-Friday window ##########")
frim = valid & (nxt_dow == 4)
fs = f[frim].dropna()
roll = fs.rolling(104).mean() * 100
for y in range(2015, 2027):
    sel = roll[(roll.index >= pd.Timestamp(f"{y}-01-01")) & (roll.index < pd.Timestamp(f"{y+1}-01-01"))]
    if len(sel):
        print(f"  {y}: trailing-104-Friday mean Fri->Mon = {sel.iloc[-1]:+.4f}%")
