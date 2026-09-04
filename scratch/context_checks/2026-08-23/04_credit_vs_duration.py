"""HYG at a 52-week high while TLT sits at a 52-week low.

Friday's tape: HYG -0.23% from its 252d high, TLT +0.86% off its 252d LOW,
^TNX -0.15% from its 252d high, LQD +0.56% off its low, IEF +0.70% off its
low. High yield is pricing the best year it has had; duration is pricing the
worst. No trigger in the engine sees this because the P9 family carries no
credit-versus-duration pair.

Cell: HYG within 1% of its trailing-252 max on the same close TLT is within
2% of its trailing-252 min. Declustered at 21 sessions so one regime is one
observation. Forward returns lag=0 from that close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, sign_test, declusters,
                       local_control, cluster_note)

TK = ["HYG", "TLT", "SPY", "^VIX", "LQD", "IEF"]
px = load_prices(TK)
ASOF = pd.Timestamp("2026-08-21")
C = {t: px[t]["Close"].astype(float).loc[:ASOF] for t in TK}
idx = C["SPY"].index.intersection(C["HYG"].index).intersection(C["TLT"].index)

hyg, tlt = C["HYG"].reindex(idx), C["TLT"].reindex(idx)
near_hi = hyg / hyg.rolling(252).max() - 1.0
near_lo = tlt / tlt.rolling(252).min() - 1.0
print(f"live: HYG {100*near_hi.iloc[-1]:+.2f}% from 252d high, "
      f"TLT {100*near_lo.iloc[-1]:+.2f}% above 252d low, bar {idx[-1].date()}")

mask = (near_hi >= -0.01) & (near_lo <= 0.02)
trig = idx[mask.fillna(False).values]
epi = declusters(trig, 21, idx)
print(f"\nraw days {len(trig)}, episodes at 21td gap {len(epi)}")
print("episodes:", [str(d.date()) for d in epi])
print("first/last raw day of each year:",
      {int(y): (str(g.min().date()), str(g.max().date()), len(g))
       for y, g in pd.Series(trig, index=trig).groupby(trig.year)})

ctl = local_control(idx, trig, 126)
for t in ["SPY", "HYG", "TLT", "^VIX"]:
    s = C[t].reindex(idx)
    print(f"\n########## {t} forward from the cell ##########")
    rows = []
    for h in (1, 5, 10, 21, 63):
        f = fwd_ret(s, h)
        v = f.reindex(epi).dropna()
        r = summarize(v.values, f"h={h} episodes")
        if r["n"]:
            up = int((v.values > 0).sum())
            r["record"] = f"{up}-{r['n'] - up}"
            r["sign_p"] = round(sign_test(up, r["n"]), 4)
            r["ctl_all_pct"] = round(100 * f.dropna().mean(), 3)
            r["ctl_local_pct"] = round(100 * f.reindex(ctl).dropna().mean(), 3)
            r["edge_vs_all"] = round(r["mean_pct"] - r["ctl_all_pct"], 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = [c for c in ["label", "n", "mean_pct", "median_pct", "hit", "t", "record",
                        "sign_p", "ctl_all_pct", "ctl_local_pct", "edge_vs_all"] if c in df]
    print(df[keep].round(3).to_string(index=False))

print("\n########## how rare is it, and what did each episode become? ##########")
f21 = fwd_ret(C["SPY"].reindex(idx), 21)
f63 = fwd_ret(C["SPY"].reindex(idx), 63)
for d in epi:
    a, b = f21.get(d, np.nan), f63.get(d, np.nan)
    print(f"   {d.date()}  SPY +21d {100*a:+6.2f}%   +63d {100*b:+6.2f}%"
          if not np.isnan(a) else f"   {d.date()}  (incomplete)")
v = f21.reindex(epi).dropna()
print("  concentration h21:", cluster_note(v.index, v.values, k=2))

print("\n########## the looser version, to check the thresholds are not the story ##########")
for hi_th, lo_th in [(-0.02, 0.03), (-0.005, 0.01), (-0.03, 0.05)]:
    m = (near_hi >= hi_th) & (near_lo <= lo_th)
    tr = idx[m.fillna(False).values]
    ep = declusters(tr, 21, idx)
    v = fwd_ret(C["SPY"].reindex(idx), 21).reindex(ep).dropna()
    up = int((v.values > 0).sum())
    st = summarize(v.values, f"HYG>={100*hi_th:.1f}% TLT<=+{100*lo_th:.0f}%")
    print(f"   {st['label']:28s} epi={st['n']:2d} SPY+21d mean={st['mean_pct']:+.2f}% "
          f"med={st['median_pct']:+.2f}% record {up}-{st['n']-up} p={sign_test(up, st['n']):.4f}")

print("\n########## TLT is the leg with the edge: threshold sensitivity ##########")
tltc = C["TLT"].reindex(idx)
for hi_th, lo_th in [(-0.005, 0.01), (-0.01, 0.02), (-0.02, 0.03), (-0.03, 0.05), (-0.05, 0.07)]:
    m = (near_hi >= hi_th) & (near_lo <= lo_th)
    tr = idx[m.fillna(False).values]
    ep = declusters(tr, 21, idx)
    lc = local_control(idx, tr, 126)
    line = f"   HYG>={100*hi_th:+.1f}% & TLT<=+{100*lo_th:.0f}%  epi={len(ep):2d}"
    for h in (1, 21, 63):
        f = fwd_ret(tltc, h)
        v = f.reindex(ep).dropna()
        if not len(v):
            continue
        up = int((v.values > 0).sum())
        st = summarize(v.values, "")
        lcm = 100 * f.reindex(lc).dropna().mean()
        line += (f" | h{h}: {st['mean_pct']:+.2f}% vs loc {lcm:+.2f}% "
                 f"({up}-{st['n']-up}, p={sign_test(up, st['n']):.3f})")
    print(line)

print("\n########## and the era split on the widest window ##########")
m = (near_hi >= -0.03) & (near_lo <= 0.05)
tr = idx[m.fillna(False).values]
ep = declusters(tr, 21, idx)
cut = pd.Timestamp("2018-01-01")
for h in (21, 63):
    f = fwd_ret(tltc, h)
    for lab, e in [("pre-2018", ep[ep < cut]), ("2018+", ep[ep >= cut])]:
        v = f.reindex(e).dropna()
        if not len(v):
            continue
        up = int((v.values > 0).sum())
        st = summarize(v.values, "")
        print(f"   h{h} {lab:9s} n={st['n']:2d} mean={st['mean_pct']:+.2f}% "
              f"med={st['median_pct']:+.2f}% record {up}-{st['n']-up} p={sign_test(up, st['n']):.4f}")
