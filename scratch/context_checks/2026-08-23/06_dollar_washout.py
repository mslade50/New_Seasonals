"""The dollar's 21-day return sits in the 0.4th percentile of its own year.

The sweep truncated P5b:rank21_extreme at eight subjects and dropped
DX-Y.NYB, EURUSD=X, UUP and USDSGD=X, which is the entire dollar complex and
the lowest reading on the whole tape. Recomputed here by hand.

Cell: DXY 21d return in the bottom 5% of its trailing 252 sessions, and then
the same state crossed with gold's 21d return in the TOP 5% of its own year,
which is what Friday actually printed (gold 21d rank 98.4, DXY 0.4).
Declustered at 21 sessions, forward returns lag=0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, sign_test, declusters,
                       local_control, cluster_note)

TK = ["DX-Y.NYB", "GC=F", "SPY", "EEM", "EURUSD=X"]
px = load_prices(TK)
ASOF = pd.Timestamp("2026-08-21")
C = {t: px[t]["Close"].astype(float).loc[:ASOF] for t in TK}


def rank21(s):
    return s.pct_change(21, fill_method=None).rolling(252).rank(pct=True) * 100.0


dxy, gold = C["DX-Y.NYB"], C["GC=F"]
r_d, r_g = rank21(dxy), rank21(gold)
print(f"live: DXY 21d rank {r_d.iloc[-1]:.1f}, gold 21d rank "
      f"{r_g.reindex(dxy.index).ffill().iloc[-1]:.1f}, bar {dxy.index[-1].date()}")

idx = dxy.index
base_mask = (r_d <= 5.0)
trig = idx[base_mask.fillna(False).values]
epi = declusters(trig, 21, idx)
print(f"\nDXY 21d bottom 5%: raw days {len(trig)}, episodes {len(epi)}")

gj = r_g.reindex(idx).ffill()
joint = base_mask & (gj >= 95.0)
tj = idx[joint.fillna(False).values]
ej = declusters(tj, 21, idx)
print(f"crossed with gold 21d TOP 5%: raw days {len(tj)}, episodes {len(ej)}")
print("joint episodes:", [str(d.date()) for d in ej])


def block(name, dates, sub_tickers=("DX-Y.NYB", "GC=F", "SPY", "EEM")):
    print(f"\n########## {name} ##########")
    ctl = local_control(idx, dates, 126)
    for t in sub_tickers:
        s = C[t].reindex(idx).ffill()
        rows = []
        for h in (1, 5, 10, 21):
            f = fwd_ret(s, h)
            v = f.reindex(dates).dropna()
            r = summarize(v.values, f"{t} h={h}")
            if r["n"]:
                up = int((v.values > 0).sum())
                r["record"] = f"{up}-{r['n'] - up}"
                r["sign_p"] = round(sign_test(up, r["n"]), 4)
                r["ctl_all"] = round(100 * f.dropna().mean(), 3)
                r["ctl_local"] = round(100 * f.reindex(ctl).dropna().mean(), 3)
            rows.append(r)
        df = pd.DataFrame(rows)
        keep = [c for c in ["label", "n", "mean_pct", "median_pct", "hit", "t",
                            "record", "sign_p", "ctl_all", "ctl_local"] if c in df]
        print(df[keep].round(3).to_string(index=False))


block("DXY 21d bottom 5%, all episodes", epi)
block("DXY bottom 5% AND gold top 5%", ej)

print("\n########## the dollar's own path, joint cell ##########")
f21 = fwd_ret(dxy, 21)
v = f21.reindex(ej).dropna()
for d, x in v.items():
    print(f"   {d.date()}  DXY +21d {100*x:+6.2f}%")
print("  concentration:", cluster_note(v.index, v.values, k=2))
cut = pd.Timestamp("2018-01-01")
for lab, w in [("pre-2018", v[v.index < cut]), ("2018+", v[v.index >= cut])]:
    if not len(w):
        continue
    up = int((w.values > 0).sum())
    st = summarize(w.values, lab)
    print(f"  {lab:9s} n={st['n']:2d} mean={st['mean_pct']:+.2f}% record {up}-{st['n']-up}")
