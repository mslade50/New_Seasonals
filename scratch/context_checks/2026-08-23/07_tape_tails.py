"""Two loose ends: coffee's -10.6% session, and the era check the dollar
cell in drill 06 still owed.

KC=F fell 10.64% on Friday, the largest single-session move anywhere on the
98-ticker tape. The engine's P6 cell (>= 2 ATR down, n=54) is too coarse for
a move that size, so this conditions on the magnitude instead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, sign_test, declusters,
                       local_control, cluster_note)

px = load_prices(["KC=F", "DX-Y.NYB", "SPY"])
ASOF = pd.Timestamp("2026-08-21")

# ---------------------------------------------------------------- coffee ---
kc = px["KC=F"]["Close"].astype(float).loc[:ASOF]
idx = kc.index
r1 = kc.pct_change(fill_method=None)
print(f"live: KC=F {100*r1.iloc[-1]:+.2f}% on {idx[-1].date()}; "
      f"prior 5 sessions {[round(100*x,1) for x in r1.iloc[-6:-1].values]}")

for th in (-0.06, -0.08, -0.10):
    m = r1 <= th
    trig = idx[m.fillna(False).values]
    epi = declusters(trig, 5, idx)
    ctl = local_control(idx, trig, 126)
    print(f"\n########## KC=F sessions of {100*th:.0f}% or worse "
          f"(raw {len(trig)}, episodes {len(epi)}) ##########")
    rows = []
    for h in (1, 3, 5, 10, 21):
        f = fwd_ret(kc, h)
        v = f.reindex(epi).dropna()
        r = summarize(v.values, f"h={h}")
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
    if th == -0.08:
        print("  dates:", [str(d.date()) for d in epi])
        v = fwd_ret(kc, 5).reindex(epi).dropna()
        print("  concentration h5:", cluster_note(v.index, v.values, k=2))
        cut = pd.Timestamp("2018-01-01")
        for lab, w in [("pre-2018", v[v.index < cut]), ("2018+", v[v.index >= cut])]:
            if not len(w):
                continue
            up = int((w.values > 0).sum())
            st = summarize(w.values, lab)
            print(f"  {lab:9s} n={st['n']:2d} mean={st['mean_pct']:+.2f}% "
                  f"med={st['median_pct']:+.2f}% record {up}-{st['n']-up}")

# ------------------------------------------------------- dollar era check ---
print("\n\n########## drill 06 follow-up: era split on DXY 21d bottom 5% ##########")
dxy = px["DX-Y.NYB"]["Close"].astype(float).loc[:ASOF]
di = dxy.index
rk = dxy.pct_change(21, fill_method=None).rolling(252).rank(pct=True) * 100.0
trig = di[(rk <= 5.0).fillna(False).values]
epi = declusters(trig, 21, di)
spy = px["SPY"]["Close"].astype(float).reindex(di).ffill()
cut = pd.Timestamp("2018-01-01")
for name, s in [("DXY", dxy), ("SPY", spy)]:
    f = fwd_ret(s, 1)
    for lab, e in [("all", epi), ("pre-2018", epi[epi < cut]), ("2018+", epi[epi >= cut])]:
        v = f.reindex(e).dropna()
        if not len(v):
            continue
        up = int((v.values > 0).sum())
        st = summarize(v.values, lab)
        print(f"   {name} h1 {lab:9s} n={st['n']:2d} mean={st['mean_pct']:+.3f}% "
              f"med={st['median_pct']:+.3f}% record {up}-{st['n']-up} "
              f"down_p={sign_test(st['n']-up, st['n']):.4f} t={st['t']:+.2f}")
    v = f.reindex(epi).dropna()
    print(f"   {name} concentration:", cluster_note(v.index, v.values, k=2))
