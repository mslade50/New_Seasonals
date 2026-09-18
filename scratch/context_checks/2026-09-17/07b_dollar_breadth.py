"""Dollar breadth, fixed.

07 passed a RETURN series into pct_rank, which takes a PRICE series and
computes the n-day return itself, so every count came back zero. Redo it, then
ask whether broad dollar strength has any bearing on the September witching
session (07's one real finding: DXY 19-7 up, +0.188%, t 2.30).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, pct_rank, declusters,
)

PAIRS = ["CHF=X", "USDSEK=X", "USDNOK=X", "CAD=X", "USDSGD=X", "JPY=X", "USDMXN=X"]
px = close_panel(["DX-Y.NYB"] + PAIRS)
dxy = px["DX-Y.NYB"].dropna()

counts = None
for p in PAIRS:
    s = px[p].dropna()
    rk = pct_rank(s, 5, 252)
    c = (rk >= 95).astype(float).reindex(px.index)
    counts = c if counts is None else counts.add(c, fill_value=0)
counts = counts.dropna()
today = counts.iloc[-1]
print("today's per-pair 5d percentile:")
for p in PAIRS:
    print(f"  {p:<10} {pct_rank(px[p].dropna(), 5, 252).iloc[-1]:.1f}")
print(f"breadth today: {today:.0f} of {len(PAIRS)} pairs in the top 5% of their year")
print(f"  sessions at >=5 of 7: {int((counts >= 5).sum())}  "
      f">=6: {int((counts >= 6).sum())}  ==7: {int((counts == 7).sum())}")

f1 = dxy.shift(-1) / dxy - 1.0
f5 = dxy.shift(-5) / dxy - 1.0
common = counts.index.intersection(f5.dropna().index)
for h, f in [(1, f1), (5, f5)]:
    rows = []
    for lab, thr in [(">=5 of 7", 5), (">=6 of 7", 6), ("<5 of 7", None)]:
        m = (counts.reindex(common) >= thr) if thr else (counts.reindex(common) < 5)
        d = common[m.values]
        r = summarize(f.loc[d].values, lab)
        if r["n"]:
            k = int((f.loc[d] > 0).sum())
            r["record"] = f"{k}-{r['n'] - k} up"
            r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
        rows.append(r)
    rows.append(summarize(f.dropna().values, "CTL all sessions"))
    show(rows, f"DXY h={h} by dollar breadth")

d = common[(counts.reindex(common) >= 5).values]
epi = declusters(d, 10, common)
print(f"\n>=5-of-7 episodes (10td declustered): {len(epi)}, "
      f"by year {dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")
for h, f in [(1, f1), (5, f5)]:
    dd = epi.intersection(f.dropna().index)
    v = f.loc[dd]
    k = int((v > 0).sum())
    print(f"  DXY h={h}: n={len(v)} mean {100 * v.mean():+.3f}% "
          f"median {100 * v.median():+.3f}% {k}-{len(v) - k} up "
          f"sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")

print("\n=== September witching DXY cell, split by how strong the dollar arrived ===")
qw = pd.DatetimeIndex(load_events(["quad_witching"])["date"])
idx = dxy.index
pos = pd.Series(range(len(idx)), index=idx)
anch = pd.DatetimeIndex([idx[pos[x] - 1] for x in qw if x in pos.index and pos[x] > 0])
sep = anch[anch.month == 9].intersection(f1.dropna().index)
rk5 = pct_rank(dxy, 5, 252)
print(f"  DXY 5d percentile today: {rk5.iloc[-1]:.1f}")
tbl = pd.DataFrame({"year": sep.year,
                    "dxy_5d_pctile": rk5.reindex(sep).round(1).values,
                    "witching_chg_pct": (100 * f1.loc[sep]).round(3).values})
print(tbl.to_string(index=False))
for lab, m in [("arrived strong (5d pctile >= 75)", rk5.reindex(sep) >= 75),
               ("arrived weak/neutral (< 75)", rk5.reindex(sep) < 75)]:
    dd = sep[m.fillna(False).values]
    r = summarize(f1.loc[dd].values, lab)
    if r["n"]:
        k = int((f1.loc[dd] > 0).sum())
        print(f"  {lab}: n={r['n']} mean {r['mean_pct']:+.3f}% {k}-{r['n'] - k} up "
              f"sign p(up) = {sign_test(k, r['n']):.4f}")
