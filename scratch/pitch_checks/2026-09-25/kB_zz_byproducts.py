"""kB byproducts (NOT pre-specified; searched cells, owe a search charge). Two
parents that beat their own candidates in rounds 1-2, measured once for the
record so the orchestrator can decide whether anything is worth parking:
  (1) FCX/COPX lag <= -8pp vs beta-HG=F over 21d at ANY copper level (M1's parent)
  (2) TNX 252 high with GOLD 21d <= -5% (the gold-led half of C1), short TLT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
TK = ["COPX", "FCX", "HG=F", "GC=F", "^TNX", "TLT"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}

for miner in ("FCX", "COPX"):
    df = pd.concat([cl[miner], cl["HG=F"]], axis=1, keys=[miner, "HG"]).dropna()
    r = df.pct_change()
    b = (r[miner].rolling(252).cov(r["HG"]) / r["HG"].rolling(252).var()).shift(1)
    rp = (r[miner] - b * r["HG"]).dropna()
    px = pd.DataFrame({"PAIR": (1 + rp).cumprod()})
    idx = px.index
    lagv = ((df[miner] / df[miner].shift(21) - 1) - (df["HG"] / df["HG"].shift(21) - 1)).reindex(idx)
    m = (lagv <= -0.08).fillna(False)
    for h in (5, 10):
        ret = vehicle_ret(px, [("PAIR", 1.0)], h)
        e = declusters(idx[(m & ret.notna()).values], 10, idx)
        v = ret.loc[e].values
        w = int((v > 0).sum())
        print(f"\n{miner} lag<=-8pp any copper level h={h}: N={len(e)} mean {100*v.mean():+.3f}% "
              f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}  all-days {100*ret.mean():+.3f}%")
        show(era_split(e, v), f"{miner} parent era split h={h}")
        print("  ", cluster_note(e, v))
        srt = np.sort(v)[::-1]
        print(f"   drop-best-2 {100*srt[2:].mean():+.3f}%")

fut = pd.concat([cl["HG=F"], cl["GC=F"]], axis=1, keys=["HG", "GC"]).dropna()
gc21 = fut["GC"] / fut["GC"].shift(21) - 1
tnx = cl["^TNX"]
px = pd.DataFrame({"TLT": cl["TLT"]})
idx = px.index
th = (tnx >= tnx.rolling(252).max() - 1e-12).reindex(idx).fillna(False).astype(bool)
g = (gc21 <= -0.05).reindex(idx).fillna(False).astype(bool)
for thr in (-0.03, -0.05, -0.07):
    gg = (gc21 <= thr).reindex(idx).fillna(False).astype(bool)
    for h in (5, 10):
        ret = vehicle_ret(px, [("TLT", -1.0)], h)
        e = declusters(idx[(th & gg & ret.notna()).values], 10, idx)
        v = ret.loc[e].values
        w = int((v > 0).sum())
        print(f"\nshort TLT | TNX 252 hi & GC 21d <= {100*thr:.0f}% h={h}: N={len(e)} mean {100*v.mean():+.3f}% "
              f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}; drop-best-2 "
              f"{100*np.sort(v)[::-1][2:].mean():+.3f}%")
        print("  ", cluster_note(e, v), " eps:", [str(d.date()) for d in e])
