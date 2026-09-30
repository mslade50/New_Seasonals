"""kB G1 round 1: short GLD after a CONFIRMED rate rise: ^TNX closes at its 252
high AND DX-Y.NYB within 0.5% of its 252 high on the same session, GLD <= -1.5%,
h=1..5. Pre-specified SHORT (continuation). Inversion of W13. Job: kill it.

Gate attribution leads: does the DX leg or the TNX-high leg filter anything over
"GLD <= -1.5% on any day" and over "GLD <= -1.5% on a TNX 252-high day"?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["GLD", "GC=F", "^TNX", "DX-Y.NYB"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}

# own-calendar states
tnx = cl["^TNX"]
tnx_hi = tnx >= tnx.rolling(252).max() - 1e-12
dx = cl["DX-Y.NYB"]
dx_off = dx / dx.rolling(252).max() - 1.0
print(f"live: TNX {tnx.iloc[-1]:.3f} at-high={bool(tnx_hi.iloc[-1])}  DX off {100*dx_off.iloc[-1]:+.3f}%  "
      f"GLD 1d {100*(cl['GLD'].iloc[-1]/cl['GLD'].iloc[-2]-1):+.2f}%  "
      f"GC=F 1d {100*(cl['GC=F'].iloc[-1]/cl['GC=F'].iloc[-2]-1):+.2f}%")
print("NOTE: DX closes -0.502% off its high; the literal 0.5% gate misses today by 0.2 bp. "
      "Threshold used: off >= -0.00505 (the map's stated intent is that today fires).")
df = raw["GLD"]
a = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
print(f"GLD close {df['Close'].iloc[-1]:.2f}  Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/df['Close'].iloc[-1]:.2f}%)  "
      f"off 252 high {100*(df['Close'].iloc[-1]/df['Close'].rolling(252).max().iloc[-1]-1):+.2f}%")


def run(veh: str, hs=(1, 3, 5)):
    px = pd.DataFrame({veh: cl[veh]})
    idx = px.index
    g1 = cl[veh].pct_change().reindex(idx)
    th = tnx_hi.reindex(idx).fillna(False).astype(bool)
    doff = dx_off.reindex(idx)
    near = lambda k: (doff >= -k).fillna(False)
    gdn = lambda k: (g1 <= -k).fillna(False)
    cell = gdn(0.015) & th & near(0.00505)
    print(f"\n\n================ vehicle {veh} ================")
    print("joint-cell days:", [str(d.date()) for d in idx[cell.values]])
    variants = {
        "GATE: GLD<=-1.5% any day": gdn(0.015),
        "GATE: GLD<=-1.5% & TNX 252 high": gdn(0.015) & th,
        "GATE: GLD<=-1.5% & DX near (0.5%)": gdn(0.015) & near(0.00505),
        "GATE: TNX high & DX near, no GLD": th & near(0.00505),
        "NB GLD<=-1.0% joint": gdn(0.010) & th & near(0.00505),
        "NB GLD<=-2.0% joint": gdn(0.020) & th & near(0.00505),
        "NB DX within 1.0% joint": gdn(0.015) & th & near(0.010),
        "NB DX within 2.0% joint": gdn(0.015) & th & near(0.020),
        "COMPLEMENT: GLD<=-1.5% & TNX high & DX NOT near": gdn(0.015) & th & ~near(0.00505),
    }
    for h in hs:
        battery(px, cell, [(veh, -1.0)], h, f"G1 short {veh}", 2.0,
                variants=variants, event_kinds=("nfp", "cpi", "fomc_decision"))
    # episode-level list for the joint at h=5
    ret5 = vehicle_ret(px, [(veh, -1.0)], 5, 1)
    ret1 = vehicle_ret(px, [(veh, -1.0)], 1, 1)
    rows = []
    for d in idx[cell.values]:
        rows.append({"date": d.date(), "gld_1d": round(100 * g1.loc[d], 2),
                     "dx_off": round(100 * doff.loc[d], 2),
                     "short_h1": round(100 * ret1.get(d, np.nan), 3),
                     "short_h5": round(100 * ret5.get(d, np.nan), 3)})
    print("\njoint firings (short-side returns, %):")
    print(pd.DataFrame(rows).to_string(index=False))
    # year split of the GLD<=-1.5% & TNX-high parent (sign test)
    for h in (1, 5):
        r = vehicle_ret(px, [(veh, -1.0)], h, 1)
        v = r.dropna().index
        for lbl, m in (("joint", cell), ("GLD<=-1.5 & TNX high", gdn(0.015) & th),
                       ("GLD<=-1.5 any", gdn(0.015))):
            s = idx[m.values].intersection(v)
            e = declusters(s, h, v)
            vals = r.loc[e].values
            w = int((vals > 0).sum())
            mid = np.asarray(e.year % 4 == 2)
            print(f"  h={h} {lbl:24s} N={len(e):4d} mean {100*vals.mean():+.3f}%  "
                  f"record {w}-{len(vals)-w} sign p {sign_test(w, len(vals)):.4f}  "
                  f"| midterm {100*vals[mid].mean() if mid.any() else np.nan:+.3f}% (n={mid.sum()})  "
                  f"non-mid {100*vals[~mid].mean() if (~mid).any() else np.nan:+.3f}%")


run("GLD")
run("GC=F", hs=(1, 5))
