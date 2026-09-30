"""C1 round 2: long gold after a >= 3.5% one-day crash.

Round 1 showed the dollar-and-yields gate has 2 prior instances (both June 2026)
and the yields leg alone inverts the parent, so this round attributes the gate,
walks the definition neighbours (% and ATR, first-in-21d vs any), splits by era
and regime, measures concentration by VALUE on the traded side, checks whether
GC=F's stronger read is dates or close timing, and asks whether the crash day
adds anything over the drawdown state it sits in (filter_vs_reanchor).
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
TK = ["GLD", "GC=F", "DX-Y.NYB", "^TNX"]
raw = load_prices(TK)
idx = raw["GLD"].loc[:BAR].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx).ffill(limit=3) for t in TK})
g = raw["GLD"].loc[:BAR]
atr = pd.Series(wilder_atr(g["High"], g["Low"], g["Close"]), index=idx)
r1 = px["GLD"].pct_change(fill_method=None)
ratr = px["GLD"].diff() / atr.shift(1)
hi = lambda s: s.rolling(252, min_periods=200).max()
dx_near = px["DX-Y.NYB"] >= 0.99 * hi(px["DX-Y.NYB"])
tnx_near = px["^TNX"] >= 0.97 * hi(px["^TNX"])
off_hi = px["GLD"] / hi(px["GLD"]) - 1
v200 = px["GLD"] / px["GLD"].rolling(200).mean() - 1


def first_in(m, n=21):
    return m & ~m.shift(1).rolling(n, min_periods=1).max().fillna(0).astype(bool)


def cell(mask, h, veh="GLD", label=""):
    r = fwd_lag(px[veh], h, 1)
    ok = r.notna()
    d = idx[mask.fillna(False).values & ok.values]
    e = declusters(d, h, idx)
    v = r.loc[e].values
    base = r[ok]
    w = int((v > 0).sum())
    o = summarize(v, label)
    if len(v):
        o["rec"] = f"{w}-{len(v)-w}"
        o["edge_pp"] = round(o["mean_pct"] - 100 * base.mean(), 3)
        o["p_coin"] = round(sign_test(w, len(v)), 4)
        o["p_vs_uprate"] = round(sign_test(w, len(v), float((base > 0).mean())), 4)
    return o, e, v


crash = r1 <= -0.035
# 1. definition neighbours
print("TODAY: r1 %.4f  ratr %.2f  first-in-21d(3.5%%) %s  first-in-21d(ATR2) %s" % (
    r1.loc[BAR], ratr.loc[BAR], bool(first_in(crash).loc[BAR]),
    bool(first_in(ratr <= -2.0).loc[BAR])))
for h in (1, 3, 5):
    rows = []
    for lbl, m in (("r1<=-3.0%", r1 <= -0.03), ("r1<=-3.5%", crash), ("r1<=-4.0%", r1 <= -0.04),
                   ("ATR<=-1.5", ratr <= -1.5), ("ATR<=-2.0", ratr <= -2.0), ("ATR<=-2.5", ratr <= -2.5),
                   ("r1<=-3.5% & ATR<=-2.0 (both)", crash & (ratr <= -2.0))):
        rows.append(cell(m, h, label=f"{lbl} any")[0])
        rows.append(cell(first_in(m), h, label=f"{lbl} first-in-21d")[0])
    show(rows, f"1. definition neighbours, GLD h={h} lag 1")

# 2. regime / era splits on the parent and on the live-matched form
for h in (1, 3, 5):
    rows = []
    for lbl, m in (("parent", crash),
                   ("pre-2018", crash & (idx < "2018-01-01")),
                   ("2018+", crash & (idx >= "2018-01-01")),
                   ("ex 2025-26", crash & (idx < "2025-01-01")),
                   ("2025-26 only", crash & (idx >= "2025-01-01")),
                   ("TNX near hi", crash & tnx_near), ("TNX NOT near", crash & ~tnx_near),
                   ("DX near hi", crash & dx_near), ("DX NOT near", crash & ~dx_near),
                   ("DX & TNX near (the gate)", crash & dx_near & tnx_near),
                   ("GLD < 200d (live)", crash & (v200 < 0)), ("GLD >= 200d", crash & (v200 >= 0)),
                   (">=15% off hi", crash & (off_hi <= -0.15)),
                   ("<200d & >=15% off (live state)", crash & (v200 < 0) & (off_hi <= -0.15)),
                   ("<200d & 2018+", crash & (v200 < 0) & (idx >= "2018-01-01"))):
        rows.append(cell(m, h, label=lbl)[0])
    show(rows, f"2. regime splits, GLD h={h}")

# 3. concentration by VALUE on the traded side
for h in (1, 5):
    o, e, v = cell(crash, h)
    order = np.argsort(-v)
    tot = v.sum()
    yrs = pd.Series(v, index=e.year).groupby(level=0).sum().sort_values(ascending=False)
    print(f"\n3. h={h}: total {100*tot:+.2f}pp; best-2 {[str(e[i].date()) for i in order[:2]]} "
          f"= {100*v[order[:2]].sum():+.2f}pp ({100*v[order[:2]].sum()/tot:.0f}%); drop-best-2 mean "
          f"{100*np.delete(v, order[:2]).mean():+.3f}% (drift {100*fwd_lag(px['GLD'], h).mean():+.3f}%); "
          f"top year {yrs.index[0]} {100*yrs.iloc[0]:+.2f}pp ({100*yrs.iloc[0]/tot:.0f}%)")
    print(f"   {cluster_note(e, v)}")

# 4. GC=F vs GLD on the SAME (GLD-defined) dates: dates or close timing?
rows = []
for h in (1, 3, 5):
    for veh in ("GLD", "GC=F"):
        rows.append(cell(crash, h, veh, f"{veh} on GLD crash dates h={h}")[0])
show(rows, "4. vehicle on identical dates")

# 5. does the crash day add anything over the drawdown state it sits in?
for h in (3, 5):
    ret = fwd_lag(px["GLD"], h, 1)
    parent = (off_hi <= -0.15) & (v200 < 0)
    pe = declusters(idx[parent.fillna(False).values & ret.notna().values], h, idx)
    child = parent & crash
    pm = pd.Series(False, index=idx)
    pm.loc[pe] = True
    fr = filter_vs_reanchor(ret, pm, child, idx, window_td=21,
                            label=f"drawdown state (>=15% off hi & <200d, declustered) -> + crash day, h={h}")
    if fr["n_matched"]:
        rn = reanchor_null(ret, [a for a, _, _ in fr["pairs"]], fr["shifts"], idx,
                           np.nanmean(ret.reindex([b for _, b, _ in fr["pairs"]]).values))
        print(f"  reanchor_null p={rn['p']:.3f}  null mean {rn['null_mean_pct']:+.3f}%  "
              f"child {rn['child_mean_pct']:+.3f}%")
    o, _, _ = cell(parent, h, label="drawdown state alone (declustered)")
    print(f"  drawdown state alone: n={o['n']} mean {o['mean_pct']:+.3f}% rec {o['rec']}")
