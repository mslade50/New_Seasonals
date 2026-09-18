"""kC c7 round 2 on the kill axis: is the live slice (midterm year, SPY above
its 200d) wrong-signed across DEFINITION NEIGHBOURS and horizons, or is the
round-1 split one lucky cut? Also the two slices that could rescue it: two or
more SPDRs flooring together above the 200d (today's shape) and ^TNX at/near a
252 max above the 200d. Date-clustered (episodes within 10 td merged) so the
cross-name correlation does not inflate the record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SP = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
raw = load_prices(SP + ["SPY", "^TNX"])
C = {t: raw[t]["Close"].dropna() for t in raw}
spy = C["SPY"]
cal = spy.index
spy200 = spy / spy.rolling(200).mean() - 1.0
tnx = C["^TNX"]
tmax = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_near = (tnx >= 0.97 * tmax).reindex(cal).fillna(False).astype(bool)
RK = {t: {w: pct_rank(C[t], w) for w in (5, 10, 21, 63)} for t in SP}


def eps(h, k, wins, names=SP):
    rows = []
    for t in names:
        s = C[t]
        r = s.shift(-(1 + h)) / s.shift(-1) - 1.0
        valid = r.dropna().index
        m = None
        for w in wins:
            x = RK[t][w] <= k
            m = x if m is None else (m & x)
        trig = s.index[m.fillna(False).values].intersection(valid)
        if len(trig) == 0:
            continue
        drift = r.loc[valid].mean()
        for d in declusters(trig, max(h, 10), valid):
            rows.append((t, d, r.loc[d], r.loc[d] - drift))
    df = pd.DataFrame(rows, columns=["tkr", "date", "ret", "exc"])
    df["above"] = spy200.reindex(df.date).values > 0
    df["mid"] = df.date.dt.year % 4 == 2
    df["tnx"] = tnx_near.reindex(df.date).values
    df["nday"] = df.groupby("date").tkr.transform("count")
    return df


def clus(df, gap=10):
    pos = pd.Series(range(len(cal)), index=cal)
    d = df.sort_values("date")
    cid, last, c = [], -10**9, -1
    for q in d.date.map(pos).values:
        if q - last >= gap:
            c += 1
        cid.append(c)
        last = q
    return d.assign(cid=cid).groupby("cid").agg(date=("date", "first"), ret=("ret", "mean"), exc=("exc", "mean"))


def fmt(df):
    if len(df) < 2:
        return f"eps {len(df):3d}"
    g = clus(df)
    w = int((g.exc > 0).sum())
    return (f"eps {len(df):3d} exc {100*df.exc.mean():+.3f}pp | clus {len(g):3d} exc {100*g.exc.mean():+.3f}pp "
            f"raw {100*g.ret.mean():+.3f}% exc>0 {w}-{len(g)-w} p {sign_test(w, len(g)):.3f}")


print("=== neighbours: live slice (midterm & above200) vs its complement ===")
for h in (5, 10):
    for k in (5, 10, 15):
        for wins in ((5, 21, 63), (10, 21, 63), (21, 63)):
            E = eps(h, k, wins)
            live = E[E.mid & E.above]
            other = E[~(E.mid & E.above)]
            ab = E[E.above]
            print(f"h={h:2d} k={k:2d} {str(wins):12s} LIVE[mid&above] {fmt(live)}")
            print(f"{'':27s} above200 all   {fmt(ab)}")
            print(f"{'':27s} complement     {fmt(other)}")

print("\n=== rescue slices, h=10 k=10 5/21/63 ===")
E = eps(10, 10, (5, 21, 63))
for lbl, m in [(">=2 same day & above200", (E.nday >= 2) & E.above),
               (">=2 same day & mid & above200", (E.nday >= 2) & E.mid & E.above),
               ("TNX near max & above200", E.tnx & E.above),
               ("TNX near max & mid & above200", E.tnx & E.mid & E.above),
               ("TNX near max & above200 ex-2006", E.tnx & E.above & (E.date.dt.year != 2006)),
               ("above200 ex-XLU", E.above & (E.tkr != "XLU")),
               ("mid & above200 ex-XLU", E.mid & E.above & (E.tkr != "XLU")),
               ("XLU+XLI only, mid & above200", E.mid & E.above & E.tkr.isin(["XLU", "XLI"]))]:
    print(f"  {lbl:34s} {fmt(E[m])}")
x = E[E.tnx & E.above].sort_values("date")
print("\n  TNX near max & above200 episodes:")
print(x.assign(ret=(100 * x.ret).round(2), exc=(100 * x.exc).round(2))[["tkr", "date", "ret", "exc", "mid"]].to_string(index=False))

print("\n=== gate attribution: does the 5d leg filter or re-anchor vs the 21/63 double floor? (pooled, h=10) ===")
agg = {"del": [], "kp": [], "kc": [], "par": []}
for t in SP:
    s = C[t]
    r = s.shift(-11) / s.shift(-1) - 1.0
    valid = r.dropna().index
    par = ((RK[t][21] <= 10) & (RK[t][63] <= 10)).fillna(False)
    chi = (par & (RK[t][5] <= 10)).fillna(False)
    pd_ = declusters(s.index[par.values].intersection(valid), 10, valid)
    cd_ = declusters(s.index[chi.values].intersection(valid), 10, valid)
    pm = pd.Series(False, index=valid); pm.loc[pd_] = True
    cm = pd.Series(False, index=valid); cm.loc[cd_] = True
    o = filter_vs_reanchor(r.loc[valid], pm, cm, valid, window_td=21)
    agg["par"] += list(r.reindex(pd_).values)
    agg["del"] += list(r.reindex(pd.DatetimeIndex(o["deleted_dates"])).values)
    agg["kp"] += [r.loc[a] for a, _, _ in o["pairs"]]
    agg["kc"] += [r.loc[b] for _, b, _ in o["pairs"]]
m = {k: 100 * np.nanmean(v) for k, v in agg.items()}
print(f"  parent all {m['par']:+.3f}% (n {len(agg['par'])}) deleted {m['del']:+.3f}% (n {len(agg['del'])}) "
      f"kept@parent {m['kp']:+.3f}% kept@child {m['kc']:+.3f}% (n {len(agg['kc'])})")
print(f"  FILTERING {m['kp']-m['par']:+.3f}pp  RE-ANCHORING {m['kc']-m['kp']:+.3f}pp")
