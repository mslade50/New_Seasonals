"""kC c9 round 2 on the two open doors from round 1: (a) the pooled country
class in midterm & above-200d years paid +0.778pp at h=10 on 64 episodes, a
slice searched this morning and inflated by same-date overlap across 12
correlated country ETFs; date-cluster it, drop the best year, and read EEM's
own slice. (b) today's three live members (EEM, EWY, EWT) as a basket.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

REF = ["EEM", "EFA", "EWJ", "EWZ", "FXI", "EWY", "EWT", "INDA", "EWW", "RSX", "KWEB", "VGK"]
raw = load_prices(REF + ["SPY"])
C = {t: raw[t]["Close"].dropna() for t in raw}
spy = C["SPY"]
cal = spy.index
spy200 = spy / spy.rolling(200).mean() - 1.0


def eps(h, k63=2, yr=0.20, names=REF):
    rows = []
    for t in names:
        s = C[t]
        r = s.shift(-(1 + h)) / s.shift(-1) - 1.0
        valid = r.dropna().index
        m = ((pct_rank(s, 63) <= k63) & (s / s.shift(252) - 1 > yr)).fillna(False)
        trig = s.index[m.values].intersection(valid)
        if len(trig) == 0:
            continue
        drift = r.loc[valid].mean()
        r5 = pct_rank(s, 5)
        for d in declusters(trig, h, valid):
            rows.append((t, d, r.loc[d], r.loc[d] - drift, r5.loc[d]))
    df = pd.DataFrame(rows, columns=["tkr", "date", "ret", "exc", "r5"])
    df["above"] = spy200.reindex(df.date).values > 0
    df["mid"] = df.date.dt.year % 4 == 2
    return df


def clus(df, gap):
    pos = pd.Series(range(len(cal)), index=cal)
    d = df.sort_values("date")
    cid, last, c = [], -10**9, -1
    for q in d.date.map(lambda x: pos.get(x, np.nan)).values:
        if q - last >= gap:
            c += 1
        cid.append(c)
        last = q
    return d.assign(cid=cid).groupby("cid").agg(date=("date", "first"), ret=("ret", "mean"),
                                                exc=("exc", "mean"), n=("tkr", "count"))


def fmt(df, gap):
    if len(df) < 2:
        return f"eps {len(df)}"
    g = clus(df, gap)
    w = int((g.exc > 0).sum())
    yr = g.groupby(g.date.dt.year).exc.sum().sort_values(ascending=False)
    ex_best = g[g.date.dt.year != yr.index[0]]
    return (f"eps {len(df):3d} exc {100*df.exc.mean():+.3f}pp | clusters {len(g):3d} exc {100*g.exc.mean():+.3f}pp "
            f"exc>0 {w}-{len(g)-w} p {sign_test(w, len(g)):.3f} | drop best yr {yr.index[0]}: {100*ex_best.exc.mean():+.3f}pp "
            f"(n {len(ex_best)})")


for h in (5, 10):
    E = eps(h)
    print(f"\n######## h={h} ########")
    for lbl, m in [("pooled all", E.index == E.index), ("pooled above200", E.above),
                   ("pooled mid & above200", E.mid & E.above), ("pooled nonmid & above200", ~E.mid & E.above),
                   ("pooled mid & above200 & r5<15", E.mid & E.above & (E.r5 < 15)),
                   ("pooled above200 & r5<15", E.above & (E.r5 < 15)),
                   ("EEM mid & above200", E.mid & E.above & (E.tkr == "EEM")),
                   ("EEM above200", E.above & (E.tkr == "EEM")),
                   ("EEM above200 & r5<15", E.above & (E.r5 < 15) & (E.tkr == "EEM"))]:
        print(f"  {lbl:32s} {fmt(E[m], h)}")
    g = clus(E[E.mid & E.above], h)
    print("  pooled mid&above200 clusters by year (sum of exc pp, n):",
          {int(y): (round(100 * v.exc.sum(), 2), len(v)) for y, v in g.groupby(g.date.dt.year)})
    x = E[(E.tkr == "EEM") & E.above].sort_values("date")
    print("  EEM above200 episodes:", ", ".join(f"{d.date()} {100*v:+.2f}" for d, v in zip(x.date, x.ret)))

# live basket: EEM + EWY + EWT equal weight on dates where >= 2 of the three are in the mask
print("\n=== live-shape basket: >=2 of {EEM, EWY, EWT} in the mask same day, EW of those in it, h=10 ===")
h = 10
px = pd.DataFrame({t: C[t] for t in ["EEM", "EWY", "EWT"]}).dropna()
M = pd.DataFrame({t: ((pct_rank(px[t], 63) <= 2) & (px[t] / px[t].shift(252) - 1 > 0.20)) for t in px}).fillna(False)
F = px.shift(-(1 + h)) / px.shift(-1) - 1.0
valid = F.dropna().index
trig = px.index[(M.sum(axis=1) >= 2).values].intersection(valid)
ep = declusters(trig, h, valid)
vals = np.array([F.loc[d, [t for t in px if M.loc[d, t]]].mean() for d in ep])
drift = F.loc[valid].mean(axis=1).mean()
w = int((vals > 0).sum())
print(f"  N={len(vals)} mean {100*vals.mean():+.3f}% vs EW drift {100*drift:+.3f}%  rec {w}-{len(vals)-w} sign p {sign_test(w, len(vals)):.3f}")
print("  ", ", ".join(f"{d.date()} {100*v:+.2f} {'A' if spy200.get(d, np.nan) > 0 else 'B'}" for d, v in zip(ep, vals)))
