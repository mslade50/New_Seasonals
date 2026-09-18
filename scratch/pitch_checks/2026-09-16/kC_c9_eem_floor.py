"""kC c9 round 1: long EEM at a 63-day rank floor (<= 2) inside a year still
up > 20% (252-session return), h=5 and h=10, lag=1, declustered gap = h.

Reference class: every country/region ETF in master_prices (EFA EWJ EWZ FXI
EEM EWY EWT INDA EWW RSX KWEB VGK). Also re-reads the 2026-09-15 23-ETF family
definition (r63 <= 5 AND 252d >= +40%) for EEM and says whether today's EEM
state is in it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)
REF = ["EEM", "EFA", "EWJ", "EWZ", "FXI", "EWY", "EWT", "INDA", "EWW", "RSX", "KWEB", "VGK"]
raw = load_prices(REF + ["SPY", "DX-Y.NYB"])
C = {t: raw[t]["Close"].dropna() for t in raw}
spy = C["SPY"]
cal = spy.index
spy200 = spy / spy.rolling(200).mean() - 1.0
dx21 = pct_rank(C["DX-Y.NYB"], 21)


def fwd(s, h, lag=1):
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def mask(s, k63=2, yr=0.20, r5max=None):
    r63 = pct_rank(s, 63)
    y = s / s.shift(252) - 1.0
    m = (r63 <= k63) & (y > yr)
    if r5max is not None:
        m &= pct_rank(s, 5) < r5max
    return m.fillna(False)


def eps(t, h, **kw):
    s = C[t]
    r = fwd(s, h)
    valid = r.dropna().index
    trig = s.index[mask(s, **kw).values].intersection(valid)
    if len(trig) == 0:
        return pd.DataFrame(), r, valid
    e = declusters(trig, h, valid)
    drift = r.loc[valid].mean()
    df = pd.DataFrame({"tkr": t, "date": e, "ret": r.loc[e].values,
                       "exc": r.loc[e].values - drift})
    df["above200"] = spy200.reindex(df.date).values > 0
    df["midterm"] = df.date.dt.year % 4 == 2
    df["post2018"] = df.date >= "2018-01-01"
    df["r5"] = pct_rank(s, 5).reindex(df.date).values
    df["dx21"] = dx21.reindex(df.date).values
    return df, r, valid


def line(df, label):
    if len(df) == 0:
        print(f"  {label:40s} N=0")
        return
    v = df.ret.values
    w = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
    print(f"  {label:40s} N={len(v):3d} mean {100*v.mean():+.3f}% exc {100*df.exc.mean():+.3f}pp "
          f"hit {100*w/len(v):5.1f}% t {t:+.2f} sign p {sign_test(w, len(v)):.4f} worst {100*v.min():+.2f}%")


s = C["EEM"]
print(f"live EEM {s.index[-1].date()}: r63 {pct_rank(s,63).iloc[-1]:.2f} r5 {pct_rank(s,5).iloc[-1]:.2f} "
      f"r21 {pct_rank(s,21).iloc[-1]:.2f} 252d {100*(s.iloc[-1]/s.iloc[-253]-1):+.2f}%  SPY vs 200d {100*spy200.iloc[-1]:+.2f}%")
print(f"  in 09-15 family base (r63<=5 & 252d>=40%)? {bool(pct_rank(s,63).iloc[-1] <= 5 and s.iloc[-1]/s.iloc[-253]-1 >= 0.40)}")
print("  live members of the c9 mask today:",
      [t for t in REF if len(C[t]) > 300 and C[t].index[-1] == s.index[-1] and bool(mask(C[t]).iloc[-1])])

for H in (5, 10):
    print(f"\n######## h={H} ########")
    E, r, valid = eps("EEM", H)
    loc = local_control(valid, E.date) if len(E) else valid
    print(f"  EEM own drift all days {100*r.loc[valid].mean():+.3f}%  local +/-126 {100*r.loc[loc].mean():+.3f}%")
    line(E, "EEM c9 cell")
    if len(E):
        print("  EEM episodes:", ", ".join(f"{d.date()} {100*x:+.2f}" for d, x in zip(E.date, E.ret)))
        for lbl, m in [("pre-2018", ~E.post2018), ("2018+", E.post2018), ("above200", E.above200),
                       ("below200", ~E.above200), ("midterm", E.midterm), ("r5<15 (live)", E.r5 < 15),
                       ("r5>=15", E.r5 >= 15), ("DX r21<60 (live)", E.dx21 < 60)]:
            line(E[m], "  " + lbl)
    # reference class
    allE, rows = [], []
    for t in REF:
        if t not in C or len(C[t]) < 600:
            continue
        e, rt, vt = eps(t, H)
        if len(e) == 0:
            rows.append({"tkr": t, "n": 0})
            continue
        allE.append(e)
        se = np.sqrt(e.ret.var(ddof=1) / len(e) + rt.loc[vt].var(ddof=1) / len(vt)) if len(e) > 1 else np.nan
        rows.append({"tkr": t, "n": len(e), "mean_pct": 100 * e.ret.mean(), "exc_pct": 100 * e.exc.mean(),
                     "se_pct": 100 * se, "t": e.exc.mean() / se if se and se > 0 else np.nan,
                     "hit": 100 * (e.ret > 0).mean(), "first": str(e.date.min().date())})
    d = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(d.round(3).to_string(index=False))
    dd = d.dropna(subset=["se_pct"])
    dd = dd[dd.n >= 3]
    w = 1 / dd.se_pct ** 2
    mu = (w * dd.exc_pct).sum() / w.sum()
    Q = float((w * (dd.exc_pct - mu) ** 2).sum())
    k = len(dd)
    print(f"  FE common {mu:+.3f}pp (t {mu*np.sqrt(w.sum()):+.2f}) Q {Q:.2f}/{k-1} p {1-sps.chi2.cdf(Q,k-1):.3f}; "
          f"EEM rank {list(d.tkr).index('EEM')+1}/{len(d)}")
    P = pd.concat(allE)
    line(P, "POOLED country class")
    for lbl, m in [("pooled above200", P.above200), ("pooled below200", ~P.above200),
                   ("pooled midterm", P.midterm), ("pooled midterm&above200", P.midterm & P.above200),
                   ("pooled pre-2018", ~P.post2018), ("pooled 2018+", P.post2018),
                   ("pooled r5<15", P.r5 < 15), ("pooled r5>=15", P.r5 >= 15),
                   ("pooled above200 & r5<15", P.above200 & (P.r5 < 15))]:
        line(P[m], lbl)
    # neighbours on EEM and on the pool
    print("  definition neighbours (EEM | pooled class):")
    for k63 in (2, 5, 10):
        for yr in (0.0, 0.10, 0.20, 0.30):
            e, _, _ = eps("EEM", H, k63=k63, yr=yr)
            pe = pd.concat([eps(t, H, k63=k63, yr=yr)[0] for t in REF if t in C and len(C[t]) >= 600])
            ev = e.ret.values if len(e) else np.array([np.nan])
            print(f"    r63<={k63:2d} yr>{int(100*yr):2d}%  EEM N={len(e):3d} mean {100*np.nanmean(ev):+.3f}% "
                  f"exc {100*e.exc.mean() if len(e) else np.nan:+.3f}pp | pool N={len(pe):3d} exc {100*pe.exc.mean():+.3f}pp "
                  f"hit {100*(pe.ret>0).mean():.1f}%")
    # does the year gate filter? EEM r63<=2 any year: kept (yr>20%) vs deleted
    e0, _, _ = eps("EEM", H, k63=2, yr=-9)
    if len(e0):
        y = (C["EEM"] / C["EEM"].shift(252) - 1).reindex(e0.date).values
        print(f"  year-gate on EEM r63<=2 episodes: kept(yr>20%) N={int((y>0.2).sum())} mean {100*e0.ret[y>0.2].mean():+.3f}% | "
              f"deleted N={int((y<=0.2).sum())} mean {100*e0.ret[y<=0.2].mean():+.3f}%")

# 09-15 family definition on EEM for the record
for H in (10,):
    s = C["EEM"]
    r = fwd(s, H)
    valid = r.dropna().index
    y = s / s.shift(252) - 1
    B = ((pct_rank(s, 63) <= 5) & (y >= 0.40)).fillna(False)
    tr = s.index[B.values].intersection(valid)
    e = declusters(tr, 10, valid)
    print(f"\n09-15 family base B on EEM h=10: N={len(e)} vals {[round(100*x,2) for x in r.loc[e].values]} "
          f"dates {[str(d.date()) for d in e]}")
