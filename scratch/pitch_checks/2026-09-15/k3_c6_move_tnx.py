import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# C6 round 1: bond vol AND yields thrusting together into an FOMC decision, long duration.
pd.set_option("future.no_silent_downcasting", True)
TK = ["SPY", "^MOVE", "^TNX", "TLT", "IEF", "^VIX"]
raw = close_panel(TK)
for t in TK:
    s = raw[t].dropna()
    print(t, "first", s.index[0].date(), "last", s.index[-1].date(), "n", len(s))
cal = raw["TLT"].dropna().index
move_r5 = pct_rank(raw["^MOVE"], 5).reindex(cal).ffill(limit=1)
tnx = raw["^TNX"].dropna()
tnx_r5 = pct_rank(raw["^TNX"], 5).reindex(cal).ffill(limit=1)
tnx_hi = tnx.rolling(252).max().reindex(cal).ffill(limit=1)
tnx_c = tnx.reindex(cal).ffill(limit=1)
vix = raw["^VIX"].reindex(cal).ffill(limit=1)
vix1 = vix / vix.shift(1) - 1
spy = raw["SPY"].reindex(cal)
spy200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
px = raw.reindex(cal)
live = pd.Timestamp("2026-09-14")
print(f"LIVE {live.date()}: MOVE r5 {move_r5.loc[live]:.1f}  TNX r5 {tnx_r5.loc[live]:.1f}  TNX {tnx_c.loc[live]:.3f} "
      f"252max {tnx_hi.loc[live]:.3f} ({100*(tnx_c.loc[live]/tnx_hi.loc[live]-1):+.2f}%)  VIX 1d {100*vix1.loc[live]:+.2f}%")
mv = raw["^MOVE"].dropna()
print(f"^MOVE valid first {mv.index[0].date()}, MOVE r5 first valid {move_r5.dropna().index[0].date()}")

fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
fpos, fkept = anchor_positions(cal, fomc, 0)
fpos = np.array(fpos)
fkept = pd.DatetimeIndex(fkept)
N = len(cal)


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


def cells_at(shift, lag=1):
    sp = fpos - 2 + shift
    ok = (sp >= 0) & (sp < N)
    sp = sp[ok]
    d = cal[sp]
    fd = fkept[ok]
    M = move_r5.reindex(d).values
    T = tnx_r5.reindex(d).values
    valid = ~np.isnan(M) & ~np.isnan(T)
    return sp, d, fd, M, T, valid


for V in ["TLT", "IEF"]:
    for h in (1, 2, 3):
        ret = fwd_lag(px[V], h, 1)
        sp, d, fd, M, T, valid = cells_at(0)
        vals = ret.reindex(d).values
        vals = np.where(valid, vals, np.nan)
        post = d >= pd.Timestamp("2018-01-01")
        mid = np.array([x.year % 4 == 2 for x in fd])
        vup = vix1.reindex(d).values > 0
        above = (spy.reindex(d) > spy200.reindex(d)).values
        nearmax = (tnx_c.reindex(d) >= 0.99 * tnx_hi.reindex(d)).values
        J = (M >= 90) & (T >= 90)
        masks = {
            "PARENT all FOMC k=-2 (MOVE era)": valid,
            "JOINT MOVE r5>=90 & TNX r5>=90": J,
            "complement of JOINT": valid & ~J,
            "MOVE>=90 only (TNX<90)": (M >= 90) & (T < 90),
            "TNX>=90 only (MOVE<90)": (T >= 90) & (M < 90),
            "rung 80/80": (M >= 80) & (T >= 80),
            "rung 85/85": (M >= 85) & (T >= 85),
            "rung 95/95": (M >= 95) & (T >= 95),
            "rung MOVE>=90 & TNX>=95": (M >= 90) & (T >= 95),
            "JOINT & TNX within 1% of 252max": J & nearmax,
            "JOINT & NOT near max": J & ~nearmax,
            "JOINT & VIX up k=-2 (LIVE)": J & vup,
            "JOINT & VIX down k=-2": J & ~vup,
            "JOINT midterm": J & mid,
            "JOINT non-midterm": J & ~mid,
            "JOINT pre-2018": J & ~post,
            "JOINT 2018+": J & post,
            "JOINT SPY>200d": J & above,
            "JOINT SPY<200d": J & ~above,
        }
        rows = [rec(vals[m], lbl) for lbl, m in masks.items()]
        allv = ret[move_r5.notna() & tnx_r5.notna()].dropna().values
        rows.append(summarize(allv, "CTRL all days (MOVE era)"))
        show(rows, f"{V} long, signal k=-2, entry MOC k=-1, h={h}")

        # event removed: same joint state on non-FOMC days
        near = np.zeros(N, bool)
        for p in fpos:
            near[max(0, p - 6):min(N, p + 2)] = True
        js = ((move_r5 >= 90) & (tnx_r5 >= 90)).values & ret.notna().values
        nonf = declusters(cal[js & ~near], max(h, 3), cal)
        jd = d[J]
        loc = local_control(cal[ret.notna().values & move_r5.notna().values], jd)
        show([rec(ret.loc[nonf].values, "JOINT state on NON-FOMC days (declustered)"),
              rec(ret.loc[loc].values, "local +/-126td ex-trigger")], f"{V} h={h} gate-without-event / local")

        # k=-1 conditioner read, lag 0 contrast
        sp1 = fpos - 1
        d1 = cal[sp1]
        J1 = ((move_r5.reindex(d1) >= 90) & (tnx_r5.reindex(d1) >= 90)).values
        r0 = fwd_ret(px[V], h).reindex(d1).values
        show([rec(r0[J1], "JOINT read at k=-1, entry same close (lag 0, not tradeable)")], f"{V} h={h} k=-1 read")

    # placebo ladder h=1..3 on JOINT 90 and 80
    for h in (1, 2, 3):
        ret = fwd_lag(px[V], h, 1)
        rows = []
        for shift in range(-5, 6):
            sp, d, fd, M, T, valid = cells_at(shift)
            vals = ret.reindex(d).values
            row = {"k_signal": -2 + shift}
            for rung in (90, 80):
                m = (M >= rung) & (T >= rung)
                x = vals[m]
                x = x[~np.isnan(x)]
                row[f"n{rung}"] = len(x)
                row[f"mean{rung}"] = round(100 * x.mean(), 3) if len(x) else np.nan
                row[f"hit{rung}"] = round(100 * (x > 0).mean(), 0) if len(x) else np.nan
            rows.append(row)
        lad = pd.DataFrame(rows)
        print(f"\n{V} h={h} PLACEBO LADDER (JOINT at signal, lag 1):")
        print(lad.to_string(index=False))
        for c in ["mean90", "mean80"]:
            t0 = lad.loc[lad.k_signal == -2, c].values[0]
            print(f"  {c}: true k=-2 ranks {int((lad[c] > t0).sum()) + 1} of {lad[c].notna().sum()}")

# episode list
ret3 = fwd_lag(px["TLT"], 3, 1)
ret1 = fwd_lag(px["TLT"], 1, 1)
dy1 = (tnx_c.shift(-2) - tnx_c.shift(-1)) * 100
dy3 = (tnx_c.shift(-4) - tnx_c.shift(-1)) * 100
sp, d, fd, M, T, valid = cells_at(0)
J = (M >= 85) & (T >= 85)
print("\nJOINT(85) episodes: signal, FOMC, MOVE r5, TNX r5, TLT h1, TLT h3, dTNX bp h1, h3, VIX1d, near252max, mid")
for x, f in zip(d[J], fd[J]):
    print(f"  {x.date()} FOMC {f.date()} M {move_r5.loc[x]:.0f} T {tnx_r5.loc[x]:.0f}  TLT {100*ret1.loc[x]:+.2f}% {100*ret3.loc[x]:+.2f}%  "
          f"dTNX {dy1.loc[x]:+.1f} {dy3.loc[x]:+.1f}bp  VIX {100*vix1.loc[x]:+.1f}%  nearmax {bool(tnx_c.loc[x] >= 0.99*tnx_hi.loc[x])}  mid {f.year%4==2}")
print("\ncost: TLT/IEF ~2.5 bp/side = 5 bp round trip; 5x bar = +0.25%")
