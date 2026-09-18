import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# C8 round 1: short gold into an FOMC after a yield-thrust week (sign flip of the 09-14 kill).
pd.set_option("future.no_silent_downcasting", True)
TK = ["GLD", "GC=F", "^TNX", "DX-Y.NYB", "SPY"]
raw = close_panel(TK)
for t in TK:
    s = raw[t].dropna()
    print(t, "first", s.index[0].date(), "last", s.index[-1].date(), "n", len(s))
tnx = raw["^TNX"].dropna()
tr5 = pct_rank(raw["^TNX"], 5)
thi = rolling_on_valid(raw["^TNX"], lambda x: x.rolling(252).max())
live = pd.Timestamp("2026-09-14")
print(f"LIVE: TNX {tnx.loc[live]:.3f} r5 {tr5.loc[live]:.1f} 252max {thi.loc[live]:.3f} gap {100*(tnx.loc[live]/thi.loc[live]-1):+.2f}%")

fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


for V in ["GLD", "GC=F"]:
    cal = raw[V].dropna().index
    px = raw[[V, "DX-Y.NYB", "SPY"]].reindex(cal)
    R5 = tr5.reindex(cal).ffill(limit=1)
    TC = raw["^TNX"].reindex(cal).ffill(limit=1)
    HI = thi.reindex(cal).ffill(limit=1)
    DX = raw["DX-Y.NYB"].reindex(cal).ffill(limit=1)
    STATE = (R5 >= 95) & (TC >= 0.99 * HI)
    N = len(cal)
    fpos, fk = anchor_positions(cal, fomc, 0)
    fpos = np.array(fpos)
    fk = pd.DatetimeIndex(fk)
    g = px[V]
    g50 = rolling_on_valid(g, lambda x: x.rolling(50).mean())
    gdd = g / rolling_on_valid(g, lambda x: x.rolling(252).max()) - 1
    print(f"\n==================== {V} ====================")
    # residual model on all days per h: GLD ret ~ DX ret + dTNX(bp) over the same hold
    for h in (1, 2, 3, 5):
        ret = -fwd_lag(g, h, 1)  # SHORT gold
        dxr = fwd_lag(DX, h, 1)
        dy = (TC.shift(-(1 + h)) - TC.shift(-1)) * 100
        ok = ret.notna() & dxr.notna() & dy.notna()
        X = np.column_stack([np.ones(ok.sum()), dxr[ok].values, dy[ok].values])
        coef = np.linalg.lstsq(X, ret[ok].values, rcond=None)[0]
        resid = ret - (coef[0] + coef[1] * dxr + coef[2] * dy)
        expl = coef[1] * dxr + coef[2] * dy

        sp = fpos - 2
        ok2 = (sp >= 0) & (sp < N)
        sp = sp[ok2]
        d = cal[sp]
        fd = fk[ok2]
        st = STATE.reindex(d).fillna(False).values.astype(bool)
        vals = ret.reindex(d).values
        post = d >= pd.Timestamp("2018-01-01")
        mid = np.array([x.year % 4 == 2 for x in fd])
        dyv = dy.reindex(d).values
        gdown = (g.reindex(d) < g50.reindex(d)).values
        deep = (gdd.reindex(d) <= -0.10).values
        masks = {
            "PARENT all FOMC k=-2": np.ones(len(d), bool),
            "CELL thrust r5>=95 & within1% max": st,
            "complement": ~st,
            "CELL & TNX ROSE over hold": st & (dyv > 0),
            "CELL & TNX FELL over hold": st & (dyv <= 0),
            "CELL midterm": st & mid,
            "CELL non-midterm": st & ~mid,
            "CELL pre-2018": st & ~post,
            "CELL 2018+": st & post,
            "CELL & GLD<50d": st & gdown,
            "CELL & GLD>=50d": st & ~gdown,
            "CELL & GLD >=10% off 252 high (LIVE)": st & deep,
            "PARENT & GLD >=10% off high (drift ctl)": deep,
        }
        rows = []
        for lbl, m in masks.items():
            r = rec(vals[m], lbl)
            if r["n"]:
                r["resid_pct"] = round(100 * np.nanmean(resid.reindex(d).values[m]), 3)
                r["expl_pct"] = round(100 * np.nanmean(expl.reindex(d).values[m]), 3)
                r["dTNX_bp"] = round(float(np.nanmean(dyv[m])), 1)
            rows.append(r)
        rows.append(summarize(ret.dropna().values, "CTRL all days (short)"))
        show(rows, f"{V} SHORT h={h} signal k=-2 lag1 (resid/expl from ret~DX+dTNX, betas {coef[1]:+.3f}/DX, {coef[2]*100:+.4f}%/bp)")

        # event removed: state on non-FOMC days; local control
        near = np.zeros(N, bool)
        for p in fpos:
            near[max(0, p - 7):min(N, p + 2)] = True
        sm = STATE.fillna(False).values.astype(bool) & ret.notna().values
        nonf = declusters(cal[sm & ~near], max(h, 5), cal)
        loc = local_control(cal[ret.notna().values], d[st])
        dd_nonf = declusters(cal[sm & ~near & (gdd <= -0.10).fillna(False).values], max(h, 5), cal)
        show([rec(ret.loc[nonf].values, "STATE on NON-FOMC days (declustered)"),
              rec(ret.loc[dd_nonf].values, "STATE non-FOMC & GLD>=10% off high"),
              rec(ret.loc[loc].values, "local +/-126td ex-cell")], f"{V} h={h} gate-without-event / local")
        if h in (1, 3, 5):
            print("  cell episodes:", [(str(x.date()), round(100 * ret.loc[x], 2), round(float(dy.loc[x]), 1)) for x in d[st]])

    # placebo ladder, SHORT, h=1..5
    rows = []
    for shift in range(-5, 6):
        row = {"k_signal": -2 + shift}
        for h in (1, 3, 5):
            ret = -fwd_lag(g, h, 1)
            sp = fpos - 2 + shift
            sp = sp[(sp >= 0) & (sp < N)]
            dd = cal[sp]
            m = STATE.reindex(dd).fillna(False).values.astype(bool)
            x = ret.reindex(dd).values[m]
            x = x[~np.isnan(x)]
            row[f"n{h}"] = len(x)
            row[f"short_h{h}"] = round(100 * x.mean(), 3) if len(x) else np.nan
        rows.append(row)
    lad = pd.DataFrame(rows)
    print(f"\n{V} PLACEBO LADDER short, state at signal:")
    print(lad.to_string(index=False))
    for h in (1, 3, 5):
        c = f"short_h{h}"
        t0 = lad.loc[lad.k_signal == -2, c].values[0]
        print(f"  {c}: true k=-2 ranks {int((lad[c] > t0).sum()) + 1} of {lad[c].notna().sum()}")
print("\ncost: GLD ~2.5 bp/side = 5 bp RT (+ short borrow negligible over days); 5x bar = +0.25%")
