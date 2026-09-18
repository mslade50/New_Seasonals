import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
TK = ["SPY", "^VIX", "^VIX3M", "SVXY", "UVXY"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
px["^VIX"] = px["^VIX"].ffill(limit=2)
for t in TK:
    s = px[t].dropna()
    print(t, "first", s.index[0].date(), "last", s.index[-1].date(), "n", len(s))

vix = px["^VIX"]
v1 = vix / vix.shift(1) - 1
v2 = vix / vix.shift(2) - 1
live = pd.Timestamp("2026-09-14")
print(f"LIVE {live.date()}: VIX {vix.loc[live]:.2f} 1d {100*v1.loc[live]:+.2f}% 2d {100*v2.loc[live]:+.2f}% "
      f"prev-day 1d {100*v1.shift(1).loc[live]:+.2f}%")

ev = load_events(["fomc_decision", "vix_expiry"])
fomc = pd.DatetimeIndex(ev[ev.event == "fomc_decision"]["date"])
vxe = set(pd.DatetimeIndex(ev[ev.event == "vix_expiry"]["date"]))
fpos, fkept = anchor_positions(cal, fomc, 0)
fpos = np.array(fpos)
coll_flag = np.array([d in vxe for d in fkept])
print(f"FOMC decisions in calendar: {len(fpos)}; collisions with VIX expiry: {coll_flag.sum()}")

ERA = pd.Timestamp("2018-03-01")
N = len(cal)


def rets(h):
    svxy = fwd_lag(px["SVXY"], h, 1)
    spy = fwd_lag(px["SPY"], h, 1)
    svix = -fwd_lag(vix, h, 1)
    ok = svxy.notna() & spy.notna() & (cal >= ERA)
    b = np.polyfit(spy[ok].values, svxy[ok].values, 1)[0]
    alpha = svxy - b * spy
    return {"SVXY": svxy, "SPY": spy, "shortVIX": svix, "alpha": alpha, "beta": b}


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


def anchors(shift):
    """signal positions at decision + (-2 + shift), with metadata."""
    sp = fpos - 2 + shift
    ok = (sp >= 2) & (sp < N)
    return sp[ok], pd.DatetimeIndex(fkept)[ok], coll_flag[ok]


def pick(series, sp, era=None):
    d = cal[sp]
    v = series.reindex(d).values
    if era == "post":
        v = np.where(d >= ERA, v, np.nan)
    elif era == "pre":
        v = np.where(d < ERA, v, np.nan)
    return v


for h in (1, 2):
    R = rets(h)
    sp, fd, cf = anchors(0)
    d = cal[sp]
    st = v1.reindex(d).values
    st2 = v2.reindex(d).values
    stprev = v1.shift(1).reindex(d).values
    mid = np.array([x.year % 4 == 2 for x in fd])
    masks = {
        "PARENT all FOMC k=-2": np.ones(len(sp), bool),
        "STATE VIX 1d>=5%": st >= 0.05,
        "complement VIX 1d<5%": st < 0.05,
        "rung 1d>=3%": st >= 0.03,
        "rung 1d>=8%": st >= 0.08,
        "rung 2d>=5%": st2 >= 0.05,
        "LIVE-form 1d>=5% & 2d<=0": (st >= 0.05) & (st2 <= 0),
        "LIVE-form 1d>=5% & prev1d<=-5%": (st >= 0.05) & (stprev <= -0.05),
        "STATE & midterm": (st >= 0.05) & mid,
        "STATE & non-midterm": (st >= 0.05) & ~mid,
        "STATE & collision": (st >= 0.05) & cf,
        "STATE & no collision": (st >= 0.05) & ~cf,
        "PARENT midterm": mid,
        "PARENT collision": cf,
    }
    print(f"\n######## h={h}  (SVXY beta on SPY, 2018-03+ all days: {R['beta']:.3f}) ########")
    for vname, era in [("SVXY", "post"), ("alpha", "post"), ("SPY", "post"), ("SPY", None),
                       ("shortVIX", None), ("shortVIX", "post"), ("shortVIX", "pre")]:
        rows = []
        for lbl, m in masks.items():
            v = pick(R[vname], sp, era)
            rows.append(rec(v[m], lbl))
        allv = R[vname].copy()
        if era == "post":
            allv = allv[cal >= ERA]
        elif era == "pre":
            allv = allv[cal < ERA]
        rows.append(summarize(allv.dropna().values, "CTRL all days"))
        show(rows, f"h={h} {vname} era={era or 'full'} (long SVXY / beta-charged alpha / SPY / short ^VIX)")

    # same state on NON-FOMC days (event removed), and local control
    near = np.zeros(N, bool)
    for p in fpos:
        near[max(0, p - 6):min(N, p + 2)] = True
    stall = (v1 >= 0.05).values
    for vname, era in [("SVXY", "post"), ("alpha", "post"), ("shortVIX", None)]:
        s = R[vname]
        eramask = (cal >= ERA) if era == "post" else np.ones(N, bool)
        nonf = cal[stall & ~near & eramask & s.notna().values]
        nonf = declusters(nonf, h, cal)
        st_f = sp[(st >= 0.05)]
        st_dates = cal[st_f]
        st_dates = st_dates[st_dates >= ERA] if era == "post" else st_dates
        loc = local_control(cal[eramask & s.notna().values], st_dates)
        show([rec(s.loc[nonf].values, f"{vname}: VIX 1d>=5% on NON-FOMC days (declustered)"),
              rec(s.loc[loc].values, f"{vname}: local +/-126td ex-trigger")],
             f"h={h} gate-without-event and local control, {vname} era={era or 'full'}")

    # placebo ladder
    rows = []
    for shift in range(-5, 6):
        sps, _, _ = anchors(shift)
        dd = cal[sps]
        m = v1.reindex(dd).values >= 0.05
        row = {"shift": shift, "k_signal": -2 + shift}
        for vname, era in [("SVXY", "post"), ("alpha", "post"), ("shortVIX", None)]:
            v = pick(R[vname], sps, era)[m]
            v = v[~np.isnan(v)]
            row[f"{vname}_n"] = len(v)
            row[f"{vname}_mean"] = round(100 * v.mean(), 3) if len(v) else np.nan
        rows.append(row)
    lad = pd.DataFrame(rows)
    print(f"\nh={h} PLACEBO LADDER (state VIX 1d>=5% at signal, lag1):")
    print(lad.to_string(index=False))
    for c in ["SVXY_mean", "alpha_mean", "shortVIX_mean"]:
        rank = int((lad[c] > lad.loc[lad["shift"] == 0, c].values[0]).sum()) + 1
        print(f"  {c}: true k=-2 ranks {rank} of {lad[c].notna().sum()}")

# episode list h=1
R1 = rets(1)
sp, fd, cf = anchors(0)
d = cal[sp]
m = v1.reindex(d).values >= 0.05
print("\nSTATE episodes (signal k=-2, FOMC date, VIX 1d, prev 1d, SVXY h1, alpha h1, SPY h1, shortVIX h1, midterm, collision):")
for x, f, c in zip(d[m], fd[m], cf[m]):
    print(f"  {x.date()} FOMC {f.date()} v1 {100*v1.loc[x]:+.1f}% prev {100*v1.shift(1).loc[x]:+.1f}%  "
          f"SVXY {100*R1['SVXY'].loc[x]:+.2f}%  a {100*R1['alpha'].loc[x]:+.2f}%  SPY {100*R1['SPY'].loc[x]:+.2f}%  "
          f"sVIX {100*R1['shortVIX'].loc[x]:+.2f}%  mid={f.year%4==2} coll={c}")
print("\ncost: SVXY ~5 bp/side = 10 bp round trip; 5x bar = +0.50% per trade")
