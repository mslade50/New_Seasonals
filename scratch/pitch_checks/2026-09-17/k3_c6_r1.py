"""K3 c6 round 1 - Short VLO against beta x XLE when the refiner prints a
trailing-252 closing high on a session crude (USO) falls >= 3% (also >= 2%).
Signal 2026-09-16.

  T0 live confirmation
  T1 cell N, pair (short VLO / +beta XLE, rolling-252 OLS as of D) h=1..10,
     outright short VLO, own-drift + all-days controls
  T2 refiner basket (VLO MPC PSX DINO PBF DK): any refiner at a 252 high on a
     crude-down day, short that refiner vs beta XLE
  T3 PLACEBO that must be beaten: on EVERY crude-down day, short the energy
     name with the best 1-day return vs beta XLE
  T4 mechanism: does the refiner-vs-crude ratio (VLO/USO) mean-revert after
     such days, and is VLO's excess over XLE doing the work
  T5 CL=F as the crude leg (pre-2006 history)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
REF = ["VLO", "MPC", "PSX", "DINO", "PBF", "DK"]
ENER = ["XLE", "XOP", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB",
        "MPC", "PSX"]
px = close_panel(sorted(set(REF + ENER + ["USO", "CL=F", "SPY"])))
px = px[px.index <= BAR]
px = px[px["XLE"].notna()]
D = px.index
r1 = px.pct_change(fill_method=None)
COST = 6.0


def beta_of(t, bench="XLE"):
    c = r1[t].rolling(252, min_periods=200).cov(r1[bench])
    v = r1[bench].rolling(252, min_periods=200).var()
    return c / v


def hi252(t):
    c = px[t]
    return c >= c.rolling(252, min_periods=252).max()


def short_pair(t, h, lag=1):
    return -(fwd_lag(px[t], h, lag) - beta_of(t) * fwd_lag(px["XLE"], h, lag))


# T0 live
print("LIVE 2026-09-16:")
for t in ["USO", "XLE", "VLO", "MPC", "PSX", "DINO"]:
    print(f"  {t} 1d {100*r1[t].iloc[-1]:+.2f}%  at252hi {bool(hi252(t).iloc[-1])}"
          f"  off-hi {100*(px[t].iloc[-1]/px[t].iloc[-252:].max()-1):+.2f}%")
print(f"  beta VLO/XLE {beta_of('VLO').iloc[-1]:.3f}")
best = r1[ENER].iloc[-1].sort_values(ascending=False)
print("  best energy name today:", best.head(3).round(4).to_dict())


def cellrows(mask, retfn, lab, hs=(1, 2, 3, 5, 10), gap=5):
    rows, keep = [], {}
    for h in hs:
        ret = retfn(h)
        valid = ret.notna()
        sig = D[(mask.reindex(D, fill_value=False) & valid).values]
        epi = declusters(sig, gap, D)
        v = ret.loc[epi].values
        r = summarize(v, f"{lab} h={h}")
        if r["n"]:
            w = int((v > 0).sum())
            r["rec"] = f"{w}-{r['n']-w}"
            r["sign_p"] = round(sign_test(w, r["n"]), 4)
            r["ctl_all"] = round(100 * ret[valid].mean(), 3)
            r["edge"] = round(r["mean_pct"] - r["ctl_all"], 3)
            r["cost_x"] = round(1e4 * v.mean() / COST, 2)
        rows.append(r)
        keep[h] = (epi, v)
    return rows, keep


for thr in (-0.03, -0.02):
    cd = r1["USO"] <= thr
    cell = hi252("VLO") & cd
    sig = D[cell.values]
    print(f"\n### USO <= {100*thr:.0f}%: crude-down days {int(cd.sum())}, "
          f"VLO-252hi days {int(hi252('VLO').sum())}, cell days {len(sig)}")
    print("   dates:", [str(d.date()) for d in sig])
    rows, keep = cellrows(cell, lambda h: short_pair("VLO", h), "short VLO/bXLE")
    rows2, _ = cellrows(cell, lambda h: -fwd_lag(px["VLO"], h), "short VLO outright")
    show(rows + rows2, f"T1 cell USO<={100*thr:.0f}%")
    if 3 in keep and len(keep[3][0]):
        for d, v in zip(keep[3][0], keep[3][1]):
            print(f"    {d.date()} h=3 pair {100*v:+.2f}%")

# T1b: gate attribution - VLO 252 high on ANY day, and on crude-UP days
for lab, m in [("VLO 252hi any day", hi252("VLO")),
               ("VLO 252hi & USO>=0", hi252("VLO") & (r1["USO"] >= 0)),
               ("VLO 252hi & USO<=-2%", hi252("VLO") & (r1["USO"] <= -0.02))]:
    rows, _ = cellrows(m, lambda h: short_pair("VLO", h), lab, hs=(1, 3, 5, 10))
    show(rows, f"T1b gate attribution: {lab}")

# T2 refiner basket: any refiner at 252 high on crude-down day
for thr in (-0.03, -0.02):
    cd = r1["USO"] <= thr
    recs = []
    for h in (1, 2, 3, 5, 10):
        vals, dts = [], []
        for t in REF:
            m = hi252(t) & cd & px[t].notna()
            ret = short_pair(t, h)
            sig = D[(m & ret.notna()).values]
            epi = declusters(sig, 5, D)
            vals += list(ret.loc[epi].values)
            dts += list(epi)
        v = np.array(vals)
        r = summarize(v, f"refiner basket USO<={100*thr:.0f}% h={h}")
        if r["n"]:
            w = int((v > 0).sum())
            r["rec"] = f"{w}-{r['n']-w}"
            r["sign_p"] = round(sign_test(w, r["n"]), 4)
            r["cost_x"] = round(1e4 * v.mean() / COST, 2)
            r["uniq_dates"] = len(set(dts))
        recs.append(r)
    show(recs, f"T2 refiner basket (name-level episodes) USO<={100*thr:.0f}%")

# T3 placebo: best energy name on every crude-down day
for thr in (-0.03, -0.02):
    cd = (r1["USO"] <= thr).values
    recs = []
    for h in (1, 2, 3, 5, 10):
        vals = []
        for i in np.where(cd)[0]:
            row = r1[ENER].iloc[i].dropna()
            if row.empty:
                continue
            t = row.idxmax()
            v = short_pair(t, h).iloc[i]
            if np.isfinite(v):
                vals.append(v)
        v = np.array(vals)
        r = summarize(v, f"PLACEBO best-name USO<={100*thr:.0f}% h={h}")
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{r['n']-w}"
        r["sign_p"] = round(sign_test(w, r["n"]), 4)
        recs.append(r)
    show(recs, "T3 placebo (day-level)")

# T3b placebo restricted: best name that is ALSO at a 252 high on crude-down day
cd = r1["USO"] <= -0.02
recs = []
for h in (1, 3, 5, 10):
    vals = []
    for t in ENER:
        if t == "XLE":
            continue
        m = hi252(t) & cd
        ret = short_pair(t, h)
        sig = D[(m & ret.notna()).values]
        epi = declusters(sig, 5, D)
        vals += list(ret.loc[epi].values)
    v = np.array(vals)
    r = summarize(v, f"any energy name 252hi & USO<=-2% h={h}")
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{r['n']-w}"
    r["sign_p"] = round(sign_test(w, r["n"]), 4)
    recs.append(r)
show(recs, "T3b reference class: any energy name at a 252 high on a crude-down day")

# T4 mechanism: crack proxy VLO/USO
ratio = px["VLO"] / px["USO"]
cell = hi252("VLO") & (r1["USO"] <= -0.02)
rows = []
for h in (1, 3, 5, 10):
    rr = fwd_lag(ratio, h)
    sig = D[(cell & rr.notna()).values]
    epi = declusters(sig, 5, D)
    r = summarize(-rr.loc[epi].values, f"short VLO/USO ratio h={h}")
    r["ctl_all"] = round(-100 * rr.mean(), 3)
    rows.append(r)
    ro = -fwd_lag(px["USO"], h)
    r2 = summarize(-ro.loc[epi].values, f"USO long leg h={h}")
    rows.append(r2)
show(rows, "T4 crack proxy mean reversion (VLO 252hi & USO<=-2%)")

# T5 CL=F crude leg
clr = px["CL=F"].pct_change(fill_method=None)
clr = clr.where(px["CL=F"] > 5)
for thr in (-0.03, -0.02):
    cell = hi252("VLO") & (clr <= thr)
    rows, keep = cellrows(cell, lambda h: short_pair("VLO", h),
                          f"CL=F<={100*thr:.0f}% short VLO/bXLE", hs=(1, 3, 5, 10))
    show(rows, f"T5 CL=F crude leg thr {thr}")
    e, v = keep[5]
    if len(e):
        show(era_split(e, v), "era h=5")
