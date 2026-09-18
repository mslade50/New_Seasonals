import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# Round 1b for C3: where does the LIVE reading sit on the dose curve, and does the
# live "re-bid after a crush" form survive on the non-levered ^VIX read?
pd.set_option("future.no_silent_downcasting", True)
TK = ["SPY", "^VIX", "^VIX3M", "SVXY"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
px["^VIX"] = px["^VIX"].ffill(limit=2)
vix = px["^VIX"]
v1 = vix / vix.shift(1) - 1
v2 = vix / vix.shift(2) - 1
term = vix / px["^VIX3M"]
live = pd.Timestamp("2026-09-14")
print(f"LIVE: v1 {100*v1.loc[live]:+.2f}%  v2 {100*v2.loc[live]:+.2f}%  prev1d {100*v1.shift(1).loc[live]:+.2f}%  VIX/VIX3M {term.loc[live]:.3f}")

ev = load_events(["fomc_decision", "vix_expiry"])
fomc = pd.DatetimeIndex(ev[ev.event == "fomc_decision"]["date"])
vxe = set(pd.DatetimeIndex(ev[ev.event == "vix_expiry"]["date"]))
fpos, fkept = anchor_positions(cal, fomc, 0)
fpos = np.array(fpos)
fkept = pd.DatetimeIndex(fkept)
ERA = pd.Timestamp("2018-03-01")
N = len(cal)

h = 1
svxy = fwd_lag(px["SVXY"], h, 1)
spy = fwd_lag(px["SPY"], h, 1)
svix = -fwd_lag(vix, h, 1)
ok = svxy.notna() & spy.notna() & (cal >= ERA)
beta = np.polyfit(spy[ok].values, svxy[ok].values, 1)[0]
alpha = svxy - beta * spy
print(f"beta {beta:.3f}")


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


sp = fpos - 2
d = cal[sp]
post = d >= ERA
V1 = v1.reindex(d).values
V2 = v2.reindex(d).values
PV = v1.shift(1).reindex(d).values
TM = term.reindex(d).values
mid = np.array([x.year % 4 == 2 for x in fkept])

buckets = {"[-inf,0)": (V1 < 0), "[0,3%)": (V1 >= 0) & (V1 < .03), "[3,5%)": (V1 >= .03) & (V1 < .05),
           "[5,8%) LIVE": (V1 >= .05) & (V1 < .08), "[8%,inf)": V1 >= .08,
           "[5,7%)": (V1 >= .05) & (V1 < .07), "[7,9%)": (V1 >= .07) & (V1 < .09), "[6.5,9.5%) +/-1.5 of live": (V1 >= .065) & (V1 < .095)}
for name, s, emask in [("long SVXY 2018-03+", svxy, post), ("alpha SVXY-b*SPY 2018-03+", alpha, post),
                       ("SPY 2018-03+", spy, post), ("short ^VIX full", svix, np.ones(len(d), bool)),
                       ("short ^VIX pre-2018", svix, ~post), ("SPY full", spy, np.ones(len(d), bool))]:
    vals = s.reindex(d).values
    show([rec(vals[m & emask], lbl) for lbl, m in buckets.items()], f"DOSE buckets, VIX 1d at k=-2 -> {name}, h=1")

forms = {
    "1d>=5% & prev<=-5%": (V1 >= .05) & (PV <= -.05),
    "1d>=5% & prev<=-10%": (V1 >= .05) & (PV <= -.10),
    "1d>=5% & 2d<=0 (LIVE)": (V1 >= .05) & (V2 <= 0),
    "1d>=5% & prev<=-5% & 2d>0": (V1 >= .05) & (PV <= -.05) & (V2 > 0),
    "1d>=3% & prev<=-5% & 2d<=0": (V1 >= .03) & (PV <= -.05) & (V2 <= 0),
    "1d>=5% & VIX/VIX3M<0.9": (V1 >= .05) & (TM < 0.9),
    "1d>=5% & VIX/VIX3M>=0.9": (V1 >= .05) & (TM >= 0.9),
    "1d>=5% & 2d<=0 & midterm": (V1 >= .05) & (V2 <= 0) & mid,
}
for name, s, emask in [("long SVXY 2018-03+", svxy, post), ("alpha 2018-03+", alpha, post),
                       ("short ^VIX full", svix, np.ones(len(d), bool)), ("short ^VIX pre-2018", svix, ~post),
                       ("SPY full", spy, np.ones(len(d), bool))]:
    vals = s.reindex(d).values
    show([rec(vals[m & emask], lbl) for lbl, m in forms.items()], f"LIVE-FORM variants -> {name}, h=1")

m = (V1 >= .05) & (V2 <= 0)
print("\nLIVE-form (1d>=5% & 2d<=0) episodes, full history:")
for x, f in zip(d[m], fkept[m]):
    print(f"  {x.date()} FOMC {f.date()} v1 {100*v1.loc[x]:+.1f}% v2 {100*v2.loc[x]:+.1f}% prev {100*v1.shift(1).loc[x]:+.1f}% "
          f"sVIX {100*svix.loc[x]:+.2f}% SPY {100*spy.loc[x]:+.2f}% SVXY {100*svxy.loc[x]:+.2f}% mid={f.year%4==2} coll={f in vxe}")

# placebo ladder on the LIVE form, short ^VIX full history (median + hit, mean is outlier-driven)
rows = []
for shift in range(-5, 6):
    spp = fpos - 2 + shift
    okp = (spp >= 2) & (spp < N - 2)
    dd = cal[spp[okp]]
    mm = (v1.reindex(dd).values >= .05) & (v2.reindex(dd).values <= 0)
    x = svix.reindex(dd).values[mm]
    x = x[~np.isnan(x)]
    mm5 = v1.reindex(dd).values >= .05
    y = svix.reindex(dd).values[mm5]
    y = y[~np.isnan(y)]
    rows.append({"k_signal": -2 + shift, "live_n": len(x), "live_mean": round(100 * x.mean(), 2) if len(x) else np.nan,
                 "live_median": round(100 * np.median(x), 2) if len(x) else np.nan,
                 "live_hit": round(100 * (x > 0).mean(), 1) if len(x) else np.nan,
                 "state5_n": len(y), "state5_median": round(100 * np.median(y), 2), "state5_hit": round(100 * (y > 0).mean(), 1)})
lad = pd.DataFrame(rows)
print("\nPLACEBO LADDER short ^VIX full history h=1:")
print(lad.to_string(index=False))
for c in ["live_median", "state5_median", "state5_hit"]:
    t0 = lad.loc[lad.k_signal == -2, c].values[0]
    print(f"  {c}: true k=-2 ranks {int((lad[c] > t0).sum()) + 1} of {lad[c].notna().sum()}")

# decision-day tail in the state, SVXY era
m5 = (V1 >= .05) & post
x = svxy.reindex(d).values[m5]
print(f"\nSVXY 2018-03+ state tail: worst {100*np.nanmin(x):.2f}%, n<=-5%: {(x <= -0.05).sum()} of {np.isfinite(x).sum()}; "
      f"sum of 2 worst {100*np.sort(x)[:2].sum():+.2f}pp vs total {100*np.nansum(x):+.2f}pp")
print(cluster_note(d[m5], x))
