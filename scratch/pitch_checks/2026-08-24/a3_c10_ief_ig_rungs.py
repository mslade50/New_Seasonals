"""C10 round 1+2 -- long IEF rather than TLT with the IG complex pinned.

W5 parks the shape on TLT and requires TLT <= 0.5% off its 52w low. Today
TLT is +0.86% and MISSES; IEF (+0.70%) and LQD (+0.56%) both clear. C10 asks
whether a rung set anchored on the two legs that ARE pinned stands on its own.

Structure (gate-OFF FIRST, per the round-2 rule):
  (0) live state, reproduced from the cache
  (1) PARENT ladder: each leg alone at each rung, no join, no freshness
  (2) the JOIN: does adding a leg add anything, and which leg is load-bearing
  (3) FRESHNESS: first trigger day in >= 10 sessions, and whether today is one
  (4) DURATION PROPORTIONALITY: is the IEF cell just the TLT cell divided by
      the duration ratio? If yes there is no separate IEF trade -- IEF pays
      the same 3 bps round trip for ~1/2.25 of the move, which is a COST kill,
      not a statistics kill. Test with the LQD-on-IEF residual too.
  (5) the grid I actually walked, and the cost of having walked it
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
COST = {"TLT": 3.0, "IEF": 3.0, "LQD": 6.0, "AGG": 4.0}
TK = ["TLT", "IEF", "LQD", "AGG"]
raw = load_prices(TK)
idx = None
for t in TK:
    i = raw[t]["Close"].dropna().index
    idx = i if idx is None else idx.intersection(i)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in TK}).dropna()
idx = px.index
print("panel %s .. %s  N=%d" % (idx[0].date(), idx[-1].date(), len(idx)))

dist = {}
for t in TK:
    lo = rolling_on_valid(px[t], lambda x: x.rolling(252).min())
    dist[t] = (px[t] / lo - 1.0) * 100.0
dist = pd.DataFrame(dist)

print("\n(0) LIVE STATE (bar %s): %s" % (idx[-1].date(),
      "  ".join("%s %+.2f%% off 52w low" % (t, dist[t].iloc[-1]) for t in TK)))

H = 1  # W5's horizon; scanned in (6)
ret = {t: fwd_lag(px[t], H, 1) for t in TK}
valid = {t: ret[t].notna() for t in TK}


def cellstats(mask, tkr, h=H, label="", fresh=False, gap=10):
    r = fwd_lag(px[tkr], h, 1)
    d = idx[mask.reindex(idx, fill_value=False).values & r.notna().values]
    if fresh:
        d = declusters(d, gap, idx)
    if len(d) == 0:
        return {"label": label, "n": 0}, d
    base = r.dropna()
    s = summarize(r.loc[d].values, label)
    s["ctrl_all_pct"] = round(100 * base.mean(), 4)
    s["excess_pp"] = round(s["mean_pct"] - 100 * base.mean(), 4)
    loc = local_control(idx[r.notna().values], d)
    s["local_pct"] = round(100 * r.loc[loc].mean(), 4)
    s["vs_local_pp"] = round(s["mean_pct"] - 100 * r.loc[loc].mean(), 4)
    w = int((r.loc[d].values > 0).sum())
    p0 = float((base > 0).mean())
    # sign_test's p != 0.5 path overflows above a few hundred n; normal-approx there
    if len(d) <= 250:
        s["sign_p_vs_base"] = round(sign_test(w, len(d), p0), 4)
    else:
        z = (w - len(d) * p0) / np.sqrt(len(d) * p0 * (1 - p0))
        from scipy.stats import norm as _n
        s["sign_p_vs_base"] = round(float(1 - _n.cdf(z)), 4)
    s["x_cost"] = round(100 * s["mean_pct"] / COST[tkr], 1)
    return s, d


# ------------------------------------------------------- (1) parent ladder
print("\n(1) GATE OFF FIRST -- each leg ALONE, no join, no freshness (long IEF h=1)")
rows = []
for t in TK:
    for rung in (0.5, 1.0, 1.5, 3.0):
        m = dist[t] <= rung
        s, _ = cellstats(m, "IEF", label=f"{t} <= {rung}% off low -> long IEF")
        rows.append(s)
show(rows, "single-leg parents, vehicle IEF")

print("\n  same single-leg parents, vehicle TLT (for the W5 comparison)")
rows = []
for t in TK:
    for rung in (0.5, 1.0):
        s, _ = cellstats(dist[t] <= rung, "TLT",
                         label=f"{t} <= {rung}% -> long TLT")
        rows.append(s)
show(rows)

# ------------------------------------------------------------ (2) the join
print("\n(2) THE JOIN -- attribution, one leg at a time")
JOINS = {
    "W5 tight (TLT.5 IEF1 LQD1)": (dist["TLT"] <= 0.5) & (dist["IEF"] <= 1.0) & (dist["LQD"] <= 1.0),
    "C10 drop TLT (IEF1 LQD1)": (dist["IEF"] <= 1.0) & (dist["LQD"] <= 1.0),
    "C10 + TLT<=1.0 (all three at 1%)": (dist["TLT"] <= 1.0) & (dist["IEF"] <= 1.0) & (dist["LQD"] <= 1.0),
    "IEF<=1 alone": dist["IEF"] <= 1.0,
    "LQD<=1 alone": dist["LQD"] <= 1.0,
    "IEF<=0.7 LQD<=0.6 (today's exact rung)": (dist["IEF"] <= 0.70) & (dist["LQD"] <= 0.56),
    "loose (IEF1.5 LQD1.5)": (dist["IEF"] <= 1.5) & (dist["LQD"] <= 1.5),
    "AGG<=1 join (IEF1 LQD1 AGG1)": (dist["IEF"] <= 1.0) & (dist["LQD"] <= 1.0) & (dist["AGG"] <= 1.0),
}
for veh in ("IEF", "TLT"):
    rows = []
    for lbl, m in JOINS.items():
        s, _ = cellstats(m, veh, label=f"{lbl} -> long {veh}")
        rows.append(s)
    show(rows, f"joins, vehicle {veh}, h=1 MOC")

print("\n  LIVE today? ", {lbl: bool(m.iloc[-1]) for lbl, m in JOINS.items()})

# ------------------------------------------------------------ (3) freshness
print("\n(3) FRESHNESS: first trigger day in >= 10 sessions (W5's actual trigger)")
for lbl in ("W5 tight (TLT.5 IEF1 LQD1)", "C10 drop TLT (IEF1 LQD1)",
            "C10 + TLT<=1.0 (all three at 1%)"):
    m = JOINS[lbl]
    for veh in ("IEF", "TLT"):
        s, d = cellstats(m, veh, label=f"{lbl} FRESH -> long {veh}", fresh=True)
        sl, dl = cellstats(m, veh, label=f"{lbl} ALL DAYS -> long {veh}")
        show([s, sl])
        if len(d):
            print("     episode dates:", ", ".join(str(x.date()) for x in d))
    # is TODAY fresh under this rung?
    dd = idx[m.reindex(idx, fill_value=False).values]
    if len(dd):
        pos = pd.Series(range(len(idx)), index=idx)
        prev = dd[dd < idx[-1]]
        gap = (pos[idx[-1]] - pos[prev[-1]]) if len(prev) else 9999
        print("     TODAY fires=%s; sessions since previous trigger day = %s "
              "(needs >= 10)" % (bool(m.iloc[-1]), gap))

# ----------------------------------------------- (4) duration proportionality
print("\n(4) DURATION PROPORTIONALITY -- is the IEF cell a scaled TLT cell?")
m = JOINS["C10 drop TLT (IEF1 LQD1)"]
d = idx[m.values]
sd = {t: px[t].pct_change().std() for t in TK}
print("  daily sd ratios vs IEF: " + "  ".join(
    "%s %.2f" % (t, sd[t] / sd["IEF"]) for t in TK))
rows = []
for t in TK:
    s, _ = cellstats(m, t, label=f"{t} on the C10 rung")
    rows.append(s)
show(rows, "same cell, every vehicle")
ex = {}
for t in TK:
    r = fwd_lag(px[t], H, 1)
    dd = idx[m.values & r.notna().values]
    ex[t] = r.loc[dd].mean() - r.dropna().mean()
print("  excess ratios: TLT/IEF %.2f (daily-sd ratio %.2f) | LQD/IEF %.2f (sd %.2f)"
      % (ex["TLT"] / ex["IEF"], sd["TLT"] / sd["IEF"],
         ex["LQD"] / ex["IEF"], sd["LQD"] / sd["IEF"]))
# LQD residual against IEF on the cell (is there a credit component at all?)
rI = fwd_lag(px["IEF"], H, 1)
rL = fwd_lag(px["LQD"], H, 1)
ok = rI.notna() & rL.notna()
beta = np.polyfit(rI[ok].values, rL[ok].values, 1)[0]
resid = (rL - beta * rI)
dd = idx[m.values & ok.values]
print("  LQD beta on IEF = %.2f; LQD residual on the cell = %+.2f bps "
      "(all-days %+.2f bps) -> credit component %s"
      % (beta, 100 * 100 * resid.loc[dd].mean(),
         100 * 100 * resid[ok].mean(),
         "absent" if abs(100*100*(resid.loc[dd].mean()-resid[ok].mean())) < 3 else "present"))

# ------------------------------------------------------ (5) horizon + grid
print("\n(5) HORIZON scan on the C10 rung, and the grid I walked")
for veh in ("IEF", "TLT"):
    rows = []
    for h in (1, 2, 3, 5, 10):
        s, _ = cellstats(m, veh, h=h, label=f"{veh} h={h}")
        rows.append(s)
    show(rows, f"horizon scan, {veh}")

# the honest grid: 4 legs x 4 rungs x 8 joins x 2 vehicles x 5 horizons
ts = []
for lbl, mm in JOINS.items():
    for veh in TK:
        for h in (1, 2, 3, 5, 10):
            r = fwd_lag(px[veh], h, 1)
            dd = idx[mm.reindex(idx, fill_value=False).values & r.notna().values]
            if len(dd) < 8:
                continue
            v = r.loc[dd].values - r.dropna().mean()
            ts.append((abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), lbl, veh, h))
ts.sort(reverse=True)
print("  grid actually walked: %d cells with N>=8. top 6 by |t|:" % len(ts))
for t_, lbl, veh, h in ts[:6]:
    print("    |t| %.2f  %s -> %s h=%d" % (t_, lbl, veh, h))
print("  Sidak familywise p for the max: 1-(1-p)^K with K=%d" % len(ts))
from scipy import stats as _st
pmax = 2 * (1 - _st.norm.cdf(ts[0][0]))
print("    best cell pointwise p = %.4f -> familywise %.3f"
      % (pmax, 1 - (1 - pmax) ** len(ts)))
