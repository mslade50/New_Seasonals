"""D3 round 1 -- EWZ (and EEM) failing to participate in a global risk-on thrust.

PRE-SPECIFIED BEFORE MEASURING (stated here so the neighbour scan below is
visibly a robustness check, not the search that found the cell):

  TRIGGER  EWZ 5d return < 0
     AND   SPY within 1.0% of its trailing-252d high
     AND   SPY 5d return >= +2.0%

  VEHICLES  (a) long EWZ outright
            (b) long EWZ / short SPY, EQUAL DOLLAR
            (c) long EWZ / short (beta x SPY), beta = trailing 252d OLS of
                EWZ daily returns on SPY daily returns, point-in-time
  ENTRY     lag=1 MOC-tomorrow.   HORIZONS  h=5 headline, 1..21 scanned.

THE OBVIOUS KILL, and therefore the control that decides this: the trigger
requires SPY at a 52w high having just run +2% in a week. That is a specific
and very good tape. The comparison that matters is NOT EWZ's unconditional
drift, it is EWZ's drift ON THOSE SAME SPY DAYS. If EWZ-decoupling adds
nothing over "EWZ during a SPY thrust to a high", the divergence is decoration
and the cell is a levered long wearing a relative-value costume.

Registry collision handled explicitly: "laggard-snapback continuation
(SMH/QQQ form) ... the trigger over-selects bear tape by +29pp vs base rate".
Here the regime-selection question is INVERTED (a 52w-high gate cannot select
bear tape), so the bear-tape fraction is computed and reported anyway to show
which way the selection runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["EWZ", "EEM", "SPY", "EFA", "EWW", "FXI"]
px = close_panel(TK).dropna(subset=["EWZ", "SPY", "EEM", "EFA"])
px = px.loc[px.index >= "2003-05-01"]          # EEM starts 2003-04-14
idx = px.index
print(f"panel span {idx[0].date()} .. {idx[-1].date()}  n={len(idx)}")

r1 = {t: px[t].pct_change() for t in TK}
r5 = {t: px[t].pct_change(5) for t in TK}
d52 = {t: px[t] / px[t].rolling(252).max() - 1.0 for t in TK}
sma200 = px["SPY"].rolling(200).mean()

print("\nTODAY (2026-08-07 close):")
for t in ["EWZ", "EEM", "SPY", "EFA"]:
    print(f"  {t:4s} ret5d {100*r5[t].iloc[-1]:+6.2f}%   dist52wh "
          f"{100*d52[t].iloc[-1]:+7.2f}%   rank5d "
          f"{pct_rank(px[t], 5).iloc[-1]:5.1f}  rank63 "
          f"{pct_rank(px[t], 63).iloc[-1]:5.1f}")

# ---------------------------------------------------------------- beta, PIT
def rolling_beta(a: pd.Series, b: pd.Series, win: int = 252) -> pd.Series:
    cov = a.rolling(win).cov(b)
    var = b.rolling(win).var()
    return cov / var


beta_ewz = rolling_beta(r1["EWZ"], r1["SPY"])
beta_eem = rolling_beta(r1["EEM"], r1["SPY"])
print(f"\nPIT beta EWZ~SPY today {beta_ewz.iloc[-1]:.2f}  "
      f"(median over history {beta_ewz.median():.2f}, "
      f"10-90 pct {beta_ewz.quantile(.1):.2f}-{beta_ewz.quantile(.9):.2f})")
print(f"PIT beta EEM~SPY today {beta_eem.iloc[-1]:.2f}  "
      f"(median {beta_eem.median():.2f})")

# ---------------------------------------------------------------- triggers
SPY_HIGH = -0.010
SPY_THRUST = 0.020

spy_gate = ((d52["SPY"] >= SPY_HIGH) & (r5["SPY"] >= SPY_THRUST)).fillna(False)
m_ewz = (spy_gate & (r5["EWZ"] < 0)).fillna(False)          # TREATMENT
m_eem = (spy_gate & (r5["EEM"] < 0)).fillna(False)          # neighbour ticker
m_eem_lag63 = (spy_gate & (pct_rank(px["EEM"], 63) < 10)).fillna(False)

print(f"\nGATE COUNTS (day level)")
print(f"  SPY gate alone (52wh within 1.0% AND 5d >= +2%): {int(spy_gate.sum())}")
print(f"  + EWZ 5d < 0  (TREATMENT):                       {int(m_ewz.sum())}"
      f"   share of parent {m_ewz.sum()/max(spy_gate.sum(),1):.3f}")
print(f"  + EEM 5d < 0:                                    {int(m_eem.sum())}"
      f"   share of parent {m_eem.sum()/max(spy_gate.sum(),1):.3f}")
print(f"  + EEM rank63 < 10 (the '63d laggard' form):       {int(m_eem_lag63.sum())}")

# cluster depth today -- mid-cluster entry is not a fresh trigger
for lbl, mm in [("TREATMENT EWZ", m_ewz), ("SPY gate", spy_gate)]:
    run = 0
    for v in mm.values[::-1]:
        if v:
            run += 1
        else:
            break
    print(f"  CLUSTER DEPTH TODAY {lbl}: {run} consecutive sessions "
          f"(fires today: {bool(mm.iloc[-1])})")

# ---------------------------------------------- regime selection, both ways
base_bear = float((px["SPY"] < sma200).mean())
for lbl, mm in [("TREATMENT EWZ", m_ewz), ("SPY gate parent", spy_gate)]:
    sel = float((px["SPY"] < sma200)[mm].mean()) if mm.sum() else np.nan
    print(f"  REGIME SELECTION {lbl}: bear tape (SPY<200d) on "
          f"{100*sel:.1f}% of trigger days vs base rate {100*base_bear:.1f}% "
          f"-> {100*(sel-base_bear):+.1f}pp")

# ---------------------------------------------------------------- variants
variants = {
    "SPY high 0.5%": (((d52["SPY"] >= -0.005) & (r5["SPY"] >= .02))
                      & (r5["EWZ"] < 0)).fillna(False),
    "SPY high 2.0%": (((d52["SPY"] >= -0.020) & (r5["SPY"] >= .02))
                      & (r5["EWZ"] < 0)).fillna(False),
    "SPY thrust >=1%": (((d52["SPY"] >= -0.01) & (r5["SPY"] >= .01))
                        & (r5["EWZ"] < 0)).fillna(False),
    "SPY thrust >=3%": (((d52["SPY"] >= -0.01) & (r5["SPY"] >= .03))
                        & (r5["EWZ"] < 0)).fillna(False),
    "EWZ rank5d < 15": (spy_gate & (pct_rank(px["EWZ"], 5) < 15)).fillna(False),
    "EWZ 5d < -2%": (spy_gate & (r5["EWZ"] < -0.02)).fillna(False),
    "EWZ 5d < -3.5% (today)": (spy_gate & (r5["EWZ"] < -0.035)).fillna(False),
    "GATE ATTR: SPY gate only (CONTROL)": spy_gate,
    "no SPY high, EWZ 5d<0 only": (r5["EWZ"] < 0).fillna(False),
}

# ---------------------------------------------------------------- batteries
for legs, cost, tag in [
        ([("EWZ", 1.0)], 5.0, "LONG EWZ outright"),
        ([("EWZ", 1.0), ("SPY", -1.0)], 3.5, "LONG EWZ / SHORT SPY equal-dollar"),
]:
    battery(px, m_ewz, legs, h=5, cost_bps=cost,
            title=f"{tag} | EWZ 5d<0 while SPY within 1% of 52wh and +2%/5d",
            lag=1, min_gap=5, event_kinds=("cpi", "ppi"),
            variants=variants if tag.startswith("LONG EWZ outright") else None)

# ------------------------------------------- beta-neutral spread, by hand
print("\n" + "=" * 78)
print("BETA-NEUTRAL SPREAD (leg sizing = PIT 252d beta at the signal date)")
print("=" * 78)
for h in (1, 2, 3, 5, 10, 21):
    fe = fwd_lag(px["EWZ"], h, 1)
    fs = fwd_lag(px["SPY"], h, 1)
    valid = fe.notna() & fs.notna() & beta_ewz.notna()
    sig = idx[m_ewz.values & valid.values]
    if len(sig) == 0:
        continue
    epi = declusters(sig, h, idx)
    bn = (fe - beta_ewz * fs)
    eq = (fe - fs)
    ctrl_idx = idx[spy_gate.values & valid.values & (idx >= sig[0])]
    ce = declusters(ctrl_idx, h, idx)
    base = bn[valid & (idx >= sig[0])]
    print(f"h={h:2d}  BETA-NEUTRAL trig {100*bn.loc[epi].mean():+.3f}% "
          f"(N={len(epi)})  SPYgate-ctrl {100*bn.loc[ce].mean():+.3f}% "
          f"(N={len(ce)})  uncond-same-span {100*base.mean():+.3f}%  || "
          f"EQ-DOLLAR trig {100*eq.loc[epi].mean():+.3f}%  ctrl "
          f"{100*eq.loc[ce].mean():+.3f}%  || avg PIT beta at trigger "
          f"{beta_ewz.loc[epi].mean():.2f}")

# ------------------------------------------- head to head vs the SPY gate
print("\n" + "=" * 78)
print("HEAD TO HEAD: EWZ-decoupling minus SPY-gate control, SAME SPAN, episodes")
print("(this is the number that decides it -- does the EWZ leg filter?)")
print("=" * 78)
for legs, tag in [([("EWZ", 1.0)], "long EWZ"),
                  ([("EWZ", 1.0), ("SPY", -1.0)], "EWZ-SPY eq$"),
                  ([("EEM", 1.0)], "long EEM"),
                  ([("SPY", 1.0)], "long SPY (what the gate alone buys)")]:
    print(f"\n-- {tag}")
    for h in (1, 3, 5, 10, 21):
        r = vehicle_ret(px, legs, h, 1)
        val = r.notna()
        mm = m_eem if tag == "long EEM" else m_ewz
        dt_ = idx[mm.values & val.values]
        if len(dt_) == 0:
            continue
        lo = dt_[0]
        dc = idx[spy_gate.values & val.values & (idx >= lo)]
        et, ec = declusters(dt_, h, idx), declusters(dc, h, idx)
        vt, vc = r.loc[et].values, r.loc[ec].values
        base = r[val & (idx >= lo)]
        se = np.sqrt(vt.var(ddof=1)/len(vt) + vc.var(ddof=1)/max(len(vc), 2))
        wins = int((vt > 0).sum())
        print(f"  h={h:2d}  TRIG {100*vt.mean():+.3f}% (N={len(vt)}, "
              f"{wins}-{len(vt)-wins}, signp {sign_test(wins, len(vt)):.3f})  "
              f"SPYGATE {100*vc.mean():+.3f}% (N={len(vc)})  "
              f"ADD {100*(vt.mean()-vc.mean()):+.3f}pp welch t "
              f"{(vt.mean()-vc.mean())/se:+.2f}  | uncond {100*base.mean():+.3f}%")

# ---------------------------------------------------------------- EEM cell
print("\n" + "=" * 78)
print("NEIGHBOUR TICKER: EEM 5d<0 under the same SPY gate  (do NOT conflate)")
print("=" * 78)
battery(px, m_eem, [("EEM", 1.0)], h=5, cost_bps=3.0,
        title="LONG EEM | EEM 5d<0 while SPY within 1% of 52wh and +2%/5d",
        lag=1, min_gap=5, event_kinds=("cpi", "ppi"))

# ---------------------------------------------------------------- scan/era
print("\n" + "=" * 78)
print("SCAN (multiplicity applies to this table): h=1..21")
print("=" * 78)
show(horizon_scan(px, idx[m_ewz.values], [("EWZ", 1.0)],
                  hs=(1, 2, 3, 5, 7, 10, 15, 21)), "EWZ decoupler, long EWZ")
show(horizon_scan(px, idx[m_ewz.values], [("EWZ", 1.0), ("SPY", -1.0)],
                  hs=(1, 2, 3, 5, 7, 10, 15, 21)), "EWZ decoupler, eq$ spread")

print("\n" + "=" * 78)
print("ERA / MIDTERM / CONCENTRATION  (h=5 episodes, long EWZ)")
print("=" * 78)
h = 5
r = vehicle_ret(px, [("EWZ", 1.0)], h, 1)
val = r.notna()
e = declusters(idx[m_ewz.values & val.values], h, idx)
v = r.loc[e].values
yr = pd.DatetimeIndex(e).year
show(era_split(e, v, "2018-01-01"), "era 2018")
show(era_split(e, v, "2013-01-01"), "era 2013")
mid = yr % 4 == 2
base = r[val]
bmid = base.index.year % 4 == 2
show([summarize(v[mid], f"MIDTERM (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})"),
      summarize(base[bmid].values, "CTRL all days midterm"),
      summarize(base[~bmid].values, "CTRL all days non-midterm")], "midterm split")
hist = pd.Series(100 * v, index=pd.DatetimeIndex(e)).groupby(yr).agg(
    ["count", "sum", "mean"])
print("\nyear histogram (episodes, h=5, pp):")
print(hist.round(2).to_string())
print("\n" + cluster_note(pd.DatetimeIndex(e), v, k=3))

order = np.argsort(v)[:5]
print("\nWORST 5 EPISODES (long EWZ, h=5) -- name the idiosyncratic tail:")
for i in order:
    print(f"  {pd.Timestamp(e[i]).date()}  {100*v[i]:+.2f}%")
order = np.argsort(-v)[:5]
print("BEST 5 EPISODES:")
for i in order:
    print(f"  {pd.Timestamp(e[i]).date()}  {100*v[i]:+.2f}%")

# same for eq-dollar spread
r2 = vehicle_ret(px, [("EWZ", 1.0), ("SPY", -1.0)], h, 1)
e2 = declusters(idx[m_ewz.values & r2.notna().values], h, idx)
v2 = r2.loc[e2].values
print("\nWORST 5 EPISODES (eq$ EWZ-SPY spread, h=5):")
for i in np.argsort(v2)[:5]:
    print(f"  {pd.Timestamp(e2[i]).date()}  {100*v2[i]:+.2f}%")
print(cluster_note(pd.DatetimeIndex(e2), v2, k=3))
