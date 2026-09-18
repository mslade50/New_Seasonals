"""C8 round 1 -- SHORT duration (TLT / IEF) on the commodity-complex-at-a-252d-high
state, with the print pair inside the hold.

The registry's standing correction is mandatory here: "CPI/PPI/FOMC work on
duration" is a TRADING-DAY-OF-MONTH profile (long TLT into CPI = +0.178% raw,
+6.7 bps tdom-matched). So every number below is quoted against a tdom-matched
control as well as the usual three.

Second burden, stated in the candidate: duration is ALREADY at a 252d low, so
this is momentum. The commodity-high state has to add something beyond
"bonds have been going down" -- that is the TLT-near-its-own-low control.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICK = ["DBC", "USO", "TLT", "IEF", "LQD", "SPY"]
px = close_panel(TICK)
raw = load_prices(["DBC", "TLT", "IEF"])
IDX = px.index


def dist_to_high(t, n=252):
    s = raw[t]["Close"]
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (1.0 - s / hi).reindex(IDX)


def dist_above_low(t, n=252):
    s = raw[t]["Close"]
    lo = rolling_on_valid(s, lambda x: x.rolling(n).min())
    return (s / lo - 1.0).reindex(IDX)


def tdom(idx):
    d = pd.DatetimeIndex(idx)
    g = pd.Series(range(len(d)), index=d).groupby([d.year, d.month]).rank(method="first")
    return g.astype(int)


def flag_in_window(kinds, h, lag=1, k=0):
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


LAG, H = 1, 3
TD = tdom(IDX)
d_dbc = dist_to_high("DBC")
pr = flag_in_window(("ppi", "cpi"), H, LAG)
state = d_dbc <= 0.0025
cell = state & pr

print("=" * 78)
print("C8  SHORT duration on DBC-at-252d-high + print in hold   h=3 lag=1")
print("=" * 78)
print(f"live: DBC {100*d_dbc.iloc[-1]:.3f}% off high | TLT "
      f"{100*dist_above_low('TLT').iloc[-1]:.2f}% above 252d low | IEF "
      f"{100*dist_above_low('IEF').iloc[-1]:.2f}% | tdom today(09-08) = 6")

for tkr, cost in (("TLT", 3.0), ("IEF", 3.0)):
    battery(px, cell, [(tkr, -1.0)], H, f"C8  SHORT {tkr}, DBC<=0.25% off high + print",
            cost_bps=cost, event_kinds=("cpi",),
            variants={
                "tol 0.10%": (d_dbc <= 0.0010) & pr,
                "tol 0.25% (LIVE)": cell,
                "tol 0.50%": (d_dbc <= 0.0050) & pr,
                "tol 1.00%": (d_dbc <= 0.0100) & pr,
                "tol 2.00%": (d_dbc <= 0.0200) & pr,
            })

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("MANDATORY tdom-MATCHED CONTROL  (registry 2026-08-10 d5b_tdom_control)")
print("=" * 78)
for tkr in ("TLT", "IEF"):
    ret = vehicle_ret(px, [(tkr, -1.0)], H, LAG)
    valid = ret.notna()
    d = IDX[cell.values & valid.values]
    e = declusters(d, H, IDX)
    # tdom-matched control: all days sharing the trigger days' tdom distribution
    tds = TD.loc[d]
    ctrl_parts, w = [], []
    for t_, cnt in tds.value_counts().items():
        pool = ret[valid & (TD == t_).values]
        if len(pool):
            ctrl_parts.append(pool.mean())
            w.append(cnt)
    tdom_ctrl = float(np.average(ctrl_parts, weights=w))
    rows = [summarize(ret.loc[d].values, f"SHORT {tkr} cell [day]"),
            summarize(ret.loc[e].values, f"SHORT {tkr} cell [epi]"),
            summarize(ret[valid].values, f"SHORT {tkr} all days")]
    show(rows, f"{tkr}")
    print(f"  tdom of trigger days: median {int(tds.median())}, "
          f"range {int(tds.min())}-{int(tds.max())}, "
          f"top: {dict(tds.value_counts().head(4))}")
    print(f"  tdom-MATCHED control mean = {100*tdom_ctrl:+.3f}%   "
          f"=> cell excess [day] {100*(ret.loc[d].mean()-tdom_ctrl):+.3f}pp, "
          f"[epi] {100*(ret.loc[e].mean()-tdom_ctrl):+.3f}pp")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- is the commodity state doing anything beyond "
      "'bonds have been going down'?")
print("=" * 78)
lo_tlt = dist_above_low("TLT")
mom = lo_tlt <= 0.02          # TLT within 2% of its own 252d low (live: 1.44%)
for tkr in ("TLT", "IEF"):
    ret = vehicle_ret(px, [(tkr, -1.0)], H, LAG)
    valid = ret.notna()
    rows = []
    for m, lbl in ((cell, "cmdty-high AND print (cell)"),
                   (state & ~pr, "cmdty-high, NO print"),
                   (pr & ~state, "print, NO cmdty-high"),
                   (mom, "TLT within 2% of 252d LOW (momentum alone)"),
                   (mom & state, "TLT low AND cmdty high"),
                   (mom & state & pr, "TLT low AND cmdty high AND print (fully live)"),
                   (pd.Series(True, index=IDX), "all days")):
        d = IDX[m.values & valid.values]
        if len(d) == 0:
            rows.append({"label": lbl, "n": 0}); continue
        e = declusters(d, H, IDX)
        rows.append(summarize(ret.loc[d].values, lbl + " [day]"))
        rows.append(summarize(ret.loc[e].values, lbl + " [epi]"))
    show(rows, f"SHORT {tkr}  h=3 lag=1")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLACEBO ANCHOR LADDER k=-5..+5")
print("=" * 78)
for tkr in ("TLT", "IEF"):
    ret = vehicle_ret(px, [(tkr, -1.0)], H, LAG)
    valid = ret.notna()
    rows, means = [], []
    for k in range(-5, 6):
        m = state & flag_in_window(("ppi", "cpi"), H, LAG, k)
        d = IDX[m.values & valid.values]
        e = declusters(d, H, IDX)
        r = summarize(ret.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r); means.append(r.get("mean_pct", np.nan))
    show(rows, f"SHORT {tkr}")
    print(f"  true anchor ranks {int(np.sum(np.asarray(means) >= means[5]))} of 11")
