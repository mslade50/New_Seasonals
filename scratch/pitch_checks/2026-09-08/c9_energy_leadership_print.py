"""C9 round 1 -- energy equity leadership continuation: long XLE / XOP at a
21-day return rank >= 83 with an inflation print inside the hold.

Live 2026-09-04: XLE r21 83.3 (band [83,90)), XOP r21 91.7 (band [90,95)).
Registry priors this must beat:
  - plain XLE at a fresh 252d high pays +0.606% over 70 episodes, only
    +0.135pp over its all-days baseline (2026-08-19 b2_c9)
  - energy 5d thrust into a 52w high: thrust alone -0.313%, near-high without
    thrust -0.298% (2026-08-17 p2_c5)
  - the CPI anchor SUBTRACTS on energy (2026-08-10 c5_energy_cpi_washout)
  - pre-holiday r21>=80: XOP ZERO prior obs, XLE two averaging -1.42%
LEFT-OPEN RULE: each rank band is quoted on its own, live band first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICK = ["XLE", "XOP", "SPY", "USO", "DBC", "OIH"]
px = close_panel(TICK)
raw = load_prices(["XLE", "XOP"])
IDX = px.index


def r21(t):
    return pct_rank(raw[t]["Close"], 21).reindex(IDX)


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
pr = flag_in_window(("ppi", "cpi"), H, LAG)
rk = {t: r21(t) for t in ("XLE", "XOP")}
print("=" * 78)
print("C9  energy leadership r21>=83 + print in hold   h=3 lag=1")
print("=" * 78)
print(f"live r21: XLE {rk['XLE'].iloc[-1]:.1f}  XOP {rk['XOP'].iloc[-1]:.1f}")

for t, cost in (("XLE", 3.0), ("XOP", 4.0)):
    m = (rk[t] >= 83) & pr
    battery(px, m, [(t, 1.0)], H, f"C9  long {t}, r21>=83 + print", cost_bps=cost,
            event_kinds=("cpi",),
            variants={
                "r21 in [83,90) LIVE-XLE": (rk[t] >= 83) & (rk[t] < 90) & pr,
                "r21 in [90,95) LIVE-XOP": (rk[t] >= 90) & (rk[t] < 95) & pr,
                "r21 in [95,100]": (rk[t] >= 95) & pr,
                "r21>=70 + print": (rk[t] >= 70) & pr,
                "r21>=83 + print (headline)": m,
                "r21>=83, NO print": (rk[t] >= 83) & ~pr,
            })
    # market-relative: is leadership just beta?
    battery(px, m, [(t, 1.0), ("SPY", -1.0)], H,
            f"C9  {t} MINUS SPY, same mask (is leadership just beta?)",
            cost_bps=cost + 3.0, event_kinds=("cpi",))

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION both ways + rank-band ladder (left-open rule)")
print("=" * 78)
for t in ("XLE", "XOP"):
    ret = vehicle_ret(px, [(t, 1.0)], H, LAG)
    valid = ret.notna()
    rows = []
    bands = [(0, 50), (50, 70), (70, 83), (83, 90), (90, 95), (95, 101)]
    for lo, hi in bands:
        m = (rk[t] >= lo) & (rk[t] < hi) & pr
        d = IDX[m.values & valid.values]
        if len(d) == 0:
            rows.append({"label": f"[{lo},{hi}) + print", "n": 0}); continue
        e = declusters(d, H, IDX)
        rows.append(summarize(ret.loc[e].values, f"[{lo},{hi}) + print [epi]"))
    for lo, hi in bands:
        m = (rk[t] >= lo) & (rk[t] < hi) & ~pr
        d = IDX[m.values & valid.values]
        if len(d) == 0:
            rows.append({"label": f"[{lo},{hi}) NO print", "n": 0}); continue
        e = declusters(d, H, IDX)
        rows.append(summarize(ret.loc[e].values, f"[{lo},{hi}) NO print [epi]"))
    rows.append(summarize(ret[valid].values, "all days"))
    show(rows, f"{t}  h=3 lag=1  (exclusive bands, both calendar halves)")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLACEBO ANCHOR LADDER k=-5..+5 on the print calendar (mask r21>=83)")
print("=" * 78)
for t in ("XLE", "XOP"):
    ret = vehicle_ret(px, [(t, 1.0)], H, LAG)
    valid = ret.notna()
    rows, means = [], []
    for k in range(-5, 6):
        m = (rk[t] >= 83) & flag_in_window(("ppi", "cpi"), H, LAG, k)
        d = IDX[m.values & valid.values]
        e = declusters(d, H, IDX)
        r = summarize(ret.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r); means.append(r.get("mean_pct", np.nan))
    show(rows, t)
    print(f"  true anchor ranks {int(np.sum(np.asarray(means) >= means[5]))} of 11")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SEPTEMBER / MIDTERM slice of the live cell")
print("=" * 78)
for t in ("XLE", "XOP"):
    ret = vehicle_ret(px, [(t, 1.0)], H, LAG)
    valid = ret.notna()
    m = (rk[t] >= 83) & pr
    d = IDX[m.values & valid.values]
    e = declusters(d, H, IDX)
    sep = pd.DatetimeIndex(e).month == 9
    mid = (pd.DatetimeIndex(e).year % 4) == 2
    show([summarize(ret.loc[e].values, "all episodes"),
          summarize(ret.loc[e[sep]].values, f"September only (N={int(sep.sum())})"),
          summarize(ret.loc[e[mid]].values, f"midterm yrs (N={int(mid.sum())})"),
          summarize(ret.loc[e[sep & mid]].values,
                    f"Sept AND midterm (N={int((sep & mid).sum())})")], t)
    if sep.sum():
        print("  September episode dates:",
              ", ".join(str(x.date()) for x in e[sep]))
