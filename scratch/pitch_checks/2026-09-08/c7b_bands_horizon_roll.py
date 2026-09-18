"""Round-2 confirmation for C7 / C8 (and the C9 live bands).

Three jobs:
 1. LEFT-OPEN RULE. The live DBC reading is 0.188% off its 252d high. Quote the
    EXCLUSIVE band it falls in on its own, not just the cumulative <= tol form.
 2. HORIZON SCAN 1..10. A kill has to hold at every horizon, otherwise I have
    only killed one h.
 3. ROLL. Price DBC/USO carry against CL=F front month over the same window,
    because the C7 thesis is "own the barrel" and the vehicle is not the barrel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["DBC", "USO", "XLE", "XOP", "TLT", "IEF", "SPY"])
raw = load_prices(["DBC", "USO", "XLE", "XOP", "CL=F"])
IDX = px.index


def dist_to_high(t, n=252):
    s = raw[t]["Close"]
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (1.0 - s / hi).reindex(IDX)


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


LAG = 1
d = dist_to_high("DBC")
live = float(d.iloc[-1])
print("=" * 78)
print(f"1. EXCLUSIVE DBC-distance bands (live reading {100*live:.3f}%), h=3, print in hold")
print("=" * 78)
pr3 = flag_in_window(("ppi", "cpi"), 3, LAG)
for tkr, ret_legs in (("DBC", [("DBC", 1.0)]), ("USO", [("USO", 1.0)]),
                      ("TLT-short", [("TLT", -1.0)]), ("IEF-short", [("IEF", -1.0)])):
    ret = vehicle_ret(px, ret_legs, 3, LAG)
    valid = ret.notna()
    rows = []
    edges = [0.0, 0.0010, 0.0025, 0.0050, 0.0100, 0.0200, 0.0500]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi) & pr3
        dd = IDX[m.values & valid.values]
        if len(dd) == 0:
            rows.append({"label": f"[{100*lo:.2f}%,{100*hi:.2f}%)", "n": 0}); continue
        e = declusters(dd, 3, IDX)
        lbl = f"[{100*lo:.2f}%,{100*hi:.2f}%)"
        if lo <= live < hi:
            lbl += "  <-- LIVE BAND"
        rows.append(summarize(ret.loc[e].values, lbl))
    rows.append(summarize(ret[valid].values, "all days"))
    show(rows, f"{tkr}  exclusive bands x print-in-hold, episode level")

print("\n" + "=" * 78)
print("2. HORIZON SCAN 1..10 on the exact live cells (episodes, lag=1)")
print("=" * 78)
for lbl, legs, mask_fn in (
        ("C7 DBC  (DBC<=0.25% off high + print)", [("DBC", 1.0)], "dbc"),
        ("C7 USO  (same state)", [("USO", 1.0)], "dbc"),
        ("C8 SHORT TLT (same state)", [("TLT", -1.0)], "dbc"),
        ("C8 SHORT IEF (same state)", [("IEF", -1.0)], "dbc"),
        ("C9 XLE  (r21>=83 + print)", [("XLE", 1.0)], "xle"),
        ("C9 XOP  (r21>=83 + print)", [("XOP", 1.0)], "xop")):
    rows = []
    for h in range(1, 11):
        pr = flag_in_window(("ppi", "cpi"), h, LAG)
        if mask_fn == "dbc":
            m = (d <= 0.0025) & pr
        else:
            rk = pct_rank(raw[mask_fn.upper() if mask_fn != "xle" else "XLE"]["Close"], 21).reindex(IDX) \
                 if mask_fn == "xle" else pct_rank(raw["XOP"]["Close"], 21).reindex(IDX)
            m = (rk >= 83) & pr
        ret = vehicle_ret(px, legs, h, LAG)
        valid = ret.dropna().index
        t = IDX[m.reindex(IDX, fill_value=False).values].intersection(valid)
        e = declusters(t, h, valid)
        r = summarize(ret.loc[e].values, f"h={h}")
        base = ret.loc[valid].mean()
        if r["n"]:
            r["ctl_all_days_pct"] = round(100 * base, 3)
            r["edge_pct"] = round(r["mean_pct"] - 100 * base, 3)
        rows.append(r)
    show(rows, lbl)
    eds = [r.get("edge_pct", np.nan) for r in rows]
    pos = int(np.sum(np.asarray(eds, dtype=float) > 0))
    print(f"  edge over own all-days drift is POSITIVE at {pos} of 10 horizons; "
          f"max edge {np.nanmax(eds):+.3f}pp, min {np.nanmin(eds):+.3f}pp")

print("\n" + "=" * 78)
print("3. ROLL: the vehicle is not the barrel")
print("=" * 78)
cl = raw["CL=F"]["Close"]
for t in ("USO", "DBC"):
    s = raw[t]["Close"]
    j = pd.concat({"v": s, "c": cl}, axis=1).dropna()
    j = j[j.index >= pd.Timestamp("2007-01-01")]
    yrs = (j.index[-1] - j.index[0]).days / 365.25
    vr = (j["v"].iloc[-1] / j["v"].iloc[0]) ** (1 / yrs) - 1
    cr = (j["c"].iloc[-1] / j["c"].iloc[0]) ** (1 / yrs) - 1
    d3 = (s.shift(-4) / s.shift(-1) - 1).mean()
    c3 = (cl.shift(-4) / cl.shift(-1) - 1).mean()
    print(f"{t}: CAGR {100*vr:+.2f}%/yr vs CL=F front {100*cr:+.2f}%/yr over "
          f"{yrs:.1f}y  => roll+fee drag {100*(vr-cr):+.2f}%/yr "
          f"({10000*(vr-cr)/252*3:+.1f} bps per 3-session hold)")
    print(f"   unconditional 3-session lag-1 drift: {t} {10000*d3:+.1f} bps, "
          f"CL=F {10000*c3:+.1f} bps")
