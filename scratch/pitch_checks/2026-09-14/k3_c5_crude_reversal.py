import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "USO", "XLE", "XOP", "CL=F"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
for t in raw.columns:
    s = raw[t].dropna()
    print(t, "first", s.index[0].date(), "last", s.index[-1].date(), "n", len(s))


def arm(src, drop=-0.02, near=0.03, rank=90, rank_lag=1):
    s = px[src]
    r1 = s / s.shift(1) - 1
    hi = rolling_on_valid(s, lambda x: x.rolling(252, min_periods=200).max())
    d52 = s / hi - 1
    r21 = pct_rank(s, 21)
    return (r1 <= drop) & (d52 >= -near) & (r21.shift(rank_lag) >= rank), r1, d52, r21


live = pd.Timestamp("2026-09-11")
for src in ["USO", "CL=F"]:
    m, r1, d52, r21 = arm(src)
    print(f"live {src}: r1 {100*r1.loc[live]:+.2f}% d52 {100*d52.loc[live]:+.2f}% r21 {r21.loc[live]:.1f} r21(d-1) {r21.shift(1).loc[live]:.1f} ARM={bool(m.loc[live])}")

mask, r1, d52, r21 = arm("USO")
variants = {
    "drop<=-1.5%": arm("USO", drop=-0.015)[0],
    "drop<=-3%": arm("USO", drop=-0.03)[0],
    "near 5%": arm("USO", near=0.05)[0],
    "near 2%": arm("USO", near=0.02)[0],
    "r21>=80": arm("USO", rank=80)[0],
    "r21>=95": arm("USO", rank=95)[0],
    "r21 same-day>=85": arm("USO", rank=85, rank_lag=0)[0],
    "NO rank gate": arm("USO", rank=-1)[0],
    "NO near-high gate": arm("USO", near=9.9)[0],
}
for tkr, w in [("XLE", 1.0), ("XLE", -1.0), ("XOP", -1.0), ("USO", -1.0)]:
    for h in [1, 3, 5]:
        battery(px, mask, [(tkr, w)], h, f"C5 USO reversal day -> {'LONG' if w>0 else 'SHORT'} {tkr}", 3.5,
                variants=variants if h == 3 else None, min_gap=5, event_kinds=("fomc_decision",))

# CL=F source trigger, forward on XLE/XOP (futures-based definition, no roll bias in the gate)
if "CL=F" in px:
    mcl = arm("CL=F")[0]
    for tkr, w in [("XLE", -1.0), ("XOP", -1.0), ("CL=F", -1.0)]:
        battery(px, mcl, [(tkr, w)], 3, f"C5 CL=F reversal day -> SHORT {tkr}", 3.5, min_gap=5,
                event_kinds=("fomc_decision",))

# gate attribution: USO -2% days NOT near high / not after thrust, forward XLE, USO
rows = []
for lbl, m in [("CELL", mask), ("-2% day, no gates", r1 <= -0.02),
               ("-2% day & NOT (near-high & thrust)", (r1 <= -0.02) & ~mask),
               ("near-high & thrust, no -2% day", (d52 >= -0.03) & (r21.shift(1) >= 90) & (r1 > -0.02))]:
    for tkr in ["XLE", "XOP", "USO"]:
        for h in [1, 3, 5]:
            r = vehicle_ret(px, [(tkr, 1.0)], h)
            d = cal[(m & r.notna()).reindex(cal, fill_value=False).values]
            d = declusters(d, 5, cal)
            s = summarize(r.loc[d].values, f"{lbl} | long {tkr} h={h}")
            rows.append(s)
show(rows, "gate attribution (LONG returns; short = negative)")
for tkr in ["XLE", "XOP", "USO"]:
    for h in [1, 3, 5]:
        print(f"all-days long {tkr} h={h}: {100*vehicle_ret(px, [(tkr, 1.0)], h).mean():+.3f}%")
