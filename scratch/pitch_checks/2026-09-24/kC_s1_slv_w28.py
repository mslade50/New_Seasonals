"""S1 round 1: short SLV after the metals complex breaks together (watchlist 28).

Faithful re-run of 2026-08-31 b4c_c19_slv_short_teardown.py on the tape through
2026-09-23. W28's CELL is GLD, SLV, GDX each <= -2% on the same session, short
SLV MOC next close. Its arm: (a) depth bucket SLV <= -4% reaching 46-36 from
35-36 (sign p vs SLV's own down-rate), or (b) family excess; AND the lag profile
must show the effect at lag 0 or lag 2 as well as lag 1.

Questions:
 0. Which legs of the arm cleared on 2026-09-23 (GLD -1.80% vs the -2% leg).
 1. Faithful cell record, lag 0/1/2, through 09-23 (does the 08-28 firing add?).
 2. Today's ACTUAL configuration (SLV & GDX <= -4%, GLD in (-2%, -1.5%]) and the
    GLD -1.5% neighbour (a walk, charged), with lag profiles.
 3. battery() on the faithful cell h=1, and on the relaxed-GLD walk h=1.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-23")
GAP = 5
BASE = ["GLD", "SLV", "GDX", "SPY", "DX-Y.NYB", "^TNX"]
px = close_panel(BASE).dropna().loc[:BAR]
print(f"panel {px.index[0].date()} .. {px.index[-1].date()} n={len(px)}")
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in BASE}
LEGS = [("SLV", -1.0)]

print("\n0. LIVE VERIFY, last 20 sessions")
tail = pd.DataFrame({t: 100 * r1[t] for t in ["GLD", "SLV", "GDX", "DX-Y.NYB"]}).tail(20)
tail["W28_cell"] = (tail[["GLD", "SLV", "GDX"]] <= -2.0).all(axis=1)
print(tail.round(2).to_string())
d = px.index[-1]
legs_ok = {t: bool(r1[t].iloc[-1] <= -0.02) for t in ["GLD", "SLV", "GDX"]}
print(f"  {d.date()}: legs <= -2%: {legs_ok}; SLV depth arm <= -4%: "
      f"{bool(r1['SLV'].iloc[-1] <= -0.04)}")
_s = load_prices(["SLV"])["SLV"].loc[:BAR]
atr = float(wilder_atr(_s["High"], _s["Low"], _s["Close"])[-1])
print(f"  SLV close {px['SLV'].iloc[-1]:.2f}  Wilder-14 ATR {atr:.3f} "
      f"({100*atr/px['SLV'].iloc[-1]:.2f}%)")


def cell(mask, h, label, lag=1, gap=GAP):
    r = vehicle_ret(px, LEGS, h, lag)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}
    epi = declusters(days, gap, px.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    p0 = float((base > 0).mean())
    o = summarize(vals, label)
    o["n_days"] = len(days)
    o["edge_pp"] = round(o["mean_pct"] - 100 * base.mean(), 3)
    o["rec"] = f"{w}-{len(vals)-w}"
    o["p_vs_downrate"] = round(sign_test(w, len(vals), p0), 4)
    o["p_coin"] = round(sign_test(w, len(vals)), 4)
    o["last"] = str(epi[-1].date())
    return o


g = {t: r1[t] <= -0.02 for t in ["GLD", "SLV", "GDX"]}
faithful = g["GLD"] & g["SLV"] & g["GDX"]
today_cfg = (r1["SLV"] <= -0.04) & (r1["GDX"] <= -0.04) & (r1["GLD"] > -0.02) & (r1["GLD"] <= -0.015)
masks = {
    "FAITHFUL W28 (all three <= -2%)": faithful,
    "W28 depth bucket (all three <=-2%, SLV <= -4%)": faithful & (r1["SLV"] <= -0.04),
    "SLV & GDX <= -2%, GLD NOT <= -2% (today's side)": g["SLV"] & g["GDX"] & ~g["GLD"],
    "TODAY CFG: SLV,GDX <= -4%, GLD (-2,-1.5]": today_cfg,
    "WALK: GLD <= -1.5%, SLV,GDX <= -2%": (r1["GLD"] <= -0.015) & g["SLV"] & g["GDX"],
    "WALK: GLD <= -1.5%, SLV <= -4%, GDX <= -2%": (r1["GLD"] <= -0.015) & (r1["SLV"] <= -0.04) & g["GDX"],
    "PARENT SLV <= -4% alone": r1["SLV"] <= -0.04,
}
print("\n1/2. LAG PROFILE at h=1 (episodes gap 5), short SLV")
for lag in (0, 1, 2):
    show([cell(m, 1, k, lag=lag) for k, m in masks.items()], f"h=1 lag={lag}")
for h in (3, 5):
    show([cell(m, h, k, lag=1) for k, m in masks.items()], f"h={h} lag=1")

print("\n  Recent faithful episodes (2026) h=1 lag=1 short returns:")
r = vehicle_ret(px, LEGS, 1, 1)
rl0 = vehicle_ret(px, LEGS, 1, 0)
rec = px.index[faithful.values & (px.index >= "2025-06-01")]
for dd in rec:
    print(f"   {dd.date()}  SLV {100*r1['SLV'][dd]:+.2f}  GLD {100*r1['GLD'][dd]:+.2f}  "
          f"GDX {100*r1['GDX'][dd]:+.2f}  -> lag0 {100*rl0[dd]:+.3f}%  lag1 {100*r[dd]:+.3f}%")

battery(px, faithful, LEGS, 1, "S1 faithful W28 cell, short SLV", cost_bps=3.0,
        variants={"SLV<=-3.5 (others -2)": faithful & (r1["SLV"] <= -0.035),
                  "SLV<=-4.0 (others -2)": faithful & (r1["SLV"] <= -0.04),
                  "SLV<=-4.5 (others -2)": faithful & (r1["SLV"] <= -0.045)},
        min_gap=GAP, event_kinds=("nfp",))
walk = (r1["GLD"] <= -0.015) & g["SLV"] & g["GDX"]
battery(px, walk, LEGS, 1, "S1 WALK GLD<=-1.5% (charged), short SLV", cost_bps=3.0,
        variants={"SLV<=-3.5": walk & (r1["SLV"] <= -0.035),
                  "SLV<=-4.0": walk & (r1["SLV"] <= -0.04),
                  "SLV<=-4.5": walk & (r1["SLV"] <= -0.045)},
        min_gap=GAP, event_kinds=("nfp",))
