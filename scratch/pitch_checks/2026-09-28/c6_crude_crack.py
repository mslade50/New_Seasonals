"""C6 round 1: short crude on the FIRST <= -3% day after a 21-day thrust.
Pre-specified: USO r21 >= 75 on D, USO 1d <= -3% on D, no other <= -3% day in
D-10..D-1. Short CL=F / USO, h=3 and h=5, lag=1 (MOC 09-28 close).
Live-gate check first: was 09-25 really the FIRST -3% day inside 10 sessions?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def own(t: str) -> pd.DataFrame:
    px = close_panel([t])[[t]].dropna()
    return px[px[t] > 0]


def masks(c: pd.Series, r21_min: float = 75, drop: float = -0.03, look: int = 10,
          r21_prev: bool = False):
    r1 = c.pct_change()
    r21 = pct_rank(c, 21)
    thr = (r21.shift(1) if r21_prev else r21) >= r21_min
    crack = r1 <= drop
    prior = crack.shift(1).rolling(look).sum() > 0
    first = thr & crack & ~prior
    anyc = thr & crack
    return first, anyc, crack, r1, r21


def eps(ret, mask, h, idx):
    s = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, h, idx)
    return e, ret.loc[e].values


uso = own("USO")
c = uso["USO"]
first, anyc, crack, r1, r21 = masks(c)
tail = pd.DataFrame({"close": c, "ret1_pct": 100 * r1, "ret21_pct": 100 * c.pct_change(21),
                     "r21": r21, "crack": crack, "first": first, "any": anyc}).tail(14)
print("LIVE GATE CHECK (USO, last 14 sessions):")
print(tail.round(2).to_string())
cl = own("CL=F")
print("\nCL=F last 14 ret1 %:")
print((100 * cl["CL=F"].pct_change()).tail(14).round(2).to_string())

# translate USO-defined masks onto each vehicle's own index
for veh, cost in (("USO", 4.0), ("CL=F", 1.5)):
    px = own(veh)
    idx = px.index
    for h in (3, 5):
        legs = [(veh, -1.0)]
        battery(px, first.reindex(idx, fill_value=False), legs, h,
                f"C6 short {veh} first crack after r21>=75", cost_bps=cost,
                variants={"ANY crack r21>=75 (live form if not first)": anyc.reindex(idx, fill_value=False),
                          "first crack, r21(D-1)>=75": masks(c, r21_prev=True)[0].reindex(idx, fill_value=False),
                          "first crack r21>=85": masks(c, 85)[0].reindex(idx, fill_value=False),
                          "first crack r21>=65": masks(c, 65)[0].reindex(idx, fill_value=False),
                          "first crack <=-2.5%": masks(c, drop=-0.025)[0].reindex(idx, fill_value=False),
                          "first crack <=-4%": masks(c, drop=-0.04)[0].reindex(idx, fill_value=False),
                          "ALL -3% days, no thrust gate": crack.reindex(idx, fill_value=False)},
                event_kinds=("nfp",))
        ret = vehicle_ret(px, legs, h)
        rows = []
        for lbl, m in (("first crack thrust", first), ("any crack thrust", anyc),
                       ("-3% day no thrust (r21<75)", crack & ~(r21 >= 75)),
                       ("all -3% days", crack)):
            e, v = eps(ret, m, h, idx)
            s = summarize(v, f"{lbl} (N={len(e)})")
            w = int((v > 0).sum())
            s["sign_p"] = sign_test(w, len(v)) if len(v) else np.nan
            rows.append(s)
        show(rows, f"{veh} h={h} gate attribution (SHORT return, episodes)")
