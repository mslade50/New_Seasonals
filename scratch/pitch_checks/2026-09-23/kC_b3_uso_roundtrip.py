"""B3 round 1: short USO after a crude thrust round-trip.
Trigger: USO r5 <= 3 AND r21 >= 90 on some session in the prior 10 sessions.
Pre-specified sign: SHORT continuation, h=5, lag=1 (MOC tomorrow).
Controls: own drift / all days / local (battery) + MATCHED: r5 <= 3 flush
WITHOUT the prior thrust. Era: pre/post 2018 and pre/post 2020-05 (USO roll
methodology change + 1:8 reverse split -> potentially two instruments)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def state(t):
    px = close_panel([t])
    px = px[[t]].dropna()
    c = px[t]
    r5, r21, r63 = pct_rank(c, 5), pct_rank(c, 21), pct_rank(c, 63)
    return px, c, r5, r21, r63


def eps(ret, mask, h, idx):
    s = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, h, idx)
    return e, ret.loc[e].values


def run(t, h=5, look=10, r5max=3, r21min=90, cost=4.0):
    px, c, r5, r21, r63 = state(t)
    prior = r21.shift(1).rolling(look).max()
    trig = (r5 <= r5max) & (prior >= r21min)
    if t == "USO":
        tail = pd.DataFrame({"close": c, "ret5": 100 * c.pct_change(5),
                             "ret21": 100 * c.pct_change(21), "r5": r5,
                             "r21": r21, "r63": r63, "prior10_r21max": prior,
                             "trig": trig}).tail(16)
        print("LIVE STATE CHECK (USO):")
        print(tail.round(2).to_string())
    legs = [(t, -1.0)]
    battery(px, trig, legs, h, f"B3 {t} short after thrust round-trip",
            cost_bps=cost, event_kinds=("nfp",))
    ret = vehicle_ret(px, legs, h)
    idx = px.index
    e_c, v_c = eps(ret, trig, h, idx)
    e_m, v_m = eps(ret, (r5 <= r5max) & (prior < r21min), h, idx)
    e_p, v_p = eps(ret, (r5 <= r5max), h, idx)
    show([summarize(v_c, f"CHILD r5<={r5max} & prior r21>={r21min} (N={len(e_c)})"),
          summarize(v_m, f"MATCHED flush, no prior thrust (N={len(e_m)})"),
          summarize(v_p, f"PARENT any r5<={r5max} flush (N={len(e_p)})")],
         f"{t} gate attribution (short return, episodes, h={h})")
    if len(v_c) > 1 and len(v_m) > 1:
        se = np.sqrt(v_c.var(ddof=1) / len(v_c) + v_m.var(ddof=1) / len(v_m))
        print(f"  child - matched = {100*(v_c.mean()-v_m.mean()):+.3f}pp  "
              f"welch t {(v_c.mean()-v_m.mean())/se:+.2f}")
    if t == "USO":
        show(era_split(e_c, v_c, "2020-05-01"), "USO era split at 2020-05 (child)")
        show(era_split(e_m, v_m, "2020-05-01"), "USO era split at 2020-05 (matched)")
        show(era_split(e_m, v_m), "USO era split at 2018 (matched)")
    for d, v in zip(e_c, v_c):
        print(f"   {d.date()}  short {100*v:+.2f}%")
    return px, trig


if __name__ == "__main__":
    run("USO")
    run("CL=F", cost=1.5)
