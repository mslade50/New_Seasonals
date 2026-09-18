import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
px = close_panel(["GLD", "GC=F", "^TNX"])
tnx = px["^TNX"].dropna()
r5 = pct_rank(tnx, 5)
hi252 = tnx.rolling(252).max()
hi63 = tnx.rolling(63).max()
DEFS = {
    "L1 r5>=95 & 252max (pitched)": (r5 >= 95) & (tnx >= hi252 - 1e-9),
    "L2 r5>=90 & 252max": (r5 >= 90) & (tnx >= hi252 - 1e-9),
    "L3 r5>=95 & within2% 252max": (r5 >= 95) & (tnx >= 0.98 * hi252),
    "L4 r5>=95 & 63max": (r5 >= 95) & (tnx >= hi63 - 1e-9),
    "L5 r5>=90 & 63max": (r5 >= 90) & (tnx >= hi63 - 1e-9),
    "L6 r5>=95 (no level)": (r5 >= 95),
    "L7 252max (no thrust)": (tnx >= hi252 - 1e-9),
}
GAP = 10  # one episode per FOMC window
cells_walked = 0


def ep_stats(ret, dates, valid, label):
    d = pd.DatetimeIndex(dates).intersection(valid)
    if len(d) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([])
    e = declusters(d, GAP, valid)
    v = ret.loc[e].values
    r = summarize(v, label)
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r, e


tnx_chg = {}
for V in ["GLD", "GC=F"]:
    pv = px[[V]].dropna()
    idx = pv.index
    vr5 = pct_rank(pv[V], 5)
    tv = tnx.reindex(idx).ffill()
    for h in (3, 4, 5):
        ret = vehicle_ret(pv, [(V, 1.0)], h)
        valid = ret.dropna().index
        fin = pd.Series(event_in_window(valid, idx, h, 1, ("fomc_decision",)), index=valid)
        dy = (tv.shift(-(1 + h)) - tv.shift(-1)).reindex(valid) * 100  # bp change over hold
        base_f, _ = ep_stats(ret, valid[fin.values], valid, "FOMC-in-hold ALL (null parent)")
        base_all = summarize(ret.loc[valid].values, "all days (day-level)")
        rows = [base_f, base_all]
        for name, m in DEFS.items():
            t = m.reindex(valid).fillna(False).astype(bool).values
            f = fin.values
            lo = (vr5.reindex(valid) <= 30).fillna(False).values
            rj, ej = ep_stats(ret, valid[t & f], valid, f"{name} & FOMC")
            rn, en = ep_stats(ret, valid[t & ~f], valid, f"{name} & noFOMC")
            rl, el = ep_stats(ret, valid[t & f & lo], valid, f"{name} & FOMC & GLDr5<=30")
            cells_walked += 3
            rj["excess_vs_FOMCall"] = round(rj.get("mean_pct", np.nan) - base_f["mean_pct"], 3) if rj["n"] else np.nan
            rj["tnx_bp_hold"] = round(float(dy.loc[ej].mean()), 1) if len(ej) else np.nan
            rn["tnx_bp_hold"] = round(float(dy.loc[en].mean()), 1) if len(en) else np.nan
            rows += [rj, rn, rl]
        show(rows, f"{V} h={h} (declustered min_gap {GAP})")

print(f"\ncells walked in this ladder: {cells_walked} (7 defs x 3 splits x 3 h x 2 vehicles)")

# mechanism: within thrust (L6, most populated) & FOMC, does gold's return track a yield reversal?
pv = px[["GLD"]].dropna()
idx = pv.index
tv = tnx.reindex(idx).ffill()
for h in (3, 5):
    ret = vehicle_ret(pv, [("GLD", 1.0)], h)
    valid = ret.dropna().index
    fin = pd.Series(event_in_window(valid, idx, h, 1, ("fomc_decision",)), index=valid)
    dy = (tv.shift(-(1 + h)) - tv.shift(-1)).reindex(valid) * 100
    for name in ("L6 r5>=95 (no level)", "L5 r5>=90 & 63max"):
        t = DEFS[name].reindex(valid).fillna(False).astype(bool).values
        e = declusters(valid[t & fin.values], GAP, valid)
        y = dy.loc[e]
        g = ret.loc[e]
        print(f"\nMECH {name} & FOMC h={h}: N={len(e)} TNX falls over hold in {int((y<0).sum())}/{len(e)}; "
              f"corr(GLD ret, dTNX)={np.corrcoef(g, y)[0,1]:+.2f}; GLD when TNX fell "
              f"{100*g[y<0].mean():+.3f}% (n={int((y<0).sum())}) / rose {100*g[y>=0].mean():+.3f}% (n={int((y>=0).sum())})")
        e2 = declusters(valid[t & ~fin.values], GAP, valid)
        y2 = dy.loc[e2]
        print(f"     same thrust, NO FOMC: TNX falls in {int((y2<0).sum())}/{len(e2)}, mean dTNX {y2.mean():+.1f}bp vs FOMC {y.mean():+.1f}bp")
        # midterm split
        mt = np.array([d.year % 4 == 2 for d in e])
        if mt.any():
            print(f"     midterm: {100*g[mt].mean():+.3f}% n={int(mt.sum())}; non-midterm {100*g[~mt].mean():+.3f}% n={int((~mt).sum())}")
        show(era_split(e, g.values), f"era split {name} & FOMC h={h}")
        print("     episodes:", [(str(d.date()), round(100*ret.loc[d], 2)) for d in e])
