import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICK = ["GLD", "GC=F", "^TNX", "DX-Y.NYB"]
px = close_panel(TICK)

tnx = px["^TNX"].dropna()
r5 = pct_rank(tnx, 5)
hi = tnx.rolling(252).max()
at_max = tnx >= hi - 1e-9
thrust = (r5 >= 95) & at_max

print("LIVE check 2026-09-11: TNX", tnx.iloc[-1], "r5", round(r5.iloc[-1], 1),
      "at_max", bool(at_max.iloc[-1]), "last date", tnx.index[-1].date())
print("  last 6 at_max:", at_max.tail(6).astype(int).tolist(), " r5:",
      r5.tail(6).round(1).tolist())
dx = px["DX-Y.NYB"].dropna()
print("  DX r21", round(pct_rank(dx, 21).iloc[-1], 1))
g = px["GLD"].dropna()
print("  GLD r5", round(pct_rank(g, 5).iloc[-1], 1), " 5d",
      round(100 * (g.iloc[-1] / g.iloc[-6] - 1), 2),
      " off 252 hi", round(100 * (g.iloc[-1] / g.rolling(252).max().iloc[-1] - 1), 1))


def cell(ret, dates, h, label):
    d = pd.DatetimeIndex(dates).intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": label, "n": 0}
    e = declusters(d, h, ret.dropna().index)
    v = ret.loc[e].values
    r = summarize(v, label)
    w = int((v > 0).sum())
    r["n_days"] = len(d)
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


for V in ["GLD", "GC=F"]:
    pv = px[[V]].dropna()
    idx = pv.index
    thr = thrust.reindex(idx).fillna(False).astype(bool)
    vr5 = pct_rank(pv[V], 5)
    for h in (3, 4, 5):
        ret = vehicle_ret(pv, [(V, 1.0)], h)
        valid = ret.dropna().index
        fin = pd.Series(event_in_window(valid, idx, h, 1, ("fomc_decision",)), index=valid)
        t = thr.reindex(valid).values
        f = fin.values
        rows = [
            cell(ret, valid[t & f], h, "JOINT thrust & FOMC-in-hold"),
            cell(ret, valid[t & ~f], h, "thrust, NO FOMC in hold"),
            cell(ret, valid[~t & f], h, "FOMC in hold, NO thrust"),
            cell(ret, valid[f], h, "FOMC in hold, all (09-01 null analog)"),
            cell(ret, valid[t], h, "thrust, all"),
            summarize(ret.loc[valid].values, "CTRL all days (day-level)"),
        ]
        lowg = (vr5.reindex(valid) <= 30).values
        rows.append(cell(ret, valid[t & f & lowg], h, "JOINT & GLD r5<=30"))
        rows.append(cell(ret, valid[t & f & ~lowg], h, "JOINT & GLD r5>30"))
        rows.append(cell(ret, valid[t & ~f & lowg], h, "thrust noFOMC & GLD r5<=30"))
        rows.append(cell(ret, valid[~t & f & lowg], h, "FOMC noThrust & GLD r5<=30"))
        show(rows, f"{V} h={h}")
        # attribution on day-level (overlap-inflated, indicative) and episodes
        J = declusters(valid[t & f], h, valid)
        Fo = declusters(valid[~t & f], h, valid)
        To = declusters(valid[t & ~f], h, valid)
        print(f"  episode welch t: JOINT vs FOMC-noThrust {welch(ret.loc[J], ret.loc[Fo]):+.2f} | "
              f"JOINT vs thrust-noFOMC {welch(ret.loc[J], ret.loc[To]):+.2f}")
        if V == "GLD" and h == 5:
            print("  JOINT episodes:", [(str(d.date()), round(100 * ret.loc[d], 2)) for d in J])

# round-1 battery on the pitched form (GLD, h=5 and h=3)
pg = px[["GLD"]].dropna()
thr_g = thrust.reindex(pg.index).fillna(False).astype(bool)
for h in (3, 5):
    fin = pd.Series(event_in_window(pg.index, pg.index, h, 1, ("fomc_decision",)), index=pg.index)
    mask = thr_g & fin
    r5v = r5.reindex(pg.index)
    atm = at_max.reindex(pg.index).fillna(False).astype(bool)
    variants = {
        "r5>=90 & max": (r5v >= 90) & atm & fin,
        "r5>=97 & max": (r5v >= 97) & atm & fin,
        "r5>=95 (no max)": (r5v >= 95) & fin,
        "max (no r5)": atm & fin,
    }
    battery(pg, mask, [("GLD", 1.0)], h, f"C7 GLD joint h={h}", 3.0,
            variants=variants, event_kinds=("cpi", "nfp"))
