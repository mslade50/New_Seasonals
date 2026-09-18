"""K3R c6 round 1 - SHORT VLO against beta x XLE when VLO closes at a trailing
252 high on a session USO falls >= 3%. Signal 2026-09-16 (live per
k3r_live_out.txt).

Pair: -VLO + beta*XLE, beta = rolling-252 OLS of VLO on XLE daily returns,
lagged 1 session. lag=1 MOC entry. Episodes = filter then decluster gap 10.
Rows: cell, parents (VLO 252 high alone; USO<=-3% alone), all-days + local
controls, era split, worst, cost, and the PLACEBO (short the day's best
1-day performer among 12 energy names vs its own beta x XLE on the same
crude-down days). Mechanism rows: forward VLO-minus-USO after trigger.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)
BAR = pd.Timestamp("2026-09-16")
NAMES = ["XLE", "XOP", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB", "MPC", "PSX"]
full = close_panel(NAMES + ["USO"])
full = full[full.index <= BAR]
px = full[["VLO", "XLE", "USO"]].dropna()
D = px.index
print(f"pair panel span {D[0].date()}..{D[-1].date()}  N days {len(D)}")
r1 = full.pct_change()


def beta_vs_xle(tk):
    return (r1[tk].rolling(252).cov(r1["XLE"]) / r1["XLE"].rolling(252).var()).shift(1)


bV = beta_vs_xle("VLO").reindex(D)
vhi = px["VLO"] >= px["VLO"].shift(1).rolling(251).max()
uso1 = px["USO"].pct_change()
cell = vhi & (uso1 <= -0.03)
parA = vhi
parB = uso1 <= -0.03
print(f"cell days {int(cell.sum())}  parent VLO-high days {int(parA.sum())}  parent USO<=-3% days {int(parB.sum())}")
print("cell days:", ", ".join(str(d.date()) for d in D[cell.values]))

COST = 7.0  # VLO 4 + XLE 3 bps round trip
MID = lambda idx: (pd.DatetimeIndex(idx).year % 4 == 2)


def pair_ret(h):
    return -fwd_lag(px["VLO"], h) + bV * fwd_lag(px["XLE"], h)


rows, era_rows = [], []
for h in (1, 2, 3, 5, 10):
    pr = pair_ret(h)
    valid = pr.dropna().index
    ctl = pr.loc[valid].mean()
    for lbl, m in [("CELL", cell), ("parent VLO 252hi", parA), ("parent USO<=-3%", parB)]:
        t = D[m.values].intersection(valid)
        e = declusters(t, 10, D)
        v = pr.loc[e].values
        r = summarize(v, f"h={h} {lbl}")
        if r["n"]:
            r["ctl_all_pct"] = 100 * ctl
            r["edge_pp"] = r["mean_pct"] - 100 * ctl
            w = int((v > 0).sum())
            r["rec"] = f"{w}-{len(v)-w}"
            r["sign_p"] = sign_test(w, len(v))
            r["x_cost"] = 100 * 100 * np.nanmean(v) / COST
        rows.append(r)
        if lbl == "CELL":
            loc = local_control(valid, t)
            r2 = summarize(pr.loc[loc].values, f"h={h} local +/-126 ctl")
            rows.append(r2)
            pre = e < pd.Timestamp("2018-01-01")
            era_rows += [summarize(v[pre], f"h={h} pre-2018"), summarize(v[~pre], f"h={h} 2018+"),
                         summarize(v[MID(e)], f"h={h} midterm"), summarize(v[~MID(e)], f"h={h} non-midterm")]
show(rows, "c6 pair: -VLO + beta*XLE (episodes gap 10)")
show(era_rows, "c6 era / midterm split (cell episodes)")

# episode detail at h=5
pr5 = pair_ret(5)
e5 = declusters(D[cell.values].intersection(pr5.dropna().index), 10, D)
print("\nh=5 episodes:", ", ".join(f"{d.date()} {100*pr5[d]:+.2f}%" for d in e5))
if len(e5):
    print("concentration:", cluster_note(e5, pr5.loc[e5].values))

# leg attribution at h=5
v5 = fwd_lag(px["VLO"], 5)
x5 = bV * fwd_lag(px["XLE"], 5)
va = v5.dropna().index
print(f"\nleg attribution h=5 (excess over own all-days drift): short VLO "
      f"{-100*(v5.loc[e5].mean()-v5.loc[va].mean()):+.3f}pp   long beta*XLE "
      f"{100*(x5.loc[e5].mean()-x5.dropna().mean()):+.3f}pp")

# mechanism: VLO minus USO forward
for h in (1, 3, 5, 10):
    m = fwd_lag(px["VLO"], h) - fwd_lag(px["USO"], h)
    vv = m.dropna().index
    e = declusters(D[cell.values].intersection(vv), 10, D)
    print(f"mechanism h={h}: fwd VLO-USO after cell {100*m.loc[e].mean():+.3f}% (n={len(e)}) "
          f"vs all days {100*m.loc[vv].mean():+.3f}%  vs USO<=-3% days "
          f"{100*m.loc[declusters(D[parB.values].intersection(vv), 10, D)].mean():+.3f}%  "
          f"(negative = VLO gives back vs crude)")

# PLACEBO: on USO<=-3% days, short the day's best 1d performer vs its own beta*XLE
print("\n=== PLACEBO: short day's best energy name vs beta*XLE on USO<=-3% days ===")
cands = [n for n in NAMES if n != "XLE"]
betas = {n: beta_vs_xle(n) for n in cands}
hi252 = {n: full[n] >= full[n].shift(1).rolling(251).max() for n in cands}
fu = full["USO"].pct_change()
crude_days = full.index[(fu <= -0.03).values]
prow = []
for h in (1, 3, 5, 10):
    best_r, best_hi_r, dates_all, dates_hi, who = [], [], [], [], []
    for d in crude_days:
        day = r1.loc[d, cands].dropna()
        if day.empty:
            continue
        b = day.idxmax()
        pr = -fwd_lag(full[b], h) + betas[b] * fwd_lag(full["XLE"], h)
        val = pr.get(d, np.nan)
        if np.isnan(val):
            continue
        dates_all.append(d); best_r.append(val); who.append(b)
        if bool(hi252[b].get(d, False)):
            dates_hi.append(d); best_hi_r.append(val)
    sa = pd.Series(best_r, index=pd.DatetimeIndex(dates_all))
    ea = declusters(sa.index, 10, full.index)
    r = summarize(sa.loc[ea].values, f"h={h} PLACEBO best name (any)")
    w = int((sa.loc[ea] > 0).sum()); r["rec"] = f"{w}-{len(ea)-w}"; r["sign_p"] = sign_test(w, len(ea))
    prow.append(r)
    if dates_hi:
        sh = pd.Series(best_hi_r, index=pd.DatetimeIndex(dates_hi))
        eh = declusters(sh.index, 10, full.index)
        r = summarize(sh.loc[eh].values, f"h={h} PLACEBO best name AT 252 high")
        w = int((sh.loc[eh] > 0).sum()); r["rec"] = f"{w}-{len(eh)-w}"; r["sign_p"] = sign_test(w, len(eh))
        prow.append(r)
    if h == 5:
        print("best-name frequency:", pd.Series(who).value_counts().to_dict())
show(prow, "placebo (episodes gap 10)")

# any-name 252 high on crude-down day, excluding VLO (reference class)
print("\n=== reference class: each name at 252 high on USO<=-3% day, short vs beta*XLE, h=5 ===")
rc = []
for n in cands:
    pr = -fwd_lag(full[n], 5) + betas[n] * fwd_lag(full["XLE"], 5)
    m = hi252[n] & (fu <= -0.03)
    t = full.index[m.fillna(False).values].intersection(pr.dropna().index)
    e = declusters(t, 10, full.index)
    r = summarize(pr.loc[e].values, n)
    if r["n"]:
        w = int((pr.loc[e] > 0).sum()); r["rec"] = f"{w}-{r['n']-w}"
    rc.append(r)
show(rc, "reference class h=5")
