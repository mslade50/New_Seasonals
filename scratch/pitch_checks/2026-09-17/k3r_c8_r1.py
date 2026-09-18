"""K3R c8 round 1 - LONG IEF / SHORT 0.523 TLT after a belly-led five-day
selloff: IEF 5d pct_rank <= 5 AND TLT 5d pct_rank >= 20. Signal 2026-09-16
(live per k3r_live_out.txt: IEF 3.17, TLT 21.83).

Episodes = filter then decluster gap 10. Controls: all days, local +/-126,
TDOM-MATCHED (mandatory for rates), parent IEF r5<=5 without the TLT gate.
Era split pre-2018 / 2018-21 / 2022-23 / 2024+. Midterm. Cost 4.4 bps two-leg
round trip incl borrow (registry basis for this pair). Neighbour table at h=5
and h=8. filter_vs_reanchor vs parent IEF r5<=5.
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
px = close_panel(["IEF", "TLT"]).dropna()
px = px[px.index <= BAR]
D = px.index
LEGS = [("IEF", 1.0), ("TLT", -0.523)]
COST = 4.4
ri = pct_rank(px["IEF"], 5)
rt = pct_rank(px["TLT"], 5)
cell = (ri <= 5) & (rt >= 20)
parent = ri <= 5
tdom = pd.Series(D.to_period("M"), index=D).groupby(D.to_period("M")).cumcount() + 1
print(f"span {D[0].date()}..{D[-1].date()}  cell days {int(cell.sum())}  parent days {int(parent.sum())}")


def ep_stats(pr, m, lbl, gap=10):
    valid = pr.dropna().index
    t = D[m.reindex(D, fill_value=False).values].intersection(valid)
    e = declusters(t, gap, D)
    v = pr.loc[e].values
    r = summarize(v, lbl)
    if not r["n"]:
        return r, e
    ctl = pr.loc[valid].mean()
    # tdom-matched control: mean over non-trigger days with the same tdom, per episode
    nt = valid.difference(t)
    tm = pr.loc[nt].groupby(tdom.loc[nt].values).mean()
    tctl = np.nanmean([tm.get(tdom[d], np.nan) for d in e])
    r["ctl_all_bps"] = 1e4 * ctl
    r["ctl_tdom_bps"] = 1e4 * tctl
    r["mean_bps"] = 1e4 * np.mean(v)
    r["edge_tdom_bps"] = 1e4 * (np.mean(v) - tctl)
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = sign_test(w, len(v))
    r["x_cost"] = 1e4 * np.mean(v) / COST
    return r, e


rows = []
for h in (1, 2, 3, 5, 8, 10):
    pr = vehicle_ret(px, LEGS, h)
    r, e = ep_stats(pr, cell, f"h={h} CELL")
    rows.append(r)
    r, _ = ep_stats(pr, parent, f"h={h} parent IEF r5<=5")
    rows.append(r)
    r, _ = ep_stats(pr, (ri <= 5) & (rt < 20), f"h={h} complement TLT r5<20")
    rows.append(r)
show(rows, "c8 pair IEF - 0.523 TLT (episodes gap 10, bps columns)")

for h in (3, 5, 8):
    pr = vehicle_ret(px, LEGS, h)
    r, e = ep_stats(pr, cell, "x")
    v = pr.loc[e].values
    eras = [("pre-2018", e < pd.Timestamp("2018-01-01")),
            ("2018-2021", (e >= pd.Timestamp("2018-01-01")) & (e < pd.Timestamp("2022-01-01"))),
            ("2022-2023", (e >= pd.Timestamp("2022-01-01")) & (e < pd.Timestamp("2024-01-01"))),
            ("2024+", e >= pd.Timestamp("2024-01-01")),
            ("midterm", e.year % 4 == 2), ("non-midterm", e.year % 4 != 2)]
    er = []
    for lbl, m in eras:
        s = summarize(v[m], f"h={h} {lbl}")
        if s["n"]:
            s["mean_bps"] = 1e4 * np.mean(v[m])
            s["wins"] = int((v[m] > 0).sum())
        er.append(s)
    show(er, f"era / midterm split h={h}")
    print(f"  worst episode h={h}: {1e4*v.min():+.1f} bps on {e[int(np.argmin(v))].date()}")
    print(f"  concentration h={h}: {cluster_note(e, v)}")
    if h == 5:
        loc = local_control(pr.dropna().index, D[cell.values].intersection(pr.dropna().index))
        print(f"  local +/-126 control h=5: {1e4*pr.loc[loc].mean():+.2f} bps")
        print("  episodes:", ", ".join(f"{d.date()} {1e4*pr[d]:+.0f}" for d in e))

# neighbours
for h in (5, 8):
    pr = vehicle_ret(px, LEGS, h)
    nb = []
    for iefc in (3, 5, 10):
        for tltc in (10, 20, 30, 40):
            r, _ = ep_stats(pr, (ri <= iefc) & (rt >= tltc), f"IEF<={iefc} TLT>={tltc}")
            nb.append({k: r.get(k) for k in ("label", "n", "mean_bps", "edge_tdom_bps", "rec", "sign_p", "x_cost")})
    show(nb, f"definition neighbours h={h}")

# filter vs re-anchor vs parent IEF r5<=5 (declustered anchors)
for h in (5, 8):
    pr = vehicle_ret(px, LEGS, h)
    valid = pr.dropna().index
    pe = declusters(D[parent.values].intersection(valid), 10, D)
    ce = declusters(D[cell.values].intersection(valid), 10, D)
    pm = pd.Series(D.isin(pe), index=D)
    cm = pd.Series(D.isin(ce), index=D)
    out = filter_vs_reanchor(pr, pm, cm, D, window_td=21, label=f"h={h} cell vs parent IEF r5<=5")
    if out["shifts"]:
        print(f"  shifts: median {np.median(out['shifts']):.0f}, zero-shift {sum(1 for s in out['shifts'] if s == 0)} of {len(out['shifts'])}")
        rn = reanchor_null(pr, [a for a, _, _ in out["pairs"]], out["shifts"], D,
                           child_mean=pr.reindex([b for _, b, _ in out["pairs"]]).mean())
        print(f"  reanchor_null p {rn['p']:.3f}")

# live-pair state: the pair's own 5d return percentile (is this just a curve-move rank?)
p5 = px["IEF"].pct_change(5) - 0.523 * px["TLT"].pct_change(5)
prank = p5.rolling(252).rank(pct=True) * 100
print(f"\nlive pair 5d return {1e4*p5.iloc[-1]:+.1f} bps, trailing-252 rank {prank.iloc[-1]:.1f}")
for h in (5, 8):
    pr = vehicle_ret(px, LEGS, h)
    for lbl, m in [("pair 5d rank<=5", prank <= 5), ("pair 5d rank<=5 & CELL", (prank <= 5) & cell),
                   ("CELL & pair rank>5", cell & (prank > 5))]:
        r, _ = ep_stats(pr, m, f"h={h} {lbl}")
        print({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k))
               for k in ("label", "n", "mean_bps", "edge_tdom_bps", "rec", "sign_p", "x_cost")})
