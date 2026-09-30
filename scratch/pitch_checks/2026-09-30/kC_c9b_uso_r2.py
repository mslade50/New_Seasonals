"""C9 round 2: long USO after a >= 4% fall with USO r63 rank >= 70 (entry lag 1).

(1) concentration: 2026 share, drop-best-2, ex-2026; (2) definition neighbours as
BUCKETS (so declustering cannot reshuffle them): drop in [-3,-3.5), [-3.5,-4), [-4,-5),
<= -5, each with r63 >= 70, day-level and episodes; r63 ladder; (3) era / regime:
pre/post 2018, midterm, NFP inside the h=3 hold, SPY above its 200d; (4) gate
attribution per vehicle ex-2026 (does r63 still filter without the 2026 regime?);
(5) session decomposition of h=3.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def own(t: str) -> pd.Series:
    s = close_panel([t])[t].dropna()
    return s[s > 0]


uso, cl, xle, spy = own("USO"), own("CL=F"), own("XLE"), own("SPY")
idx = uso.index
r1 = uso.pct_change()
r63 = pct_rank(uso, 63)
gate = r63 >= 70
drop4 = r1 <= -0.04
sig = (drop4 & gate).fillna(False)


def eps(s, mask, h, lo=None, hi=None):
    ret = fwd_lag(s, h, 1)
    m = mask.reindex(s.index, fill_value=False).fillna(False).astype(bool)
    t = s.index[m.values & ret.notna().values]
    if lo is not None:
        t = t[(t >= lo)]
    if hi is not None:
        t = t[(t < hi)]
    e = declusters(t, h, s.index)
    return ret.loc[e]


def rs(v, lbl):
    s = summarize(v.values, lbl)
    w = int((v > 0).sum())
    s["rec"] = f"{w}-{len(v)-w}"
    s["sign_p"] = sign_test(w, len(v)) if len(v) else np.nan
    return s


H = 3
v = eps(uso, sig, H)
# (1) concentration
print("(1) concentration, USO h=3 episodes")
print("  ", cluster_note(v.index, v.values))
by_y = v.groupby(v.index.year).agg(["sum", "count"])
print("   by year (sum pp, n):", {int(y): (round(100 * r["sum"], 2), int(r["count"])) for y, r in by_y.iterrows()})
tot = v.sum()
print(f"   2026 share of total: {100*v[v.index.year==2026].sum()/tot:.0f}%")
ex = v[v.index.year < 2026]
srt = np.sort(v.values)[::-1]
show([rs(v, "all"), rs(ex, "ex-2026"), rs(pd.Series(srt[2:]), "drop best 2"),
      rs(v[v.index.year != 2022], "ex-2022"), rs(v[(v.index.year < 2022)], "pre-2022")],
     "concentration views (USO h=3)")
plain_ex = eps(uso, drop4, H, hi="2026-01-01")
comp_ex = eps(uso, drop4 & ~gate, H, hi="2026-01-01")
print(f"   ex-2026 controls: plain -4% {100*plain_ex.mean():+.3f}% (n {len(plain_ex)}), "
      f"complement {100*comp_ex.mean():+.3f}% (n {len(comp_ex)})")

# (2) buckets
print("\n(2) drop-size buckets with r63 >= 70 (USO long, h=3)")
rows = []
for lbl, lo_, hi_ in (("(-3.0,-2.5]", -0.030, -0.025), ("(-3.5,-3.0]", -0.035, -0.030),
                      ("(-4.0,-3.5]", -0.040, -0.035), ("(-5.0,-4.0]", -0.050, -0.040),
                      ("<= -5.0", -1.0, -0.050)):
    m = gate & (r1 <= hi_) & (r1 > lo_)
    ret = fwd_lag(uso, H, 1)
    d = ret[m.fillna(False) & ret.notna()]
    r = rs(d, f"bucket {lbl} day-level")
    r["ex2026_mean"] = 100 * d[d.index.year < 2026].mean()
    r["n_ex2026"] = int((d.index.year < 2026).sum())
    rows.append(r)
    # same bucket WITHOUT the gate
    m2 = (r1 <= hi_) & (r1 > lo_)
    d2 = ret[m2.fillna(False) & ret.notna()]
    r2 = rs(d2, f"   same bucket, no gate")
    rows.append(r2)
show(rows)
rows = []
for q in (50, 60, 70, 80, 90):
    vv = eps(uso, drop4 & (r63 >= q), H)
    r = rs(vv, f"-4% & r63 >= {q}")
    r["ex2026"] = 100 * vv[vv.index.year < 2026].mean()
    rows.append(r)
for thr in (-0.03, -0.035, -0.04, -0.045, -0.05):
    vv = eps(uso, (r1 <= thr) & gate, H)
    r = rs(vv, f"<= {100*thr:.1f}% & r63 >= 70")
    r["ex2026"] = 100 * vv[vv.index.year < 2026].mean()
    rows.append(r)
show(rows, "threshold ladders (episodes)")

# (3) era / regime
print("\n(3) era / regime (USO h=3 episodes)")
spy200 = (spy > spy.rolling(200).mean()).reindex(idx).fillna(False)
nfp_in = pd.Series(event_in_window(v.index, idx, H, 1, ("nfp",)), index=v.index)
rows = [rs(v[v.index < "2018"], "pre-2018"), rs(v[v.index >= "2018"], "2018+"),
        rs(v[v.index.year % 4 == 2], "midterm"), rs(v[v.index.year % 4 != 2], "non-midterm"),
        rs(v[nfp_in.values], "NFP inside h=3 hold (live case)"), rs(v[~nfp_in.values], "no NFP in hold"),
        rs(v[spy200.reindex(v.index).values], "SPY > 200d (live)"),
        rs(v[~spy200.reindex(v.index).values], "SPY < 200d")]
show(rows)
exn = v[(v.index.year < 2026) & nfp_in.values]
print(f"   NFP-in-hold ex-2026: {100*exn.mean():+.3f}% on {len(exn)}")
# NFP-in-hold on the plain -4% day family, to see if the NFP split is a family fact
pv = eps(uso, drop4, H)
pn = pd.Series(event_in_window(pv.index, idx, H, 1, ("nfp",)), index=pv.index)
print(f"   plain -4% family: NFP in hold {100*pv[pn.values].mean():+.3f}% (n {int(pn.sum())}) vs out "
      f"{100*pv[~pn.values].mean():+.3f}%")

# (4) gate attribution ex-2026, three vehicles
print("\n(4) gate attribution ex-2026 (episodes; the gate must filter without the 2026 regime)")
rows = []
for veh, s in (("USO", uso), ("CL=F", cl), ("XLE", xle)):
    for h in (2, 3, 5):
        g_ = eps(s, sig, h, hi="2026-01-01")
        p_ = eps(s, drop4, h, hi="2026-01-01")
        c_ = eps(s, drop4 & ~gate, h, hi="2026-01-01")
        rows.append({"veh": veh, "h": h, "gated": 100 * g_.mean(), "gated_rec": rs(g_, "")["rec"],
                     "plain": 100 * p_.mean(), "complement": 100 * c_.mean(),
                     "gate_minus_comp_pp": 100 * (g_.mean() - c_.mean())})
show(rows)

# (5) session decomposition
print("\n(5) per-session decomposition, USO h=5 path (episodes):")
paths = episode_paths(pd.DataFrame({"USO": uso}), v.index, [("USO", 1.0)], 5, lag=1)
sess = paths.diff(axis=1)
sess[1] = paths[1]
print("   all: mean by session", (100 * sess.mean()).round(3).to_dict())
pe = paths[paths.index.year < 2026]
se = pe.diff(axis=1)
se[1] = pe[1]
print("   ex-2026: mean by session", (100 * se.mean()).round(3).to_dict())
