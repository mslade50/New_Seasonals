"""Refutation probe 1.6: OVS bottom-extremity tier (mean of 4 short-window ranks < 94 -> 0.7x).
Recomputes from signal_quality_features.parquet (one row per collapsed OVS trade).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

T = pd.read_parquet(D / "signal_quality_features.parquet")
o = T[T.Strategy == "Overbot Vol Spike"].copy()
o["rank_mean"] = o[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
o = o.dropna(subset=["rank_mean"]).sort_values("Signal Date").reset_index(drop=True)
o["path1"] = (o.gap_atr > 0.25).astype(int)
o["midterm"] = (o.year % 4 == 2).astype(int)


def episodes(dates, gap_td=5):
    d = dates.values.astype("datetime64[D]")
    ep = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        ep[i] = ep[i - 1] + (np.busday_count(d[i - 1], d[i]) > gap_td)
    return ep


o["ep"] = episodes(o["Signal Date"])
o["day"] = o["Signal Date"].dt.strftime("%Y%m%d")


def cl_diff(g, mask, cl="ep"):
    """mean(mask) - mean(~mask) with cluster-robust SE (clusters = cl)."""
    a, b = g[mask], g[~mask]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    d = a.R.mean() - b.R.mean()
    x = np.where(mask, 1.0, 0.0)
    X = np.column_stack([np.ones(len(g)), x])
    XtX = np.linalg.inv(X.T @ X)
    beta = XtX @ X.T @ g.R.values
    e = g.R.values - X @ beta
    meat = np.zeros((2, 2))
    for c in np.unique(g[cl].values):
        m = g[cl].values == c
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = g[cl].nunique()
    V = XtX @ meat @ XtX * G / (G - 1)
    return float(d), float(d / np.sqrt(V[1, 1]))


def cell(g):
    return dict(N=int(len(g)), avgR=float(g.R.mean()) if len(g) else None, win=float(g.win.mean()) if len(g) else None,
                ppr=float(g.PnL_flat_750k.sum() / g.Risk_flat_750k.sum()) if len(g) and g.Risk_flat_750k.sum() else None)


print(f"OVS trades {len(o)}  years {o.year.min()}-{o.year.max()}  episodes {o.ep.nunique()}  signal days {o.day.nunique()}")
lo = o.rank_mean < 94
print("cells:", "<94", cell(o[lo]), ">=94", cell(o[~lo]))
d_ep, t_ep = cl_diff(o, lo, "ep"); d_day, t_day = cl_diff(o, lo, "day")
print(f"binary cut diff {d_ep:+.3f}R  t(episode) {t_ep:.2f}  t(day) {t_day:.2f}  N_ep {o.ep.nunique()}")
OUT["headline"] = dict(cell_lo=cell(o[lo]), cell_hi=cell(o[~lo]), diff=d_ep, t_episode=t_ep, t_day=t_day, n_episodes=int(o.ep.nunique()))

# ---- forking paths: cut-point grid on rank_mean and the alternatives tried in the study
rows = []
for c in range(88, 99):
    m = o.rank_mean < c
    d, t = cl_diff(o, m)
    rows.append(dict(cut=c, N_lo=int(m.sum()), avgR_lo=float(o[m].R.mean()), avgR_hi=float(o[~m].R.mean()), diff=d, t=t))
G = pd.DataFrame(rows); print("\n-- cut-point grid on mean rank --"); print(G.round(3).to_string(index=False))
OUT["cut_grid"] = G.round(4).to_dict("records")
# alternatives enumerated in signal_quality_03/04: filt_extremity terciles; rank_mean fixed 3-tier; rank_min fixed 3-tier; 4 single windows x 3 tiers; gap x ext 2x3; wf on 4 features; R1 with top boost; R1b bottom only
OUT["alternatives_tried_in_study"] = ["filt_extremity terciles (02)", "rank_mean <94/94-97/>=97 (03)", "rank_min <90/90-95/>=95 (03)",
                                       "single windows 2/5/10/21 at <92/92-97/>=97 (04)", "gap x extremity 2x3 (03)", "wf on filt_extremity, rank_mean, book_sig_5td, gap_atr (03)",
                                       "R1 (0.5x + 1.25x top) and R1b (0.5x only) (04)", "R1xR2 with density (04)"]

# ---- LOYO and drop-year robustness on the binary cut
loyo = []
for y in sorted(o.year.unique()):
    g = o[o.year != y]
    d, t = cl_diff(g, g.rank_mean < 94)
    loyo.append(dict(drop_year=int(y), diff=d, t=t))
L = pd.DataFrame(loyo)
print("\nLOYO diff range", L["diff"].min().round(3), "to", L["diff"].max().round(3), "| t min", L.t.min().round(2))
OUT["loyo_min_diff"] = float(L["diff"].min()); OUT["loyo_min_t"] = float(L.t.min())
for lab, keep in [("ex 2020+2022", ~o.year.isin([2020, 2022])), ("ex 2020-2022", ~o.year.isin([2020, 2021, 2022])), ("ex 2026", o.year != 2026),
                  ("2003-2015", o.year <= 2015), ("2016-2026", o.year >= 2016), ("midterm years", o.midterm == 1), ("non-midterm", o.midterm == 0),
                  ("P1 only", o.path1 == 1), ("P2 only", o.path1 == 0), ("Liquid", o.Tier == "Liquid"), ("Overflow", o.Tier == "Overflow")]:
    g = o[keep]
    d, t = cl_diff(g, g.rank_mean < 94)
    print(f"{lab:16s} N {len(g):5d}  <94 {cell(g[g.rank_mean < 94])['avgR']:+.3f} (N {int((g.rank_mean < 94).sum())})  >=94 {cell(g[g.rank_mean >= 94])['avgR']:+.3f}  diff {d:+.3f}  t {t:.2f}")
    OUT.setdefault("subsets", {})[lab] = dict(N=int(len(g)), lo=cell(g[g.rank_mean < 94]), hi=cell(g[g.rank_mean >= 94]), diff=d, t=t)

# per-year cell table
Y = o.groupby("year").apply(lambda g: pd.Series(dict(n=len(g), n_lo=int((g.rank_mean < 94).sum()), lo=g[g.rank_mean < 94].R.mean(), hi=g[g.rank_mean >= 94].R.mean())))
Y["diff"] = Y.lo - Y.hi
print("\nper-year (lo - hi):", Y["diff"].round(2).to_dict()); print("years with lo < hi:", int((Y["diff"] < 0).sum()), "of", Y["diff"].notna().sum())
OUT["per_year"] = Y.round(4).reset_index().to_dict("records")

# ---- shipped form 0.7x vs study 0.5x at equal total risk, per year
def eval_rule(df, mult):
    risk = df.Risk_flat_750k.values; m = mult / ((mult * risk).sum() / risk.sum())
    flat = risk * df.R.values; tier = risk * m * df.R.values
    Yt = pd.DataFrame(dict(y=df.year.values, f=flat, t=tier)).groupby("y").sum()
    d = Yt.t - Yt.f
    return dict(gain_pct=float(d.sum() / abs(Yt.f.sum()) * 100), years_better=int((d > 0).sum()), years=len(Yt), worst_year=float(d.min()),
                ppr_flat=float(flat.sum() / risk.sum()), ppr_rule=float(tier.sum() / (risk * m).sum()), raw_dpnl_no_rescale=float(((mult - 1) * risk * df.R.values).sum()))
for mk, mult in [("0.5x", 0.5), ("0.7x (shipped)", 0.7), ("0.8x", 0.8)]:
    r = eval_rule(o, np.where(o.rank_mean < 94, mult, 1.0)); print(mk, {k: round(v, 3) for k, v in r.items()}); OUT.setdefault("rule_forms", {})[mk] = r
# ex 2020+2022 for the shipped form
r = eval_rule(o[~o.year.isin([2020, 2022])], np.where(o[~o.year.isin([2020, 2022])].rank_mean < 94, 0.7, 1.0)); print("0.7x ex 2020+2022", {k: round(v, 3) for k, v in r.items()}); OUT["rule_forms"]["0.7x ex2020+2022"] = r

# ---- is the cut redundant with the T+1 gap path (P2 already 0.2x)? share of <94 risk that is P1
lo_g = o[lo]
print(f"\n<94 cell: P1 share of trades {lo_g.path1.mean():.2f}, P1 share of risk {lo_g[lo_g.path1 == 1].Risk_flat_750k.sum() / lo_g.Risk_flat_750k.sum():.2f}")
# monotone within P1?
p1 = o[o.path1 == 1]
print("P1 by rank_mean band:", {b: cell(p1[(p1.rank_mean >= a) & (p1.rank_mean < c)]) for b, (a, c) in {"<90": (0, 90), "90-94": (90, 94), "94-97": (94, 97), ">=97": (97, 101)}.items()})
OUT["p1_bands"] = {b: cell(p1[(p1.rank_mean >= a) & (p1.rank_mean < c)]) for b, (a, c) in {"<90": (0, 90), "90-94": (90, 94), "94-97": (94, 97), ">=97": (97, 101)}.items()}
# episode concentration: share of the (hi - lo) sumR gap in the top 5 episodes
o["contrib"] = np.where(lo, -(o.R - o.R.mean()), 0.0)
ec = o.groupby("ep").contrib.sum().sort_values(ascending=False)
print("top-5 episode share of the lo-cell deficit:", round(ec.head(5).sum() / ec[ec > 0].sum(), 3))
OUT["top5_episode_share_of_deficit"] = float(ec.head(5).sum() / ec[ec > 0].sum())

json.dump(OUT, open(HERE / "a_ovs_extremity.json", "w"), indent=1, default=float)
print("wrote a_ovs_extremity.json")
