"""Refutation probes 1.9 / 1.10: family 5d raw-candidate flow thresholds (dip_buy >= 6, oversold_hold >= 7,
short_fade >= 104) are in-sample terciles. Recompute walk-forward per-year terciles from prior years only and
report the hi-flow cell's edge, the 1.2x up-size at equal risk, episode counts, and era/regime concentration.
Uses flow_trades_candidates.parquet (trades with f5 attached from raw engine candidates, 2005+).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[1]
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
tr = pd.read_parquet(D / "flow_trades_candidates.parquet").sort_values(["Signal Date", "Strategy", "Ticker"]).reset_index(drop=True)
print("columns:", [c for c in tr.columns if c in ("f5", "s1", "ep", "cap_scale", "family", "year", "R", "PnL", "Risk")])
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
tr["dial"] = frag["63d"].rolling(10).mean().shift(1).reindex(tr["Signal Date"]).values
tr["day"] = tr["Signal Date"].dt.strftime("%Y%m%d")
FAM = ["dip_buy", "oversold_hold", "short_fade"]
THR_IS = {"dip_buy": 6, "oversold_hold": 7, "short_fade": 104}


def cl_diff(g, mask, cl="ep"):
    a, b = g[mask], g[~mask]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    x = np.where(mask, 1.0, 0.0); X = np.column_stack([np.ones(len(g)), x])
    XtX = np.linalg.inv(X.T @ X); beta = XtX @ X.T @ g.R.values; e = g.R.values - X @ beta
    meat = np.zeros((2, 2))
    for c in np.unique(g[cl].values):
        m = g[cl].values == c; s = X[m].T @ e[m]; meat += np.outer(s, s)
    G = g[cl].nunique(); V = XtX @ meat @ XtX * G / (G - 1)
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def ppr(g):
    return float(g.PnL.sum() / g.Risk.sum()) if g.Risk.sum() else np.nan


def eval_up(g, hi, mult):
    """up-size hi rows by mult, equal total risk; per-year comparison of PnL."""
    m = np.where(hi, mult, 1.0); risk = g.Risk.values; m = m / ((m * risk).sum() / risk.sum())
    flat = risk * g.R.values; rule = risk * m * g.R.values
    Y = pd.DataFrame(dict(y=g.year.values, f=flat, r=rule)).groupby("y").sum(); d = Y.r - Y.f
    return dict(gain_pct=float(d.sum() / abs(Y.f.sum()) * 100), years_better=int((d > 0).sum()), years=len(Y), worst_year=float(d.min()))


for f in FAM:
    F = tr[tr.family == f].copy()
    print(f"\n==================== {f}  N={len(F)}  years {F.year.min()}-{F.year.max()}")
    # in-sample threshold as shipped
    hi_is = F.f5 >= THR_IS[f]
    if f == "dip_buy":
        hi_is = hi_is & ~(F.dial >= 50)
    d, t = cl_diff(F, hi_is); dd_, td_ = cl_diff(F, hi_is, "day")
    print(f"IN-SAMPLE thr {THR_IS[f]}: hi N {int(hi_is.sum())} avgR {F[hi_is].R.mean():+.3f} ppr {ppr(F[hi_is]):.3f} | rest avgR {F[~hi_is].R.mean():+.3f} ppr {ppr(F[~hi_is]):.3f} | ratio ppr {ppr(F[hi_is]) / ppr(F[~hi_is]):.2f} | diff {d:+.3f} t_ep {t:.2f} t_day {td_:.2f} | hi episodes {F[hi_is].ep.nunique()} hi days {F[hi_is].day.nunique()}")
    OUT.setdefault(f, {})["in_sample"] = dict(N_hi=int(hi_is.sum()), avgR_hi=float(F[hi_is].R.mean()), ppr_hi=ppr(F[hi_is]), avgR_rest=float(F[~hi_is].R.mean()), ppr_rest=ppr(F[~hi_is]),
                                              ppr_ratio=ppr(F[hi_is]) / ppr(F[~hi_is]), diff=d, t_ep=t, t_day=td_, hi_episodes=int(F[hi_is].ep.nunique()), hi_days=int(F[hi_is].day.nunique()))
    # walk-forward thresholds: tercile of f5 over trades in years < y (expanding, 2005+), test years 2010-2026
    hi_wf = pd.Series(False, index=F.index, dtype=bool); thr_by_year = {}
    for y in range(2010, 2027):
        trn = F[(F.year < y)]
        if len(trn) < 60:
            continue
        thr = float(trn.f5.quantile(2 / 3)); thr_by_year[y] = thr
        te = F.year == y
        hi_wf.loc[te] = (F.f5[te] >= np.ceil(thr + 1e-9)).values.astype(bool)
    if f == "dip_buy":
        hi_wf = (hi_wf & ~(F.dial >= 50)).astype(bool)
    Fo = F[F.year >= 2010]; hw = hi_wf[F.year >= 2010].astype(bool); hi_is_o = hi_is[F.year >= 2010]
    print("WF thresholds by year (ceil of prior-years 2/3 quantile):", {k: int(np.ceil(v)) for k, v in thr_by_year.items()})
    d, t = cl_diff(Fo, hw)
    print(f"WALK-FORWARD thr 2010+: hi N {int(hw.sum())} avgR {Fo[hw].R.mean():+.3f} ppr {ppr(Fo[hw]):.3f} | rest avgR {Fo[~hw].R.mean():+.3f} ppr {ppr(Fo[~hw]):.3f} | ratio {ppr(Fo[hw]) / ppr(Fo[~hw]):.2f} | diff {d:+.3f} t {t:.2f} | agreement with IS flag {(hw == hi_is_o).mean():.3f}")
    d2, t2 = cl_diff(Fo, hi_is_o)
    print(f"IN-SAMPLE thr on 2010+: hi N {int(hi_is_o.sum())} avgR {Fo[hi_is_o].R.mean():+.3f} ppr {ppr(Fo[hi_is_o]):.3f} | rest {Fo[~hi_is_o].R.mean():+.3f} ppr {ppr(Fo[~hi_is_o]):.3f} | ratio {ppr(Fo[hi_is_o]) / ppr(Fo[~hi_is_o]):.2f} | diff {d2:+.3f} t {t2:.2f}")
    OUT[f]["walk_forward"] = dict(thresholds={int(k): int(np.ceil(v)) for k, v in thr_by_year.items()}, N_hi=int(hw.sum()), avgR_hi=float(Fo[hw].R.mean()), ppr_hi=ppr(Fo[hw]), avgR_rest=float(Fo[~hw].R.mean()), ppr_rest=ppr(Fo[~hw]),
                                  ppr_ratio=ppr(Fo[hw]) / ppr(Fo[~hw]), diff=d, t=t, agreement_with_is=float((hw == hi_is_o).mean()))
    OUT[f]["in_sample_2010plus"] = dict(N_hi=int(hi_is_o.sum()), ppr_ratio=ppr(Fo[hi_is_o]) / ppr(Fo[~hi_is_o]), diff=d2, t=t2)
    # up-size at equal risk, 2010+: 1.25 (study) / 1.2 (shipped), IS vs WF thresholds
    for lab, mask in [("IS thr", hi_is_o), ("WF thr", hw)]:
        for mk in (1.2, 1.25):
            r = eval_up(Fo, mask.values, mk); print(f"   up-size {mk} on {lab}: {r}"); OUT[f].setdefault("upsize", {})[f"{lab}|{mk}"] = r
    # per-year hi-vs-rest sign with WF thresholds
    yr = Fo.groupby("year").apply(lambda g: pd.Series(dict(n_hi=int(hw[g.index].sum()), hi=g[hw[g.index]].R.mean(), rest=g[~hw[g.index]].R.mean())))
    yr["diff"] = yr.hi - yr.rest
    print("per-year WF hi - rest:", yr["diff"].round(2).to_dict(), "| years hi>rest:", int((yr["diff"] > 0).sum()), "of", int(yr["diff"].notna().sum()))
    OUT[f]["wf_years_positive"] = [int((yr["diff"] > 0).sum()), int(yr["diff"].notna().sum())]
    # era / regime concentration of the hi cell (IS thresholds, full 2005+)
    for lab, keep in [("2005-2015", F.year <= 2015), ("2016-2026", F.year >= 2016), ("ex 2020", F.year != 2020), ("ex 2020+2022", ~F.year.isin([2020, 2022])), ("ex 2026", F.year != 2026), ("2026", F.year == 2026)]:
        g = F[keep]; m = hi_is[keep]; d, t = cl_diff(g, m)
        print(f"   {lab:12s} hi N {int(m.sum()):4d} avgR {g[m].R.mean() if m.sum() else np.nan:+.3f} rest {g[~m].R.mean():+.3f} ratio_ppr {ppr(g[m]) / ppr(g[~m]) if m.sum() and ppr(g[~m]) else np.nan:.2f} diff {d:+.3f} t {t:.2f}")
        OUT[f].setdefault("subsets", {})[lab] = dict(N_hi=int(m.sum()), avgR_hi=float(g[m].R.mean()) if m.sum() else None, avgR_rest=float(g[~m].R.mean()), diff=d, t=t)
    # top-5 hi-flow episodes' share of hi-cell PnL and the hi-cell's worst episode
    hp = F[hi_is].groupby("ep").PnL.sum().sort_values(ascending=False)
    print(f"   hi-flow episodes {len(hp)}; top-5 share of hi PnL {hp.head(5).sum() / hp[hp > 0].sum():.2f}; worst episode ${hp.min():,.0f}; episodes negative {int((hp < 0).sum())}")
    OUT[f]["hi_episode_concentration"] = dict(n=int(len(hp)), top5_share=float(hp.head(5).sum() / hp[hp > 0].sum()), worst=float(hp.min()), n_negative=int((hp < 0).sum()))
    # threshold sensitivity (IS): +-1, +-2
    sens = {}
    for k in [-2, -1, 0, 1, 2]:
        m = F.f5 >= THR_IS[f] + k
        if f == "dip_buy":
            m = m & ~(F.dial >= 50)
        sens[THR_IS[f] + k] = dict(N=int(m.sum()), ratio=float(ppr(F[m]) / ppr(F[~m])), diff=float(cl_diff(F, m)[0]))
    print("   thr sensitivity:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sens.items()}); OUT[f]["thr_sensitivity"] = sens

# ---- 1.9 cap relief: cap-bound trades by year and the share of the cap-bound PnL in the top years
tb = tr[tr.cap_scale < 0.999]
by = tb.groupby("year").agg(n=("R", "size"), pnl=("PnL", "sum"), avgR=("R", "mean"))
print("\ncap-bound trades by year:"); print(by.round(2).T.to_string())
OUT["cap_bound_by_year"] = by.round(3).reset_index().to_dict("records")
print("cap-bound avgR 2026:", round(tb[tb.year == 2026].R.mean(), 3), "N", int((tb.year == 2026).sum()), "| top-3 years share of cap-bound PnL:", round(by.pnl.sort_values(ascending=False).head(3).sum() / by.pnl[by.pnl > 0].sum(), 2))

json.dump(OUT, open(HERE / "d_flow_wf.json", "w"), indent=1, default=float)
print("wrote d_flow_wf.json")
