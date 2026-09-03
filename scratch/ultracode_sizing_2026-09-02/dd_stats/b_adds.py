"""Refutation probes 1.4 (OLV depth ladder) and 1.5 (Weak Close / LT Trend solo 0.8x / adds 1.2x).
Recomputes from within_strategy_adds_features.parquet (per-leg features as built by the study).
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
RNG = np.random.default_rng(7)
df = pd.read_parquet(D / "within_strategy_adds_features.parquet")
print("columns:", [c for c in df.columns if c in ("episode", "n_open", "same_day_prior", "rung_ladder", "residual_mult", "yr", "euler_var")])
df["R"] = df["R_Multiple"]
df["day"] = df["Signal Date"].dt.strftime("%Y%m%d")
df["n_sig_day"] = df.groupby(["Strategy", "Signal Date"]).Ticker.transform("size")


def cl_diff(g, mask, cl="episode"):
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


def cell(g):
    return dict(N=int(len(g)), avgR=float(g.R.mean()) if len(g) else None, win=float((g.R > 0).mean()) if len(g) else None)


def robustness(g, mask, label):
    d, t = cl_diff(g, mask); d_day, t_day = cl_diff(g, mask, "day")
    res = dict(label=label, a=cell(g[mask]), b=cell(g[~mask]), diff=d, t_episode=t, t_day=t_day, n_episodes=int(g.episode.nunique()))
    loyo = []
    for y in sorted(g.yr.unique()):
        h = g[g.yr != y]; dd, tt = cl_diff(h, mask[g.yr != y]); loyo.append((int(y), dd, tt))
    L = pd.DataFrame(loyo, columns=["drop", "diff", "t"]).dropna()
    res["loyo_min_diff"] = float(L["diff"].min()); res["loyo_min_t"] = float(L.t.min()); res["loyo_worst_year"] = int(L.loc[L["diff"].idxmin(), "drop"])
    # drop best episode (by contribution of the masked cell to the gap)
    con = pd.Series(np.where(mask, g.R - g[~mask].R.mean(), 0.0), index=g.index).groupby(g.episode).sum()
    best = con.idxmax(); h = g[g.episode != best]; dd, tt = cl_diff(h, mask[g.episode != best])
    res["drop_best_episode"] = dict(episode=int(best), diff=dd, t=tt, best_share_of_gap=float(con.max() / con[con > 0].sum()))
    for lab, keep in [("2003-2015", g.yr <= 2015), ("2016-2026", g.yr >= 2016), ("ex 2020+2022", ~g.yr.isin([2020, 2022])), ("ex 2026", g.yr != 2026), ("ex 2021", g.yr != 2021)]:
        h = g[keep]; dd, tt = cl_diff(h, mask[keep])
        res.setdefault("subsets", {})[lab] = dict(a=cell(h[mask[keep]]), b=cell(h[~mask[keep]]), diff=dd, t=tt)
    yrs = g.groupby("yr").apply(lambda h: h[mask[h.index]].R.mean() - h[~mask[h.index]].R.mean()).dropna()
    res["years_positive"] = [int((yrs > 0).sum()), int(len(yrs))]
    print(f"\n[{label}] a {res['a']} b {res['b']} diff {d:+.3f} t_ep {t:.2f} t_day {t_day:.2f} | LOYO min diff {res['loyo_min_diff']:+.3f} (drop {res['loyo_worst_year']}) t {res['loyo_min_t']:.2f} | "
          f"drop-best-ep diff {dd if False else res['drop_best_episode']['diff']:+.3f} t {res['drop_best_episode']['t']:.2f} share {res['drop_best_episode']['best_share_of_gap']:.2f} | yrs+ {res['years_positive']}")
    for k, v in res["subsets"].items():
        print(f"    {k:12s} a {v['a']['avgR']:+.3f}(N{v['a']['N']}) b {v['b']['avgR']:+.3f}(N{v['b']['N']}) diff {v['diff']:+.3f} t {v['t']:.2f}")
    return res


def eval_rule(g, mult):
    risk = g.Risk_flat_750k.values; m = mult / ((mult * risk).sum() / risk.sum())
    flat = risk * g.R.values; tier = risk * m * g.R.values
    Yt = pd.DataFrame(dict(y=g.yr.values, f=flat, t=tier)).groupby("y").sum(); d = Yt.t - Yt.f
    return dict(gain_pct=float(d.sum() / abs(Yt.f.sum()) * 100), years_better=int((d > 0).sum()), years=len(Yt), worst_year=float(d.min()), best_year=float(d.max()),
                ppr_flat=float(flat.sum() / risk.sum()), ppr_rule=float(tier.sum() / (risk * m).sum()))


# ================================================================ 1.5 Weak Close / LT Trend
for s in ["Weak Close Decent Sznls", "LT Trend ST OS"]:
    g = df[df.Strategy == s].copy().reset_index(drop=True)
    print(f"\n==================== {s} N={len(g)} episodes={g.episode.nunique()} signal-days={g.day.nunique()}")
    adds = g.n_open >= 1
    OUT.setdefault(s, {})["adds_vs_solo"] = robustness(g, adds, "adds(n_open>=1) vs solo")
    # same-day ordering artefact: on a k-signal day the study labels leg 1 'solo' and legs 2..k 'adds' by trade_id order
    sd = g.n_sig_day >= 2
    print(f"   same-day cluster days: {int(sd.sum())} legs on {g[sd].day.nunique()} days; of the {int(adds.sum())} adds, {int((adds & (g.same_day_prior >= 1)).sum())} are same-day adds; "
          f"'solo' legs that sit on a cluster day: {int((~adds & sd).sum())}")
    OUT[s]["sameday_artefact"] = dict(adds=int(adds.sum()), sameday_adds=int((adds & (g.same_day_prior >= 1)).sum()), solo_on_cluster_day=int((~adds & sd).sum()))
    # implementable relabel: cluster-day OR prior open leg (a same-day first leg cannot be told from the second at staging)
    impl = adds | sd
    OUT[s]["implementable_cluster_or_open"] = robustness(g, impl, "cluster-day OR open leg vs true solo")
    prior_only = (g.n_open - g.same_day_prior) >= 1
    OUT[s]["prior_day_open_only"] = robustness(g, prior_only, "prior-day open leg (excl. same-day) vs rest")
    # rule forms at equal risk
    for lab, (a, b) in {"0.75/1.25 (study)": (0.75, 1.25), "0.8/1.2 (shipped)": (0.8, 1.2), "0.8/1.2 on cluster-or-open": (0.8, 1.2)}.items():
        mk = impl if "cluster" in lab else adds
        r = eval_rule(g, np.where(mk, b, a)); print(f"   {lab:28s} {r}"); OUT[s].setdefault("rule_forms", {})[lab] = r
    # Size_Mult basis (WCDS legacy seasonal mult): is 'adds' confounded with Size_Mult?
    if s == "Weak Close Decent Sznls":
        tab = g.groupby(pd.cut(g.Size_Mult, [0, 0.9, 1.1, 2]), observed=True).agg(N=("R", "size"), avgR=("R", "mean"), adds_share=("n_open", lambda x: (x >= 1).mean()))
        print("   by Size_Mult band:\n", tab.round(3).to_string()); OUT[s]["by_size_mult"] = tab.round(4).reset_index().astype(str).to_dict("records")

# ================================================================ 1.4 OLV depth
g = df[df.Strategy == "Oversold Low Volume"].copy().reset_index(drop=True)
print(f"\n==================== OLV N={len(g)} episodes={g.episode.nunique()} signal-days={g.day.nunique()}")
OUT["OLV"] = {}
OUT["OLV"]["depth3_vs_solo"] = robustness(g[(g.n_open >= 3) | (g.n_open == 0)].reset_index(drop=True), (g[(g.n_open >= 3) | (g.n_open == 0)].n_open >= 3).reset_index(drop=True), "depth 3+ vs solo")
OUT["OLV"]["adds_vs_solo"] = robustness(g, g.n_open >= 1, "adds vs solo")
OUT["OLV"]["depth3_vs_rest"] = robustness(g, g.n_open >= 3, "depth 3+ vs rest")
dep = g.groupby(pd.cut(g.n_open, [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"]), observed=True).agg(N=("R", "size"), avgR=("R", "mean"), yrs=("yr", "nunique"), eps=("episode", "nunique"), n2021=("yr", lambda x: (x == 2021).sum()), n2026=("yr", lambda x: (x == 2026).sum()), n2025=("yr", lambda x: (x == 2025).sum()))
print(dep.round(3).to_string()); OUT["OLV"]["depth_table"] = dep.round(4).reset_index().astype(str).to_dict("records")
# the shipped re-key: max(recency rung, depth rung) with filled-only depth (the study variable); per year at equal risk; drop years
resid = g.Size_Mult / g.rung_ladder
new = resid * np.maximum(g.rung_ladder, np.select([g.n_open == 0, g.n_open <= 2], [0.5, 0.7], 1.0))
fac = (new / g.Size_Mult).values
r = eval_rule(g, fac); print("re-key (filled depth) equal-risk:", r); OUT["OLV"]["rekey_equal_risk"] = r
for lab, keep in [("ex 2021", g.yr != 2021), ("ex 2026", g.yr != 2026), ("ex 2021+2025+2026", ~g.yr.isin([2021, 2025, 2026])), ("2016+", g.yr >= 2016)]:
    r = eval_rule(g[keep], fac[keep.values]); print(f"   {lab:18s} {r}"); OUT["OLV"].setdefault("rekey_subsets", {})[lab] = r
# how much of the re-key's raw PnL gain sits in 2021 / 2025 / 2026
raw = pd.Series(g.PnL_flat_750k.values * (fac - 1), index=g.yr).groupby(level=0).sum()
print("raw d_pnl by year (top):", raw.sort_values(ascending=False).head(5).round(0).to_dict(), "| total", round(raw.sum()), "| share 2021+2025+2026:", round(raw[[2021, 2025, 2026]].sum() / raw.sum(), 2))
OUT["OLV"]["rekey_raw_dpnl_share_2021_2025_2026"] = float(raw[[2021, 2025, 2026]].sum() / raw.sum())
# composite clip: how often does tilt 1.17 x depth 1.0 x pullback 1.15 x flow 1.2 exceed 1.5?
OUT["OLV"]["composite_max_product"] = 1.17 * 1.0 * 1.15 * 1.2

json.dump(OUT, open(HERE / "b_adds.json", "w"), indent=1, default=float)
print("\nwrote b_adds.json")
