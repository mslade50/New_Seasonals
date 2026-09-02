"""Signal-quality study (2026-09-02), step 2: within-strategy expectancy tiers.

For every strategy with >= 50 collapsed trades and every signal-time feature:
  * within-strategy terciles -> N, avgR, win rate, median R, sd, PnL/risk
  * Spearman rho(feature, R)
  * episode-clustered OLS t-stat of R on the standardised feature
    (episodes = same-strategy signals chained while gaps <= 5 td)
  * leave-one-year-out: tercile cut-points from the training years, sign of
    (top - bottom avgR) on the held-out year vs the training sign
Selection rule (written before the run): N >= 60, tercile avgR strictly
monotone, |cluster t| >= 2, LOYO agreement >= 70% over >= 5 usable years,
|top - bottom| >= 0.30R. Everything else is a negative result.
Also: overlay-capture checks (OLV recency ladder, earnings overrides, gap
derate, same-day derate), pooled strategy-FE tests for market-state features,
and the win-rate vs expectancy split for every surviving tier.

Output: signal_quality_results.json (beside this script).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:,.3f}".format)

T = pd.read_parquet(HERE / "signal_quality_features.parquet")
T["td_to_next_earn_c"] = T["td_to_next_earn"].clip(upper=60)
T["td_since_last_earn_c"] = T["td_since_last_earn"].clip(upper=60)
T["log_dollar_vol"] = np.log(T["dollar_vol_m"].clip(lower=0.01))

TICKER_FEATS = ["filt_extremity", "rank_2d", "rank_5d", "rank_10d", "rank_21d", "rank_252d", "ret_10d", "ret_63d",
                "rel_ret_21d", "rel_ret_63d", "atr_pct", "atr_pct_rank", "dist200", "dist50", "dist10_atr",
                "sma200_slope", "hi252_dist", "lo252_dist", "vol_ratio", "vol_ratio_10d_rank", "log_dollar_vol",
                "range_pct", "move1_atr", "consec_down", "consec_up", "rv21", "vol_of_vol", "age_years"]
MARKET_FEATS = ["spy_dist200", "spy_ret10", "spy_ret21", "spy_ret63", "spy_rv21", "spy_hi252_dist", "vix",
                "vix_pct252", "vrp", "dial", "dial_raw21", "dial_pit", "pc_pct_lag1", "breadth200", "breadth_chg21",
                "sector_breadth200"]
LEDGER_FEATS = ["days_since_last_sig_c", "prior_sig_21td", "n_sig_strat_day", "n_sig_book_day", "book_sig_5td",
                "strat_sig_21td", "open_legs_strat", "td_to_next_earn_c", "td_since_last_earn_c", "wait_td",
                "gap_atr", "overflow"]
FEATS = TICKER_FEATS + MARKET_FEATS + LEDGER_FEATS
MIN_N = 50


def episodes(dates: pd.Series, gap_td: int = 5) -> np.ndarray:
    d = np.sort(dates.values.astype("datetime64[D]"))
    order = np.argsort(dates.values)
    ep = np.zeros(len(d), dtype=int)
    cur = 0
    for i in range(1, len(d)):
        if np.busday_count(d[i - 1], d[i]) > gap_td:
            cur += 1
        ep[i] = cur
    out = np.empty(len(d), dtype=int)
    out[order] = ep
    return out


def cluster_t(y: np.ndarray, x: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x with cluster-robust SE. Returns (slope, t, n_clusters)."""
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((2, 2))
    G = len(np.unique(cl))
    for g in np.unique(cl):
        m = cl == g
        Xg = X[m]
        eg = e[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    se = np.sqrt(max(V[1, 1], 1e-18))
    return float(b[1]), float(b[1] / se), int(G)


def terciles(x: pd.Series, cuts=None):
    if cuts is None:
        cuts = x.quantile([1 / 3, 2 / 3]).values
    return pd.Series(np.digitize(x.values, cuts), index=x.index), cuts


def tier_stats(g: pd.DataFrame, lab) -> dict:
    return dict(tier=lab, n=int(len(g)), avgR=float(g["R"].mean()), win=float(g["win"].mean()),
                medR=float(g["R"].median()), sdR=float(g["R"].std()),
                pnl_per_risk=float(g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum()),
                worstR=float(g["R"].min()))


def analyse(df: pd.DataFrame, feat: str) -> dict | None:
    d = df[[feat, "R", "win", "PnL_flat_750k", "Risk_flat_750k", "year", "ep"]].dropna()
    if len(d) < MIN_N or d[feat].nunique() < 3:
        return None
    x = d[feat].astype(float)
    tq, cuts = terciles(x)
    if tq.nunique() < 3:
        return None
    tiers = [tier_stats(d[tq == k], k) for k in range(3)]
    avg = [t["avgR"] for t in tiers]
    mono = (avg[0] < avg[1] < avg[2]) or (avg[0] > avg[1] > avg[2])
    rho = float(x.rank().corr(d["R"].rank()))
    z = ((x - x.mean()) / (x.std() + 1e-12)).values
    slope, t, G = cluster_t(d["R"].values, z, d["ep"].values)
    # LOYO on tercile top-bottom sign
    agree = []
    yrs = sorted(d["year"].unique())
    for y in yrs:
        tr = d[d["year"] != y]
        te = d[d["year"] == y]
        if len(te) < 4:
            continue
        _, c = terciles(tr[feat].astype(float))
        trq = pd.Series(np.digitize(tr[feat].values, c), index=tr.index)
        teq = pd.Series(np.digitize(te[feat].values, c), index=te.index)
        if (teq == 0).sum() < 2 or (teq == 2).sum() < 2:
            continue
        tr_diff = tr.loc[trq == 2, "R"].mean() - tr.loc[trq == 0, "R"].mean()
        te_diff = te.loc[teq == 2, "R"].mean() - te.loc[teq == 0, "R"].mean()
        agree.append(int(np.sign(tr_diff) == np.sign(te_diff)))
    loyo = float(np.mean(agree)) if agree else np.nan
    top_bot = avg[2] - avg[0]
    win_split = tiers[2]["win"] - tiers[0]["win"]
    passes = bool(len(d) >= 60 and mono and abs(t) >= 2.0 and len(agree) >= 5 and loyo >= 0.70 and abs(top_bot) >= 0.30)
    return dict(feature=feat, n=int(len(d)), n_clusters=G, cuts=[float(c) for c in cuts], tiers=tiers,
                monotone=bool(mono), spearman=rho, slope_per_sd=slope, cluster_t=t, loyo_agree=loyo,
                loyo_years=len(agree), top_minus_bottom_avgR=float(top_bot), top_minus_bottom_win=float(win_split),
                passes=passes)


results: dict = {"meta": dict(
    trades=int(len(T)), min_n=MIN_N, features=FEATS,
    dial_vintage="rd2_fragility.parquet 10d MA of 63d, current-weights recompute before 2026-07-02, PIT append after; "
                 "dial_pit = sibling PIT vintage-lagged series (cross_strategy_regime_pit_dial.parquet) where available",
    selection_rule="N>=60, strictly monotone tercile avgR, |episode-cluster t|>=2, LOYO sign agreement >=70% over >=5 years, |top-bottom avgR|>=0.30",
    episode_def="same-strategy signals chained while consecutive gaps <= 5 trading days")}
per_strat: dict = {}
all_tests = []
for s, g in T.groupby("Strategy"):
    if len(g) < MIN_N:
        continue
    g = g.copy()
    g["ep"] = episodes(g["Signal Date"])
    out = {}
    for f in FEATS:
        if f == "overflow" and g["overflow"].nunique() < 2:
            continue
        r = analyse(g, f)
        if r is None:
            continue
        out[f] = r
        all_tests.append(dict(strategy=s, **{k: v for k, v in r.items() if k not in ("tiers", "cuts")}))
    per_strat[s] = dict(n=int(len(g)), avgR=float(g["R"].mean()), win=float(g["win"].mean()),
                        n_episodes=int(g["ep"].nunique()), features=out)
results["per_strategy"] = per_strat

A = pd.DataFrame(all_tests)
results["n_tests"] = int(len(A))
results["expected_false_positives_at_t2"] = float(len(A) * 0.046)
strong = A[(A["cluster_t"].abs() >= 2) & A["monotone"]].sort_values("cluster_t", key=np.abs, ascending=False)
print(f"\n=== {len(A)} strategy x feature tests; {len(strong)} with |t|>=2 and monotone; {A.passes.sum()} pass the full rule ===")
cols = ["strategy", "feature", "n", "n_clusters", "spearman", "cluster_t", "loyo_agree", "loyo_years", "top_minus_bottom_avgR", "top_minus_bottom_win", "passes"]
print(strong[cols].to_string(index=False))
results["strong_table"] = strong[cols].round(3).to_dict("records")
results["passing"] = A[A.passes][cols].round(3).to_dict("records")

# --------------------------------------------------------------- quintile + detail view for passers and near-passers
detail = {}
near = A[(A["cluster_t"].abs() >= 2) & (A["loyo_agree"] >= 0.6)]
for _, r in near.iterrows():
    g = T[T["Strategy"] == r["strategy"]].copy()
    g["ep"] = episodes(g["Signal Date"])
    d = g[[r["feature"], "R", "win", "PnL_flat_750k", "Risk_flat_750k", "year"]].dropna()
    q = pd.qcut(d[r["feature"]].rank(method="first"), 5, labels=False)
    quint = [tier_stats(d[q == k], k) for k in range(5)]
    by_year = d.groupby("year").apply(lambda h: pd.Series(dict(n=len(h), rho=h[r["feature"]].rank().corr(h["R"].rank()) if len(h) > 3 else np.nan))).round(3)
    detail[f"{r['strategy']}|{r['feature']}"] = dict(quintiles=quint, by_year=by_year.reset_index().to_dict("records"))
results["detail"] = detail

# --------------------------------------------------------------- overlay-capture checks
ov: dict = {}
olv = T[T["Strategy"] == "Oversold Low Volume"].copy(); olv["ep"] = episodes(olv["Signal Date"])
rung = olv["prior_sig_21td"].clip(upper=2)
ov["olv_recency_ladder"] = dict(
    note="rung = prior FILLED OLV signals in the ticker inside 21 td (ledger proxy for the live mask count; understates the true count). Live mults 0.5/0.7/1.0.",
    tiers=[tier_stats(olv[rung == k], f"rung{k}") for k in range(3)],
    tiers_2016plus=[tier_stats(olv[(rung == k) & (olv.year >= 2016)], f"rung{k}") for k in range(3)])
e_in = olv["td_to_next_earn"].between(0, 10)
ov["olv_earnings_override"] = dict(note="OLV override: -10..0 td to earnings -> 10 bps nominal (from 35).",
                                   tiers=[tier_stats(olv[e_in], "in_window_0_10td"), tier_stats(olv[~e_in & olv.td_to_next_earn.notna()], "outside_with_data"), tier_stats(olv[olv.td_to_next_earn.isna()], "no_earn_data")])
sts = T[T["Strategy"] == "St OS Sznl"]
e_in = sts["td_to_next_earn"].between(1, 5)
ov["stos_earnings_override"] = dict(note="St OS Sznl override: -5..-1 td -> 6 bps nominal (from 40).",
                                    tiers=[tier_stats(sts[e_in], "in_window_1_5td"), tier_stats(sts[~e_in & sts.td_to_next_earn.notna()], "outside_with_data"), tier_stats(sts[sts.td_to_next_earn.isna()], "no_earn_data")])
for s in ["Monday Dip", "SPY QQQ MonFri Reversion", "Weak Close Decent Sznls", "Indices Oversold Bounce"]:
    g = T[T["Strategy"] == s]
    gu = g["gap_atr"] > 0.25
    gd = g["gap_atr"] < -0.25
    ov[f"gap_derate|{s}"] = dict(note="live derate (Monday Dip, MonFri only): T+1 open > close + 0.25 ATR -> 0.5x",
                                tiers=[tier_stats(g[gu], "gap_up>0.25"), tier_stats(g[~gu & ~gd], "flat"), tier_stats(g[gd], "gap_dn<-0.25")])
bear = T[T["Strategy"] == "3x Bear ETF Overbot Fade"]
ov["bear_same_day_derate"] = dict(note="same_day_signal_derate 0.10/floor 0.30 on the day's signal count (staged, not filled; ledger count is fills)",
                                  tiers=[tier_stats(bear[bear.n_sig_strat_day == 1], "n=1"), tier_stats(bear[bear.n_sig_strat_day == 2], "n=2"), tier_stats(bear[bear.n_sig_strat_day >= 3], "n>=3")])
# does the OLV recency rung already capture any OLV gradient that passes/near-passes?
olv_near = [k for k in detail if k.startswith("Oversold Low Volume|")]
cond = {}
for k in olv_near:
    f = k.split("|")[1]
    d = olv[[f, "R", "ep", "prior_sig_21td"]].dropna()
    z = ((d[f] - d[f].mean()) / d[f].std()).values
    rung_d = d["prior_sig_21td"].clip(upper=2).values.astype(float)
    # partial: residualise R and z on rung dummies
    D = np.column_stack([np.ones(len(d)), rung_d == 1, rung_d == 2]).astype(float)
    P = np.eye(len(d)) - D @ np.linalg.pinv(D)
    _, t_partial, _ = cluster_t(P @ d["R"].values, P @ z, d["ep"].values)
    cond[f] = dict(t_raw=float(cluster_t(d["R"].values, z, d["ep"].values)[1]), t_partial_on_rung=float(t_partial))
ov["olv_gradients_conditional_on_rung"] = cond
results["overlay_capture"] = ov

# --------------------------------------------------------------- pooled strategy-FE tests for market-state features
pool = {}
TT = T.copy()
TT["ep"] = 0
# episodes book-wide: signals chained within 5 td, per strategy label offset
off = 0
for s, g in TT.groupby("Strategy"):
    e = episodes(g["Signal Date"]) + off
    TT.loc[g.index, "ep"] = e
    off = e.max() + 1
for f in MARKET_FEATS + ["book_sig_5td", "n_sig_book_day", "overflow"]:
    d = TT[[f, "R", "Strategy", "ep", "year"]].dropna()
    if len(d) < 200:
        continue
    # demean R and feature within strategy (strategy fixed effects)
    d = d.copy()
    d["Rd"] = d["R"] - d.groupby("Strategy")["R"].transform("mean")
    d["xd"] = d[f] - d.groupby("Strategy")[f].transform("mean")
    z = (d["xd"] / (d[f].std() + 1e-12)).values
    slope, t, G = cluster_t(d["Rd"].values, z, d["ep"].values)
    # LOYO sign on slope
    ag = []
    for y in sorted(d.year.unique()):
        tr, te = d[d.year != y], d[d.year == y]
        if len(te) < 30:
            continue
        s1 = cluster_t(tr["Rd"].values, (tr["xd"] / (d[f].std() + 1e-12)).values, tr["ep"].values)[0]
        s2 = cluster_t(te["Rd"].values, (te["xd"] / (d[f].std() + 1e-12)).values, te["ep"].values)[0]
        ag.append(int(np.sign(s1) == np.sign(s2)))
    pool[f] = dict(n=int(len(d)), slope_R_per_sd=slope, cluster_t=t, n_clusters=G, loyo_agree=float(np.mean(ag)) if ag else np.nan, loyo_years=len(ag))
P = pd.DataFrame(pool).T
print("\n=== pooled strategy-FE slope of R on market-state features (R per 1 sd) ===")
print(P.to_string())
results["pooled_market_state"] = P.round(4).to_dict("index")

# --------------------------------------------------------------- liquid vs overflow within strategy
tier = {}
for s, g in T.groupby("Strategy"):
    if g["overflow"].nunique() < 2:
        continue
    g = g.copy(); g["ep"] = episodes(g["Signal Date"])
    _, t, G = cluster_t(g["R"].values, g["overflow"].values.astype(float), g["ep"].values)
    tier[s] = dict(liquid=tier_stats(g[g.overflow == 0], "liquid"), overflow=tier_stats(g[g.overflow == 1], "overflow"), cluster_t_overflow=float(t))
results["liquid_vs_overflow"] = tier
print("\n=== liquid vs overflow ===")
for s, v in tier.items():
    print(f"{s:28s} liquid avgR {v['liquid']['avgR']:+.2f} (N {v['liquid']['n']}, win {v['liquid']['win']:.2f}) | overflow {v['overflow']['avgR']:+.2f} (N {v['overflow']['n']}, win {v['overflow']['win']:.2f}) | t {v['cluster_t_overflow']:+.2f}")

# --------------------------------------------------------------- summary prints
print("\n=== overlay capture ===")
for k, v in ov.items():
    if "tiers" in v:
        print(k, "|", " ; ".join(f"{t['tier']}: N {t['n']} avgR {t['avgR']:+.2f} win {t['win']:.2f}" for t in v["tiers"] if t["n"] > 0))
print("olv conditional on rung:", cond)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


json.dump(_clean(results), open(HERE / "signal_quality_results.json", "w"), indent=1)
print("\nwrote signal_quality_results.json")
