"""Refutation probes: 1.7 (OVS P2 cap), 1.8 (IOB clones), 2.1 (hedge multiplicity + episode table),
1.11 (gate arithmetic), 1.1/S.1 (drawdown-rule switch), 1.2 (overflow evidence), 0.6 (haircut components).
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

# ================================================================ 1.7 OVS P2 cap: capped vs uncapped confounded with cluster size?
T = pd.read_parquet(D / "signal_quality_features.parquet")
o = T[T.Strategy == "Overbot Vol Spike"].copy()
p2 = o[o.gap_atr <= 0.25].copy()
p2["cyc"] = np.where(p2.year % 4 == 2, 0.75, 1.0)
p2["uncapped_risk"] = 12.0 * p2.cyc * NAV / 1e4
p2["capped"] = p2.Risk_flat_750k < p2.uncapped_risk * 0.999
p2["n_day"] = p2.groupby("Signal Date").R.transform("size")          # P2 fills that day
p2["n_ovs_day"] = o.groupby("Signal Date").R.transform("size").reindex(p2.index)
print("=== 1.7 P2 cap ===")
print("capped vs uncapped overall:", p2.groupby("capped").R.agg(["size", "mean"]).round(3).to_dict())
tab = p2.groupby([pd.cut(p2.n_day, [0, 1, 2, 4, 8, 99], labels=["1", "2", "3-4", "5-8", "9+"]), "capped"], observed=True).R.agg(["size", "mean"]).unstack()
print("by P2 fills that day x capped:\n", tab.round(3).to_string())
OUT["p2_cap"] = dict(overall=p2.groupby("capped").R.agg(["size", "mean"]).round(4).to_dict(), by_cluster={f"{a}|{b}": v for (a, b), v in tab.round(4).astype(str).to_dict().items()})
# within cluster days only (n_day >= 3): capped vs uncapped
c3 = p2[p2.n_day >= 3]; print("cluster days (>=3 P2 fills): capped", c3[c3.capped].R.agg(["size", "mean"]).round(3).to_dict(), "uncapped", c3[~c3.capped].R.agg(["size", "mean"]).round(3).to_dict())
OUT["p2_cap"]["cluster_days_only"] = dict(capped=c3[c3.capped].R.agg(["size", "mean"]).round(4).to_dict(), uncapped=c3[~c3.capped].R.agg(["size", "mean"]).round(4).to_dict())
# extra PnL if cap lifted 0.75 -> 1.0 (i.e. 1/3 more room): by year and share of 2020
p2["extra_full"] = (p2.uncapped_risk - p2.Risk_flat_750k).clip(lower=0) * p2.R
p2["extra_step"] = np.minimum((p2.uncapped_risk - p2.Risk_flat_750k).clip(lower=0), p2.Risk_flat_750k / 3.0) * p2.R  # approx: cap 1.0/0.75 = 1.333x room
by = p2.groupby("year")[["extra_full", "extra_step"]].sum()
print("extra PnL full uncapping by year (top):", by.extra_full.sort_values(ascending=False).head(4).round(0).to_dict(), "| total", round(by.extra_full.sum()), "| 2020 share", round(by.extra_full[2020] / by.extra_full.sum(), 2))
print("approx half-step (cap x1.333) extra PnL total", round(by.extra_step.sum()), "| 2020 share", round(by.extra_step[2020] / by.extra_step.sum(), 2), "| years positive", int((by.extra_step > 0).sum()), "of", int((by.extra_step != 0).sum()))
OUT["p2_cap"]["extra_pnl"] = dict(full=float(by.extra_full.sum()), full_2020_share=float(by.extra_full[2020] / by.extra_full.sum()), step=float(by.extra_step.sum()), step_2020_share=float(by.extra_step[2020] / by.extra_step.sum()),
                                  years_pos=int((by.extra_step > 0).sum()), years_touched=int((by.extra_step != 0).sum()))
# capped P2 trades: episode/day clustered t of capped - uncapped
p2["day"] = p2["Signal Date"].dt.strftime("%Y%m%d")
x = np.where(p2.capped, 1.0, 0.0); X = np.column_stack([np.ones(len(p2)), x]); XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ p2.R.values; e = p2.R.values - X @ b
meat = np.zeros((2, 2))
for c in p2.day.unique():
    m = (p2.day == c).values; s = X[m].T @ e[m]; meat += np.outer(s, s)
G = p2.day.nunique(); V = XtX @ meat @ XtX * G / (G - 1)
print(f"capped - uncapped diff {b[1]:+.3f} day-clustered t {b[1] / np.sqrt(V[1, 1]):.2f} (days {G})")
OUT["p2_cap"]["diff_t_day"] = [float(b[1]), float(b[1] / np.sqrt(V[1, 1]))]

# ================================================================ 1.8 IOB clones
iob = T[T.Strategy == "Indices Oversold Bounce"].copy()
both = iob.groupby("Signal Date").Ticker.transform("size") == 2
piv = iob[both].pivot_table(index="Signal Date", columns="Ticker", values="R")
print("\n=== 1.8 IOB SPY+QQQ both-fire ===")
print(f"both-fire days {len(piv)} of {iob['Signal Date'].nunique()} signal days; R corr {piv.corr().iloc[0, 1]:.3f}; avgR single-fire {iob[~both].R.mean():+.3f} (N {int((~both).sum())}) vs both-fire {iob[both].R.mean():+.3f} (N {int(both.sum())})")
day = iob.groupby("Signal Date").agg(pnl=("PnL_flat_750k", "sum"), n=("R", "size"))
print("day PnL mean/sd: single", day[day.n == 1].pnl.agg(["mean", "std"]).round(0).to_dict(), "both", day[day.n == 2].pnl.agg(["mean", "std"]).round(0).to_dict())
OUT["iob_clone"] = dict(both_days=int(len(piv)), corr=float(piv.corr().iloc[0, 1]), avgR_single=float(iob[~both].R.mean()), avgR_both=float(iob[both].R.mean()),
                        day_single=day[day.n == 1].pnl.agg(["mean", "std"]).round(1).to_dict(), day_both=day[day.n == 2].pnl.agg(["mean", "std"]).round(1).to_dict())

# ================================================================ 2.1 hedge: grid multiplicity + episode table
Gd = pd.read_csv(D / "cross_strategy_regime_hedge_grid.csv")
print("\n=== 2.1 hedge grid ===", Gd.shape)
pit = Gd[(Gd.vintage == "pit") & (Gd.mult == 1.0)]
rule_pass = pit[(pit.ep_mean_usd > 0) & (pit.t_clustered >= 1.0) & (pit.sharpe_hedged >= pit.sharpe_unhedged)]
print(f"PIT rows {len(pit)}; passing (ep_mean>0, t>=1, Sharpe not worse): {len(rule_pass)}; of which dial-armed {int(rule_pass.arming.str.startswith('dial').sum())}, control-armed {int((~rule_pass.arming.str.startswith('dial')).sum())}")
print("by arming rule (PIT, book, SPY): "); print(pit[(pit.target == "book") & (pit.instrument == "SPY")][["arming", "window", "armed_days", "n_episodes", "hedge_total_usd", "t_clustered", "sharpe_unhedged", "sharpe_hedged", "ep_pos_share"]].round(2).to_string(index=False))
OUT["hedge_grid"] = dict(pit_rows=int(len(pit)), passing=int(len(rule_pass)), passing_dial=int(rule_pass.arming.str.startswith("dial").sum()),
                         book_spy=pit[(pit.target == "book") & (pit.instrument == "SPY")][["arming", "window", "n_episodes", "hedge_total_usd", "t_clustered", "sharpe_hedged"]].round(3).to_dict("records"))
R3 = json.load(open(D / "cross_strategy_regime_results_3_refine.json"))
eps = R3["episodes"]["pit|dial50_h45"]["episodes"]
E = pd.DataFrame(eps); print("PIT dial50_h45 episodes:"); print(E[["start", "end", "armed_days", "hedge_usd", "book_usd", "spy_ret_pct", "beta_mean"]].round(1).to_string(index=False))
h = E.hedge_usd.values
print(f"episodes {len(h)}, positive {(h > 0).sum()}, top-2 share of positive total {np.sort(h)[-2:].sum() / h[h > 0].sum():.2f}, median ${np.median(h):,.0f}, mean ${h.mean():,.0f}, sign-test P(>= {(h > 0).sum()} of {len(h)} | p=.5) = {sum(__import__('math').comb(len(h), k) for k in range((h > 0).sum(), len(h) + 1)) / 2 ** len(h):.3f}")
OUT["hedge_episodes"] = dict(n=int(len(h)), pos=int((h > 0).sum()), top2_share=float(np.sort(h)[-2:].sum() / h[h > 0].sum()), median=float(np.median(h)), table=E.round(1).to_dict("records"))
live_eps = pd.DataFrame(R3["episodes"]["live|dial50_h45"]["episodes"])
print("LIVE vintage dial50_h45:", len(live_eps), "episodes, total", round(live_eps.hedge_usd.sum()), "positive", int((live_eps.hedge_usd > 0).sum()))

# ================================================================ 1.11 gate arithmetic and 1.1/S.1 drawdown rule
P = json.load(open(D / "practitioner_02_package_replay.json"))
base = P["configs"]["baseline"]["grm"]["grm1.5"]["windows"]["2005-2026"]
print("\n=== 1.11 gate arithmetic ===")
print(f"baseline ann PnL {base['ann_pnl_pct']:.2f}% at GRM 1.5 -> x1.25 with levers OFF = {base['ann_pnl_pct'] * 1.25:.2f}% (+{base['ann_pnl_pct'] * 0.25:.2f} pts). Gate needs >= +4 pts: levers may cost up to {base['ann_pnl_pct'] * 0.25 - 4:.2f} pts and still pass.")
OUT["gate"] = dict(base_ann=base["ann_pnl_pct"], grm_only_gain_pts=base["ann_pnl_pct"] * 0.25, slack_pts=base["ann_pnl_pct"] * 0.25 - 4)
cfgs = {k: v["grm"]["grm1.5"]["windows"]["2005-2026"] for k, v in P["configs"].items() if "grm1.5" in v["grm"]}
tab = pd.DataFrame({k: dict(ann=v["ann_pnl_pct"], sharpe=v["sharpe"], maxdd=v["maxdd_pct"], worst21=v["worst21_pct"]) for k, v in cfgs.items()}).T
print(tab.round(2).to_string()); OUT["package_configs_grm15"] = tab.round(3).to_dict("index")
U = json.load(open(D / "unconstrained_growth_results.json"))
dd = P["dd_frontier_bootstrap"]["baseline"]
print("\n=== 1.1 / S.1: P(DD>15%) by haircut at m=1.25 (GRM 1.875), package replay bootstrap ===")
for k in sorted(dd):
    if "m1.25" in k or "m1.0" in k:
        print(k, {kk: round(vv, 3) for kk, vv in dd[k].items() if kk in ("grm", "median_maxdd_pct", "p95_maxdd_pct", "P_dd_gt15", "P_dd_gt20")})
OUT["dd_at_1875"] = {k: v for k, v in dd.items() if "m1.25" in k}
print("unconstrained_growth drawdown_distribution keys:", list(U["drawdown_distribution"].keys())[:10])
try:
    ddu = U["drawdown_distribution"]
    s = json.dumps(ddu); print(s[:1500])
except Exception as ex:
    print(ex)

# ================================================================ 1.2 overflow exclusion evidence
SQ = json.load(open(D / "signal_quality_results.json"))["liquid_vs_overflow"]
print("\n=== 1.2 overflow vs liquid (within strategy) ===")
for s, v in SQ.items():
    print(f"{s:26s} liquid {v['liquid']['avgR']:+.2f} (N{v['liquid']['n']}) overflow {v['overflow']['avgR']:+.2f} (N{v['overflow']['n']}) t {v['cluster_t_overflow']:+.2f}")
OUT["overflow_t"] = {s: v["cluster_t_overflow"] for s, v in SQ.items()}

# ================================================================ 0.6 haircut components
H = json.load(open(D / "estimation_haircut_results.json"))["book"]
print("\n=== 0.6 haircut ===")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in H.items() if k in ("keep_bottom_up_pnl_weighted", "keep_direct_blend", "keep_recommended", "haircut_recommended")})
de = H["direct_evidence"]
print("direct: structural", round(de["structural_keep"], 3), "realized_2026", round(de["realized_2026_keep"], 3), "= 0.5*", round(de["riskweighted_2026_ratio_vs_trailing5_at_risk_mix"], 3), "+ 0.5*min(1,", round(de["riskweighted_midterm_conditioned_2026_ratio"], 3), ")")
print("rule-freeze OOS ratio", round(de["rule_freeze_oos_ratio"], 3), "N", de["rule_freeze_oos_N"], "CI", [round(x, 2) for x in de["rule_freeze_oos_ci95"]], "| raw 2026 ratio CI", [round(x, 2) for x in de["raw_2026_ratio_ci95_dayblock"]])
OUT["haircut"] = dict(bottom_up=H["keep_bottom_up_pnl_weighted"], direct=H["keep_direct_blend"], recommended=H["keep_recommended"], structural=de["structural_keep"], realized=de["realized_2026_keep"],
                      oos_ratio=de["rule_freeze_oos_ratio"], oos_ci=de["rule_freeze_oos_ci95"])

json.dump(OUT, open(HERE / "f_misc.json", "w"), indent=1, default=float)
print("wrote f_misc.json")
