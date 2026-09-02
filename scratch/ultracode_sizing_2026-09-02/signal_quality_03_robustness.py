"""Signal-quality study (2026-09-02), step 3: robustness + walk-forward tier
sizing for the candidate gradients from step 2.

  A. OVS: filter extremity vs gap (2-path) vs signal density; within-path;
     era table; fixed-cutpoint tiers; walk-forward tier sizing.
  B. OLV: market-pullback features (SPY off-high / 21d return / breadth
     change), alpha-vs-beta decomposition over the hold, PIT dial check,
     post-earnings window, cross-strategy mechanism check, walk-forward.
  C. Signal density across strategies + daily-cap interaction.
  D. Gap-up derate extension check (WCDS) + dip-buy family pooled.
  E. Walk-forward tier sizing for every near-passer (generic).

Walk-forward tier rule (written before the run): expanding training window
(>= 5 years and >= 60 trades), tercile cut-points and multipliers
clip(avgR_tier / avgR_all, 0.5, 1.5) from training trades, rescaled so the
risk-weighted mean multiplier over training trades is 1.0 (risk-neutral),
applied to the held-out year. Report gain in PnL per unit risk vs flat.

Output: signal_quality_results.json is UPDATED in place (keys 'robustness').
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:,.3f}".format)

T = pd.read_parquet(HERE / "signal_quality_features.parquet")
T["td_to_next_earn_c"] = T["td_to_next_earn"].clip(upper=60)
T["td_since_last_earn_c"] = T["td_since_last_earn"].clip(upper=60)
T["era"] = pd.cut(T["year"], [2002, 2009, 2016, 2021, 2027], labels=["2003-09", "2010-16", "2017-21", "2022-26"])
OUT: dict = {}


def episodes(dates: pd.Series, gap_td: int = 5) -> np.ndarray:
    order = np.argsort(dates.values)
    d = np.sort(dates.values.astype("datetime64[D]"))
    ep = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        ep[i] = ep[i - 1] + (np.busday_count(d[i - 1], d[i]) > gap_td)
    out = np.empty(len(d), dtype=int)
    out[order] = ep
    return out


def cluster_ols(y, X, cl):
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = len(np.unique(cl))
    for g in np.unique(cl):
        m = cl == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    se = np.sqrt(np.clip(np.diag(V), 1e-18, None))
    return b[1:], (b / se)[1:], G


def z(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / (s.std() + 1e-12)).values


def cell(g: pd.DataFrame, lab) -> dict:
    return dict(tier=str(lab), n=int(len(g)), avgR=float(g["R"].mean()) if len(g) else None,
                win=float(g["win"].mean()) if len(g) else None, sdR=float(g["R"].std()) if len(g) > 1 else None,
                worstR=float(g["R"].min()) if len(g) else None,
                mu_over_var=float(g["R"].mean() / g["R"].var()) if len(g) > 2 else None)


def wf_tier_sim(df: pd.DataFrame, feat: str, min_years: int = 5, min_train: int = 60, cap=(0.5, 1.5)) -> dict:
    """Walk-forward tercile-tier sizing; returns gain vs flat on equal total risk."""
    d = df[[feat, "R", "year", "Risk_flat_750k"]].dropna().sort_values("year")
    years = sorted(d.year.unique())
    rows = []
    for y in years:
        tr = d[d.year < y]
        te = d[d.year == y]
        if len(tr) < min_train or tr.year.nunique() < min_years or len(te) == 0:
            continue
        cuts = tr[feat].quantile([1 / 3, 2 / 3]).values
        tq = np.digitize(tr[feat].values, cuts)
        base = tr["R"].mean()
        mults = np.array([np.clip(tr["R"][tq == k].mean() / base if base > 0 else 1.0, *cap) for k in range(3)])
        w = np.array([(tr["Risk_flat_750k"][tq == k]).sum() for k in range(3)])
        mults = mults / (mults @ w / w.sum())  # risk-neutral on training risk
        teq = np.digitize(te[feat].values, cuts)
        m = mults[teq]
        risk = te["Risk_flat_750k"].values
        flat = float((risk * te["R"].values).sum())
        tiered = float((risk * m * te["R"].values).sum())
        rows.append(dict(year=int(y), n=int(len(te)), flat_pnl=flat, tiered_pnl=tiered, mults=[float(x) for x in mults],
                         risk_ratio=float((risk * m).sum() / risk.sum())))
    if not rows:
        return dict(years=0)
    Y = pd.DataFrame(rows)
    flat, tiered = Y.flat_pnl.sum(), Y.tiered_pnl.sum()
    return dict(years=int(len(Y)), flat_pnl=float(flat), tiered_pnl=float(tiered), gain_pct=float((tiered - flat) / abs(flat) * 100),
                years_better=int((Y.tiered_pnl > Y.flat_pnl).sum()), worst_year_delta=float((Y.tiered_pnl - Y.flat_pnl).min()),
                mean_risk_ratio=float(Y.risk_ratio.mean()), yearly=Y.round(1).to_dict("records"))


# ================================================================ A. OVS
ovs = T[T.Strategy == "Overbot Vol Spike"].copy()
ovs["ep"] = episodes(ovs["Signal Date"])
ovs["path1"] = (ovs["gap_atr"] > 0.25).astype(int)  # the live 2-path rule (Risk bps column is nominal 60 for both paths; P2 shows as Size_Mult 0.2)
ovs["rank_min"] = ovs[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].min(axis=1)
ovs["rank_mean"] = ovs[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
A: dict = {}
X = np.column_stack([z(ovs.filt_extremity), z(ovs.gap_atr), z(ovs.book_sig_5td), z(ovs.rank_252d.fillna(50))])
b, t, G = cluster_ols(ovs["R"].values, X, ovs["ep"].values)
A["multivariate"] = dict(features=["filt_extremity", "gap_atr", "book_sig_5td", "rank_252d"], slope_R_per_sd=[float(x) for x in b], cluster_t=[float(x) for x in t], clusters=G)
print("\n=== A. OVS multivariate (R per sd; cluster t) ===")
print(pd.DataFrame(dict(feature=A["multivariate"]["features"], slope=b, t=t)).to_string(index=False))
A["by_path"] = {}
for p, lab in [(1, "P1"), (0, "P2")]:
    g = ovs[ovs.path1 == p]
    cuts = ovs.filt_extremity.quantile([1 / 3, 2 / 3]).values
    q = np.digitize(g.filt_extremity.values, cuts)
    A["by_path"][lab] = [cell(g[q == k], f"T{k}") for k in range(3)]
    print(lab, [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2)) for c in A["by_path"][lab]])
A["by_era"] = {}
cuts = ovs.filt_extremity.quantile([1 / 3, 2 / 3]).values
q = np.digitize(ovs.filt_extremity.values, cuts)
for e, g in ovs.groupby("era", observed=True):
    qq = q[ovs.index.get_indexer(g.index)]
    A["by_era"][str(e)] = [cell(g[qq == k], f"T{k}") for k in range(3)]
    print(e, [(c["tier"], c["n"], round(c["avgR"], 2)) for c in A["by_era"][str(e)]])
# fixed-cutpoint tiers on mean rank of the four short windows (implementable in daily_scan from the same rank columns)
A["fixed_tiers_rank_mean"] = [cell(ovs[ovs.rank_mean < 94], "mean_rank<94"), cell(ovs[(ovs.rank_mean >= 94) & (ovs.rank_mean < 97)], "94-97"), cell(ovs[ovs.rank_mean >= 97], ">=97")]
A["fixed_tiers_rank_min"] = [cell(ovs[ovs.rank_min < 90], "min_rank<90"), cell(ovs[(ovs.rank_min >= 90) & (ovs.rank_min < 95)], "90-95"), cell(ovs[ovs.rank_min >= 95], ">=95")]
print("fixed tiers mean rank:", [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2), round(c["sdR"], 2)) for c in A["fixed_tiers_rank_mean"]])
print("fixed tiers min rank :", [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2), round(c["sdR"], 2)) for c in A["fixed_tiers_rank_min"]])
# gap x extremity 2x2
A["gap_x_extremity"] = {}
for gl, gm in [("gap<=0.25", ovs.gap_atr <= 0.25), ("gap>0.25", ovs.gap_atr > 0.25)]:
    for el, em in [("ext_low", q == 0), ("ext_mid", q == 1), ("ext_high", q == 2)]:
        A["gap_x_extremity"][f"{gl}|{el}"] = cell(ovs[gm & em], f"{gl}|{el}")
print("gap x extremity:", {k: (v["n"], round(v["avgR"], 2)) for k, v in A["gap_x_extremity"].items()})
A["wf_extremity"] = wf_tier_sim(ovs, "filt_extremity")
A["wf_rank_mean"] = wf_tier_sim(ovs, "rank_mean")
A["wf_book_sig_5td"] = wf_tier_sim(ovs, "book_sig_5td")
A["wf_gap_atr_reference"] = wf_tier_sim(ovs, "gap_atr")
for k in ["wf_extremity", "wf_rank_mean", "wf_book_sig_5td", "wf_gap_atr_reference"]:
    r = A[k]
    print(f"OVS {k}: years {r['years']} gain {r.get('gain_pct', 0):+.1f}% years_better {r.get('years_better')} worst_delta {r.get('worst_year_delta', 0):,.0f} risk_ratio {r.get('mean_risk_ratio', 1):.3f}")
# same-day signal count vs cap (P1 60 bps eff -> 4 signals = 240 bps)
ovs_day = ovs.groupby("Signal Date").agg(n=("R", "size"), risk=("Risk_flat_750k", "sum"), avgR=("R", "mean"))
ovs_day["risk_bps"] = ovs_day["risk"] / 750_000 * 1e4
A["by_day_signal_count"] = [dict(bucket=lab, days=int(len(g)), trades=int(g.n.sum()), avgR_trade_weighted=float((g.avgR * g.n).sum() / g.n.sum()), mean_staged_bps=float(g.risk_bps.mean()))
                            for lab, g in [("1", ovs_day[ovs_day.n == 1]), ("2-3", ovs_day[ovs_day.n.between(2, 3)]), ("4-6", ovs_day[ovs_day.n.between(4, 6)]), ("7+", ovs_day[ovs_day.n >= 7])]]
print("OVS by day signal count:", A["by_day_signal_count"])
OUT["A_ovs"] = A

# ================================================================ B. OLV
olv = T[T.Strategy == "Oversold Low Volume"].copy()
olv["ep"] = episodes(olv["Signal Date"])
B: dict = {}
spy = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"], filters=[("ticker", "=", "SPY")]).to_pandas().set_index("date")["Close"].sort_index()
# SPY return over the hold (entry close -> exit close) and over the 10 sessions after the signal
olv["spy_hold_ret"] = (spy.reindex(olv["Exit Date"]).values / spy.reindex(olv["Entry Date"]).values - 1) * 100
idx = spy.index
pos = idx.searchsorted(olv["Signal Date"].values)
pos10 = np.minimum(pos + 10, len(idx) - 1)
olv["spy_fwd10"] = (spy.values[pos10] / spy.values[pos] - 1) * 100
X = np.column_stack([z(olv.spy_hi252_dist), z(olv.spy_ret21), z(olv.breadth_chg21)])
b, t, G = cluster_ols(olv["R"].values, X, olv["ep"].values)
B["multivariate_full"] = dict(features=["spy_hi252_dist", "spy_ret21", "breadth_chg21"], slope=[float(x) for x in b], t=[float(x) for x in t], clusters=G, n=int(len(olv)))
print("\n=== B. OLV multivariate full sample ===")
print(pd.DataFrame(dict(feature=B["multivariate_full"]["features"], slope=b, t=t)).to_string(index=False))
o16 = olv.dropna(subset=["dial"])
X = np.column_stack([z(o16.spy_hi252_dist), z(o16.spy_ret21), z(o16.dial)])
b, t, G = cluster_ols(o16["R"].values, X, o16["ep"].values)
B["multivariate_2016_with_dial"] = dict(features=["spy_hi252_dist", "spy_ret21", "dial(current-weights)"], slope=[float(x) for x in b], t=[float(x) for x in t], clusters=G, n=int(len(o16)))
print(pd.DataFrame(dict(feature=B["multivariate_2016_with_dial"]["features"], slope=b, t=t)).to_string(index=False))
op = olv.dropna(subset=["dial_pit"])
X = np.column_stack([z(op.spy_hi252_dist), z(op.spy_ret21), z(op.dial_pit)])
b, t, G = cluster_ols(op["R"].values, X, op["ep"].values)
B["multivariate_with_dial_pit"] = dict(features=["spy_hi252_dist", "spy_ret21", "dial_pit"], slope=[float(x) for x in b], t=[float(x) for x in t], clusters=G, n=int(len(op)))
print(pd.DataFrame(dict(feature=B["multivariate_with_dial_pit"]["features"], slope=b, t=t)).to_string(index=False))
# dial_pit univariate tiers on the same trades as the current-weights dial
B["dial_vintage_compare"] = dict(
    current=[cell(op[np.digitize(op.dial.values, op.dial.quantile([1 / 3, 2 / 3]).values) == k], f"T{k}") for k in range(3)],
    pit=[cell(op[np.digitize(op.dial_pit.values, op.dial_pit.quantile([1 / 3, 2 / 3]).values) == k], f"T{k}") for k in range(3)],
    n=int(len(op)))
print("dial vintages (same trades):", {k: [(c["n"], round(c["avgR"], 2)) for c in v] for k, v in B["dial_vintage_compare"].items() if k != "n"})
# alpha vs beta: R on SPY hold return; gradient on residual
b1, t1, _ = cluster_ols(olv["R"].values, olv["spy_hold_ret"].values[:, None], olv["ep"].values)
resid = olv["R"].values - b1[0] * (olv["spy_hold_ret"].values - olv["spy_hold_ret"].mean())
b2, t2, _ = cluster_ols(resid, z(olv.spy_hi252_dist)[:, None], olv["ep"].values)
b3, t3, _ = cluster_ols(olv["R"].values, z(olv.spy_hi252_dist)[:, None], olv["ep"].values)
cuts = olv.spy_hi252_dist.quantile([1 / 3, 2 / 3]).values
q = np.digitize(olv.spy_hi252_dist.values, cuts)
B["alpha_beta"] = dict(R_on_spy_hold_slope=float(b1[0]), t=float(t1[0]),
                       gradient_raw_t=float(t3[0]), gradient_residual_t=float(t2[0]), gradient_raw_slope=float(b3[0]), gradient_residual_slope=float(b2[0]),
                       spy_fwd10_by_tier=[float(olv.spy_fwd10[q == k].mean()) for k in range(3)],
                       spy_hold_by_tier=[float(olv.spy_hold_ret[q == k].mean()) for k in range(3)],
                       R_by_tier=[float(olv.R[q == k].mean()) for k in range(3)],
                       resid_by_tier=[float(resid[q == k].mean()) for k in range(3)])
print("alpha/beta:", B["alpha_beta"])
# fixed cutpoints for implementability: SPY dist from 252d high
B["fixed_tiers_spy_off_high"] = [cell(olv[olv.spy_hi252_dist <= -5], "SPY<=-5%"), cell(olv[(olv.spy_hi252_dist > -5) & (olv.spy_hi252_dist <= -2)], "-5..-2%"), cell(olv[olv.spy_hi252_dist > -2], ">-2%")]
print("OLV SPY off-high fixed:", [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2), round(c["sdR"], 2)) for c in B["fixed_tiers_spy_off_high"]])
B["by_era_spy_off_high"] = {str(e): [cell(g[np.digitize(g.spy_hi252_dist.values, cuts) == k], f"T{k}") for k in range(3)] for e, g in olv.groupby("era", observed=True)}
print("by era:", {e: [(c["n"], round(c["avgR"], 2)) for c in v] for e, v in B["by_era_spy_off_high"].items()})
# post-earnings window
has = olv[olv.td_since_last_earn.notna()]
B["post_earnings"] = [cell(has[has.td_since_last_earn <= 10], "0-10 td since"), cell(has[has.td_since_last_earn.between(11, 21)], "11-21"),
                      cell(has[has.td_since_last_earn.between(22, 42)], "22-42"), cell(has[has.td_since_last_earn > 42], "43+"), cell(olv[olv.td_since_last_earn.isna()], "no data")]
print("OLV post-earnings:", [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2), c["worstR"]) for c in B["post_earnings"]])
pe = (olv.td_since_last_earn <= 21).astype(float).values
pre = olv.td_to_next_earn.between(0, 10).astype(float).values
X = np.column_stack([pe, pre, z(olv.spy_hi252_dist)])
b, t, G = cluster_ols(olv["R"].values, X, olv["ep"].values)
B["earnings_window_regression"] = dict(features=["post_earn_<=21td", "pre_earn_0..10td", "spy_hi252_dist_z"], slope=[float(x) for x in b], t=[float(x) for x in t])
print("earnings windows:", B["earnings_window_regression"])
# LOYO on post-earnings flag
ag = []
for y in sorted(olv.year.unique()):
    te = olv[olv.year == y]
    a, bb = te[te.td_since_last_earn <= 21], te[~(te.td_since_last_earn <= 21)]
    if len(a) >= 2 and len(bb) >= 2:
        ag.append(int(a.R.mean() < bb.R.mean()))
B["post_earnings_loyo"] = dict(years=len(ag), share_years_post_worse=float(np.mean(ag)))
B["earn_coverage_by_era"] = olv.groupby("era", observed=True)["has_earn"].mean().round(2).to_dict()
print("post-earn LOYO:", B["post_earnings_loyo"], "coverage:", B["earn_coverage_by_era"])
# recency rung x pullback
rung = olv.prior_sig_21td.clip(upper=2)
B["rung_x_pullback"] = {f"rung{r}|T{k}": cell(olv[(rung == r) & (q == k)], "") for r in range(3) for k in range(3)}
# cross-strategy mechanism check
B["spy_off_high_other_strats"] = {}
for s in ["LT Trend ST OS", "St OS Sznl", "Weak Close Decent Sznls", "Monday Dip", "SPY QQQ MonFri Reversion", "Indices Oversold Bounce", "52wh Breakout", "Overbot Vol Spike"]:
    g = T[T.Strategy == s]
    B["spy_off_high_other_strats"][s] = [cell(g[g.spy_hi252_dist <= -5], "<=-5%"), cell(g[(g.spy_hi252_dist > -5) & (g.spy_hi252_dist <= -2)], "-5..-2"), cell(g[g.spy_hi252_dist > -2], ">-2%")]
    print(f"  {s:26s}", [(c["tier"], c["n"], round(c["avgR"], 2)) for c in B["spy_off_high_other_strats"][s]])
B["wf_spy_hi252_dist"] = wf_tier_sim(olv, "spy_hi252_dist")
B["wf_spy_ret21"] = wf_tier_sim(olv, "spy_ret21")
B["wf_breadth_chg21"] = wf_tier_sim(olv, "breadth_chg21")
olv["post_earn_flag"] = (olv.td_since_last_earn <= 21).astype(float)
B["wf_post_earn"] = wf_tier_sim(olv, "post_earn_flag")
for k in ["wf_spy_hi252_dist", "wf_spy_ret21", "wf_breadth_chg21", "wf_post_earn"]:
    r = B[k]
    print(f"OLV {k}: years {r['years']} gain {r.get('gain_pct', 0):+.1f}% years_better {r.get('years_better')} worst_delta {r.get('worst_year_delta', 0):,.0f}")
OUT["B_olv"] = B

# ================================================================ C. signal density across strategies + cap interaction
C: dict = {"per_strategy": {}}
for s, g in T.groupby("Strategy"):
    if len(g) < 50:
        continue
    cuts = g.book_sig_5td.quantile([1 / 3, 2 / 3]).values
    q = np.digitize(g.book_sig_5td.values, cuts)
    lo, hi = g[q == 0], g[q == 2]
    C["per_strategy"][s] = dict(lo=cell(lo, "lo"), hi=cell(hi, "hi"), diff=float(hi.R.mean() - lo.R.mean()))
print("\n=== C. book_sig_5td lo vs hi tercile by strategy ===")
for s, v in C["per_strategy"].items():
    print(f"  {s:26s} lo N{v['lo']['n']} {v['lo']['avgR']:+.2f} | hi N{v['hi']['n']} {v['hi']['avgR']:+.2f} | diff {v['diff']:+.2f}")
C["n_strats_positive"] = int(sum(v["diff"] > 0 for v in C["per_strategy"].values()))
C["n_strats"] = len(C["per_strategy"])
# cap interaction: per-strategy day staged (filled) risk in bps, avgR by day-risk bucket
day = T.groupby(["Strategy", "Signal Date"]).agg(n=("R", "size"), risk=("Risk_flat_750k", "sum"), sumRxrisk=("PnL_flat_750k", "sum")).reset_index()
day["bps"] = day.risk / 750_000 * 1e4
day["R_riskw"] = day.sumRxrisk / day.risk
C["day_risk_buckets"] = [dict(bucket=lab, days=int(len(g)), trades=int(g.n.sum()), riskw_avgR=float(g.sumRxrisk.sum() / g.risk.sum()))
                         for lab, g in [("<100bps", day[day.bps < 100]), ("100-200", day[day.bps.between(100, 200)]), ("200-250", day[day.bps.between(200, 250)]), ("cap-bound>=240", day[day.bps >= 240])]]
print("day risk buckets (all strategies):", C["day_risk_buckets"])
OUT["C_density"] = C

# ================================================================ D. gap-up derate extension
D: dict = {}
for s in ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip", "Indices Oversold Bounce", "Monthly Weak Close"]:
    g = T[T.Strategy == s].copy()
    g["ep"] = episodes(g["Signal Date"])
    up = (g.gap_atr > 0.25).astype(float).values
    if up.sum() < 5:
        continue
    b, t, G = cluster_ols(g["R"].values, up[:, None], g["ep"].values)
    ag = []
    for y in sorted(g.year.unique()):
        te = g[g.year == y]
        a, bb = te[te.gap_atr > 0.25], te[~(te.gap_atr > 0.25)]
        if len(a) >= 2 and len(bb) >= 2:
            ag.append(int(a.R.mean() < bb.R.mean()))
    D[s] = dict(gap_up=cell(g[g.gap_atr > 0.25], "gap_up>0.25"), rest=cell(g[~(g.gap_atr > 0.25)], "rest"), cluster_t_gapup=float(t[0]), loyo_years=len(ag), share_years_gapup_worse=float(np.mean(ag)) if ag else None,
              entry=str(g["Entry Criteria"].iloc[0]))
    print(f"  {s:26s} gap_up N{D[s]['gap_up']['n']} {D[s]['gap_up']['avgR']:+.2f} vs rest N{D[s]['rest']['n']} {D[s]['rest']['avgR']:+.2f} t {t[0]:+.2f} LOYO {D[s]['share_years_gapup_worse']}/{len(ag)} entry {D[s]['entry']}")
fam = T[T.Strategy.isin(["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip"])].copy()
fam["ep"] = episodes(fam["Signal Date"])
fam["Rd"] = fam.R - fam.groupby("Strategy").R.transform("mean")
b, t, G = cluster_ols(fam["Rd"].values, (fam.gap_atr > 0.25).astype(float).values[:, None], fam["ep"].values)
D["family_pooled_FE"] = dict(n=int(len(fam)), slope=float(b[0]), t=float(t[0]), clusters=G)
print("family pooled:", D["family_pooled_FE"])
OUT["D_gap"] = D

# ================================================================ E. generic walk-forward for every near-passer
E: dict = {}
R2 = json.load(open(HERE / "signal_quality_results.json"))
near = [(r["strategy"], r["feature"]) for r in R2["strong_table"] if abs(r["cluster_t"]) >= 2 and (r["loyo_agree"] or 0) >= 0.6 and r["n"] >= 60]
for s, f in near:
    g = T[T.Strategy == s]
    if g[f].nunique() < 3:
        continue
    r = wf_tier_sim(g, f)
    r.pop("yearly", None)
    E[f"{s}|{f}"] = r
print("\n=== E. walk-forward tier sizing, near-passers (gain vs flat on equal risk) ===")
for k, r in sorted(E.items(), key=lambda kv: -kv[1].get("gain_pct", -99)):
    if r.get("years", 0):
        print(f"  {k:55s} years {r['years']:2d} gain {r['gain_pct']:+6.1f}% better {r['years_better']:2d}/{r['years']} worst {r['worst_year_delta']:>9,.0f}")
OUT["E_walk_forward"] = E


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


R2["robustness"] = _clean(OUT)
json.dump(R2, open(HERE / "signal_quality_results.json", "w"), indent=1)
print("\nupdated signal_quality_results.json")
