"""Signal-quality study (2026-09-02), step 4: candidate RULES with fixed,
implementable cut-points, evaluated year by year on equal total risk
(risk-neutral rescale on the full sample so a rule cannot win by simply
adding risk), plus the OVS path-2 aggregate-cap cost and a family-level
test of the single-stock dip-buy 'market pullback' gradient.

Rules evaluated (cut-points chosen from step 2/3 tables, multipliers fixed
BEFORE this run; year-by-year consistency is the test):
  R1 OVS extremity: mean(rank_2d,5d,10d,21d) < 94 -> 0.5x ; >= 97 -> 1.25x
  R1b OVS extremity bottom-only: < 94 -> 0.5x, else 1.0x
  R2 OVS signal density: book_sig_5td (trailing 5 sessions, ex-today) <= 3 -> 0.5x
  R3 52wh: SPY realized 21d vol < 10% -> 0.5x
  R4 WCDS: signal-day move <= -1.0 ATR -> 1.25x ; > -0.5 ATR -> 0.5x
  R5 OLV: <= 21 td since last earnings -> 0.5x
  R6 LT Trend ST OS: book_sig_5td <= 2 -> 0.5x
  R7 single-stock dip-buy family (OLV, LT Trend, WCDS, Monday Dip, St OS Sznl):
     SPY > -2% from 252d high -> 0.5x
  R8 WCDS gap-up derate extension: T+1 open > close + 0.25 ATR -> 0.5x
Output: results JSON updated with key 'rules'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 250, "display.float_format", "{:,.3f}".format)
T = pd.read_parquet(HERE / "signal_quality_features.parquet")
T["rank_mean"] = T[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
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
    return b[1:], (b / np.sqrt(np.clip(np.diag(V), 1e-18, None)))[1:], G


def cell(g, lab):
    return dict(tier=lab, n=int(len(g)), avgR=float(g.R.mean()) if len(g) else None, win=float(g.win.mean()) if len(g) else None,
                sdR=float(g.R.std()) if len(g) > 1 else None, worstR=float(g.R.min()) if len(g) else None,
                sumR=float(g.R.sum()) if len(g) else None)


def evaluate_rule(df: pd.DataFrame, mult: np.ndarray, label: str, min_year_n: int = 4) -> dict:
    """Fixed rule: per-year PnL on equal total risk (mult rescaled so risk-weighted mean = 1 over the whole sample)."""
    risk = df["Risk_flat_750k"].values
    m = mult / ((mult * risk).sum() / risk.sum())
    flat = risk * df.R.values
    tiered = risk * m * df.R.values
    Y = pd.DataFrame(dict(year=df.year.values, flat=flat, tiered=tiered, n=1)).groupby("year").sum()
    Y = Y[Y.n >= min_year_n]
    d = Y.tiered - Y.flat
    # daily PnL sd on active days (variance cost of the rule)
    D = pd.DataFrame(dict(day=df["Signal Date"].values, flat=flat, tiered=tiered)).groupby("day").sum()
    ep = episodes(df["Signal Date"])
    b, t, G = cluster_ols(df.R.values, (m - 1)[:, None], ep)  # slope of R on the multiplier deviation
    return dict(rule=label, n=int(len(df)), years=int(len(Y)), years_better=int((d > 0).sum()), gain_pct=float(d.sum() / abs(Y.flat.sum()) * 100),
                worst_year_delta=float(d.min()), best_year_delta=float(d.max()), share_trades_cut=float((mult < 1).mean()), share_trades_up=float((mult > 1).mean()),
                day_sd_ratio=float(D.tiered.std() / D.flat.std()), worst_day_ratio=float(D.tiered.min() / D.flat.min()) if D.flat.min() < 0 else None,
                cluster_t_R_on_mult=float(t[0]), clusters=G,
                pnl_per_risk_flat=float(flat.sum() / risk.sum()), pnl_per_risk_tiered=float(tiered.sum() / (risk * m).sum()))


def loyo_binary(df, flag, worse=True):
    ag = []
    for y in sorted(df.year.unique()):
        te = df[df.year == y]
        a, b = te[flag[df.year == y]], te[~flag[df.year == y]]
        if len(a) >= 2 and len(b) >= 2:
            ag.append(int((a.R.mean() < b.R.mean()) == worse))
    return dict(years=len(ag), share_agree=float(np.mean(ag)) if ag else None)


rules = {}
# ---------------------------------------------------------------- R1 OVS extremity
ovs = T[T.Strategy == "Overbot Vol Spike"].copy()
m = np.where(ovs.rank_mean < 94, 0.5, np.where(ovs.rank_mean >= 97, 1.25, 1.0))
rules["R1_ovs_extremity"] = evaluate_rule(ovs, m, "OVS mean rank <94 -> 0.5x, >=97 -> 1.25x")
rules["R1b_ovs_extremity_bottom_only"] = evaluate_rule(ovs, np.where(ovs.rank_mean < 94, 0.5, 1.0), "OVS mean rank <94 -> 0.5x only")
rules["R1_cells"] = [cell(ovs[ovs.rank_mean < 94], "<94"), cell(ovs[(ovs.rank_mean >= 94) & (ovs.rank_mean < 97)], "94-97"), cell(ovs[ovs.rank_mean >= 97], ">=97")]
rules["R1_cells_by_tier"] = {t: [cell(g[g.rank_mean < 94], "<94"), cell(g[(g.rank_mean >= 94) & (g.rank_mean < 97)], "94-97"), cell(g[g.rank_mean >= 97], ">=97")] for t, g in ovs.groupby("Tier")}
rules["R1_cells_by_path"] = {p: [cell(g[g.rank_mean < 94], "<94"), cell(g[(g.rank_mean >= 94) & (g.rank_mean < 97)], "94-97"), cell(g[g.rank_mean >= 97], ">=97")] for p, g in ovs.groupby(ovs.gap_atr > 0.25)}
rules["R1_loyo_bottom"] = loyo_binary(ovs, (ovs.rank_mean < 94).values, worse=True)
rules["R1_loyo_top"] = loyo_binary(ovs, (ovs.rank_mean >= 97).values, worse=False)
# which window carries it? single-window fixed cells
rules["R1_single_windows"] = {w: [cell(ovs[ovs[f"rank_{w}d"] < 92], "<92"), cell(ovs[(ovs[f"rank_{w}d"] >= 92) & (ovs[f"rank_{w}d"] < 97)], "92-97"), cell(ovs[ovs[f"rank_{w}d"] >= 97], ">=97")] for w in [2, 5, 10, 21]}
# ---------------------------------------------------------------- R2 OVS signal density
rules["R2_ovs_density"] = evaluate_rule(ovs, np.where(ovs.book_sig_5td <= 3, 0.5, 1.0), "OVS book_sig_5td <=3 -> 0.5x")
rules["R2_cells"] = [cell(ovs[ovs.book_sig_5td <= 3], "<=3"), cell(ovs[(ovs.book_sig_5td > 3) & (ovs.book_sig_5td <= 6)], "4-6"), cell(ovs[ovs.book_sig_5td > 6], ">6")]
rules["R2_loyo"] = loyo_binary(ovs, (ovs.book_sig_5td <= 3).values, worse=True)
# combined R1+R2 (multiplicative) for OVS
m12 = np.where(ovs.rank_mean < 94, 0.5, np.where(ovs.rank_mean >= 97, 1.25, 1.0)) * np.where(ovs.book_sig_5td <= 3, 0.5, 1.0)
rules["R1xR2_ovs"] = evaluate_rule(ovs, m12, "OVS R1 x R2")
# OVS path-2 aggregate cap cost (ledger: P2 rows carry Size_Mult 0.2 (or 0.15 midterm) uncapped; smaller = capped)
p2 = ovs[ovs.gap_atr <= 0.25].copy()
p2["cyc"] = np.where(p2.year % 4 == 2, 0.75, 1.0)
p2["uncapped_risk"] = 12.0 * p2.cyc * 750_000 / 1e4
p2["capped"] = p2.Risk_flat_750k < p2.uncapped_risk * 0.999
p2["extra_pnl"] = (p2.uncapped_risk - p2.Risk_flat_750k).clip(lower=0) * p2.R
byy = p2.groupby("year").agg(capped_trades=("capped", "sum"), extra_pnl=("extra_pnl", "sum"))
dayx = p2.groupby("Signal Date").extra_pnl.sum()
rules["ovs_p2_cap_cost"] = dict(p2_trades=int(len(p2)), capped_trades=int(p2.capped.sum()), avgR_capped=float(p2[p2.capped].R.mean()), avgR_uncapped=float(p2[~p2.capped].R.mean()),
                                extra_pnl_if_uncapped=float(p2.extra_pnl.sum()), worst_day_extra=float(dayx.min()), best_day_extra=float(dayx.max()),
                                years_positive=int((byy.extra_pnl > 0).sum()), years_with_capping=int((byy.capped_trades > 0).sum()), by_year=byy.round(0).reset_index().to_dict("records"))
print("OVS P2 cap cost:", {k: v for k, v in rules["ovs_p2_cap_cost"].items() if k != "by_year"})
# ---------------------------------------------------------------- R3 52wh SPY rv21
b52 = T[T.Strategy == "52wh Breakout"].copy()
rules["R3_52wh_spyrv"] = evaluate_rule(b52, np.where(b52.spy_rv21 < 10, 0.5, 1.0), "52wh SPY rv21 <10% -> 0.5x")
rules["R3_cells"] = [cell(b52[b52.spy_rv21 < 10], "<10%"), cell(b52[(b52.spy_rv21 >= 10) & (b52.spy_rv21 < 14)], "10-14%"), cell(b52[b52.spy_rv21 >= 14], ">=14%")]
rules["R3_loyo"] = loyo_binary(b52, (b52.spy_rv21 < 10).values, worse=True)
b52["ep"] = episodes(b52["Signal Date"])
X = np.column_stack([(b52.spy_rv21.values - b52.spy_rv21.mean()) / b52.spy_rv21.std(), (b52.vix.values - b52.vix.mean()) / b52.vix.std(), (b52.dist200.values - b52.dist200.mean()) / b52.dist200.std()])
bb, tt, G = cluster_ols(b52.R.values, X, b52.ep.values)
rules["R3_multivariate"] = dict(features=["spy_rv21", "vix", "dist200"], slope=[float(x) for x in bb], t=[float(x) for x in tt])
# ---------------------------------------------------------------- R4 WCDS move1_atr
w = T[T.Strategy == "Weak Close Decent Sznls"].copy()
m4 = np.where(w.move1_atr <= -1.0, 1.25, np.where(w.move1_atr > -0.5, 0.5, 1.0))
rules["R4_wcds_move"] = evaluate_rule(w, m4, "WCDS move <=-1 ATR -> 1.25x, >-0.5 -> 0.5x")
rules["R4b_wcds_move_bottom_only"] = evaluate_rule(w, np.where(w.move1_atr > -0.5, 0.5, 1.0), "WCDS move >-0.5 ATR -> 0.5x only")
rules["R4_cells"] = [cell(w[w.move1_atr <= -1.0], "<=-1.0"), cell(w[(w.move1_atr > -1.0) & (w.move1_atr <= -0.5)], "-1.0..-0.5"), cell(w[w.move1_atr > -0.5], ">-0.5")]
rules["R4_loyo_top"] = loyo_binary(w, (w.move1_atr <= -1.0).values, worse=False)
rules["R4_loyo_bottom"] = loyo_binary(w, (w.move1_atr > -0.5).values, worse=True)
w["ep"] = episodes(w["Signal Date"])
zz = lambda s: ((s - s.mean()) / s.std()).values
X = np.column_stack([zz(w.move1_atr), zz(w.gap_atr), zz(w.rank_2d), zz(w.range_pct), zz(w.sector_breadth200.fillna(w.sector_breadth200.mean()))])
bb, tt, G = cluster_ols(w.R.values, X, w.ep.values)
rules["R4_multivariate"] = dict(features=["move1_atr", "gap_atr", "rank_2d", "range_pct", "sector_breadth200"], slope=[float(x) for x in bb], t=[float(x) for x in tt])
rules["R4_by_era"] = {str(e): [cell(g[g.move1_atr <= -1.0], "<=-1"), cell(g[(g.move1_atr > -1.0) & (g.move1_atr <= -0.5)], "mid"), cell(g[g.move1_atr > -0.5], ">-0.5")]
                      for e, g in w.groupby(pd.cut(w.year, [2002, 2009, 2016, 2021, 2027], labels=["2003-09", "2010-16", "2017-21", "2022-26"]), observed=True)}
# ---------------------------------------------------------------- R5 OLV post-earnings
olv = T[T.Strategy == "Oversold Low Volume"].copy()
f5 = (olv.td_since_last_earn <= 21).values
rules["R5_olv_post_earn"] = evaluate_rule(olv, np.where(f5, 0.5, 1.0), "OLV <=21 td since earnings -> 0.5x")
rules["R5_cells"] = [cell(olv[f5], "<=21td since"), cell(olv[~f5], "rest")]
rules["R5_loyo"] = loyo_binary(olv, f5, worse=True)
rules["R5_by_era"] = {str(e): [cell(g[g.td_since_last_earn <= 21], "<=21"), cell(g[~(g.td_since_last_earn <= 21)], "rest")]
                      for e, g in olv.groupby(pd.cut(olv.year, [2002, 2009, 2016, 2021, 2027], labels=["2003-09", "2010-16", "2017-21", "2022-26"]), observed=True)}
# the existing pre-earnings override cell for reference
f5b = olv.td_to_next_earn.between(0, 10).values
rules["R5_pre_earn_override_cell"] = [cell(olv[f5b], "0..10 td to earnings (override cell)"), cell(olv[~f5b], "rest")]
rules["R5_pre_loyo"] = loyo_binary(olv, f5b, worse=True)
# ---------------------------------------------------------------- R6 LT Trend density
lt = T[T.Strategy == "LT Trend ST OS"].copy()
rules["R6_lttrend_density"] = evaluate_rule(lt, np.where(lt.book_sig_5td <= 2, 0.5, 1.0), "LT Trend book_sig_5td <=2 -> 0.5x")
rules["R6_cells"] = [cell(lt[lt.book_sig_5td <= 2], "<=2"), cell(lt[(lt.book_sig_5td > 2) & (lt.book_sig_5td <= 5)], "3-5"), cell(lt[lt.book_sig_5td > 5], ">5")]
rules["R6_loyo"] = loyo_binary(lt, (lt.book_sig_5td <= 2).values, worse=True)
# ---------------------------------------------------------------- R7 family market-pullback
fam_names = ["Oversold Low Volume", "LT Trend ST OS", "Weak Close Decent Sznls", "Monday Dip", "St OS Sznl"]
fam = T[T.Strategy.isin(fam_names)].copy()
fam["ep"] = episodes(fam["Signal Date"])
fam["Rd"] = fam.R - fam.groupby("Strategy").R.transform("mean")
flag7 = (fam.spy_hi252_dist > -2).values
bb, tt, G = cluster_ols(fam.Rd.values, flag7.astype(float)[:, None], fam.ep.values)
rules["R7_family_pullback"] = evaluate_rule(fam, np.where(flag7, 0.5, 1.0), "single-stock dip-buys: SPY within 2% of 252d high -> 0.5x")
rules["R7_family_pullback"].update(dict(pooled_FE_slope=float(bb[0]), pooled_FE_t=float(tt[0]), clusters=G))
rules["R7_cells"] = {s: [cell(g[g.spy_hi252_dist > -2], ">-2%"), cell(g[g.spy_hi252_dist <= -2], "<=-2%")] for s, g in fam.groupby("Strategy")}
rules["R7_loyo"] = loyo_binary(fam, flag7, worse=True)
rules["R7_per_strategy_eval"] = {s: evaluate_rule(g, np.where(g.spy_hi252_dist > -2, 0.5, 1.0), s) for s, g in fam.groupby("Strategy") if len(g) >= 50}
# ---------------------------------------------------------------- R8 WCDS gap-up derate extension
rules["R8_wcds_gapup"] = evaluate_rule(w, np.where(w.gap_atr > 0.25, 0.5, 1.0), "WCDS T+1 gap-up >0.25 ATR -> 0.5x")
rules["R8_loyo"] = loyo_binary(w, (w.gap_atr > 0.25).values, worse=True)
for s in ["SPY QQQ MonFri Reversion", "Monday Dip"]:
    g = T[T.Strategy == s]
    rules[f"R8_reference_{s}"] = evaluate_rule(g, np.where(g.gap_atr > 0.25, 0.5, 1.0), f"{s} gap-up derate (live)")

print("\n=== fixed-rule evaluation (equal total risk) ===")
for k, v in rules.items():
    if isinstance(v, dict) and "gain_pct" in v:
        print(f"{k:34s} N{v['n']:5d} years {v['years']:2d} better {v['years_better']:2d} gain {v['gain_pct']:+6.1f}% worst {v['worst_year_delta']:>9,.0f} daySD x{v['day_sd_ratio']:.2f} cut {v['share_trades_cut']:.2f} up {v['share_trades_up']:.2f} t {v['cluster_t_R_on_mult']:+.2f} R/risk {v['pnl_per_risk_flat']:.3f}->{v['pnl_per_risk_tiered']:.3f}")
for k in ["R1_cells", "R2_cells", "R3_cells", "R4_cells", "R5_cells", "R5_pre_earn_override_cell", "R6_cells"]:
    print(k, [(c["tier"], c["n"], round(c["avgR"], 2), round(c["win"], 2), round(c["sdR"], 2)) for c in rules[k]])
for k in ["R1_loyo_bottom", "R1_loyo_top", "R2_loyo", "R3_loyo", "R4_loyo_top", "R4_loyo_bottom", "R5_loyo", "R5_pre_loyo", "R6_loyo", "R7_loyo", "R8_loyo"]:
    print(k, rules[k])
print("R1 by tier:", {t: [(c["n"], round(c["avgR"], 2)) for c in v] for t, v in rules["R1_cells_by_tier"].items()})
print("R1 by path (True=P1):", {str(t): [(c["n"], round(c["avgR"], 2)) for c in v] for t, v in rules["R1_cells_by_path"].items()})
print("R1 single windows:", {w_: [(c["n"], round(c["avgR"], 2)) for c in v] for w_, v in rules["R1_single_windows"].items()})
print("R3 multivariate:", rules["R3_multivariate"]); print("R4 multivariate:", rules["R4_multivariate"])
print("R4 by era:", {e: [(c["n"], round(c["avgR"], 2)) for c in v] for e, v in rules["R4_by_era"].items()})
print("R5 by era:", {e: [(c["n"], round(c["avgR"], 2)) for c in v] for e, v in rules["R5_by_era"].items()})
print("R7 cells:", {s: [(c["n"], round(c["avgR"], 2)) for c in v] for s, v in rules["R7_cells"].items()})
print("R7 pooled FE t:", rules["R7_family_pullback"]["pooled_FE_t"])


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
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


R = json.load(open(HERE / "signal_quality_results.json"))
R["rules"] = _clean(rules)
json.dump(R, open(HERE / "signal_quality_results.json", "w"), indent=1)
print("updated signal_quality_results.json")
