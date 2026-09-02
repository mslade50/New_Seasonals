"""Signal-quality study (2026-09-02), step 5: dollar impacts, sensitivity and
the final candidate table (expectancy vs win-rate split) -> results JSON key
'summary'."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = pd.read_parquet(HERE / "signal_quality_features.parquet")
T["rank_mean"] = T[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
R = json.load(open(HERE / "signal_quality_results.json"))
S: dict = {}


def cell(g, lab):
    return dict(tier=lab, n=int(len(g)), avgR=float(g.R.mean()), win=float(g.win.mean()), sdR=float(g.R.std()),
                mu_over_var=float(g.R.mean() / g.R.var()), pnl=float(g.PnL_flat_750k.sum()), risk=float(g.Risk_flat_750k.sum()))


def yearly_eval(df, mult):
    risk = df.Risk_flat_750k.values
    m = mult / ((mult * risk).sum() / risk.sum())
    Y = pd.DataFrame(dict(year=df.year.values, flat=risk * df.R.values, tiered=risk * m * df.R.values, n=1)).groupby("year").sum()
    Y = Y[Y.n >= 4]
    d = Y.tiered - Y.flat
    return dict(years=int(len(Y)), years_better=int((d > 0).sum()), gain_pct=float(d.sum() / abs(Y.flat.sum()) * 100), worst=float(d.min()))


# ---- OVS R1 dollar impact
ovs = T[T.Strategy == "Overbot Vol Spike"]
lo = ovs[ovs.rank_mean < 94]
S["ovs_r1_dollars"] = dict(ovs_total_pnl=float(ovs.PnL_flat_750k.sum()), ovs_total_risk=float(ovs.Risk_flat_750k.sum()),
                           low_cell_pnl=float(lo.PnL_flat_750k.sum()), low_cell_risk=float(lo.Risk_flat_750k.sum()), low_cell_share_of_risk=float(lo.Risk_flat_750k.sum() / ovs.Risk_flat_750k.sum()),
                           cost_of_halving_low_cell_no_redeploy=float(-0.5 * lo.PnL_flat_750k.sum()),
                           years=int(ovs.year.nunique()))
# by-year deltas for R1b
risk = ovs.Risk_flat_750k.values
m = np.where(ovs.rank_mean < 94, 0.5, 1.0)
m = m / ((m * risk).sum() / risk.sum())
Y = pd.DataFrame(dict(year=ovs.year.values, flat=risk * ovs.R.values, tiered=risk * m * ovs.R.values, n=1, n_low=(ovs.rank_mean < 94).values.astype(int))).groupby("year").sum()
Y["delta"] = Y.tiered - Y.flat
S["ovs_r1b_by_year"] = Y.round(0).reset_index().to_dict("records")
print("OVS R1b by year:\n", Y.round(0).to_string())
# midterm-year check (cycle mult overlay) and P2-cap interaction
S["ovs_r1_midterm"] = {lab: [cell(g[g.rank_mean < 94], "<94"), cell(g[g.rank_mean >= 94], ">=94")] for lab, g in [("midterm", ovs[ovs.year % 4 == 2]), ("other", ovs[ovs.year % 4 != 2])]}
S["ovs_r1_by_day_density"] = {lab: [cell(g[g.rank_mean < 94], "<94"), cell(g[g.rank_mean >= 94], ">=94")] for lab, g in [("book_sig_5td<=3", ovs[ovs.book_sig_5td <= 3]), ("book_sig_5td>3", ovs[ovs.book_sig_5td > 3])]}
print("midterm:", {k: [(c["n"], round(c["avgR"], 2)) for c in v] for k, v in S["ovs_r1_midterm"].items()})
print("by density:", {k: [(c["n"], round(c["avgR"], 2)) for c in v] for k, v in S["ovs_r1_by_day_density"].items()})
# extremity threshold sensitivity (bottom cut only)
S["ovs_r1_threshold_sensitivity"] = {}
for thr in [92, 93, 94, 95, 96]:
    e = yearly_eval(ovs, np.where(ovs.rank_mean < thr, 0.5, 1.0))
    e["cell"] = cell(ovs[ovs.rank_mean < thr], f"<{thr}")
    S["ovs_r1_threshold_sensitivity"][thr] = e
    print(f"OVS bottom cut <{thr}: N {e['cell']['n']} avgR {e['cell']['avgR']:+.2f} | years better {e['years_better']}/{e['years']} gain {e['gain_pct']:+.1f}%")
# ---- R7 family pullback: per-strategy and threshold sensitivity
fam_names = ["Oversold Low Volume", "LT Trend ST OS", "Weak Close Decent Sznls", "Monday Dip", "St OS Sznl"]
fam = T[T.Strategy.isin(fam_names)]
S["r7_per_strategy"] = {s: yearly_eval(g, np.where(g.spy_hi252_dist > -2, 0.5, 1.0)) for s, g in fam.groupby("Strategy")}
print("R7 per strategy:", S["r7_per_strategy"])
S["r7_threshold_sensitivity"] = {}
for thr in [-1, -2, -3, -5]:
    e = yearly_eval(fam, np.where(fam.spy_hi252_dist > thr, 0.5, 1.0))
    e["cell"] = cell(fam[fam.spy_hi252_dist > thr], f">{thr}%")
    S["r7_threshold_sensitivity"][thr] = e
    print(f"R7 family SPY > {thr}% of high -> 0.5x: N {e['cell']['n']} avgR {e['cell']['avgR']:+.2f} | years better {e['years_better']}/{e['years']} gain {e['gain_pct']:+.1f}% worst {e['worst']:,.0f}")
S["r7_family_by_era"] = {str(e): [cell(g[g.spy_hi252_dist > -2], ">-2"), cell(g[g.spy_hi252_dist <= -2], "<=-2")] for e, g in fam.groupby(pd.cut(fam.year, [2002, 2009, 2016, 2021, 2027], labels=["2003-09", "2010-16", "2017-21", "2022-26"]), observed=True)}
print("R7 by era:", {e: [(c["n"], round(c["avgR"], 2)) for c in v] for e, v in S["r7_family_by_era"].items()})
# SPY forward 10d by the family flag (the beta channel), lag-0 from signal close
import pyarrow.parquet as pq
ROOT = HERE.parents[1]
spy = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"], filters=[("ticker", "=", "SPY")]).to_pandas().set_index("date")["Close"].sort_index()
pos = spy.index.searchsorted(fam["Signal Date"].values)
fwd = (spy.values[np.minimum(pos + 10, len(spy) - 1)] / spy.values[pos] - 1) * 100
S["r7_spy_fwd10"] = dict(near_high=float(fwd[(fam.spy_hi252_dist > -2).values].mean()), off_high=float(fwd[(fam.spy_hi252_dist <= -2).values].mean()))
print("SPY fwd10 by flag:", S["r7_spy_fwd10"])
# ---- OVS P2 cap cost as share
p2c = R["rules"]["ovs_p2_cap_cost"]
S["ovs_p2_cap_cost_share_of_ovs_pnl"] = float(p2c["extra_pnl_if_uncapped"] / ovs.PnL_flat_750k.sum())
# ---- candidate table with expectancy vs win split
cands = []
def add(name, strategy, cells, rule_key, grade, mechanism, verdict):
    lo_, hi_ = cells[0], cells[-1]
    ev = R["rules"].get(rule_key, {})
    cands.append(dict(name=name, strategy=strategy, cut_cell=lo_, rest_or_top_cell=hi_, avgR_gap=float(hi_["avgR"] - lo_["avgR"]), win_gap=float(hi_["win"] - lo_["win"]),
                      mu_over_var_ratio=float(hi_["mu_over_var"] / lo_["mu_over_var"]) if lo_["mu_over_var"] > 0 else None,
                      years_better=f"{ev.get('years_better')}/{ev.get('years')}", gain_pct_equal_risk=ev.get("gain_pct"), cluster_t=ev.get("cluster_t_R_on_mult"), grade=grade, mechanism=mechanism, verdict=verdict))
add("OVS short-window rank extremity (bottom cut)", "Overbot Vol Spike", [cell(ovs[ovs.rank_mean < 94], "mean rank <94"), cell(ovs[ovs.rank_mean >= 94], ">=94")], "R1b_ovs_extremity_bottom_only", "moderate-strong",
    "larger prior overbought move -> larger 2-day reversal (short-term reversal literature: reversal scales with the prior move); holds in both tiers, both paths, all four eras", "propose 0.5x")
add("52wh Breakout in very quiet SPY tape", "52wh Breakout", [cell(T[(T.Strategy == '52wh Breakout') & (T.spy_rv21 < 10)], "SPY rv21 <10%"), cell(T[(T.Strategy == '52wh Breakout') & (T.spy_rv21 >= 10)], ">=10%")], "R3_52wh_spyrv", "weak-moderate",
    "vol clustering + leverage effect: a 63d momentum hold opened at a realized-vol trough spans a likely vol expansion, which for equities is down-skewed", "near-miss (LOYO 64%); prereg only")
add("LT Trend ST OS in a quiet book", "LT Trend ST OS", [cell(T[(T.Strategy == 'LT Trend ST OS') & (T.book_sig_5td <= 2)], "book_sig_5td <=2"), cell(T[(T.Strategy == 'LT Trend ST OS') & (T.book_sig_5td > 2)], ">2")], "R6_lttrend_density", "moderate",
    "signal flow marks market dislocation days; a 1-day single-stock dip on a quiet book is idiosyncratic (no market bounce to ride)", "propose 0.5x, family-level with R7 preferred")
add("Single-stock dip-buys with SPY within 2% of its 252d high", "family: OLV, LT Trend, WCDS, Monday Dip, St OS Sznl", [cell(fam[fam.spy_hi252_dist > -2], "SPY >-2% of high"), cell(fam[fam.spy_hi252_dist <= -2], "<=-2%")], "R7_family_pullback", "moderate",
    "beta channel: SPY 10d forward return after the signal is ~+1.6% when off-high vs ~0% near-high (index pullbacks in uptrends recover); single-stock dips near index highs are idiosyncratic. 4 of 5 strategies agree; OLV alone does not walk forward", "propose 0.5x family band, prereg PIT check on SPY basis")
add("WCDS gap-up derate extension", "Weak Close Decent Sznls", [cell(T[(T.Strategy == 'Weak Close Decent Sznls') & (T.gap_atr > 0.25)], "T+1 gap >0.25 ATR"), cell(T[(T.Strategy == 'Weak Close Decent Sznls') & ~(T.gap_atr > 0.25)], "rest")], "R8_wcds_gapup", "weak",
    "same as the live Monday Dip / MonFri derate: the bounce plays out at the open and the Open-0.25 ATR limit fills worse", "alignment: extend the existing overlay")
add("OLV post-earnings window", "Oversold Low Volume", [cell(T[(T.Strategy == 'Oversold Low Volume') & (T.td_since_last_earn <= 21)], "<=21 td since print"), cell(T[(T.Strategy == 'Oversold Low Volume') & ~(T.td_since_last_earn <= 21)], "rest")], "R5_olv_post_earn", "weak",
    "post-earnings-announcement drift: an oversold low-volume name inside a month of a print is drifting on news, not washing out", "near-miss (LOYO 60%, era-concentrated 2022-26); re-exam trigger")
add("WCDS signal-day move size", "Weak Close Decent Sznls", [cell(T[(T.Strategy == 'Weak Close Decent Sznls') & (T.move1_atr > -0.5)], "move >-0.5 ATR"), cell(T[(T.Strategy == 'Weak Close Decent Sznls') & (T.move1_atr <= -0.5)], "<=-0.5")], "R4b_wcds_move_bottom_only", "weak",
    "a weak close on a small move is drift, on a large move is capitulation", "negative: fails LOYO (62%), reversed in 2022-26")
S["candidates"] = cands
for c in cands:
    print(f"{c['name']:55s} cut N{c['cut_cell']['n']} R{c['cut_cell']['avgR']:+.2f} w{c['cut_cell']['win']:.2f} | rest N{c['rest_or_top_cell']['n']} R{c['rest_or_top_cell']['avgR']:+.2f} w{c['rest_or_top_cell']['win']:.2f} | dR {c['avgR_gap']:+.2f} dWin {c['win_gap']:+.2f} mu/var x{c['mu_over_var_ratio']:.1f} | {c['years_better']} {c['gain_pct_equal_risk']:+.1f}% | {c['verdict']}")


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


R["summary"] = _clean(S)
json.dump(R, open(HERE / "signal_quality_results.json", "w"), indent=1)
print("updated signal_quality_results.json")
