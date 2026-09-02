"""Synthesis: per-strategy and book-level haircut on the ledger mean, with ranges.

keep_s = sel_keep x oos_adj x surv_keep x cost_keep      haircut = 1 - keep
  sel_keep : Bailey/LdP deflation, trials = config versions / 3 (roughly two thirds of the git
             versions are sizing-only and do not re-select the signal set);
             range = [trials = versions, trials = versions/8]; floor 0.25, cap 0.95;
             N < 30 -> base-rate prior 0.50 (range 0.35-0.65)
  oos_adj  : 2026 realized vs trailing-5y, shrunk toward the book's 2026 ratio by
             N; mapped to [0.70, 1.00]; uninformative (N_2026 < 15) -> 0.90
  surv_keep: by instrument/direction (see SURV)
  cost_keep: 1 - cost_share_of_avgR - 0.02 (marginal-fill give-up)
Book keep = PnL-weighted average of strategy keeps, cross-checked against the
direct book-level numbers (DSR grid, raw 2026 ratio, midterm-conditioned ratio).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
G = 0.5772156649
J = lambda n: json.load(open(OUT / n))
dsr, oos, reg, surv, cost, live, freeze = (J("estimation_haircut_dsr.json"), J("estimation_haircut_oos.json"), J("estimation_haircut_regime.json"),
                                            J("estimation_haircut_survivorship.json"), J("estimation_haircut_costs.json"), J("estimation_haircut_live_vs_ledger.json"),
                                            J("estimation_haircut_freeze_dates.json"))
rw = J("estimation_haircut_riskweighted.json")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
var_sr = dsr["per_trade_cs_var_SR"]


def emax(n):
    n = max(2, int(round(n)))
    return float(np.sqrt(var_sr) * ((1 - G) * st.norm.ppf(1 - 1 / n) + G * st.norm.ppf(1 - 1 / (n * np.e))))


# instrument class per strategy from the ledger tickers
etf_like = set(["SPY", "QQQ", "^GSPC", "^NDX", "IWM", "DIA"])
cls = {}
for s, g in led.groupby("Strategy"):
    tk = g["Ticker"]
    share_etf = tk.isin(etf_like).mean()
    share_of = (g["Tier"] == "Overflow").mean()
    d = g["Direction"].mode().iloc[0]
    if share_etf > 0.9 or s in ("Sector BO", "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade"):
        cls[s] = ("etf", d, share_of)
    else:
        cls[s] = ("stock", d, share_of)
# survivorship keep: long single-stock names carry it; the missing collapsed names are
# ~half of the 2003 listing population; long collapse-bucket avgR is ~0 vs 0.56 (04_survivorship);
# scenario: missing terminal-year names = 2.5x base delist rate (perf-related 3%/yr overflow,
# 1%/yr liquid) of long signals at avgR -0.3 -> inflation ~ share x (avgR + 0.3)
SURV_LONG_STOCK = {"overflow": (0.85, 0.80, 0.92), "liquid": (0.95, 0.92, 0.98)}  # (central, low, high) keep
rows = {}
for s, g in led.groupby("Strategy"):
    n = len(g); avg = float(g["R_Multiple"].mean()); sr = avg / float(g["R_Multiple"].std(ddof=1))
    ver = freeze[s]["n_versions"]
    kind, d, share_of = cls[s]
    # selection keep
    if n < 30:
        sel = (0.50, 0.35, 0.65); sel_note = f"N={n}: base-rate prior (pilot); DSR meaningless"
    else:
        k = lambda tr: float(np.clip(1 - emax(tr) / sr, 0.25, 0.95))
        sel = (k(ver / 3), k(ver), k(ver / 8)); sel_note = f"trials versions/3={ver/3:.0f}: keep {sel[0]:.2f} [ver {sel[1]:.2f}, ver/8 {sel[2]:.2f}]"
    # oos adj
    p = rw["per_strategy_2026_vs_2021_25_riskweighted"].get(s)
    book_ratio = rw["yearly_ratio_riskweighted_summary"]["ratio_2026"]   # 0.52, risk-weighted, at 2026 strategy risk mix
    if p and p["N_2026"] >= 15 and p["ratio"] is not None:
        w = p["N_2026"] / (p["N_2026"] + 40)
        r_shr = w * p["ratio"] + (1 - w) * book_ratio
        oos_adj = (float(np.clip(0.70 + 0.30 * min(1.0, r_shr), 0.70, 1.0)), 0.70, 1.0)
        oos_note = f"2026 N={p['N_2026']} PnL/risk {p['Rrisk_2026']:.2f} vs 2021-25 {p['Rrisk_2021_25']:.2f} ratio {p['ratio']:.2f}, shrunk {r_shr:.2f} (risk-weighted)"
    elif p and p["ratio"] is not None:
        oos_adj = (0.90, 0.75, 1.0); oos_note = f"2026 N={p['N_2026']} (uninformative) PnL/risk {p['Rrisk_2026']:.2f}"
    else:
        oos_adj = (0.90, 0.75, 1.0); oos_note = "no 2026 trades (gated or rare)"
    # survivorship
    if kind == "stock" and d == "Long":
        c, lo, hi = SURV_LONG_STOCK["overflow"] if share_of > 0.5 else SURV_LONG_STOCK["liquid"]
        sv = (c, lo, hi); sv_note = f"long single-stock, overflow share {share_of:.0%}"
    elif kind == "stock" and d == "Short":
        sv = (1.0, 0.95, 1.03); sv_note = "short single-stock: missing collapsers understate, missing takeover targets overstate; net ~0"
    else:
        sv = (1.0, 1.0, 1.0); sv_note = "index/ETF: no survivorship"
    # cost
    cs = cost["cost_by_strategy"][s]["cost_share_of_avgR"]
    ck = (float(1 - cs - 0.02), float(1 - 1.5 * cs - 0.04), float(1 - 0.7 * cs - 0.01))
    keep_c = sel[0] * oos_adj[0] * sv[0] * ck[0]
    keep_lo = sel[1] * oos_adj[1] * sv[1] * ck[1]
    keep_hi = sel[2] * oos_adj[2] * sv[2] * ck[2]
    rows[s] = {"N": n, "avgR_ledger": avg, "versions": ver, "class": f"{kind}/{d}", "overflow_share": share_of,
               "sel_keep": sel, "sel_note": sel_note, "oos_adj": oos_adj, "oos_note": oos_note, "surv_keep": sv, "surv_note": sv_note,
               "cost_keep": ck, "keep_central": keep_c, "keep_range": [keep_lo, keep_hi], "haircut_central": 1 - keep_c, "haircut_range": [1 - keep_hi, 1 - keep_lo],
               "avgR_haircut_central": avg * keep_c, "pnl_share": float(g["PnL_flat_750k"].sum() / led["PnL_flat_750k"].sum())}
tab = pd.DataFrame({s: {"N": r["N"], "avgR": round(r["avgR_ledger"], 2), "ver": r["versions"], "cls": r["class"], "sel": round(r["sel_keep"][0], 2), "oos": round(r["oos_adj"][0], 2), "surv": r["surv_keep"][0], "cost": round(r["cost_keep"][0], 2),
                        "keep": round(r["keep_central"], 2), "lo": round(r["keep_range"][0], 2), "hi": round(r["keep_range"][1], 2), "haircut": round(r["haircut_central"], 2), "avgR_hc": round(r["avgR_haircut_central"], 2)} for s, r in rows.items()}).T
pd.set_option("display.width", 250)
print(tab.sort_values("keep", ascending=False).to_string())

# book-level
w = pd.Series({s: r["pnl_share"] for s, r in rows.items()}).clip(lower=0)
w = w / w.sum()
book_keep = float(sum(w[s] * rows[s]["keep_central"] for s in rows))
book_lo = float(sum(w[s] * rows[s]["keep_range"][0] for s in rows))
book_hi = float(sum(w[s] * rows[s]["keep_range"][1] for s in rows))
direct = {
    "dsr_keep_2016plus_N31_N250": [dsr["book_2016+"]["grid"]["31"]["keep_fraction"], dsr["book_2016+"]["grid"]["250"]["keep_fraction"]],
    "dsr_keep_2003plus_N31_N250": [dsr["book_2003+"]["grid"]["31"]["keep_fraction"], dsr["book_2003+"]["grid"]["250"]["keep_fraction"]],
    "raw_2026_ratio_vs_trailing5_at_mix": reg["yearly_ratio_summary"]["ratio_2026"],
    "raw_2026_ratio_vs_prerepo_at_mix": oos["pooled"]["post_repo_ratio"], "raw_2026_ratio_ci95_dayblock": oos["pooled"]["post_repo_ratio_ci95_dayblock"],
    "rule_freeze_oos_ratio": oos["pooled"]["OOS_over_IS_mix_ratio"], "rule_freeze_oos_N": oos["pooled"]["OOS_rule"]["N"], "rule_freeze_oos_ci95": oos["pooled"]["OOS_ratio_ci95_dayblock"],
    "midterm_conditioned_2026_ratio": reg["midterm"]["book_2026_JanAug"]["avgR"] / reg["midterm"]["book_midterm_JanAug_pre2026"]["avgR"],
    "strategy_x_dial_mix_ratio_2026": reg["strategy_x_dial_mix_ratio_2026"],
    "historical_yearly_ratio_dist": reg["yearly_ratio_summary"],
    "book_cost_share": cost["book_cost_share"], "marginal_fill_giveup_share": 1 - cost["entry_marginal_<=0.02 ATR"]["avgR_if_marginal_fill_50pct"] / cost["entry_marginal_<=0.02 ATR"]["avgR_all"],
    "cross_sectional_shrink_signal_share": dsr["cross_sectional_shrinkage"]["signal_share"],
    "live_staged_ledger_R_2026": live["ledger_R_on_live_staged"],
}
# a book number that blends the bottom-up composite with the direct OOS evidence
direct["riskweighted_2026_ratio_vs_trailing5_at_risk_mix"] = rw["yearly_ratio_riskweighted_summary"]["ratio_2026"]
direct["riskweighted_yearly_ratio_dist"] = rw["yearly_ratio_riskweighted_summary"]
direct["riskweighted_midterm_conditioned_2026_ratio"] = rw["midterm_riskweighted"]["by_year"]["2026"]["R_per_risk"] / rw["midterm_riskweighted"]["midterm_pre2026_JanAug_R_per_risk"]
direct["riskweighted_2026_vs_nonmidterm_JanAug"] = rw["midterm_riskweighted"]["by_year"]["2026"]["R_per_risk"] / rw["midterm_riskweighted"]["nonmidterm_JanAug_R_per_risk"]
# direct blend: half structural (DSR x costs x survivorship), half realized-2026 (raw risk-weighted ratio and the
# midterm-conditioned ratio capped at 1, averaged)
structural = direct["dsr_keep_2016plus_N31_N250"][1] * (1 - cost["book_cost_share"] - 0.02) * 0.95
realized = 0.5 * direct["riskweighted_2026_ratio_vs_trailing5_at_risk_mix"] + 0.5 * min(1.0, direct["riskweighted_midterm_conditioned_2026_ratio"])
direct_component = 0.5 * structural + 0.5 * realized
direct["structural_keep"] = float(structural); direct["realized_2026_keep"] = float(realized)
book = {"keep_bottom_up_pnl_weighted": book_keep, "keep_range_bottom_up": [book_lo, book_hi],
        "keep_direct_blend": float(direct_component),
        "keep_recommended": float(0.5 * book_keep + 0.5 * direct_component),
        "haircut_recommended": float(1 - (0.5 * book_keep + 0.5 * direct_component)),
        "haircut_range": [float(1 - book_hi), float(1 - min(book_lo, direct["raw_2026_ratio_vs_prerepo_at_mix"]))],
        "haircut_range_note": "low end = every component at its generous bound; high end = bottom-up pessimistic bound or the raw unweighted 2026 ratio, whichever is worse",
        "direct_evidence": direct}
print("\nBOOK:", json.dumps({k: v for k, v in book.items() if k != "direct_evidence"}, indent=1))
print("direct:", json.dumps(direct, indent=1, default=str)[:3000])
# daily-series cross-check (the basis the plan sizes on): 2026 YTD Sharpe vs history, book + strategies
sd_ = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd_["dates"])
tot = pd.Series(sd_["total_flat"], index=dates, dtype=float) / 750_000.0
S = pd.DataFrame(sd_["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / 750_000.0
def shp(x):
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else None
daily = {"book_2026ytd_sharpe": shp(tot[tot.index >= "2026-01-01"]), "book_2016_2025_sharpe": shp(tot[(tot.index >= "2016-01-01") & (tot.index < "2026-01-01")]),
         "book_2026ytd_ann_ret": float(tot[tot.index >= "2026-01-01"].mean() * 252), "book_2016_2025_ann_ret": float(tot[(tot.index >= "2016-01-01") & (tot.index < "2026-01-01")].mean() * 252),
         "book_midterm_years_JanJul_sharpe": {y: shp(tot[(tot.index >= f"{y}-01-01") & (tot.index <= f"{y}-08-07")]) for y in (2006, 2010, 2014, 2018, 2022, 2026)},
         "book_midterm_years_JanJul_ann_ret": {y: float(tot[(tot.index >= f"{y}-01-01") & (tot.index <= f"{y}-08-07")].mean() * 252) for y in (2006, 2010, 2014, 2018, 2022, 2026)},
         "series_end": str(dates.max().date()),
         "per_strategy_2026ytd_vs_2016_25": {c: {"sh_2026": shp(strat[c][strat.index >= "2026-01-01"]), "sh_2016_25": shp(strat[c][(strat.index >= "2016-01-01") & (strat.index < "2026-01-01")]),
                                                  "ret_2026_pct_nav": float(strat[c][strat.index >= "2026-01-01"].sum() * 100), "ret_2016_25_avg_yr_pct_nav": float(strat[c][(strat.index >= "2016-01-01") & (strat.index < "2026-01-01")].sum() * 10)} for c in strat.columns}}
print("\nDAILY-SERIES CHECK:", json.dumps({k: v for k, v in daily.items() if k != "per_strategy_2026ytd_vs_2016_25"}, indent=1))
print(pd.DataFrame(daily["per_strategy_2026ytd_vs_2016_25"]).T.round(2).to_string())
book["daily_series_check"] = daily

# trust tiers (judgement on top of the numbers; reasons in the findings)
TRUST = {
    "more": ["SPY QQQ MonFri Reversion", "Indices Oversold Bounce", "3x ETF Overbot Fade", "Overbot Vol Spike"],
    "average": ["Monday Dip", "ATR Extended Gap Up", "3x Bear ETF Overbot Fade", "Sector BO", "Oversold Low Volume"],
    "less": ["Weak Close Decent Sznls", "52wh Breakout", "LT Trend ST OS", "St OS Sznl", "3x Leader Gap Fade", "Monthly Weak Close"],
}
res = {"per_strategy": rows, "book": book, "trust_tiers": TRUST, "method": __doc__}
(OUT / "estimation_haircut_results.json").write_text(json.dumps(res, indent=1, default=str))
