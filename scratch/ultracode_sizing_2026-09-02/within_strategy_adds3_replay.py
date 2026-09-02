"""Within-strategy adds, step 3: derive the growth-optimal add rule per strategy and
replay candidate rules on the ledger (flat $750k, realized-at-exit AND leg-MTM).

Rule forms (m = multiplier on the leg's base size; composes with every overlay that
is not the subject of the rule):
  flat            m = 1 (for OLV this REMOVES the recency ladder; the residual
                  earnings override / ticker-cap clip is kept)
  current         what the ledger booked (Size_Mult)
  edge_wf         walk-forward edge-proportional: m = clip(mu_bucket / mu_solo, lo, hi)
                  with bucket means (by open-leg depth) estimated on all OTHER years,
                  shrunk toward the solo mean with N0 = 20 pseudo-trades
  kelly_wf        edge_wf divided by the Rung-5 concurrency factor 1 + n_open * rho_s
                  (rho_s = strategy-level implied pairwise rho from leg MTM)
  var_parity      m = 1 / sqrt(1 + n_open * rho_s)   (pure variance control)
  adds_1.25       m = 1.25 on any add (n_open >= 1), 1.0 solo
  + per-strategy specials (OLV ladder re-keys, 52wh deep-stack cuts, 3x Bear derate
    on/off, sector / same-ticker / late-add cuts).
Metrics: total PnL, risk deployed, PnL per $ risk, realized-at-exit worst 21d and
maxDD, MTM worst 21d / maxDD / ann. Sharpe on the sleeve, and the same at EQUAL
total risk (rule rescaled so it deploys the current rule's risk).
Writes within_strategy_adds_replay.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
df = pd.read_parquet(OUT / "within_strategy_adds_features.parquet")
M = pd.read_parquet(OUT / "within_strategy_adds_mtm.parquet")
MARG = json.load(open(OUT / "within_strategy_adds_marginal.json"))
RES: dict = {}

RHO_S = {s: max(0.0, float(np.nanmean([r["implied_rho"] for r in rows if r["legs"] != "1"]))) if len(rows) > 1 else 0.0
         for s, rows in MARG["_implied_rho_from_leg_mtm"].items()}
print("strategy-level implied rho (from leg MTM):", {k: round(v, 2) for k, v in RHO_S.items()})

DEPTH_EDGES = {  # open-leg depth buckets used by the walk-forward edge rule
    "Oversold Low Volume": [-1, 0, 2, 5, 99], "52wh Breakout": [-1, 0, 2, 5, 99], "Overbot Vol Spike": [-1, 0, 2, 5, 12, 99],
    "Weak Close Decent Sznls": [-1, 0, 1, 99], "LT Trend ST OS": [-1, 0, 2, 99], "3x ETF Overbot Fade": [-1, 0, 2, 99],
    "3x Bear ETF Overbot Fade": [-1, 0, 99], "3x Leader Gap Fade": [-1, 0, 99]}

def metrics(g, factor, label):
    """factor = new size / booked size per leg."""
    f = pd.Series(np.asarray(factor, float), index=g.index)
    pnl = g.PnL_flat_750k * f; risk = g.Risk_flat_750k * f
    ex = pnl.groupby(g["Exit Date"]).sum()
    ex = ex.reindex(pd.bdate_range(ex.index.min(), ex.index.max())).fillna(0)
    eq = ex.cumsum(); dd_ex = float((eq - eq.cummax()).min()); w21_ex = float(ex.rolling(21).sum().min())
    m = M[M.Strategy == g.Strategy.iloc[0]]
    mp = (m.pnl * m.idx.map(f)).groupby(m.date).sum()
    mp = mp.reindex(pd.bdate_range(mp.index.min(), mp.index.max())).fillna(0)
    eqm = mp.cumsum(); dd_m = float((eqm - eqm.cummax()).min()); w21_m = float(mp.rolling(21).sum().min())
    act = mp[mp != 0]
    sh = float(act.mean() / act.std() * np.sqrt(252)) if act.std() > 0 else np.nan
    return dict(rule=label, total_pnl=float(pnl.sum()), risk_deployed=float(risk.sum()), pnl_per_risk=float(pnl.sum() / risk.sum()),
                worst21_exit=w21_ex, maxdd_exit=dd_ex, worst21_mtm=w21_m, maxdd_mtm=dd_m, sleeve_sharpe_active=sh,
                sd_day_mtm=float(act.std()), worst_day_mtm=float(mp.min()), legs_changed=int((np.abs(f - 1) > 1e-6).sum()),
                mean_factor=float(f.mean()))

def add_equal_risk(rows):
    cur = [r for r in rows if r["rule"] == "current"][0]
    for r in rows:
        k = cur["risk_deployed"] / r["risk_deployed"]
        r["pnl_at_equal_risk"] = r["total_pnl"] * k; r["maxdd_mtm_at_equal_risk"] = r["maxdd_mtm"] * k; r["worst21_mtm_at_equal_risk"] = r["worst21_mtm"] * k
        r["d_pnl_vs_current"] = r["total_pnl"] - cur["total_pnl"]; r["d_pnl_equal_risk_vs_current"] = r["pnl_at_equal_risk"] - cur["total_pnl"]
    return rows

def wf_edge_mult(g, edges, lo, hi, n0=20, conc=False):
    """walk-forward edge-proportional multiplier per leg."""
    b = pd.cut(g.n_open, edges, labels=False)
    m = pd.Series(1.0, index=g.index)
    for y in sorted(g.yr.unique()):
        tr = g[g.yr != y]; btr = b[g.yr != y]
        mu0 = tr.R_Multiple[btr == 0].mean(); n_0 = (btr == 0).sum()
        if not np.isfinite(mu0) or n_0 < 5:
            continue
        for k in b[g.yr == y].unique():
            x = tr.R_Multiple[btr == k]
            mu_shr = (x.sum() + n0 * mu0) / (len(x) + n0)
            ratio = mu_shr / mu0 if mu0 > 0 else 1.0
            m[(g.yr == y) & (b == k)] = float(np.clip(ratio, lo, hi))
    if conc:
        m = m / (1 + g.n_open * RHO_S.get(g.Strategy.iloc[0], 0.0))
    return m

for s in ["Oversold Low Volume", "52wh Breakout", "Overbot Vol Spike", "Weak Close Decent Sznls", "LT Trend ST OS",
          "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade"]:
    g = df[df.Strategy == s].copy()
    base = g.Size_Mult.copy()                        # booked
    if s == "Oversold Low Volume":
        core = g.rung_ladder                          # the subject overlay; residual (earnings override / cap clip) is kept
    else:
        core = pd.Series(1.0, index=g.index)
    resid = base / core
    rules = {"current": base, "flat": resid * 1.0}
    rules["edge_wf"] = resid * wf_edge_mult(g, DEPTH_EDGES[s], 0.5, 1.5)
    rules["edge_wf_wide"] = resid * wf_edge_mult(g, DEPTH_EDGES[s], 0.5, 2.0)
    rules["kelly_wf"] = resid * wf_edge_mult(g, DEPTH_EDGES[s], 0.5, 2.0, conc=True)
    rules["var_parity"] = resid / np.sqrt(1 + g.n_open * RHO_S.get(s, 0.0))
    rules["adds_1.25"] = resid * np.where(g.n_open >= 1, 1.25, 1.0)
    rules["adds_1.5"] = resid * np.where(g.n_open >= 1, 1.5, 1.0)
    rules["same_ticker_0.5"] = resid * np.where(g.n_same_ticker >= 1, 0.5, 1.0)
    rules["same_sector2_0.5"] = resid * np.where(g.n_same_sector >= 2, 0.5, 1.0)
    rules["late_add_gt5td_0.5"] = resid * np.where(g.stack_age_td > 5, 0.5, 1.0)
    rules["rho_gt.6_0.5"] = resid * np.where(g.rho63_mean.fillna(0) > 0.6, 0.5, 1.0)
    if s == "Oversold Low Volume":
        rules["ladder_current_by_ticker"] = base
        rules["ladder_by_depth[.5,.7,1]"] = resid * np.select([g.n_open == 0, g.n_open <= 2], [0.5, 0.7], 1.0)
        rules["solo_0.5_adds_1.0"] = resid * np.where(g.n_open == 0, 0.5, 1.0)
        rules["solo_0.5_adds_1.0_sameTk_1.25"] = resid * np.where(g.n_open == 0, 0.5, np.where(g.n_same_ticker >= 1, 1.25, 1.0))
        rules["ladder_ticker_OR_depth"] = resid * np.maximum(g.rung_ladder, np.select([g.n_open == 0, g.n_open <= 2], [0.5, 0.7], 1.0))
        rules["ladder_ticker_x_depth_boost"] = resid * g.rung_ladder * np.where(g.n_open >= 3, 1.5, 1.0)
        rules["depth_3plus_1.5"] = resid * np.where(g.n_open >= 3, 1.5, 1.0)
        rules["flat_then_cap_sameTk_2nd_0.5"] = resid * np.where(g.n_same_ticker >= 2, 0.5, 1.0)
    if s == "52wh Breakout":
        rules["open>=5_0.5 (plan D3)"] = resid * np.where(g.n_open >= 5, 0.5, 1.0)
        rules["open>=6_0.5"] = resid * np.where(g.n_open >= 6, 0.5, 1.0)
        rules["open>=5_0.5_2010+"] = resid * np.where((g.n_open >= 5) & (g.yr >= 2010), 0.5, 1.0)
        rules["sec_share>=50%_and_open>=3_0.5"] = resid * np.where((g.n_open >= 3) & (g.n_same_sector / g.n_open.clip(lower=1) >= 0.5), 0.5, 1.0)
        rules["open>=5_0"] = resid * np.where(g.n_open >= 5, 0.0, 1.0)
    if s == "3x Bear ETF Overbot Fade":
        nsig = g.groupby("Signal Date").Ticker.transform("size")
        derate = np.maximum(0.3, 1 - 0.1 * (nsig - 1))
        rules["no_same_day_derate"] = base / derate
        rules["same_day_n>=3_0.75"] = base / derate * np.where(nsig >= 3, 0.75, 1.0)
    if s == "Overbot Vol Spike":
        rules["depth>=6_1.25"] = resid * np.where(g.n_open >= 6, 1.25, 1.0)
        rules["same_sector2_1.25"] = resid * np.where(g.n_same_sector >= 2, 1.25, 1.0)
    rows = []
    for k, v in rules.items():
        rows.append(metrics(g, np.asarray(v) / np.asarray(base), k))
    rows = add_equal_risk(rows)
    R = pd.DataFrame(rows)
    print(f"\n================ {s} (N={len(g)}, {g.yr.min()}-{g.yr.max()}) rho_s={RHO_S.get(s, 0):.2f}")
    print(R[["rule", "total_pnl", "risk_deployed", "pnl_per_risk", "worst21_exit", "maxdd_exit", "worst21_mtm", "maxdd_mtm", "sleeve_sharpe_active", "legs_changed", "pnl_at_equal_risk", "d_pnl_vs_current", "d_pnl_equal_risk_vs_current"]].to_string(index=False))
    RES[s] = rows
    # walk-forward multipliers actually used by edge_wf (by depth bucket, last fit = all years but 2026)
    b = pd.cut(g.n_open, DEPTH_EDGES[s], labels=False)
    mult_tab = pd.DataFrame({"bucket": b, "edge_wf": rules["edge_wf"] / resid, "kelly_wf": rules["kelly_wf"] / resid}).groupby("bucket").agg(["mean", "min", "max"]).round(2)
    print("edge_wf / kelly_wf multipliers by depth bucket (mean/min/max across held-out years):"); print(mult_tab.to_string())
    RES[s + "_wf_mults"] = {str(k): {"edge_wf_mean": float(v[("edge_wf", "mean")]), "edge_wf_min": float(v[("edge_wf", "min")]), "edge_wf_max": float(v[("edge_wf", "max")]),
                                     "kelly_wf_mean": float(v[("kelly_wf", "mean")])} for k, v in mult_tab.iterrows()}

# ---- OVS: does the per-strategy 250 bps daily cap bind on the high-edge cluster days?
o = df[df.Strategy == "Overbot Vol Spike"].copy()
day = o.groupby("Signal Date").agg(n=("Ticker", "size"), risk=("Risk_flat_750k", "sum"), pnl=("PnL_flat_750k", "sum"), avgR=("R_Multiple", "mean"), risk_full=("unit_risk", "sum"))
day["risk_bps"] = day.risk / NAV * 1e4; day["risk_full_bps"] = day.risk_full / NAV * 1e4
day["nb"] = pd.cut(day.n, [0, 1, 2, 5, 12, 99], labels=["1", "2", "3-5", "6-12", "13+"])
cap = day.groupby("nb", observed=True).agg(days=("n", "size"), fills=("n", "sum"), avgR=("avgR", "mean"), risk_bps_mean=("risk_bps", "mean"), risk_bps_max=("risk_bps", "max"),
                                            unit_risk_bps_mean=("risk_full_bps", "mean"), pnl_day=("pnl", "mean"), pnl_tot=("pnl", "sum"), days_at_cap=("risk_bps", lambda x: int((x >= 240).sum())))
print("\n==== OVS: fills per signal day vs booked risk (cap 250 bps/day effective; P2 rows at 0.2x; unit = every fill at 60 bps) ====")
print(cap.round(2).to_string())
RES["OVS_daily_cap_footprint"] = cap.round(3).reset_index().to_dict("records")
big = day[day.n >= 6]
print(f"OVS days with >= 6 fills: {len(big)}; mean booked risk {big.risk_bps.mean():.0f} bps vs {big.risk_full_bps.mean():.0f} bps if every fill were 60 bps; avgR {big.avgR.mean():.2f}; total PnL ${big.pnl.sum():,.0f}")

# ---- 52wh 2010+ depth table for parity with the plan's D3 cell
b = df[(df.Strategy == "52wh Breakout") & (df.yr >= 2010)].copy()
b["nb"] = pd.cut(b.n_open, [-1, 0, 2, 4, 5, 99], labels=["0", "1-2", "3-4", "5", "6+"])
t = b.groupby("nb", observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), win=("R_Multiple", lambda s: (s > 0).mean()), pnl=("PnL_flat_750k", "sum"), episodes=("episode", "nunique"), rho=("rho63_mean", "mean"), same_sec=("n_same_sector", "mean"))
print("\n==== 52wh Breakout 2010+ by open legs (plan D3 parity) ===="); print(t.round(3).to_string())
RES["b52_2010_depth"] = t.round(4).reset_index().to_dict("records")
bb = b[b.n_open >= 5]
print("52wh legs with >=5 open, by year:", bb.groupby("yr").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum")).round(2).to_dict("index"))
RES["b52_deep_by_year"] = {int(k): v for k, v in bb.groupby("yr").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum")).round(3).to_dict("index").items()}

json.dump(RES, open(OUT / "within_strategy_adds_replay.json", "w"), indent=1, default=float)
print("\nwrote within_strategy_adds_replay.json")
