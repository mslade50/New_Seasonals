"""Robust-Bayesian per-strategy allocation (2026-09-02).

Re-runs the baseline plan's L1 daily-series route (Ledoit-Wolf-shrunk Sigma,
mu shrunk 50% toward an equal-Sharpe prior, w = Sigma^-1 mu, half-blend to 1)
with TWO changes from the skeptic's lens:
  (1) each strategy's mean is first multiplied by its estimation-haircut keep
      factor keep_s (estimation_haircut_results.json), so the tilt allocates
      on TRUSTED edge rather than ledger edge;
  (2) the shipping clip is narrowed from [0.6, 1.4] to [0.7, 1.3].
Walk-forward 2014-2026 (fit on data before Jan 1 of year Y, apply to Y, in-sample
vol-matched to equal weights) for: equal, plan (half-blend expanding, clip
0.6-1.4), keep-adjusted (clip 0.7-1.3), keep-adjusted with the plan's clip,
trust-only (mult = keep_s / mean keep, no covariance solve), and a
minimum-regret average of plan and trust-only.
CAVEAT: keep_s is estimated on the full sample incl. 2026, so the keep variants
carry lookahead in the walk-forward; the comparison is a sensitivity, not an
out-of-sample test of the keep vector itself.
Writes robust_bayes_03_allocation.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
full = strat[strat.index >= "2006-01-01"]
keep = {s: v["keep_central"] for s, v in json.load(open(HERE / "estimation_haircut_results.json"))["per_strategy"].items()}
# judgement overrides flagged by the haircut study itself (mechanical outliers)
keep_j = dict(keep)
keep_j["Overbot Vol Spike"] = 0.48
keep_j["52wh Breakout"] = 0.42
KEEP = pd.Series(keep_j)
print("keep factors used (judgement-adjusted OVS 0.48, 52wh 0.42):")
print(KEEP.sort_values().round(2).to_string())

def shrink_cov(X, delta=0.3):
    mu = X.mean(0); Xc = X - mu; Sig = Xc.T @ Xc / len(X)
    F = np.diag(np.diag(Sig))
    return delta * F + (1 - delta) * Sig, mu

def weights(X: pd.DataFrame, mu_shrink=0.5, keep_vec: pd.Series | None = None):
    Sig, mu = shrink_cov(X.values)
    if keep_vec is not None:
        mu = mu * keep_vec.reindex(X.columns).fillna(0.5).values
    sig = np.sqrt(np.diag(Sig)); Sbar = np.mean(mu / sig)
    mt = mu_shrink * mu + (1 - mu_shrink) * Sbar * sig
    w = np.linalg.solve(Sig, mt); w = w / np.abs(w).sum() * len(w)
    return pd.Series(w, index=X.columns)

def stats(r: pd.Series):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252),
                maxdd=dd * 100, pnl_over_maxdd=r.mean() * 252 / abs(dd))

variants = ["equal", "plan_0.6_1.4", "keep_0.7_1.3", "keep_0.6_1.4", "trust_only_0.7_1.3", "minregret"]
res = {k: [] for k in variants}; yr = []
for Y in range(2014, 2027):
    tr = full[full.index < f"{Y}-01-01"]; te = full[(full.index >= f"{Y}-01-01") & (full.index < f"{Y+1}-01-01")]
    cols = tr.columns[(tr != 0).mean() > 0.02]; tr = tr[cols]; te = te.reindex(columns=cols).fillna(0)
    base_vol = tr.sum(1).std()
    sc = lambda w: w * (base_vol / (tr @ w).std())
    wE = weights(tr); wK = weights(tr, keep_vec=KEEP)
    kmean = KEEP.reindex(cols).fillna(0.5).mean()
    trust = (KEEP.reindex(cols).fillna(0.5) / kmean).clip(0.7, 1.3)
    W = {"equal": pd.Series(1.0, index=cols),
         "plan_0.6_1.4": (0.5 * wE + 0.5).clip(0.6, 1.4),
         "keep_0.7_1.3": (0.5 * wK + 0.5).clip(0.7, 1.3),
         "keep_0.6_1.4": (0.5 * wK + 0.5).clip(0.6, 1.4),
         "trust_only_0.7_1.3": trust}
    W["minregret"] = 0.5 * W["plan_0.6_1.4"] + 0.5 * W["trust_only_0.7_1.3"]
    row = {"year": Y}
    for k in variants:
        w = sc(W[k]); r = te @ w; res[k].append(r); row[k] = r.sum() * 100
    yr.append(row)
Y = pd.DataFrame(yr)
print("\nper-year PnL (% NAV), in-sample vol-matched to equal:")
print(Y.round(1).to_string(index=False))
T = pd.DataFrame({k: stats(pd.concat(v)) for k, v in res.items()}).T
gate = {}
for k in variants:
    if k == "equal": continue
    d = (Y[k] - Y["equal"]); loss = (d / Y["equal"].abs()).min()
    gate[k] = dict(years_better=int((d > 0).sum()), worst_year_vs_equal_pct=float(loss * 100), total_gain_pctnav=float(d.sum()),
                   pnl_over_maxdd_gain_pct=float((T.loc[k, "pnl_over_maxdd"] / T.loc["equal", "pnl_over_maxdd"] - 1) * 100))
print("\nwalk-forward 2014-2026 summary:")
print(T.round(3).to_string())
print(pd.DataFrame(gate).T.round(2).to_string())

# shipping multipliers from the fit through 2025 (plan convention) and through 2026-08 (all data)
tr = full[full.index < "2026-01-01"]; cols = tr.columns[(tr != 0).mean() > 0.02]
wE = weights(tr[cols]); wK = weights(tr[cols], keep_vec=KEEP)
kmean = KEEP.reindex(cols).mean()
ship = pd.DataFrame({"plan_mult": (0.5 * wE + 0.5).clip(0.6, 1.4), "keep_w": wK, "keep_mult_0.7_1.3": (0.5 * wK + 0.5).clip(0.7, 1.3),
                     "trust_only": (KEEP.reindex(cols) / kmean).clip(0.7, 1.3)})
ship["minregret"] = 0.5 * ship["plan_mult"] + 0.5 * ship["trust_only"]
ship["keep_s"] = KEEP.reindex(cols)
ship = ship.sort_values("keep_mult_0.7_1.3")
print("\nshipping multipliers (fit through 2025):")
print(ship.round(2).to_string())
# per-strategy daily Sharpe and PnL share for reference
ref = pd.DataFrame({"sharpe_2010": full[full.index >= "2010"].apply(lambda c: c.mean() / c.std() * np.sqrt(252)),
                    "pnl_share_2010": full[full.index >= "2010"].sum() / full[full.index >= "2010"].sum().sum()})
json.dump(dict(keep_used=KEEP.round(3).to_dict(), per_year=Y.round(2).to_dict("records"), summary=T.round(4).to_dict("index"),
               gate=gate, shipping=ship.round(3).to_dict("index"), reference=ref.round(3).to_dict("index")),
          open(HERE / "robust_bayes_03_allocation.json", "w"), indent=1)
print("\nwrote robust_bayes_03_allocation.json")
