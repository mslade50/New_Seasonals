"""Kelly phase-3 estimation, allocation, and drawdown computation.

Consumes only the scratch replay produced by
kelly_build_current_replay_2026_08_05.py. Outputs reproducible CSV/JSON/PNG
artifacts under scratch/. No production engine/config/data file is modified.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_config import ACCOUNT_VALUE, GLOBAL_RISK_MULTIPLIER, STRATEGY_BOOK


STAMP = "2026-08-05"
SCRATCH = ROOT / "scratch"
SIGNALS_IN = SCRATCH / f"kelly_current_signals_{STAMP}.parquet"
DAILY_IN = SCRATCH / f"kelly_current_daily_components_{STAMP}.parquet"
DAILY_TIER_IN = SCRATCH / f"kelly_current_daily_tier_components_{STAMP}.parquet"

EST_OUT = SCRATCH / f"kelly_estimates_{STAMP}.csv"
EST_LIQ_OUT = SCRATCH / f"kelly_estimates_liquid_{STAMP}.csv"
ALLOC_OUT = SCRATCH / f"kelly_allocations_{STAMP}.csv"
SCENARIO_OUT = SCRATCH / f"kelly_scenario_multipliers_{STAMP}.csv"
CURVE_OUT = SCRATCH / f"kelly_growth_drawdown_curve_{STAMP}.csv"
CURVE_PNG = SCRATCH / f"kelly_growth_drawdown_curve_{STAMP}.png"
CORR_FULL_OUT = SCRATCH / f"kelly_corr_full_{STAMP}.csv"
CORR_CRISIS_OUT = SCRATCH / f"kelly_corr_crisis_{STAMP}.csv"
SUMMARY_OUT = SCRATCH / f"kelly_phase3_analytic_summary_{STAMP}.json"

NAV = float(ACCOUNT_VALUE)
PILOTS = {
    "3x Bear ETF Overbot Fade",
    "3x Leader Gap Fade",
    "Monthly Weak Close",
}
CRISIS_MASK_LABEL = "2020-03 + 2022 + 2024-08 + 2025-04"
N_BOOT = 20_000
BOOT_HORIZON = 252
BOOT_MEAN_BLOCK = 10
BOOT_SEED = 20260805
DD_LIMIT = -0.20 * NAV


def load_inputs():
    signals = pd.read_parquet(SIGNALS_IN)
    for col in ("Date", "Entry Date", "Exit Date"):
        if col in signals.columns:
            signals[col] = pd.to_datetime(signals[col])
    daily = pd.read_parquet(DAILY_IN)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index().fillna(0.0)
    daily_tier = pd.read_parquet(DAILY_TIER_IN)
    daily_tier["date"] = pd.to_datetime(daily_tier["date"])
    daily_tier = daily_tier.set_index("date").sort_index().fillna(0.0)
    return signals, daily, daily_tier


def episode_labels(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates).sort_values().values.astype("datetime64[D]")
    out = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        gap = np.busday_count(d[i - 1], d[i])
        out[i] = out[i - 1] + int(gap > 5)
    return out


def cluster_stats(frame: pd.DataFrame, component: str) -> dict[str, float | str]:
    g = frame[["Date", "R", "Risk_Fraction"]].dropna().copy()
    g = g[g["Risk_Fraction"] > 0].sort_values("Date").reset_index(drop=True)
    n = len(g)
    if not n:
        return {
            "n": 0, "theta": np.nan, "avg_r": np.nan, "se": np.nan,
            "n_eff": 0.0, "clusters": 0, "cluster_mode": "none",
        }
    r = g["R"].to_numpy(float)
    w = g["Risk_Fraction"].to_numpy(float)
    theta = float(np.dot(w, r) / w.sum())
    avg_r = float(np.mean(r))
    wvar = float(np.dot(w, (r - theta) ** 2) / w.sum()) if n > 1 else 0.0
    kish = float(w.sum() ** 2 / np.dot(w, w)) if np.dot(w, w) > 0 else 1.0

    def one(labels: np.ndarray, mode: str):
        labels = np.asarray(labels)
        clusters = int(pd.Series(labels).nunique())
        scores = pd.Series(w * (r - theta)).groupby(labels).sum().to_numpy(float)
        if clusters > 1:
            se2 = clusters / (clusters - 1.0) * float(np.dot(scores, scores)) / (w.sum() ** 2)
            se = math.sqrt(max(se2, 0.0))
        else:
            se = math.sqrt(wvar) if wvar > 0 else 1.0
        if se > 0 and wvar > 0:
            neff = wvar / (se * se)
        else:
            neff = 1.0
        neff = float(np.clip(neff, 1.0, max(1.0, min(kish, clusters, n))))
        return {"se": se, "n_eff": neff, "clusters": clusters, "cluster_mode": mode}

    ep = one(episode_labels(g["Date"]), "gap_gt_5td")
    chosen = ep
    if component.startswith("Overbot Vol Spike"):
        month_labels = g["Date"].dt.to_period("M").astype(str).to_numpy()
        month = one(month_labels, "calendar_month")
        if month["se"] > ep["se"]:
            chosen = month
        selected_mode = chosen["cluster_mode"]
        alternate_mode = "calendar_month" if selected_mode == "gap_gt_5td" else "gap_gt_5td"
        chosen = dict(chosen)
        chosen["cluster_mode"] += "_conservative_vs_" + alternate_mode
    return {
        "n": n,
        "theta": theta,
        "avg_r": avg_r,
        "sd_r": float(np.std(r, ddof=1)) if n > 1 else np.nan,
        "worst_r": float(np.min(r)),
        "best_r": float(np.max(r)),
        "kish_n": kish,
        **chosen,
    }


def weighted_mean(g: pd.DataFrame) -> float:
    h = g[["R", "Risk_Fraction"]].dropna()
    h = h[h["Risk_Fraction"] > 0]
    if h.empty:
        return np.nan
    return float(np.average(h["R"], weights=h["Risk_Fraction"]))


def loyo_conservative(g: pd.DataFrame) -> float:
    years = sorted(pd.to_datetime(g["Date"]).dt.year.unique())
    if len(years) < 2:
        return weighted_mean(g)
    values = []
    gy = pd.to_datetime(g["Date"]).dt.year
    for year in years:
        v = weighted_mean(g[gy != year])
        if np.isfinite(v):
            values.append(v)
    return float(min(values)) if values else weighted_mean(g)


def eb_fit(y: np.ndarray, se: np.ndarray, strength: float = 1.0):
    y = np.asarray(y, float)
    se = np.asarray(se, float)
    finite = np.isfinite(y) & np.isfinite(se) & (se > 0)
    if not finite.any():
        return np.full_like(y, np.nan), np.full_like(y, np.nan), {
            "theta0": np.nan, "tau2": np.nan, "strength": strength,
        }
    yf, sf = y[finite], se[finite]
    max_tau = max(float(np.var(yf, ddof=1)) * 10.0 if len(yf) > 1 else 0.1, 1e-6)

    def nll(tau2: float):
        prior_var = max(tau2 / strength, 1e-14)
        v = sf * sf + prior_var
        theta0 = float(np.sum(yf / v) / np.sum(1.0 / v))
        return 0.5 * float(np.sum(np.log(v) + (yf - theta0) ** 2 / v))

    opt = minimize_scalar(nll, bounds=(0.0, max_tau), method="bounded")
    candidates = [(0.0, nll(0.0)), (float(opt.x), float(opt.fun))]
    tau2 = min(candidates, key=lambda z: z[1])[0]
    prior_var = max(tau2 / strength, 1e-14)
    v = sf * sf + prior_var
    theta0 = float(np.sum(yf / v) / np.sum(1.0 / v))
    kappa_f = prior_var / (prior_var + sf * sf)
    post_f = theta0 + kappa_f * (yf - theta0)

    post = np.full_like(y, np.nan)
    kappa = np.full_like(y, np.nan)
    post[finite] = post_f
    kappa[finite] = kappa_f
    return post, kappa, {"theta0": theta0, "tau2": tau2, "strength": strength}


def standalone_kelly(r: np.ndarray, q: np.ndarray | None = None) -> float:
    r = np.asarray(r, float)
    good = np.isfinite(r)
    if q is not None:
        q = np.asarray(q, float)
        good &= np.isfinite(q) & (q > 0)
    r = r[good]
    if q is not None:
        q = q[good]
        x = q * r
    else:
        x = r
    if len(x) == 0 or float(np.mean(x)) <= 0:
        return 0.0
    neg = x[x < 0]
    if len(neg) == 0:
        return np.inf
    upper = float(np.min(-1.0 / neg)) * 0.999999
    if upper <= 0:
        return 0.0
    res = minimize_scalar(
        lambda f: -float(np.mean(np.log1p(f * x))),
        bounds=(0.0, upper), method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x)


def base_bps_map() -> dict[str, tuple[float, float, float | None]]:
    by_name = {s["name"]: s for s in STRATEGY_BOOK}
    out = {}
    for name, strat in by_name.items():
        eff = float(strat["execution"]["risk_bps"])
        out[name] = (eff / GLOBAL_RISK_MULTIPLIER, eff, None)
    ovs = by_name["Overbot Vol Spike"]["execution"]
    p1e = float(ovs["path1_bps"])
    p2e = float(ovs["path2_bps"])
    out["Overbot Vol Spike P1"] = (p1e / GLOBAL_RISK_MULTIPLIER, p1e, p1e)
    out["Overbot Vol Spike P2"] = (p2e / GLOBAL_RISK_MULTIPLIER, p2e, p2e)
    out.pop("Overbot Vol Spike", None)
    # Only OLV has a different overflow base: 25 nominal -> 37.5 effective.
    n, e, _ = out["Oversold Low Volume"]
    out["Oversold Low Volume"] = (n, e, 25.0 * GLOBAL_RISK_MULTIPLIER)
    return out


def build_estimates(signals: pd.DataFrame, daily: pd.DataFrame,
                    tier: str | None = None, strength: float = 1.0):
    if tier is not None:
        sig = signals[signals["Tier"].eq(tier)].copy()
    else:
        sig = signals.copy()
    components = list(daily.columns)
    rows = []
    recent_start = pd.Timestamp("2018-01-01")
    n_days = len(daily)
    n_days_recent = int((daily.index >= recent_start).sum())
    for comp in components:
        g = sig[sig["Component"].eq(comp)].copy()
        full = cluster_stats(g, comp)
        recent_g = g[g["Date"] >= recent_start]
        recent = cluster_stats(recent_g, comp)
        risk_total = float(g["Risk_Fraction"].sum())
        risk_recent = float(recent_g["Risk_Fraction"].sum())
        r = g["R"].to_numpy(float)
        q = g["Risk_Fraction"].to_numpy(float)
        rows.append({
            "Component": comp,
            "Tier_Basis": tier or "Full",
            "Ledger_Avg_R": full["avg_r"],
            "Risk_Weighted_R_Full": full["theta"],
            "Risk_Weighted_R_LOYO": loyo_conservative(g),
            "Risk_Weighted_R_2018": recent["theta"],
            "SE_Cluster_Full": full["se"],
            "SE_Cluster_2018": recent["se"],
            "Signals": full["n"],
            "Signals_2018": recent["n"],
            "Episodes": full["clusters"],
            "Effective_N": full["n_eff"],
            "Cluster_Mode": full["cluster_mode"],
            "Kish_N": full.get("kish_n", np.nan),
            "Worst_R": full.get("worst_r", np.nan),
            "Best_R": full.get("best_r", np.nan),
            "Annual_Risk_Budget": 252.0 * risk_total / max(n_days, 1),
            "Annual_Risk_Budget_2018": 252.0 * risk_recent / max(n_days_recent, 1),
            "Actual_Daily_Mean": float(daily[comp].mean()),
            "Standalone_Full_Kelly_Fraction": standalone_kelly(r),
            "Standalone_Full_Kelly_Mult_HeteroQ": standalone_kelly(r, q),
        })
    est = pd.DataFrame(rows).set_index("Component").reindex(components)

    priors = {}
    for label, mean_col, se_col in (
        ("Full", "Risk_Weighted_R_Full", "SE_Cluster_Full"),
        ("LOYO", "Risk_Weighted_R_LOYO", "SE_Cluster_Full"),
        ("2018", "Risk_Weighted_R_2018", "SE_Cluster_2018"),
    ):
        post, kappa, prior = eb_fit(
            est[mean_col].to_numpy(float), est[se_col].to_numpy(float), strength
        )
        est[f"Shrunk_R_{label}"] = post
        est[f"Shrinkage_Kappa_{label}"] = kappa
        priors[label] = prior
    return est.reset_index(), priors


def lw_cov(returns: pd.DataFrame):
    model = LedoitWolf(assume_centered=False).fit(returns.to_numpy(float))
    cov = pd.DataFrame(model.covariance_, index=returns.columns, columns=returns.columns)
    return cov, float(model.shrinkage_)


def crisis_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return (
        ((index.year == 2020) & (index.month == 3))
        | (index.year == 2022)
        | ((index.year == 2024) & (index.month == 8))
        | ((index.year == 2025) & (index.month == 4))
    )


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.to_numpy(float)))
    denom = np.outer(sd, sd)
    arr = np.divide(cov.to_numpy(float), denom, out=np.zeros_like(denom), where=denom > 0)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)


def make_mu(est: pd.DataFrame, label: str) -> pd.Series:
    e = est.set_index("Component")
    if label == "2018":
        risk_daily = e["Annual_Risk_Budget_2018"] / 252.0
    else:
        risk_daily = e["Annual_Risk_Budget"] / 252.0
    return risk_daily * e[f"Shrunk_R_{label}"]


def optimize_quadratic(mu_daily: pd.Series, cov_daily: pd.DataFrame,
                       pilots: set[str], budget: pd.Series | None = None):
    cols = list(cov_daily.columns)
    mu = mu_daily.reindex(cols).to_numpy(float) * 252.0
    cov = cov_daily.loc[cols, cols].to_numpy(float) * 252.0
    x0 = np.ones(len(cols), dtype=float)
    bounds = [(1.0, 1.0) if c in pilots else (0.0, None) for c in cols]
    constraints = []
    if budget is not None:
        a = budget.reindex(cols).to_numpy(float)
        target = float(a.sum())
        constraints.append({
            "type": "eq",
            "fun": lambda x, a=a, target=target: float(np.dot(a, x) - target),
            "jac": lambda x, a=a: a,
        })

    def fun(x):
        return -float(np.dot(mu, x) - 0.5 * x @ cov @ x)

    def jac(x):
        return -(mu - cov @ x)

    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"quadratic optimization failed: {res.message}")
    return pd.Series(res.x, index=cols), float(-res.fun)


def optimize_empirical(returns: pd.DataFrame, mu_target: pd.Series,
                       pilots: set[str], budget: pd.Series | None = None):
    cols = list(returns.columns)
    arr = returns.to_numpy(float)
    raw_mu = returns.mean().to_numpy(float)
    target_mu = mu_target.reindex(cols).to_numpy(float)
    delta = target_mu - raw_mu
    x0 = np.ones(len(cols), dtype=float)
    bounds = [(1.0, 1.0) if c in pilots else (0.0, None) for c in cols]
    constraints = []
    if budget is not None:
        a = budget.reindex(cols).to_numpy(float)
        target = float(a.sum())
        constraints.append({
            "type": "eq",
            "fun": lambda x, a=a, target=target: float(np.dot(a, x) - target),
            "jac": lambda x, a=a: a,
        })

    def fun(x):
        factor = 1.0 + arr @ x
        if np.min(factor) <= 1e-10:
            return 1e12 + float(np.sum(np.minimum(factor, 0.0) ** 2)) * 1e12
        return -252.0 * float(np.mean(np.log(factor)) + np.dot(delta, x))

    def jac(x):
        factor = 1.0 + arr @ x
        if np.min(factor) <= 1e-10:
            return np.zeros_like(x)
        return -252.0 * (np.mean(arr / factor[:, None], axis=0) + delta)

    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"empirical optimization failed: {res.message}")
    return pd.Series(res.x, index=cols), float(-res.fun)


def optimize_subset_empirical(returns: pd.DataFrame, mu_target: pd.Series,
                              free_components: set[str],
                              subset_budget: pd.Series):
    """Optimize only named components; hold every other multiplier at one.

    The equality constraint preserves the current filled-risk budget inside
    the subset. Used for the OVS P1/P2 split so the global Kelly corner does
    not obscure the path decision.
    """
    cols = list(returns.columns)
    arr = returns.to_numpy(float)
    raw_mu = returns.mean().to_numpy(float)
    target_mu = mu_target.reindex(cols).to_numpy(float)
    delta = target_mu - raw_mu
    x0 = np.ones(len(cols), dtype=float)
    bounds = [(0.0, None) if c in free_components else (1.0, 1.0) for c in cols]
    a = subset_budget.reindex(cols).fillna(0.0).to_numpy(float)
    target = float(a.sum())
    constraint = {
        "type": "eq",
        "fun": lambda x, a=a, target=target: float(np.dot(a, x) - target),
        "jac": lambda x, a=a: a,
    }

    def fun(x):
        factor = 1.0 + arr @ x
        if np.min(factor) <= 1e-10:
            return 1e12 + float(np.sum(np.minimum(factor, 0.0) ** 2)) * 1e12
        return -252.0 * float(np.mean(np.log(factor)) + np.dot(delta, x))

    def jac(x):
        factor = 1.0 + arr @ x
        if np.min(factor) <= 1e-10:
            return np.zeros_like(x)
        return -252.0 * (np.mean(arr / factor[:, None], axis=0) + delta)

    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=[constraint],
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"subset empirical optimization failed: {res.message}")
    return pd.Series(res.x, index=cols), float(-res.fun)


def optimize_subset_quadratic(mu_daily: pd.Series, cov_daily: pd.DataFrame,
                              free_components: set[str],
                              subset_budget: pd.Series):
    cols = list(cov_daily.columns)
    mu = mu_daily.reindex(cols).to_numpy(float) * 252.0
    cov = cov_daily.loc[cols, cols].to_numpy(float) * 252.0
    x0 = np.ones(len(cols), dtype=float)
    bounds = [(0.0, None) if c in free_components else (1.0, 1.0) for c in cols]
    a = subset_budget.reindex(cols).fillna(0.0).to_numpy(float)
    target = float(a.sum())
    constraint = {
        "type": "eq",
        "fun": lambda x, a=a, target=target: float(np.dot(a, x) - target),
        "jac": lambda x, a=a: a,
    }

    def fun(x):
        return -float(np.dot(mu, x) - 0.5 * x @ cov @ x)

    def jac(x):
        return -(mu - cov @ x)

    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=[constraint],
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"subset quadratic optimization failed: {res.message}")
    return pd.Series(res.x, index=cols), float(-res.fun)


def stationary_indices(n: int, sims=N_BOOT, horizon=BOOT_HORIZON,
                       mean_block=BOOT_MEAN_BLOCK, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.empty((sims, horizon), dtype=np.int32)
    cur = rng.integers(0, n, sims, dtype=np.int32)
    p_new = 1.0 / mean_block
    for t in range(horizon):
        idx[:, t] = cur
        restart = rng.random(sims) < p_new
        cur = np.where(restart, rng.integers(0, n, sims), (cur + 1) % n)
    return idx


def bootstrap_dd(port_ret: np.ndarray, idx: np.ndarray):
    pnl = port_ret[idx] * NAV
    cum = np.cumsum(pnl, axis=1)
    peaks = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    dd = np.min(cum - peaks, axis=1)
    terminal = cum[:, -1]
    return {
        "dd_median": float(np.percentile(dd, 50)),
        "dd_p05": float(np.percentile(dd, 5)),
        "dd_p01": float(np.percentile(dd, 1)),
        "p_dd_gt_20pct": float(np.mean(dd < DD_LIMIT)),
        "terminal_median": float(np.percentile(terminal, 50)),
        "terminal_p05": float(np.percentile(terminal, 5)),
        "p_terminal_negative": float(np.mean(terminal < 0)),
    }


def growth_metrics(m: pd.Series, returns: pd.DataFrame,
                   mu_target: pd.Series, cov: pd.DataFrame):
    cols = list(returns.columns)
    x = m.reindex(cols).to_numpy(float)
    port = returns.to_numpy(float) @ x
    raw_mu = returns.mean().to_numpy(float)
    target = mu_target.reindex(cols).to_numpy(float)
    factor = 1.0 + port
    raw_log = float(np.mean(np.log(factor)) * 252.0) if np.min(factor) > 0 else -np.inf
    adjusted_log = raw_log + float(np.dot(target - raw_mu, x) * 252.0)
    mu_ann = target * 252.0
    cov_ann = cov.loc[cols, cols].to_numpy(float) * 252.0
    gauss = float(np.dot(mu_ann, x) - 0.5 * x @ cov_ann @ x)
    return raw_log, adjusted_log, gauss, port


def location_diagnostics(mk: pd.Series, cov: pd.DataFrame, budget: pd.Series):
    cols = list(cov.columns)
    free = np.array([c not in PILOTS for c in cols], dtype=bool)
    p = np.array([1.0 if c in PILOTS else 0.0 for c in cols])
    f = np.where(free, mk.reindex(cols).to_numpy(float), 0.0)
    one = np.ones(len(cols))
    a = budget.reindex(cols).to_numpy(float)
    denom = float(np.dot(a, f))
    c_budget = float((np.dot(a, one) - np.dot(a, p)) / denom) if denom > 0 else np.nan

    s = cov.loc[cols, cols].to_numpy(float)
    target_var = float(one @ s @ one)
    aa = float(f @ s @ f)
    bb = float(2.0 * p @ s @ f)
    cc = float(p @ s @ p - target_var)
    roots = np.roots([aa, bb, cc]) if aa > 0 else []
    pos = sorted(float(np.real(x)) for x in roots if abs(np.imag(x)) < 1e-8 and np.real(x) >= 0)
    c_sigma = pos[0] if pos else np.nan
    num = float(one @ s @ mk.reindex(cols).to_numpy(float))
    den = math.sqrt(max(float(one @ s @ one) * float(mk.reindex(cols).to_numpy(float) @ s @ mk.reindex(cols).to_numpy(float)), 0.0))
    cosine = num / den if den > 0 else np.nan
    return {"c_risk_budget": c_budget, "c_variance": c_sigma,
            "covariance_metric_cosine": cosine}


def plot_curve(curve: pd.DataFrame, location: dict, current_p: float):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(curve["c"], curve["Adjusted_Log_Growth"], color="#135f9b", lw=2,
             label="Shrunk annual log growth")
    ax1.plot(curve["c"], curve["Gaussian_Growth"], color="#6baed6", lw=1.5,
             ls="--", label="Gaussian approximation")
    ax1.set_xlabel("Fraction of conditional full-Kelly seasoned sleeve")
    ax1.set_ylabel("Annualized log growth")
    ax1.axvline(0.25, color="#5a9b48", ls=":", label="Quarter-Kelly")
    ax1.axvline(0.50, color="#c78c22", ls=":", label="Half-Kelly")
    ax1.axvline(1.00, color="#8a3b3b", ls=":", label="Full Kelly")
    if np.isfinite(location.get("c_risk_budget", np.nan)):
        ax1.axvline(location["c_risk_budget"], color="#333", ls="-.",
                    label="Current risk-budget equivalent")

    ax2 = ax1.twinx()
    ax2.plot(curve["c"], curve["P_DD_Worse_20pct"], color="#b33b3b", lw=2,
             label="P(1y maxDD worse than 20%)")
    ax2.axhline(0.05, color="#b33b3b", ls="--", alpha=0.7,
                label="5% drawdown gate")
    ax2.scatter([location.get("c_risk_budget", np.nan)], [current_p],
                color="black", zorder=5, label="Current-book bootstrap risk")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(bottom=0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.set_title("Kelly growth and one-year drawdown tradeoff")
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=170)
    plt.close(fig)


def main() -> int:
    signals, daily_dollars, daily_tier_dollars = load_inputs()
    returns = daily_dollars / NAV
    components = list(returns.columns)
    print(f"inputs: {len(signals)} collapsed signals, {len(returns)} days, {len(components)} components")

    est, priors = build_estimates(signals, returns, strength=1.0)
    est_liq, priors_liq = build_estimates(
        signals, daily_tier_dollars[[c for c in daily_tier_dollars if c.endswith("||Liquid")]]
        .rename(columns=lambda c: c.rsplit("||", 1)[0]) / NAV,
        tier="Liquid", strength=1.0,
    )
    est_liq.to_csv(EST_LIQ_OUT, index=False)
    bps = base_bps_map()
    est["Nominal_Bps"] = est["Component"].map(lambda c: bps.get(c, (np.nan, np.nan, None))[0])
    est["Effective_Bps_Liquid"] = est["Component"].map(lambda c: bps.get(c, (np.nan, np.nan, None))[1])
    est["Effective_Bps_Overflow"] = est["Component"].map(lambda c: bps.get(c, (np.nan, np.nan, None))[2])
    est["Pilot_Frozen"] = est["Component"].isin(PILOTS)
    current_frac = est["Effective_Bps_Liquid"] / 10000.0
    quarter_f = 0.25 * est["Standalone_Full_Kelly_Fraction"]
    est["Current_Base_over_Quarter_Standalone_Kelly"] = current_frac / quarter_f.replace(0, np.nan)
    est.to_csv(EST_OUT, index=False)

    cov_full, lw_full = lw_cov(returns)
    crisis_returns = returns.loc[crisis_mask(returns.index)]
    cov_crisis, lw_crisis = lw_cov(crisis_returns)
    corr_full = returns.corr()
    corr_crisis = crisis_returns.corr()
    corr_full.to_csv(CORR_FULL_OUT)
    corr_crisis.to_csv(CORR_CRISIS_OUT)

    mu_full = make_mu(est, "Full")
    mu_loyo = make_mu(est, "LOYO")
    recent_returns = returns.loc[returns.index >= "2018-01-01"]
    cov_2018, lw_2018 = lw_cov(recent_returns)
    mu_2018 = make_mu(est, "2018")
    budget = est.set_index("Component")["Annual_Risk_Budget"]

    # Primary: LOYO-conservative shrunk mean, full-sample LW covariance.
    mk_quad, gk_quad = optimize_quadratic(mu_loyo, cov_full, PILOTS)
    mk_exact, gk_exact = optimize_empirical(returns, mu_loyo, PILOTS)
    rel_quad, grel_quad = optimize_quadratic(mu_loyo, cov_full, PILOTS, budget)
    rel_exact, grel_exact = optimize_empirical(returns, mu_loyo, PILOTS, budget)

    scenarios: dict[str, pd.Series] = {
        "Primary_LOYO_FullCov_RawFullKelly": mk_exact,
        "Primary_LOYO_FullCov_EqualRisk": rel_exact,
        "Primary_LOYO_FullCov_EqualRisk_Quadratic": rel_quad,
    }
    ovs_paths = {"Overbot Vol Spike P1", "Overbot Vol Spike P2"}
    ovs_budget = budget.where(budget.index.isin(ovs_paths), 0.0)
    scenarios["OVS_Internal_LOYO_FixedRisk"], _ = optimize_subset_empirical(
        returns, mu_loyo, ovs_paths, ovs_budget
    )
    scenarios["OVS_Internal_FullMean_FixedRisk"], _ = optimize_subset_empirical(
        returns, mu_full, ovs_paths, ovs_budget
    )
    scenarios["OVS_Internal_CrisisCov_FixedRisk"], _ = optimize_subset_quadratic(
        mu_loyo, cov_crisis, ovs_paths, ovs_budget
    )
    scenarios["FullMean_FullCov_EqualRisk"], _ = optimize_empirical(
        returns, mu_full, PILOTS, budget
    )
    scenarios["LOYO_CrisisCov_EqualRisk"], _ = optimize_quadratic(
        mu_loyo, cov_crisis, PILOTS, budget
    )
    scenarios["2018Mean_2018Cov_EqualRisk"], _ = optimize_empirical(
        recent_returns, mu_2018, PILOTS,
        est.set_index("Component")["Annual_Risk_Budget_2018"],
    )
    budget_2018 = est.set_index("Component")["Annual_Risk_Budget_2018"]
    ovs_budget_2018 = budget_2018.where(budget_2018.index.isin(ovs_paths), 0.0)
    scenarios["OVS_Internal_2018_FixedRisk"], _ = optimize_subset_empirical(
        recent_returns, mu_2018, ovs_paths, ovs_budget_2018
    )
    diag_cov = pd.DataFrame(np.diag(np.diag(cov_full)), index=components, columns=components)
    scenarios["LOYO_DiagonalCov_EqualRisk"], _ = optimize_quadratic(
        mu_loyo, diag_cov, PILOTS, budget
    )

    # Shrinkage-strength sensitivity.
    shrink_priors = {}
    for strength in (0.5, 2.0):
        est_s, pr_s = build_estimates(signals, returns, strength=strength)
        shrink_priors[str(strength)] = pr_s
        mu_s = make_mu(est_s, "LOYO")
        scenarios[f"LOYO_ShrinkStrength_{strength:g}_EqualRisk"], _ = optimize_empirical(
            returns, mu_s, PILOTS, budget
        )

    # Liquid-only sensitivity.
    returns_liq = daily_tier_dollars[[c for c in daily_tier_dollars if c.endswith("||Liquid")]]
    returns_liq = returns_liq.rename(columns=lambda c: c.rsplit("||", 1)[0]) / NAV
    returns_liq = returns_liq.reindex(columns=components, fill_value=0.0)
    cov_liq, lw_liq = lw_cov(returns_liq)
    mu_liq = make_mu(est_liq, "LOYO").reindex(components)
    budget_liq = est_liq.set_index("Component")["Annual_Risk_Budget"].reindex(components)
    scenarios["LiquidOnly_LOYO_EqualRisk"], _ = optimize_empirical(
        returns_liq, mu_liq, PILOTS, budget_liq
    )
    ovs_budget_liq = budget_liq.where(budget_liq.index.isin(ovs_paths), 0.0)
    scenarios["OVS_Internal_LiquidOnly_FixedRisk"], _ = optimize_subset_empirical(
        returns_liq, mu_liq, ovs_paths, ovs_budget_liq
    )

    # Full covariance/exposure with overflow-eligible mean capped at liquid evidence.
    mu_liq_support = budget / 252.0 * est_liq.set_index("Component")["Shrunk_R_LOYO"].reindex(components)
    scenarios["FullCov_LiquidSupportedMeans_EqualRisk"], _ = optimize_empirical(
        returns, mu_liq_support, PILOTS, budget
    )
    scenarios["OVS_Internal_LiquidSupportedMeans_FixedRisk"], _ = optimize_subset_empirical(
        returns, mu_liq_support, ovs_paths, ovs_budget
    )

    # Pooled OVS robustness: both paths share one multiplier.
    pooled_returns = returns.copy()
    pooled_returns["Overbot Vol Spike"] = (
        pooled_returns.pop("Overbot Vol Spike P1")
        + pooled_returns.pop("Overbot Vol Spike P2")
    )
    pooled_signals = signals.copy()
    pooled_signals["Component"] = pooled_signals["Component"].replace({
        "Overbot Vol Spike P1": "Overbot Vol Spike",
        "Overbot Vol Spike P2": "Overbot Vol Spike",
    })
    est_pool, prior_pool = build_estimates(pooled_signals, pooled_returns)
    cov_pool, lw_pool = lw_cov(pooled_returns)
    mu_pool = make_mu(est_pool, "LOYO")
    budget_pool = est_pool.set_index("Component")["Annual_Risk_Budget"]
    pool_rel, _ = optimize_empirical(pooled_returns, mu_pool, PILOTS, budget_pool)
    pool_mapped = pd.Series(index=components, dtype=float)
    for comp in components:
        pool_mapped[comp] = (
            pool_rel["Overbot Vol Spike"]
            if comp.startswith("Overbot Vol Spike") else pool_rel[comp]
        )
    scenarios["OVS_Pooled_LOYO_EqualRisk"] = pool_mapped

    scenario_df = pd.DataFrame(scenarios).T
    scenario_df.index.name = "Scenario"
    scenario_df.to_csv(SCENARIO_OUT)

    # Headline component allocation table.
    alloc = est.set_index("Component").copy()
    alloc["Shrunk_Daily_Mu_LOYO"] = mu_loyo
    alloc["Shrunk_Annual_Return_LOYO"] = mu_loyo * 252.0
    alloc["Raw_FullKelly_Quadratic"] = mk_quad
    alloc["Raw_FullKelly_Exact"] = mk_exact
    alloc["Quarter_Kelly_Mult"] = np.where(alloc.index.isin(PILOTS), 1.0, 0.25 * mk_exact)
    alloc["Half_Kelly_Mult"] = np.where(alloc.index.isin(PILOTS), 1.0, 0.50 * mk_exact)
    alloc["EqualRisk_Relative_Mult"] = rel_exact
    alloc["EqualRisk_Relative_Mult_Quadratic"] = rel_quad
    alloc["Current_over_Quarter_Correlated"] = 1.0 / alloc["Quarter_Kelly_Mult"].replace(0, np.nan)
    alloc["CrisisCov_EqualRisk_Mult"] = scenarios["LOYO_CrisisCov_EqualRisk"]
    alloc["LiquidOnly_EqualRisk_Mult"] = scenarios["LiquidOnly_LOYO_EqualRisk"]
    alloc["LiquidSupportedMeans_EqualRisk_Mult"] = scenarios["FullCov_LiquidSupportedMeans_EqualRisk"]
    alloc["OVS_Internal_LOYO_FixedRisk_Mult"] = scenarios["OVS_Internal_LOYO_FixedRisk"]
    alloc["OVS_Internal_CrisisCov_FixedRisk_Mult"] = scenarios["OVS_Internal_CrisisCov_FixedRisk"]
    alloc["OVS_Internal_2018_FixedRisk_Mult"] = scenarios["OVS_Internal_2018_FixedRisk"]
    alloc["OVS_Internal_LiquidOnly_FixedRisk_Mult"] = scenarios["OVS_Internal_LiquidOnly_FixedRisk"]
    alloc.to_csv(ALLOC_OUT)

    # Kelly-ray growth/drawdown curve, pilots fixed at 1.
    idx = stationary_indices(len(returns))
    location = location_diagnostics(mk_exact, cov_full, budget)
    current_m = pd.Series(1.0, index=components)
    _, _, _, current_port = growth_metrics(current_m, returns, mu_loyo, cov_full)
    current_boot = bootstrap_dd(current_port, idx)
    free_mk = mk_exact.copy()
    rows = []
    for c in np.round(np.arange(0.0, 1.5001, 0.05), 10):
        m = pd.Series(
            [1.0 if name in PILOTS else c * free_mk[name] for name in components],
            index=components,
        )
        raw_g, adj_g, gauss_g, port = growth_metrics(m, returns, mu_loyo, cov_full)
        boot = bootstrap_dd(port, idx)
        rows.append({
            "c": c,
            "Raw_Log_Growth": raw_g,
            "Adjusted_Log_Growth": adj_g,
            "Gaussian_Growth": gauss_g,
            "Historical_Total_PnL": float(port.sum() * NAV),
            "Historical_MaxDD": float((NAV + pd.Series(port * NAV).cumsum()
                                         - (NAV + pd.Series(port * NAV).cumsum()).cummax()).min()),
            "Bootstrap_DD_Median": boot["dd_median"],
            "Bootstrap_DD_P05": boot["dd_p05"],
            "P_DD_Worse_20pct": boot["p_dd_gt_20pct"],
            "Bootstrap_Terminal_Median": boot["terminal_median"],
            "Bootstrap_Terminal_P05": boot["terminal_p05"],
            "P_Terminal_Negative": boot["p_terminal_negative"],
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(CURVE_OUT, index=False)
    passing = curve[curve["P_DD_Worse_20pct"] < 0.05]
    c_gate_grid = float(passing["c"].max()) if not passing.empty else np.nan
    c_gate_exact = np.nan
    p_gate_exact = np.nan
    if not passing.empty:
        lo = c_gate_grid
        higher_fail = curve[(curve["c"] > lo) & (curve["P_DD_Worse_20pct"] >= 0.05)]
        if not higher_fail.empty:
            hi = float(higher_fail["c"].min())
            for _ in range(14):
                mid = (lo + hi) / 2.0
                m_mid = pd.Series(
                    [1.0 if name in PILOTS else mid * free_mk[name]
                     for name in components], index=components,
                )
                _, _, _, port_mid = growth_metrics(m_mid, returns, mu_loyo, cov_full)
                p_mid = bootstrap_dd(port_mid, idx)["p_dd_gt_20pct"]
                if p_mid < 0.05:
                    lo = mid
                else:
                    hi = mid
            c_gate_exact = float(lo)
            m_gate = pd.Series(
                [1.0 if name in PILOTS else c_gate_exact * free_mk[name]
                 for name in components], index=components,
            )
            _, _, _, port_gate = growth_metrics(m_gate, returns, mu_loyo, cov_full)
            p_gate_exact = bootstrap_dd(port_gate, idx)["p_dd_gt_20pct"]
    plot_curve(curve, location, current_boot["p_dd_gt_20pct"])

    summary = {
        "study_date": STAMP,
        "objective": "annualized empirical log growth with shrunk LOYO means",
        "components": components,
        "pilots_frozen": sorted(PILOTS),
        "signal_count": int(len(signals)),
        "daily_rows": int(len(returns)),
        "crisis_definition": CRISIS_MASK_LABEL,
        "crisis_days": int(crisis_mask(returns.index).sum()),
        "covariance": {
            "full_ledoit_wolf_shrinkage": lw_full,
            "crisis_ledoit_wolf_shrinkage": lw_crisis,
            "since_2018_ledoit_wolf_shrinkage": lw_2018,
            "liquid_ledoit_wolf_shrinkage": lw_liq,
            "pooled_ovs_ledoit_wolf_shrinkage": lw_pool,
        },
        "eb_priors": priors,
        "eb_priors_liquid": priors_liq,
        "eb_priors_shrink_sensitivity": shrink_priors,
        "eb_prior_pooled_ovs": prior_pool,
        "optimizer": {
            "conditional_full_kelly_empirical_growth": gk_exact,
            "conditional_full_kelly_quadratic_growth": gk_quad,
            "equal_risk_empirical_growth": grel_exact,
            "equal_risk_quadratic_growth": grel_quad,
        },
        "current_location_on_kelly_ray": location,
        "bootstrap": {
            "simulations": N_BOOT,
            "horizon_td": BOOT_HORIZON,
            "mean_block_td": BOOT_MEAN_BLOCK,
            "seed": BOOT_SEED,
            "drawdown_limit_dollars": DD_LIMIT,
            "probability_limit": 0.05,
            "current_book": current_boot,
            "largest_passing_c_on_0.05_grid": c_gate_grid,
            "first_crossing_c_bisection": c_gate_exact,
            "probability_at_bisection_c": p_gate_exact,
        },
        "outputs": {
            "estimates": str(EST_OUT.relative_to(ROOT)),
            "estimates_liquid": str(EST_LIQ_OUT.relative_to(ROOT)),
            "allocations": str(ALLOC_OUT.relative_to(ROOT)),
            "scenarios": str(SCENARIO_OUT.relative_to(ROOT)),
            "curve": str(CURVE_OUT.relative_to(ROOT)),
            "curve_png": str(CURVE_PNG.relative_to(ROOT)),
            "correlation_full": str(CORR_FULL_OUT.relative_to(ROOT)),
            "correlation_crisis": str(CORR_CRISIS_OUT.relative_to(ROOT)),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("\nPRIMARY EQUAL-RISK RELATIVE MULTIPLIERS")
    print(rel_exact.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nCURRENT LOCATION", json.dumps(location, indent=2))
    print("CURRENT BOOTSTRAP", json.dumps(current_boot, indent=2))
    print(f"drawdown gate largest c on grid: {c_gate_grid}")
    print(f"wrote {EST_OUT}, {ALLOC_OUT}, {SCENARIO_OUT}, {CURVE_OUT}, {CURVE_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
