"""Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the book and per
strategy, plus the implied haircut on the mean.

  E[max SR over N trials] = sqrt(V[SR_trials]) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)))   g = 0.5772
  DSR = Phi( (SR_hat - SR0) sqrt(T-1) / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

SR in per-period units (daily for the book, per-trade for strategies). Trials
N is the uncertain input; a grid is reported. V[SR_trials] is taken from the
cross-section of per-strategy Sharpes actually observed (live + retired names
that still have a ledger footprint are unavailable, so the live spread is used
and a wider prior is also shown).

Implied mean haircut: the fraction of the observed Sharpe that survives after
subtracting the expected maximum of N null trials with the observed variance:
  keep = max(0, 1 - SR0 / SR_hat)   -> haircut = 1 - keep
plus the shrinkage form (James-Stein-like): SR_true ~ SR_hat * (1 - noise_var / total_var)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
NAV = 750_000.0
G = 0.5772156649


def emax_sr(n: int, var_sr: float) -> float:
    if n <= 1:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - G) * st.norm.ppf(1 - 1 / n) + G * st.norm.ppf(1 - 1 / (n * np.e))))


def dsr(sr: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    denom = np.sqrt(max(1e-9, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(st.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


res: dict = {}
# ---- book daily series (flat basis)
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T
for label, start in [("2003+", "2003-01-01"), ("2010+", "2010-01-01"), ("2016+", "2016-01-01")]:
    r = (tot[tot.index >= start] / NAV)
    r = r[r.index <= "2026-08-07"]
    T = len(r)
    sr_d = r.mean() / r.std(ddof=1)
    sk = float(st.skew(r)); ku = float(st.kurtosis(r, fisher=False))
    # per-strategy daily Sharpe cross-section (active-days-inclusive, same basis as the plan's L1 table)
    ss = strat[strat.index >= start]
    ss = ss[ss.index <= "2026-08-07"]
    srs = (ss.mean() / ss.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna()
    srs = srs[ss.abs().gt(0).sum() > 100]
    var_cs = float(srs.var(ddof=1))
    grid = {}
    for n in [15, 31, 60, 120, 250, 500, 1000]:
        sr0 = emax_sr(n, var_cs)
        grid[n] = {"SR0_daily": sr0, "SR0_ann": sr0 * np.sqrt(252), "DSR": dsr(sr_d, sr0, T, sk, ku),
                   "keep_fraction": max(0.0, 1 - sr0 / sr_d)}
    res[f"book_{label}"] = {"T": T, "SR_daily": float(sr_d), "SR_ann": float(sr_d * np.sqrt(252)), "skew": sk, "kurt": ku,
                            "cs_var_SR_daily": var_cs, "cs_sd_SR_ann": float(np.sqrt(var_cs) * np.sqrt(252)),
                            "per_strategy_daily_SR_ann": {k: float(v * np.sqrt(252)) for k, v in srs.items()},
                            "grid": grid}
    print(f"\nBOOK {label}: T={T} SR_ann={sr_d*np.sqrt(252):.2f} skew={sk:.2f} kurt={ku:.1f} cs_sd(SR_ann)={np.sqrt(var_cs)*np.sqrt(252):.2f}")
    for n, v in grid.items():
        print(f"  N={n:5d}: E[max SR_ann]={v['SR0_ann']:.2f}  DSR={v['DSR']:.4f}  keep={v['keep_fraction']:.2f}")

# ---- per-strategy per-trade DSR (trials = versions of that strategy's dict, from git)
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
freeze = json.load(open(OUT / "estimation_haircut_freeze_dates.json"))
per = {}
# variance of trial Sharpes per trade: use the per-trade cross-section of strategy avgR/sdR
cs = led.groupby("Strategy")["R_Multiple"].agg(["mean", "std", "size"])
cs["sr_trade"] = cs["mean"] / cs["std"]
# trial variance: strategies with N >= 50 only (Monthly Weak Close's SR 3.5 on N=14 is
# an estimation artefact, not a plausible variant-to-variant spread)
var_trade = float(cs.loc[cs["size"] >= 50, "sr_trade"].var(ddof=1))
res["per_trade_cs_var_SR_note"] = "variance of per-trade SR across live strategies with N>=50"
print(f"\nper-trade SR cross-section (N>=50): mean {cs.loc[cs['size']>=50,'sr_trade'].mean():.3f} sd {np.sqrt(var_trade):.3f}")
for s, g in led.groupby("Strategy"):
    r = g["R_Multiple"].to_numpy(float)
    n = len(r); sr = r.mean() / r.std(ddof=1)
    sk = float(st.skew(r)); ku = float(st.kurtosis(r, fisher=False))
    nv = freeze[s]["n_versions"]
    out = {"N_trades": n, "SR_trade": float(sr), "avgR": float(r.mean()), "skew": sk, "kurt": ku, "n_versions_git": nv}
    for label, ntr in [("versions", max(2, nv)), ("versions_x3", max(2, 3 * nv)), ("30", 30), ("100", 100)]:
        sr0 = emax_sr(ntr, var_trade)
        out[f"trials_{label}"] = {"N": ntr, "SR0": sr0, "DSR": dsr(sr, sr0, n, sk, ku), "keep": max(0.0, 1 - sr0 / sr)}
    # plain t-stat / PSR at SR0=0
    out["PSR0"] = dsr(sr, 0.0, n, sk, ku)
    per[s] = out
    print(f"{s:26s} N={n:4d} SR={sr:.3f} avgR={r.mean():.2f} ver={nv:2d} | keep@ver {out['trials_versions']['keep']:.2f} DSR {out['trials_versions']['DSR']:.3f} | keep@3ver {out['trials_versions_x3']['keep']:.2f} DSR {out['trials_versions_x3']['DSR']:.3f} | keep@100 {out['trials_100']['keep']:.2f} DSR {out['trials_100']['DSR']:.3f} | PSR0 {out['PSR0']:.3f}")
res["per_strategy"] = per
res["per_trade_cs_var_SR"] = var_trade

# ---- shrinkage view: how much of the cross-sectional spread in per-trade avgR is noise
# noise var of each strategy's avgR estimate = sdR^2 / N ; signal var = cs var(avgR) - mean(noise var)
cs["noise_var"] = cs["std"] ** 2 / cs["size"]
total_var = float(cs["mean"].var(ddof=1)); noise = float(cs["noise_var"].mean())
shrink = max(0.0, 1 - noise / total_var)
res["cross_sectional_shrinkage"] = {"var_avgR_across_strats": total_var, "mean_noise_var": noise, "signal_share": shrink,
                                    "note": "James-Stein factor: how much of a strategy's deviation from the book-mean avgR to believe"}
print(f"\ncross-sectional shrinkage: total var {total_var:.4f} noise {noise:.4f} -> believe {shrink:.2f} of each strategy's deviation from the mean")

# ---- trials inventory (for the N grid): strategies ever + config versions + scratch studies
inv = {
    "strategies_ever_in_config": 31, "strategies_live": 15, "strategies_retired": 16,
    "config_versions_total": int(sum(v["n_versions"] for v in freeze.values())),
    "config_versions_live": {k: v["n_versions"] for k, v in freeze.items()},
    "scratch_py_scripts": len(list((ROOT / "scratch").glob("*.py"))),
    "ultracode_research_py": len(list((ROOT / "scratch/ultracode_research").glob("*.py"))),
    "claude_md_dead_overlays_listed": ["ML meta-label", "book throttle/taper", "OVS tilt", "put hedges", "VXX proxy", "21d fast confirm", "trend-sleeve gate", ">1.0x boosts", "sub-50 ramps", "sector loss gate", "OLV ladder v1/v2", "OLV band", "OLV book cap", "P1 budget gate", "LT OS Sznl", "SPX OB Fade", "Weak Close Reversion", "No Accumulation Days", "Deep Oversold x4", "Liquid Seasonals x2", "Index Seasonals", "OLV moc companion", "OVS LOC companion"],
}
res["trials_inventory"] = inv
print("\ntrials inventory:", {k: v for k, v in inv.items() if k not in ("config_versions_live", "claude_md_dead_overlays_listed")})
(OUT / "estimation_haircut_dsr.json").write_text(json.dumps(res, indent=1, default=str))
