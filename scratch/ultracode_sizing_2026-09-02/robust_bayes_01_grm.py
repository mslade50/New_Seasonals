"""Robust-Bayesian GRM study (2026-09-02).

(A) Expected log growth by multiple m of current sizing, integrated over a
    PRIOR on the keep fraction of the ledger mean (the estimation-haircut
    study's central 0.60, range 0.29-0.73 -> Beta(6.6, 4.8)), on the flat
    $750k daily book series (dist/data/strategy_daily.json total_flat).
    Reports the prior-mean growth, the prior-25th-percentile growth (the
    robust criterion) and the growth at the pessimistic keep 0.30.
(B) Drawdown distribution by m and keep (stationary block bootstrap, mean
    block 10, 1y and 3y paths), so the drawdown accepted at each GRM is
    stated under the haircut prior rather than at the raw mean.
(C) Margin boundary re-derived from the ledger with TIMS-style class rates:
    single stock 15%, BROAD index ETF 8% (the pack's margin study charged
    broad ETFs 15%, which is the rules-based rate, not the TIMS +6/-8%
    stress), 3x ETFs 45%, small-cap index 10%. Feasibility multiples on the
    max / p99 / p95 requirement day on $750k and on a $632k live NLV.
(D) Kaminski-Lo check: autocorrelation of the daily book PnL at lags 1-21,
    variance ratios, and the 21d-forward mean conditional on drawdown state.
    A drawdown-triggered cut can only pay if this autocorrelation is positive.

Writes robust_bayes_01_grm.json beside this file. Reads only; modifies nothing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
NAV = 750_000.0
LIVE_NLV = 632_000.0
RNG = np.random.default_rng(7)
OUT: dict = {}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
ret_all = (tot / NAV).astype(float)
windows = {"2003+": ret_all, "2016+": ret_all[ret_all.index >= "2016-01-01"]}

# ---------------------------------------------------------------- A. growth under a keep prior
a_, b_ = 6.63, 4.80        # Beta prior on keep: mean 0.58, sd 0.14, 5-95% ~ [0.34, 0.79]
keep_draws = RNG.beta(a_, b_, 400)
OUT["keep_prior"] = dict(alpha=a_, beta=b_, mean=float(keep_draws.mean()), p05=float(np.percentile(keep_draws, 5)),
                         p25=float(np.percentile(keep_draws, 25)), p75=float(np.percentile(keep_draws, 75)),
                         p95=float(np.percentile(keep_draws, 95)))
m_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]

def g_ann(r: np.ndarray, m: float, keep: float) -> float:
    mu = r.mean()
    rh = r - (1 - keep) * mu
    x = 1 + m * rh
    if (x <= 0).any():
        return -np.inf
    return float(np.log(x).mean() * 252)

growth = {}
for wname, r in windows.items():
    rv = r.values
    rows = []
    for m in m_grid:
        gs = np.array([g_ann(rv, m, k) for k in keep_draws])
        rows.append(dict(m=m, grm=1.5 * m, g_prior_mean=float(gs.mean()), g_prior_p25=float(np.percentile(gs, 25)),
                         g_prior_p10=float(np.percentile(gs, 10)), g_keep030=g_ann(rv, m, 0.30), g_keep045=g_ann(rv, m, 0.45),
                         g_keep060=g_ann(rv, m, 0.60), g_keep100=g_ann(rv, m, 1.0),
                         sharpe_keep060=float((rv.mean() * 0.6) / rv.std() * np.sqrt(252)),
                         vol_ann=float(rv.std() * np.sqrt(252) * m)))
    df = pd.DataFrame(rows)
    growth[wname] = df
    print(f"\n=== A. growth by multiple, window {wname} (N={len(rv)}) ===")
    print(df.round(4).to_string(index=False))
    for col in ["g_prior_mean", "g_prior_p25", "g_keep030"]:
        best = df.loc[df[col].idxmax()]
        print(f"  argmax {col}: m={best['m']} (GRM {best['grm']}) g={best[col]:.3f}")
    OUT[f"growth_{wname}"] = df.round(5).to_dict("records")

# ---------------------------------------------------------------- B. drawdown by m and keep (block bootstrap)
def block_bootstrap(rv: np.ndarray, n_days: int, n_paths: int, mean_block: int = 10) -> np.ndarray:
    N = len(rv)
    out = np.empty((n_paths, n_days))
    p = 1.0 / mean_block
    for i in range(n_paths):
        pos = 0
        while pos < n_days:
            L = RNG.geometric(p)
            start = RNG.integers(0, N)
            idx = (start + np.arange(L)) % N
            take = min(L, n_days - pos)
            out[i, pos:pos + take] = rv[idx[:take]]
            pos += take
    return out

def maxdd_paths(paths: np.ndarray, m: float, keep: float, mu: float) -> np.ndarray:
    rh = paths - (1 - keep) * mu
    eq = np.cumprod(1 + m * rh, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = 1 - eq / peak
    return dd.max(axis=1)

dd_rows = []
for wname, r in windows.items():
    rv = r.values
    mu = rv.mean()
    paths1 = block_bootstrap(rv, 252, 2000)
    paths3 = block_bootstrap(rv, 756, 1500)
    for keep in [1.0, 0.60, 0.45, 0.30]:
        for m in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            d1 = maxdd_paths(paths1, m, keep, mu)
            d3 = maxdd_paths(paths3, m, keep, mu)
            dd_rows.append(dict(window=wname, keep=keep, m=m, grm=1.5 * m,
                                dd1_med=float(np.median(d1)), dd1_p95=float(np.percentile(d1, 95)),
                                p1_gt15=float((d1 > .15).mean()), p1_gt20=float((d1 > .20).mean()), p1_gt30=float((d1 > .30).mean()),
                                dd3_med=float(np.median(d3)), dd3_p95=float(np.percentile(d3, 95)),
                                p3_gt20=float((d3 > .20).mean()), p3_gt30=float((d3 > .30).mean()), p3_gt40=float((d3 > .40).mean())))
DD = pd.DataFrame(dd_rows)
print("\n=== B. drawdown distribution by window x keep x multiple (block bootstrap) ===")
print(DD.round(3).to_string(index=False))
OUT["drawdown"] = DD.round(4).to_dict("records")

# ---------------------------------------------------------------- C. margin boundary with TIMS class rates
from strategy_config import LEV3X_ALL  # noqa: E402  (pure data module)
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
BROAD = {"SPY", "QQQ", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "MDY", "IJR", "IJH", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALL = {"IWM"}
lev = set(LEV3X_ALL)
def rate(t: str, broad_rate: float) -> float:
    if t in lev: return 0.45
    if t in BROAD: return broad_rate
    if t in SMALL: return 0.10
    return 0.15
idx = pd.bdate_range("2003-01-01", "2026-09-01")
gross = pd.Series(0.0, index=idx)
req8 = pd.Series(0.0, index=idx)
req15 = pd.Series(0.0, index=idx)
for a, b, t, n in zip(led["Entry Date"], led["Exit Date"], led["Ticker"], (led["Entry Price"] * led["Shares_flat"]).abs()):
    sl = (idx >= a) & (idx <= b)
    gross[sl] += n
    req8[sl] += n * rate(t, 0.08)
    req15[sl] += n * rate(t, 0.15)
W16 = idx >= "2016-01-01"
marg = {}
for label, req in [("broad8", req8), ("broad15", req15)]:
    for wl, sl in [("2003+", np.ones(len(idx), bool)), ("2016+", W16)]:
        q = req[sl] / NAV
        marg[f"{label}_{wl}"] = dict(req_max=float(q.max()), req_p99=float(q.quantile(.99)), req_p95=float(q.quantile(.95)),
                                     max_day=str(q.idxmax().date()),
                                     m_max_750=float(1 / q.max()), m_p99_750=float(1 / q.quantile(.99)),
                                     m_max_live=float(LIVE_NLV / NAV / q.max()), m_p99_live=float(LIVE_NLV / NAV / q.quantile(.99)),
                                     gross_max=float((gross[sl] / NAV).max()), gross_p99=float((gross[sl] / NAV).quantile(.99)))
print("\n=== C. margin requirement / NAV and feasibility multiples (m; GRM = 1.5 m) ===")
print(pd.DataFrame(marg).T.round(3).to_string())
OUT["margin"] = marg
# composition of the top-1% requirement days
top = req8[W16].nlargest(int(0.01 * W16.sum()))
comp = {}
for d in top.index:
    open_ = led[(led["Entry Date"] <= d) & (led["Exit Date"] >= d)]
    n = (open_["Entry Price"] * open_["Shares_flat"]).abs()
    comp[str(d.date())] = dict(gross=float(n.sum() / NAV), broad=float(n[open_["Ticker"].isin(BROAD | SMALL)].sum() / NAV),
                               lev=float(n[open_["Ticker"].isin(lev)].sum() / NAV),
                               top_strats=open_.groupby("Strategy").apply(lambda g: float((g["Entry Price"] * g["Shares_flat"]).abs().sum() / NAV)).sort_values(ascending=False).head(3).round(2).to_dict())
OUT["top_req_days_2016plus"] = comp
print(pd.DataFrame(comp).T.head(12).to_string())

# ---------------------------------------------------------------- D. Kaminski-Lo: autocorrelation of book PnL
def acf(x: np.ndarray, k: int) -> float:
    x = x - x.mean()
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())
kl = {}
for wname, r in windows.items():
    rv = r.values
    ac = {k: acf(rv, k) for k in range(1, 22)}
    N = len(rv)
    lb10 = N * (N + 2) * sum(ac[k] ** 2 / (N - k) for k in range(1, 11))
    vr5 = float(np.var(pd.Series(rv).rolling(5).sum().dropna()) / (5 * np.var(rv)))
    vr21 = float(np.var(pd.Series(rv).rolling(21).sum().dropna()) / (21 * np.var(rv)))
    # monthly
    mo = r.resample("ME").sum()
    ac_m1 = acf(mo.values, 1)
    # drawdown-state conditional forward mean
    eq = r.cumsum()
    dd = eq - eq.cummax()
    fwd21 = r.rolling(21).sum().shift(-21)
    st = pd.cut(dd * 100, [-100, -10, -5, -2.5, 0.0001], labels=["<-10", "-10..-5", "-5..-2.5", ">-2.5"])
    cond = fwd21.groupby(st, observed=True).agg(["mean", "count"])
    cond["ann_pct"] = cond["mean"] * 12 * 100
    kl[wname] = dict(acf=ac, ljung_box_10=float(lb10), acf_se=float(1 / np.sqrt(N)), vr5=vr5, vr21=vr21,
                     monthly_acf1=ac_m1, monthly_N=int(len(mo)), fwd21_by_dd_state=cond.round(4).to_dict())
    print(f"\n=== D. {wname}: ACF1..5 {[round(ac[k],3) for k in range(1,6)]}  se {1/np.sqrt(N):.3f}  LB10 {lb10:.1f}  VR5 {vr5:.2f} VR21 {vr21:.2f}  monthly ACF1 {ac_m1:.3f} ===")
    print(cond.round(4).to_string())
OUT["kaminski_lo"] = kl

json.dump(OUT, open(HERE / "robust_bayes_01_grm.json", "w"), indent=1, default=str)
print("\nwrote robust_bayes_01_grm.json")
