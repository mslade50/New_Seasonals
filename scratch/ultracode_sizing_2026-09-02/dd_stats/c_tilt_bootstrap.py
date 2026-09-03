"""Refutation probe 1.3: keep-adjusted allocation tilt. Bootstraps the Sigma^-1 mu weights
(same construction as robust_bayes_03_allocation.py) and reports per-strategy sign-flip rates,
plus a year-jackknife of the shipped multipliers and a walk-forward drop-year check.
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
RNG = np.random.default_rng(11)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
full = strat[strat.index >= "2006-01-01"]
print("daily series", full.index.min().date(), "->", full.index.max().date(), full.shape)
keep = {s: v["keep_central"] for s, v in json.load(open(D / "estimation_haircut_results.json"))["per_strategy"].items()}
keep["Overbot Vol Spike"] = 0.48; keep["52wh Breakout"] = 0.42
KEEP = pd.Series(keep)


def shrink_cov(X, delta=0.3):
    mu = X.mean(0); Xc = X - mu; Sig = Xc.T @ Xc / len(X)
    return delta * np.diag(np.diag(Sig)) + (1 - delta) * Sig, mu


def weights(X: pd.DataFrame, keep_vec=None, mu_shrink=0.5):
    Sig, mu = shrink_cov(X.values)
    if keep_vec is not None:
        mu = mu * keep_vec.reindex(X.columns).fillna(0.5).values
    sig = np.sqrt(np.diag(Sig)); Sbar = np.mean(mu / sig)
    mt = mu_shrink * mu + (1 - mu_shrink) * Sbar * sig
    w = np.linalg.solve(Sig, mt); w = w / np.abs(w).sum() * len(w)
    return pd.Series(w, index=X.columns)


tr = full[full.index < "2026-01-01"]; cols = tr.columns[(tr != 0).mean() > 0.02]; tr = tr[cols]
w_ship = weights(tr, KEEP); m_ship = (0.5 * w_ship + 0.5).clip(0.7, 1.3)
print("\nshipped (fit through 2025):"); print(pd.DataFrame({"w": w_ship, "mult": m_ship}).round(3).sort_values("mult").to_string())
print("mean shipped mult", round(m_ship.mean(), 3), "| risk-weighted by 2016+ deployed risk not computed here")
OUT["shipped"] = pd.DataFrame({"w": w_ship, "mult": m_ship}).round(4).to_dict("index")

# ---- (a) circular block bootstrap of the daily panel, block 21 td, 400 reps
n = len(tr); B = 400; L = 21
X = tr.values
ws, ms = [], []
for b in range(B):
    starts = RNG.integers(0, n, size=n // L + 1)
    idx = np.concatenate([np.arange(s, s + L) % n for s in starts])[:n]
    Xb = pd.DataFrame(X[idx], columns=cols)
    w = weights(Xb, KEEP); ws.append(w.values); ms.append((0.5 * w + 0.5).clip(0.7, 1.3).values)
W = pd.DataFrame(ws, columns=cols); M = pd.DataFrame(ms, columns=cols)
res = pd.DataFrame({"mult_ship": m_ship, "boot_mean": M.mean(), "boot_sd": M.std(), "p5": M.quantile(0.05), "p95": M.quantile(0.95),
                    "P(mult>1)": (M > 1).mean(), "P(at_clip_floor)": (M <= 0.7001).mean(), "P(at_clip_cap)": (M >= 1.2999).mean(),
                    "sign_flip_vs_ship": (np.sign(W - 1) != np.sign(w_ship - 1)).mean(), "P(|mult-ship|>0.15)": ((M - m_ship).abs() > 0.15).mean()})
res = res.sort_values("mult_ship")
print("\n-- block bootstrap (21d blocks, 400 reps) of the shipped multipliers --"); print(res.round(3).to_string())
OUT["block_bootstrap"] = res.round(4).to_dict("index")
print("strategies with sign-flip rate > 0.25:", list(res.index[res["sign_flip_vs_ship"] > 0.25]))
print("strategies with P(mult>1) between 0.25 and 0.75 (undetermined direction):", list(res.index[res["P(mult>1)"].between(0.25, 0.75)]))

# ---- (b) year jackknife of the shipped fit
jk = {}
for y in range(2006, 2026):
    t2 = tr[tr.index.year != y]; w = weights(t2, KEEP); jk[y] = (0.5 * w + 0.5).clip(0.7, 1.3)
J = pd.DataFrame(jk).T
jres = pd.DataFrame({"ship": m_ship, "jk_min": J.min(), "jk_max": J.max(), "jk_range": J.max() - J.min(), "year_of_min": J.idxmin(), "year_of_max": J.idxmax()})
print("\n-- drop-one-year jackknife of the shipped multipliers --"); print(jres.round(3).sort_values("ship").to_string())
OUT["year_jackknife"] = jres.round(4).astype(str).to_dict("index")
# which year moves the fit most (sum of abs changes)
mv = (J - m_ship).abs().sum(axis=1).sort_values(ascending=False)
print("years that move the multiplier vector most:", mv.head(4).round(2).to_dict())
OUT["most_influential_years"] = mv.head(5).round(3).to_dict()

# ---- (c) walk-forward 2014-2026 equal vs keep_0.7_1.3, WITHOUT vol matching, and drop-2020/2021 sensitivity; also 'no keep' variant
def stats(r):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd * 100)


rows = []; parts = {"equal": [], "keep_0.7_1.3": [], "nokeep_0.7_1.3": [], "keep_0.7_1.3_volmatch": []}
for Y in range(2014, 2027):
    t_ = full[full.index < f"{Y}-01-01"]; te = full[(full.index >= f"{Y}-01-01") & (full.index < f"{Y+1}-01-01")]
    c = t_.columns[(t_ != 0).mean() > 0.02]; t_ = t_[c]; te = te.reindex(columns=c).fillna(0)
    wK = weights(t_, KEEP); wE = weights(t_)
    W_ = {"equal": pd.Series(1.0, index=c), "keep_0.7_1.3": (0.5 * wK + 0.5).clip(0.7, 1.3), "nokeep_0.7_1.3": (0.5 * wE + 0.5).clip(0.7, 1.3)}
    base_vol = t_.sum(1).std(); W_["keep_0.7_1.3_volmatch"] = W_["keep_0.7_1.3"] * (base_vol / (t_ @ W_["keep_0.7_1.3"]).std())
    row = {"year": Y}
    for k, w in W_.items():
        r = te @ w; parts[k].append(r); row[k] = r.sum() * 100
    rows.append(row)
Yr = pd.DataFrame(rows); print("\n-- walk-forward, raw multipliers (no vol match) --"); print(Yr.round(1).to_string(index=False))
summ = pd.DataFrame({k: stats(pd.concat(v)) for k, v in parts.items()}).T
summ["years_better_vs_equal"] = [int((Yr[k] > Yr["equal"]).sum()) for k in summ.index]
summ["worst_year_vs_equal_pct"] = [float(((Yr[k] - Yr["equal"]) / Yr["equal"].abs()).min() * 100) for k in summ.index]
print(summ.round(3).to_string()); OUT["walkforward_raw"] = summ.round(4).to_dict("index"); OUT["walkforward_years"] = Yr.round(2).to_dict("records")
# gain concentration: which years carry the WF gain of keep_0.7_1.3 vs equal (vol-matched, as the plan reports)
gain = (Yr["keep_0.7_1.3_volmatch"] - Yr["equal"])
print("vol-matched gain by year (pts NAV):", gain.round(1).tolist(), "| total", round(gain.sum(), 1), "| ex 2015+2016+2021:", round(gain.sum() - gain[Yr.year.isin([2015, 2016, 2021])].sum(), 1))
OUT["wf_gain_by_year_volmatched"] = dict(zip(Yr.year.astype(int), gain.round(3)))
OUT["wf_gain_ex_2015_2016_2021"] = float(gain.sum() - gain[Yr.year.isin([2015, 2016, 2021])].sum())
OUT["wf_gain_total"] = float(gain.sum())

json.dump(OUT, open(HERE / "c_tilt_bootstrap.json", "w"), indent=1, default=float)
print("wrote c_tilt_bootstrap.json")
