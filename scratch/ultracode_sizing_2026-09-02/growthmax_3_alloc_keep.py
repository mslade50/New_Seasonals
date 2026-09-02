"""Growth-maximizer lens, part 3: per-strategy allocation with the MEASURED
per-strategy keep factors (estimation_haircut_results.json) as the mu
shrinkage instead of the baseline's flat 50% pull toward an equal-Sharpe
prior, plus a margin shadow price (the Lagrangian of max mu'w - 1/2 w'Sw
s.t. c'w <= M is w = S^-1 (mu - lambda c), with c_s the strategy's average
open margin requirement per unit weight at current size).

Reuses the baseline's daily-series machinery (dynamic_sizing_study3:
Ledoit-Wolf-shrunk covariance of per-strategy daily MTM on the flat $750k
basis, weights normalised to mean |w| = 1, walk-forward 2014-2026 with each
weight set vol-matched in sample to equal weights).  Writes
growthmax_3_alloc_keep.json.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

NAV = 750_000.0
OUT: dict = {}
pd.set_option("display.width", 250, "display.max_columns", 40)

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
keeps = {k: v["keep_central"] for k, v in json.load(open(HERE / "estimation_haircut_results.json"))["per_strategy"].items()}
keep_rng = {k: v["keep_range"] for k, v in json.load(open(HERE / "estimation_haircut_results.json"))["per_strategy"].items()}
# judgement overrides recorded by the haircut analyst for the two mechanical outliers
keeps_j = dict(keeps); keeps_j["Overbot Vol Spike"] = 0.50; keeps_j["52wh Breakout"] = 0.42

# ---------------------------------------------------------------- margin per unit weight (c_s): mean daily open requirement / NAV at current size, TIMS tiered
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["notional"] = led["Entry Price"] * led["Shares_flat"]
LEV3X = set(sc.LEV3X_ALL)
BROAD = {"SPY", "QQQ", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALL = {"IWM", "MDY", "IJR", "IJH"}
def rate(t):
    return 0.45 if t in LEV3X else 0.08 if t in BROAD else 0.10 if t in SMALL else 0.15
led["req"] = led["Ticker"].map(rate) * led["notional"]
idx = pd.bdate_range("2010-01-01", "2026-08-07")
d0 = idx.searchsorted(led["Entry Date"].values); d1 = idx.searchsorted(led["Exit Date"].values)
creq = {}
for s in led["Strategy"].unique():
    m = (led["Strategy"] == s).values
    out = np.zeros(len(idx))
    for a, b, v in zip(d0[m], d1[m], led.loc[m, "req"].values):
        if a < len(idx):
            out[a:min(b, len(idx) - 1) + 1] += v
    creq[s] = out.mean() / NAV
c_s = pd.Series(creq)
print("=== mean daily margin requirement / NAV per strategy at current size (TIMS tiered), 2010+ ===")
print((c_s.sort_values(ascending=False) * 100).round(2).to_string())
OUT["margin_per_unit_weight_pct_nav"] = (c_s * 100).round(3).to_dict()


# ---------------------------------------------------------------- covariance + weights (baseline machinery)
def shrink_cov(X):
    Sig = np.cov(X.T); n, p = X.shape
    F = np.diag(np.diag(Sig))
    Xc = X - X.mean(0)
    if n < 3000:
        pi_hat = ((Xc[:, :, None] * Xc[:, None, :]) ** 2).mean(0).sum() - (Sig ** 2).sum()
    else:
        step = max(1, n // 400)
        pi_hat = sum(np.outer(Xc[i] ** 2, Xc[i] ** 2).sum() for i in range(0, n, step)) / len(range(0, n, step))
    gamma = ((Sig - F) ** 2).sum()
    delta = float(np.clip((pi_hat / n) / gamma if gamma > 0 else 1, 0, 1))
    return delta * F + (1 - delta) * Sig, delta


def weights(X: pd.DataFrame, mode: str, lam: float = 0.0, keep_map: dict | None = None):
    Sig, delta = shrink_cov(X.values)
    mu = X.mean(0).values; sig = np.sqrt(np.diag(Sig))
    S_bar = np.mean(mu / sig)
    if mode == "baseline":          # 50% toward equal-Sharpe prior (the plan's L1)
        mu_t = 0.5 * mu + 0.5 * S_bar * sig
    elif mode == "keep":            # measured per-strategy keep as the shrinkage, no equal-Sharpe pull
        k = np.array([keep_map.get(c, 0.5) for c in X.columns])
        mu_t = k * mu
    elif mode == "keep_half":       # keep, then half toward the (keep-scaled) equal-Sharpe prior
        k = np.array([keep_map.get(c, 0.5) for c in X.columns])
        mk = k * mu; Sk = np.mean(mk / sig)
        mu_t = 0.5 * mk + 0.5 * Sk * sig
    elif mode == "raw":
        mu_t = mu
    else:
        raise ValueError(mode)
    c = np.array([creq.get(col, c_s.mean()) for col in X.columns])
    w = np.linalg.solve(Sig, mu_t - lam * c)
    w = w / np.abs(w).sum() * len(w)
    return pd.Series(w, index=X.columns), delta


W = strat[strat.index >= "2010-01-01"]
W = W.loc[:, (W != 0).mean() > 0.02]
tr = W[W.index < "2026-01-01"]
res = {}
for mode, km in [("baseline", None), ("keep", keeps), ("keep_judged", keeps_j), ("keep_half", keeps), ("raw", None)]:
    w, delta = weights(tr, mode.replace("_judged", ""), keep_map=km)
    res[mode] = w
R = pd.DataFrame(res)
R["sharpe_daily"] = tr.mean() / tr.std() * np.sqrt(252)
R["keep"] = pd.Series(keeps).reindex(R.index)
R["margin_pct_nav"] = (c_s * 100).reindex(R.index)
print("\n=== weights fit through 2025 (mean |w| = 1): baseline (plan L1) vs keep-shrunk vs raw ===")
print(R.sort_values("keep").round(3).to_string())
OUT["weights_2025"] = R.round(4).reset_index().rename(columns={"index": "strategy"}).to_dict("records")

# margin shadow price sweep on the keep weights: how much book margin per unit vol can be bought
print("\n=== margin shadow price lambda on keep weights: book mean-margin and in-sample Sharpe at equal vol ===")
Sig, _ = shrink_cov(tr.values)
base_vol = tr.sum(1).std()
rows = []
mu_keep = np.array([keeps.get(c, 0.5) for c in tr.columns]) * tr.mean(0).values
cvec = np.array([creq.get(c, c_s.mean()) for c in tr.columns])
for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
    w = np.linalg.solve(Sig, mu_keep - lam * 1e-4 * cvec)      # lam in units of 1e-4 (daily return per unit of margin/NAV)
    w = w / np.abs(w).sum() * len(w)
    ws = w * base_vol / (tr.values @ w).std()
    r = tr.values @ ws
    rows.append(dict(lam=lam, sharpe=float(r.mean() / r.std() * np.sqrt(252)), ann=float(r.mean() * 252 * 100), margin_pct_nav=float((ws * cvec).sum() * 100),
                     margin_per_vol=float((ws * cvec).sum() / r.std()), w=dict(zip(tr.columns, np.round(ws, 2)))))
    print(f"  lam {lam:3.1f}: Sharpe {rows[-1]['sharpe']:.2f} ann {rows[-1]['ann']:.1f}% mean margin {rows[-1]['margin_pct_nav']:.1f}% NAV  margin/vol {rows[-1]['margin_per_vol']:.1f}  w_min {min(ws):.2f} w_max {max(ws):.2f}")
eq = np.ones(len(tr.columns)); req_eq = (eq * cvec).sum() * 100
print(f"  equal weights: mean margin {req_eq:.1f}% NAV, Sharpe {tr.sum(1).mean()/tr.sum(1).std()*np.sqrt(252):.2f}")
OUT["margin_shadow_price"] = rows; OUT["equal_weight_margin_pct_nav"] = float(req_eq)


# ---------------------------------------------------------------- walk-forward 2014-2026
def port_stats(r):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd * 100, calmar=(r.mean() * 252) / abs(dd))


full = strat[strat.index >= "2006-01-01"]
VARIANTS = ["equal", "baseline_half", "keep_full", "keep_half_blend", "keep_judged_full", "keep_judged_half_blend", "keep_clip05_16", "raw_full"]
res = {k: [] for k in VARIANTS}; yearly = []
print("\n=== walk-forward 2014-2026 (fit < Y, apply Y, vol-matched in sample) ===")
for Y in range(2014, 2027):
    trn = full[full.index < f"{Y}-01-01"]; te = full[(full.index >= f"{Y}-01-01") & (full.index < f"{Y+1}-01-01")]
    cols = trn.columns[(trn != 0).mean() > 0.02]
    trn = trn[cols]; te = te.reindex(columns=cols).fillna(0)
    wb, _ = weights(trn, "baseline"); wk, _ = weights(trn, "keep", keep_map=keeps); wkj, _ = weights(trn, "keep", keep_map=keeps_j); wr, _ = weights(trn, "raw")
    one = pd.Series(1.0, index=cols)
    def scale(w):
        return w * (trn.sum(1).std() / (trn @ w).std())
    sets = {"equal": one, "baseline_half": scale(0.5 * wb + 0.5 * one), "keep_full": scale(wk), "keep_half_blend": scale(0.5 * wk + 0.5 * one),
            "keep_judged_full": scale(wkj), "keep_judged_half_blend": scale(0.5 * wkj + 0.5 * one), "keep_clip05_16": scale(wk.clip(0.5, 1.6)), "raw_full": scale(wr)}
    yr = dict(year=Y)
    for k, w in sets.items():
        res[k].append(te @ w); yr[k] = port_stats(te @ w)["ann"]
    yearly.append(yr)
Yr = pd.DataFrame(yearly)
summ = pd.DataFrame({k: port_stats(pd.concat(v)) for k, v in res.items()}).T
summ["years_better_pnl_vs_equal"] = [int((Yr[k] > Yr["equal"]).sum()) for k in summ.index]
summ["worst_year_vs_equal_pct"] = [float(((Yr[k] - Yr["equal"]) / Yr["equal"].abs()).min() * 100) for k in summ.index]
summ["cum_gain_vs_equal_pctNAV"] = [float((Yr[k] - Yr["equal"]).sum()) for k in summ.index]
print(summ.round(3).to_string())
print(Yr.round(1).to_string(index=False))
OUT["walk_forward"] = dict(summary=summ.round(4).to_dict(), yearly=Yr.round(3).to_dict("records"))

# ---------------------------------------------------------------- shipping multipliers under the growth lens: keep weights, half-blend, clip [0.5, 1.6]
wk, _ = weights(tr, "keep", keep_map=keeps); wkj, _ = weights(tr, "keep", keep_map=keeps_j); wb, _ = weights(tr, "baseline")
ship = pd.DataFrame({"plan_L1_clip06_14": (0.5 * wb + 0.5).clip(0.6, 1.4), "growth_keep_half_clip05_16": (0.5 * wk + 0.5).clip(0.5, 1.6),
                     "growth_keepjudged_half_clip05_16": (0.5 * wkj + 0.5).clip(0.5, 1.6), "growth_keep_full_clip05_16": wk.clip(0.5, 1.6)})
ship["keep"] = pd.Series(keeps).reindex(ship.index); ship["margin_pct_nav"] = (c_s * 100).reindex(ship.index)
print("\n=== candidate base-bps multipliers (fit through 2025) ===")
print(ship.round(2).sort_values("growth_keep_half_clip05_16").to_string())
OUT["ship_multipliers"] = ship.round(3).reset_index().rename(columns={"index": "strategy"}).to_dict("records")
json.dump(OUT, open(HERE / "growthmax_3_alloc_keep.json", "w"), indent=1, default=float)
print("\nwrote growthmax_3_alloc_keep.json")
