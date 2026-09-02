"""Dynamic sizing study (2026-09-02): is the risk dial a VARIANCE instrument,
how do cross-strategy correlations and within-strategy adds behave by regime,
and what do vol-managed / drawdown-constrained overlays do to the ledger.

Inputs: dist/data/strategy_daily.json (per Strategy||Tier daily MTM, flat
$750k), data/rd2_fragility.parquet (63d dial; 10d MA; rows before
2026-07-02 are the recompute vintage), data/backtest_trades_full.parquet,
data/master_prices.parquet (SPY, ^VIX). Writes scratch/dynamic_sizing_results_2026-09-02.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
RNG = np.random.default_rng(42)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

# ---------------------------------------------------------------- load
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame({k: v for k, v in sd["series"].items()}, index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T  # collapse tiers
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)

frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().rename("dial")          # the sizing statistic
dial_lag = dial.shift(1)                                       # known at t-1 close, sizes day t

px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY", "^VIX"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
spy_ret = px["SPY"].pct_change()
vix = px["^VIX"]

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()

# open legs per strategy per day + open notional + signals per day
idx = pd.bdate_range("2003-01-01", "2026-09-01")
def legs_by_day(df):
    out = pd.Series(0.0, index=idx)
    for a, b in zip(df["Entry Date"], df["Exit Date"]):
        out[(out.index >= a) & (out.index <= b)] += 1
    return out
open_legs = {s: legs_by_day(g) for s, g in led.groupby("Strategy")}
open_legs = pd.DataFrame(open_legs)
open_total = open_legs.sum(axis=1)
notional = pd.Series(0.0, index=idx)
for a, b, n in zip(led["Entry Date"], led["Exit Date"], led["Entry Price"] * led["Shares_flat"]):
    notional[(notional.index >= a) & (notional.index <= b)] += n
signals = led.groupby("Signal Date").size().reindex(idx).fillna(0.0)

D = pd.DataFrame({"pnl": tot, "dial": dial_lag, "spy": spy_ret, "vix": vix.shift(1),
                  "open": open_total.shift(1), "notional": notional.shift(1), "sig5": signals.rolling(5).sum().shift(1)}).dropna(subset=["pnl"])
D["ret"] = D["pnl"] / NAV
D["rv21"] = D["ret"].rolling(21).std().shift(1) * np.sqrt(252)        # trailing realized book vol (ann)
D["ewv"] = np.sqrt((D["ret"] ** 2).ewm(alpha=0.06).mean().shift(1) * 252)  # EWMA vol (~ 16d half-life)
D["fwd21_vol"] = D["ret"].rolling(21).std().shift(-21) * np.sqrt(252)
D["fwd21_mean"] = D["ret"].rolling(21).mean().shift(-21) * 252
D["fwd21_pnl"] = D["pnl"].rolling(21).sum().shift(-21)

W = D[D.index >= "2016-07-20"].dropna(subset=["dial"]).copy()   # dial window
print(f"dial window {W.index.min().date()} .. {W.index.max().date()}  N={len(W)}")

# ---------------------------------------------------------------- 1. regime-conditional book stats
bins = [0, 30, 50, 65, 80, 101]
labels = ["<30", "30-50", "50-65", "65-80", "80+"]
W["bucket"] = pd.cut(W["dial"], bins, labels=labels, right=False)
base_mu, base_sd = W["ret"].mean(), W["ret"].std()
rows = []
for b, g in W.groupby("bucket", observed=True):
    mu, s = g["ret"].mean(), g["ret"].std()
    sub = strat.reindex(g.index)
    act = sub.loc[:, (sub != 0).mean() > 0.10]
    c = act.corr()
    off = c.values[np.triu_indices(len(c), 1)]
    w = act.std() / act.std().sum()
    eff_n = 1 / float(w.values @ c.fillna(0).values @ w.values) if len(c) > 1 else np.nan
    rows.append(dict(bucket=b, days=len(g), share=len(g) / len(W), mean_bps=mu * 1e4, sd_bps=s * 1e4,
                     sharpe=mu / s * np.sqrt(252), worst_pct=g["ret"].min() * 100,
                     p_lt_1pct=(g["ret"] < -0.01).mean(), cvar5_bps=g["ret"][g["ret"] <= g["ret"].quantile(.05)].mean() * 1e4,
                     kelly_rel=(mu / s**2) / (base_mu / base_sd**2), open_legs=g["open"].mean(),
                     notional_pct=g["notional"].mean() / NAV * 100, sig5=g["sig5"].mean(),
                     avg_corr=float(np.nanmean(off)), eff_n=eff_n, spy_vol=g["spy"].std() * np.sqrt(252) * 100,
                     book_beta=np.polyfit(g["spy"].fillna(0), g["ret"], 1)[0]))
R1 = pd.DataFrame(rows)
print("\n=== 1. book by lagged dial bucket (bps of NAV/day; kelly_rel = mu/sd^2 vs unconditional) ===")
print(R1.to_string(index=False))
OUT["regime_table"] = R1.round(4).to_dict("records")

# ---------------------------------------------------------------- 2. what predicts NEXT-21d book variance vs mean?
print("\n=== 2. forecasting next-21d realized book vol and mean (Spearman rho; OLS R^2) ===")
def spear(a, b):
    m = W[[a, b]].dropna()
    return float(m[a].rank().corr(m[b].rank()))
def r2(xcols, y):
    m = W[xcols + [y]].dropna()
    X = np.column_stack([np.ones(len(m))] + [m[c].values for c in xcols])
    beta, *_ = np.linalg.lstsq(X, m[y].values, rcond=None)
    resid = m[y].values - X @ beta
    return float(1 - resid.var() / m[y].values.var())
pred = {}
for x in ["dial", "vix", "rv21", "ewv", "open", "notional", "sig5"]:
    pred[x] = dict(vol_spearman=spear(x, "fwd21_vol"), vol_r2=r2([x], "fwd21_vol"),
                   mean_spearman=spear(x, "fwd21_mean"), mean_r2=r2([x], "fwd21_mean"))
pred["dial+rv21"] = dict(vol_r2=r2(["dial", "rv21"], "fwd21_vol"), mean_r2=r2(["dial", "rv21"], "fwd21_mean"))
pred["dial+rv21+open"] = dict(vol_r2=r2(["dial", "rv21", "open"], "fwd21_vol"), mean_r2=r2(["dial", "rv21", "open"], "fwd21_mean"))
pred["vix+rv21+open"] = dict(vol_r2=r2(["vix", "rv21", "open"], "fwd21_vol"), mean_r2=r2(["vix", "rv21", "open"], "fwd21_mean"))
print(pd.DataFrame(pred).T.to_string())
OUT["forecast"] = pred
# realized vol by dial bucket (next 21d) vs mean
fb = W.groupby("bucket", observed=True)[["fwd21_vol", "fwd21_mean"]].mean()
print("\nnext-21d realized book vol / ann mean by dial bucket:\n", fb.to_string())
OUT["fwd_by_bucket"] = fb.round(4).to_dict()

# ---------------------------------------------------------------- 3. within-strategy adds: implied pairwise rho from sleeve pnl vs leg count
print("\n=== 3. within-strategy: daily sleeve pnl sd vs # open legs -> implied avg pairwise rho ===")
rows = []
for s in ["Oversold Low Volume", "Overbot Vol Spike", "Weak Close Decent Sznls", "3x Bear ETF Overbot Fade", "52wh Breakout", "SPY QQQ MonFri Reversion", "Indices Oversold Bounce", "LT Trend ST OS"]:
    if s not in strat or s not in open_legs:
        continue
    p = pd.DataFrame({"pnl": strat[s], "n": open_legs[s].reindex(strat.index)}).dropna()
    p = p[(p.index >= "2010-01-01") & (p.n >= 1)]
    one = p[p.n == 1].pnl.std()
    for lo, hi, lab in [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 99, "13+")]:
        g = p[(p.n >= lo) & (p.n <= hi)]
        if len(g) < 30:
            continue
        n = g.n.mean(); v = g.pnl.var()
        rho = (v / (n * one**2) - 1) / (n - 1) if n > 1 else np.nan
        rows.append(dict(strategy=s, legs=lab, days=len(g), n_mean=n, sd_per_leg=g.pnl.std() / n,
                         sd_vs_sqrtn=g.pnl.std() / (one * np.sqrt(n)), implied_rho=rho, mean_per_leg=g.pnl.mean() / n))
R3 = pd.DataFrame(rows)
print(R3.to_string(index=False))
OUT["adds"] = R3.round(4).to_dict("records")
# OLV implied rho by regime
p = pd.DataFrame({"pnl": strat["Oversold Low Volume"], "n": open_legs["Oversold Low Volume"].reindex(strat.index), "dial": dial_lag}).dropna()
p = p[p.n >= 2]
one = strat["Oversold Low Volume"][open_legs["Oversold Low Volume"].reindex(strat.index) == 1].std()
for lab, g in [("dial<50", p[p.dial < 50]), ("dial>=50", p[p.dial >= 50]), ("dial>=65", p[p.dial >= 65])]:
    n = g.n.mean(); rho = (g.pnl.var() / (n * one**2) - 1) / (n - 1)
    print(f"OLV stacks {lab}: days {len(g)}, mean legs {n:.1f}, implied rho {rho:.2f}, mean/leg ${g.pnl.mean()/n:,.0f}, sd/leg ${g.pnl.std()/n:,.0f}")
    OUT.setdefault("olv_rho_by_regime", {})[lab] = dict(days=len(g), legs=n, rho=rho, mean_per_leg=g.pnl.mean() / n, sd_per_leg=g.pnl.std() / n)

# ---------------------------------------------------------------- 4. overlay simulations (daily pnl scaling, lag-1 instruments)
print("\n=== 4. overlay sims on daily book pnl, 2016-07+ (equal-vol comparison) ===")
def metrics(r, label):
    eq = (r).cumsum(); dd = eq - eq.cummax()
    return dict(label=label, ann_ret=r.mean() * 252 * 100, ann_vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252),
                maxdd=dd.min() * 100, worst_day=r.min() * 100, cvar5=r[r <= r.quantile(.05)].mean() * 100,
                skew=r.skew(), calmar=(r.mean() * 252) / abs(dd.min()))
def equal_vol(r, target_sd):
    return r * (target_sd / r.std())
base = W["ret"]
sims = {"baseline": pd.Series(1.0, index=W.index)}
# A: realized-vol targeting (EWMA), never above 1.5x, floor 0.25
tv = W["ewv"].median()
sims["A_ewma_voltarget"] = (tv / W["ewv"]).clip(0.25, 1.5)
sims["A2_ewma_cutonly"] = (tv / W["ewv"]).clip(0.25, 1.0)
# B: dial bucket table = bucket Kelly relative, capped at 1 (never over), floor .25
kmap = {r["bucket"]: min(1.0, max(0.25, r["kelly_rel"])) for r in OUT["regime_table"]}
sims["B_dial_kelly_insample"] = W["bucket"].map(kmap).astype(float)
sims["B2_dial_step_50"] = np.where(W["dial"] >= 50, 0.5, 1.0)
sims["B3_dial_step_65"] = np.where(W["dial"] >= 65, 0.5, 1.0)
sims["B4_dial_linear"] = (1.0 - 0.75 * ((W["dial"] - 30) / 60).clip(0, 1))
# C: concurrency: scale by 1/sqrt(open/median)
sims["C_open_sqrt"] = (np.sqrt(W["open"].median() / W["open"].clip(lower=1))).clip(0.25, 1.5)
# D: combined ewma vol + dial step
sims["D_ewma_x_dial65"] = sims["A_ewma_voltarget"] * sims["B3_dial_step_65"]
# E: VIX-based (outside the dial)
sims["E_vix"] = (W["vix"].median() / W["vix"]).clip(0.25, 1.5)
res = []
for k, m in sims.items():
    m = pd.Series(np.asarray(m, dtype=float), index=W.index)
    r = base * m
    raw = metrics(r, k + " (raw)")
    ev = metrics(equal_vol(r, base.std()), k + " (equal-vol)")
    raw["avg_mult"] = float(m.mean()); ev["avg_mult"] = float(m.mean())
    res += [raw, ev]
R4 = pd.DataFrame(res)
print(R4.to_string(index=False))
OUT["overlays"] = R4.round(4).to_dict("records")

# leave-one-year-out for the in-sample dial table and the ewma target
print("\nLOYO (equal-vol Sharpe by held-out year): B_dial_kelly vs A_ewma vs baseline")
loyo = []
for y in sorted(W.index.year.unique()):
    tr, te = W[W.index.year != y], W[W.index.year == y]
    if len(te) < 60:
        continue
    bm, bs = tr["ret"].mean(), tr["ret"].std()
    km = {}
    for b, g in tr.groupby("bucket", observed=True):
        km[b] = min(1.0, max(0.25, (g["ret"].mean() / g["ret"].std()**2) / (bm / bs**2))) if len(g) > 40 else 1.0
    mB = te["bucket"].map(km).astype(float)
    mA = (tr["ewv"].median() / te["ewv"]).clip(0.25, 1.5)
    loyo.append(dict(year=y, base=metrics(te["ret"], "")["sharpe"], dial_table=metrics(te["ret"] * mB, "")["sharpe"],
                     ewma=metrics(te["ret"] * mA, "")["sharpe"], step65=metrics(te["ret"] * np.where(te["dial"] >= 65, .5, 1), "")["sharpe"]))
L = pd.DataFrame(loyo)
print(L.round(2).to_string(index=False))
print("years overlay beats baseline (Sharpe):", {c: int((L[c] > L["base"]).sum()) for c in ["dial_table", "ewma", "step65"]}, "of", len(L))
OUT["loyo"] = L.round(3).to_dict("records")

# ---------------------------------------------------------------- 5. drawdown-constrained sizing frontier (block bootstrap)
print("\n=== 5. drawdown frontier: block bootstrap 252d paths, P(maxDD > x) by multiplier of current sizing ===")
def block_boot(r, n_paths=3000, horizon=252, mean_block=10):
    r = r.values; N = len(r); out = np.empty((n_paths, horizon))
    for i in range(n_paths):
        pos = 0
        while pos < horizon:
            L_ = RNG.geometric(1 / mean_block); st = RNG.integers(0, N)
            seg = r[st:st + L_]
            if len(seg) < L_:
                seg = np.concatenate([seg, r[:L_ - len(seg)]])
            out[i, pos:pos + L_] = seg[:horizon - pos]; pos += L_
    return out
paths = block_boot(D[D.index >= "2016-01-01"]["ret"])
front = []
for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    p = paths * mult
    eq = p.cumsum(axis=1); dd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
    front.append(dict(mult_of_current=mult, grm_equiv=1.5 * mult, median_ann_pct=np.median(p.sum(axis=1)) * 100,
                      p_dd_gt10=(dd < -0.10).mean(), p_dd_gt15=(dd < -0.15).mean(), p_dd_gt20=(dd < -0.20).mean(), p_dd_gt30=(dd < -0.30).mean(),
                      median_dd=np.median(dd) * 100, p5_dd=np.percentile(dd, 5) * 100))
R5 = pd.DataFrame(front)
print(R5.to_string(index=False))
OUT["dd_frontier"] = R5.round(4).to_dict("records")
# same frontier under overlay A (ewma vol target) scaled to equal mean multiplier
mA = pd.Series(np.asarray(sims["A_ewma_voltarget"], dtype=float), index=W.index)
rA = (W["ret"] * mA); rA = rA * (W["ret"].std() / rA.std())
pathsA = block_boot(rA)
frontA = []
for mult in [1.0, 1.5, 2.0]:
    p = pathsA * mult; eq = p.cumsum(axis=1); dd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
    frontA.append(dict(mult=mult, median_ann_pct=np.median(p.sum(axis=1)) * 100, p_dd_gt10=(dd < -0.10).mean(), p_dd_gt15=(dd < -0.15).mean(), p_dd_gt20=(dd < -0.20).mean()))
print("frontier under EWMA vol-target overlay (equal-vol):\n", pd.DataFrame(frontA).to_string(index=False))
OUT["dd_frontier_overlayA"] = pd.DataFrame(frontA).round(4).to_dict("records")

# ---------------------------------------------------------------- 6. drawdown-state predictability (Grossman-Zhou would be pure insurance if none)
print("\n=== 6. does the book's own drawdown state predict next-21d pnl? ===")
eq = D["ret"].cumsum(); ddst = (eq - eq.rolling(252, min_periods=60).max()).shift(1)
D["dd_state"] = ddst
for lab, m in [("dd < 2%", ddst > -0.02), ("2-5%", (ddst <= -0.02) & (ddst > -0.05)), ("5-10%", (ddst <= -0.05) & (ddst > -0.10)), ("> 10%", ddst <= -0.10)]:
    g = D[m & (D.index >= "2010-01-01")].dropna(subset=["fwd21_mean"])
    print(f"{lab:8s} days {len(g):5d}  next-21d ann mean {g['fwd21_mean'].mean()*100:6.1f}%  next-21d vol {g['fwd21_vol'].mean()*100:5.1f}%")
    OUT.setdefault("dd_state", {})[lab] = dict(days=len(g), fwd_mean=g["fwd21_mean"].mean(), fwd_vol=g["fwd21_vol"].mean())

# ---------------------------------------------------------------- 7. cross-strategy correlation by regime (top strategies), and beta to SPY by regime
print("\n=== 7. strategy correlation matrices: low (<50) vs high (>=65) dial ===")
top = ["Oversold Low Volume", "Overbot Vol Spike", "Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "52wh Breakout", "Indices Oversold Bounce", "LT Trend ST OS", "3x ETF Overbot Fade"]
for lab, m in [("dial<50", W["dial"] < 50), ("dial>=65", W["dial"] >= 65)]:
    sub = strat.reindex(W.index[m])[top]
    c = sub.corr(); off = c.values[np.triu_indices(len(c), 1)]
    print(f"{lab}: avg pairwise corr {np.nanmean(off):.3f}; pairs > 0.3: {(off > 0.3).sum()} of {len(off)}")
    print(c.round(2).to_string())
    OUT.setdefault("corr_by_regime", {})[lab] = dict(avg=float(np.nanmean(off)), n_gt_03=int((off > 0.3).sum()), matrix=c.round(3).to_dict())

json.dump(OUT, open(ROOT / "scratch/dynamic_sizing_results_2026-09-02.json", "w"), indent=1, default=float)
print("\nwrote scratch/dynamic_sizing_results_2026-09-02.json")
