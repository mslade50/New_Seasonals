"""Refutation probe 2.2: OLV market-pullback tilt (SPY 3-10% below 252d high -> 1.15x).
Recomputes from signal_quality_features.parquet + master_prices (SPY close-based drawdown, the cycle_macro basis).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[1]
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

T = pd.read_parquet(D / "signal_quality_features.parquet")
o = T[T.Strategy == "Oversold Low Volume"].copy().sort_values("Signal Date").reset_index(drop=True)
spy = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"], filters=[("ticker", "=", "SPY")]).to_pandas().set_index("date")["Close"].sort_index()
spy.index = pd.to_datetime(spy.index)
dd_close = (spy / spy.rolling(252, min_periods=120).max() - 1) * 100
o["dd_close"] = dd_close.reindex(o["Signal Date"]).values          # cycle_macro basis (close vs rolling close max)
o["dd_high"] = o["spy_hi252_dist"]                                  # signal_quality basis (close vs rolling High max)


def episodes(dates, gap_td=5):
    d = dates.values.astype("datetime64[D]")
    ep = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        ep[i] = ep[i - 1] + (np.busday_count(d[i - 1], d[i]) > gap_td)
    return ep


o["ep"] = episodes(o["Signal Date"])


def cl_diff(g, mask, cl="ep"):
    a, b = g[mask], g[~mask]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    x = np.where(mask, 1.0, 0.0); X = np.column_stack([np.ones(len(g)), x])
    XtX = np.linalg.inv(X.T @ X); beta = XtX @ X.T @ g.R.values; e = g.R.values - X @ beta
    meat = np.zeros((2, 2))
    for c in np.unique(g[cl].values):
        m = g[cl].values == c; s = X[m].T @ e[m]; meat += np.outer(s, s)
    G = g[cl].nunique(); V = XtX @ meat @ XtX * G / (G - 1)
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def cell(g):
    return dict(N=int(len(g)), avgR=float(g.R.mean()) if len(g) else None, win=float(g.win.mean()) if len(g) else None)


for basis in ["dd_close", "dd_high"]:
    m = (o[basis] < -3) & (o[basis] >= -10)
    d, t = cl_diff(o, m)
    print(f"{basis}: cell [-10,-3) {cell(o[m])} vs rest {cell(o[~m])}  diff {d:+.3f} t(ep) {t:.2f}")
    OUT[f"headline_{basis}"] = dict(cell=cell(o[m]), rest=cell(o[~m]), diff=d, t=t)
o["inband"] = (o.dd_close < -3) & (o.dd_close >= -10)

# ---- band grid (forking paths): lower bound x upper bound
rows = []
for lo_b in [-6, -8, -10, -12, -15, -20]:
    for hi_b in [-1, -2, -3, -4, -5]:
        m = (o.dd_close < hi_b) & (o.dd_close >= lo_b)
        d, t = cl_diff(o, m)
        rows.append(dict(lo=lo_b, hi=hi_b, N=int(m.sum()), avgR_in=float(o[m].R.mean()) if m.sum() else np.nan, avgR_out=float(o[~m].R.mean()), diff=d, t=t))
G = pd.DataFrame(rows); print("\n-- band grid (SPY close drawdown, %) --"); print(G.round(3).to_string(index=False))
OUT["band_grid"] = G.round(4).to_dict("records")
OUT["bins_tried_across_studies"] = ["signal_quality_02: terciles of spy_hi252_dist", "signal_quality_03: <=-5 / -5..-2 / >-2", "signal_quality_04 R7: >-2 (family 0.5x)",
                                     "cycle_macro_lib: <3 / 3-10 / 10-20 / bear>20", "cycle_macro_04: OLV 1.5x at 3-10; book 1.25x at 3-10; book 0.8x<3 & 1.25x 3-10", "robust_bayes_02: posterior on 3-10 cell"]

# ---- LOYO / drop-year / era / midterm
loyo = []
for y in sorted(o.year.unique()):
    g = o[o.year != y]; d, t = cl_diff(g, g.inband); loyo.append(dict(drop=int(y), diff=d, t=t))
L = pd.DataFrame(loyo); print("\nLOYO diff min/max", L["diff"].min().round(3), L["diff"].max().round(3), "t min", L.t.min().round(2))
OUT["loyo"] = L.round(4).to_dict("records")
Y = o.groupby("year").apply(lambda g: pd.Series(dict(n=len(g), n_in=int(g.inband.sum()), in_R=g[g.inband].R.mean(), out_R=g[~g.inband].R.mean(), sumR_in=g[g.inband].R.sum())))
print("per-year:"); print(Y.round(2).to_string())
OUT["per_year"] = Y.round(4).reset_index().to_dict("records")
Y2 = Y.dropna(subset=["in_R", "out_R"]); print("years in>out:", int((Y2.in_R > Y2.out_R).sum()), "of", len(Y2))
OUT["years_in_gt_out"] = [int((Y2.in_R > Y2.out_R).sum()), int(len(Y2))]
for lab, keep in [("ex 2020+2022", ~o.year.isin([2020, 2022])), ("ex 2020-2022", ~o.year.isin([2020, 2021, 2022])), ("ex 2026", o.year != 2026), ("ex 2025+2026", ~o.year.isin([2025, 2026])),
                  ("2003-2015", o.year <= 2015), ("2016-2026", o.year >= 2016), ("ex best year", o.year != int(Y.sumR_in.idxmax()))]:
    g = o[keep]; d, t = cl_diff(g, g.inband)
    print(f"{lab:14s} N {len(g):4d} in {cell(g[g.inband])} out {cell(g[~g.inband])} diff {d:+.3f} t {t:.2f}")
    OUT.setdefault("subsets", {})[lab] = dict(N=int(len(g)), inb=cell(g[g.inband]), outb=cell(g[~g.inband]), diff=d, t=t)
print("best year by in-band sumR:", int(Y.sumR_in.idxmax()), "share of in-band sumR:", round(Y.sumR_in.max() / Y.sumR_in.sum(), 3))
OUT["best_year_share"] = [int(Y.sumR_in.idxmax()), float(Y.sumR_in.max() / Y.sumR_in.sum())]

# ---- beta share: residualise R on SPY hold return
spy_hold = (spy.reindex(o["Exit Date"]).values / spy.reindex(o["Entry Date"]).values - 1) * 100
o["spy_hold"] = spy_hold
b = np.polyfit(o.spy_hold.fillna(0), o.R, 1)[0]
o["R_resid"] = o.R - b * (o.spy_hold.fillna(0) - o.spy_hold.mean())
print(f"\nR on SPY hold-return slope {b:.3f} R per 1%; in-band SPY hold mean {o[o.inband].spy_hold.mean():+.2f}% vs out {o[~o.inband].spy_hold.mean():+.2f}%")
tmp = o.copy(); tmp["R"] = tmp["R_resid"]; d_r, t_r = cl_diff(tmp, tmp.inband)
print(f"residual (beta-stripped) in-band diff {d_r:+.3f} t {t_r:.2f}  vs raw diff {OUT['headline_dd_close']['diff']:+.3f}")
OUT["beta_strip"] = dict(slope=float(b), spy_hold_in=float(o[o.inband].spy_hold.mean()), spy_hold_out=float(o[~o.inband].spy_hold.mean()), resid_diff=d_r, resid_t=t_r)

# ---- the two walk-forwards: the tercile one (signal_quality, -6.2%) vs the 4-bin one (cycle_macro, +11%); reproduce a simple fixed-rule per-year test of 1.15x / 1.5x
def eval_rule(df, mult):
    risk = df.Risk_flat_750k.values; m = mult / ((mult * risk).sum() / risk.sum())
    flat = risk * df.R.values; tier = risk * m * df.R.values
    Yt = pd.DataFrame(dict(y=df.year.values, f=flat, t=tier)).groupby("y").sum(); Yt = Yt[(Yt.f != 0)]
    d = Yt.t - Yt.f
    return dict(gain_pct=float(d.sum() / abs(Yt.f.sum()) * 100), years_better=int((d > 0).sum()), years=len(Yt), worst_year=float(d.min()))
for mk, mult in [("1.15x (shipped)", 1.15), ("1.5x (study)", 1.5)]:
    r = eval_rule(o, np.where(o.inband, mult, 1.0)); print(mk, r); OUT.setdefault("rule_forms", {})[mk] = r
# expanding walk-forward with FIXED band but multiplier fit on prior years (mult = clip(mean_in/mean_out, 1, 1.5)), 2010+
rows = []
for y in range(2010, 2027):
    tr, te = o[(o.year < y)], o[o.year == y]
    if len(te) == 0 or tr.inband.sum() < 10:
        continue
    ratio = tr[tr.inband].R.mean() / tr[~tr.inband].R.mean() if tr[~tr.inband].R.mean() > 0 else 1.0
    mult = float(np.clip(ratio, 1.0, 1.5))
    risk = te.Risk_flat_750k.values; mm = np.where(te.inband, mult, 1.0)
    rows.append(dict(year=y, mult=mult, n=len(te), n_in=int(te.inband.sum()), ppr_flat=(risk * te.R).sum() / risk.sum(), ppr_rule=(risk * mm * te.R).sum() / (risk * mm).sum()))
W = pd.DataFrame(rows); W["better"] = W.ppr_rule > W.ppr_flat
print("\nfixed-band WF (mult fit on prior years):"); print(W.round(3).to_string(index=False)); print("years better", int(W.better.sum()), "of", len(W), "| years with any in-band trade", int((W.n_in > 0).sum()))
OUT["fixed_band_wf"] = dict(years_better=int(W.better.sum()), years=int(len(W)), years_touched=int((W.n_in > 0).sum()), table=W.round(4).to_dict("records"))

json.dump(OUT, open(HERE / "e_olv_pullback.json", "w"), indent=1, default=float)
print("wrote e_olv_pullback.json")
