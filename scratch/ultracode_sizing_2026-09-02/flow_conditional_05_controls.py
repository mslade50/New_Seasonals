"""Flow-conditional sizing, part 4: is the flow effect just the dial / VIX / market drawdown in disguise?
Controls: lag-1 10d-MA 63d dial (2016-07+ only, current-weights vintage from data/rd2_fragility.parquet;
rows before 2026-07-02 are the recompute vintage), VIX close on the signal day, SPY 21d return on the
signal day, SPY 21d realized vol. Rank-regression of R on flow rank + control rank with episode-cluster
bootstrap SE, and hi-vs-lo flow inside each control tercile.
Reads flow_trades_candidates.parquet. Writes flow_conditional_controls.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from flow_conditional_lib import ROOT, OUT_DIR, FAMILIES, cluster_boot_diff, spearman

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
tr = pd.read_parquet(OUT_DIR / "flow_trades_candidates.parquet")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().shift(1)
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY", "^VIX"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index)
spy = px["SPY"]
ctl = pd.DataFrame({"vix": px["^VIX"], "spy21": spy.pct_change(21), "spyvol21": spy.pct_change().rolling(21).std() * np.sqrt(252),
                    "spy_dd": spy / spy.rolling(252).max() - 1})
tr = tr.join(ctl, on="Signal Date")
tr["dial"] = dial.reindex(tr["Signal Date"]).values


def rank01(x):
    return x.rank(pct=True)


def cluster_ols(df, ycol, xcols, n=500, seed=3):
    """OLS of y on ranks of xcols with episode-cluster bootstrap SE."""
    d = df[[ycol] + xcols + ["ep"]].dropna()
    X = np.column_stack([np.ones(len(d))] + [rank01(d[c]).values for c in xcols])
    y = d[ycol].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    eps = d["ep"].values; ids = np.unique(eps)
    groups = [np.where(eps == g)[0] for g in ids]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
    se = np.std(np.array(bs), axis=0)
    return dict(N=len(d), coef={c: float(b) for c, b in zip(["const"] + xcols, beta)}, t={c: float(b / s) if s > 0 else np.nan for c, b, s in zip(["const"] + xcols, beta, se)})


print("=== A. rank regression: R ~ flow(f5) + control, episode-clustered t (coefficient = R change from bottom to top of the rank) ===")
rows = []
for f in FAMILIES:
    g = tr[tr.family == f]
    for ctrl in [None, "vix", "spy21", "spyvol21", "spy_dd"]:
        for var in ["f5", "f21"]:
            xs = [var] + ([ctrl] if ctrl else [])
            r = cluster_ols(g, "R", xs)
            rows.append(dict(family=f, var=var, control=ctrl or "none", N=r["N"], b_flow=r["coef"][var], t_flow=r["t"][var],
                             b_ctrl=r["coef"].get(ctrl, np.nan) if ctrl else np.nan, t_ctrl=r["t"].get(ctrl, np.nan) if ctrl else np.nan))
    g16 = g[g["Signal Date"] >= "2016-07-20"]
    for var in ["f5", "f21"]:
        for xs in [[var], [var, "dial"]]:
            r = cluster_ols(g16, "R", xs)
            rows.append(dict(family=f, var=var, control="dial (2016-07+)" if len(xs) > 1 else "none (2016-07+)", N=r["N"], b_flow=r["coef"][var], t_flow=r["t"][var],
                             b_ctrl=r["coef"].get("dial", np.nan), t_ctrl=r["t"].get("dial", np.nan)))
A = pd.DataFrame(rows)
print(A.round(3).to_string(index=False))
OUT["rank_regression"] = A.round(4).to_dict("records")

print("\n=== B. hi-vs-lo f5 flow INSIDE each control tercile (avgR, cluster t) ===")
rows = []
for f in FAMILIES:
    g = tr[tr.family == f].copy()
    q1, q2 = g.f5.quantile(1 / 3), g.f5.quantile(2 / 3)
    g["fb"] = np.where(g.f5 <= q1, "lo", np.where(g.f5 <= q2, "mid", "hi"))
    for ctrl in ["vix", "spy21", "spyvol21", "dial"]:
        gg = g.dropna(subset=[ctrl])
        c1, c2 = gg[ctrl].quantile(1 / 3), gg[ctrl].quantile(2 / 3)
        cb = np.where(gg[ctrl] <= c1, "lo", np.where(gg[ctrl] <= c2, "mid", "hi"))
        for lab in ["lo", "mid", "hi"]:
            h = gg[(cb == lab) & (gg.fb == "hi")]; l = gg[(cb == lab) & (gg.fb == "lo")]
            if len(h) >= 8 and len(l) >= 8:
                r = cluster_boot_diff(h.R.values, h.ep.values, l.R.values, l.ep.values, n=400, seed=4)
                rows.append(dict(family=f, control=ctrl, ctrl_tercile=lab, ctrl_range=f"{gg[ctrl][cb == lab].min():.2f}..{gg[ctrl][cb == lab].max():.2f}",
                                 hi_N=len(h), lo_N=len(l), hi_avgR=h.R.mean(), lo_avgR=l.R.mean(), diff=r["diff"], t_cl=r["t"]))
B = pd.DataFrame(rows)
print(B.round(3).to_string(index=False))
OUT["within_control"] = B.round(4).to_dict("records")

print("\n=== C. correlation of family flow with the controls (trade level) ===")
rows = []
for f in FAMILIES:
    g = tr[tr.family == f]
    rows.append(dict(family=f, **{f"rho_f5_{c}": spearman(g.f5, g[c]) for c in ["vix", "spy21", "spyvol21", "spy_dd", "dial"]}))
C = pd.DataFrame(rows); print(C.round(3).to_string(index=False))
OUT["flow_vs_controls"] = C.round(4).to_dict("records")

# the live OLV recency ladder is itself a flow rule (down-size at LOW recent OLV flow): show the per-ticker recency cell vs strategy flow
print("\n=== D. OLV: per-ticker recency (ladder rung) vs strategy-level flow (s21) ===")
olv = tr[tr.Strategy == "Oversold Low Volume"].copy()
q1, q2 = olv.s21.quantile(1 / 3), olv.s21.quantile(2 / 3)
olv["s21b"] = np.where(olv.s21 <= q1, "lo", np.where(olv.s21 <= q2, "mid", "hi"))
olv["rung"] = np.where(olv.SizeMult <= 0.55, "0.5x", np.where(olv.SizeMult <= 0.75, "0.7x", "1.0x"))
t = olv.pivot_table(index="rung", columns="s21b", values="R", aggfunc=["mean", "size"])
print(t.round(2).to_string())
OUT["olv_rung_vs_flow"] = {str(k): v for k, v in t.round(3).to_dict().items()}

json.dump(OUT, open(OUT_DIR / "flow_conditional_controls.json", "w"), indent=1, default=float)
print("wrote flow_conditional_controls.json")
