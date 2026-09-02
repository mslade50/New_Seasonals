"""Dynamic sizing study, part 3 (2026-09-02): per-strategy allocation done
properly. (A) bootstrap log-growth optimum per strategy with CI; (B) the
concurrency term; (C) an independent estimate from the daily series
(shrunk Sigma^-1 mu and ERC); (D) walk-forward out-of-sample test of
adopting either; (E) tail ownership (component CVaR, worst windows);
(F) era stability; (G) per-strategy daily Sharpe by dial bucket.
Writes scratch/dynamic_sizing_results3_2026-09-02.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
RNG = np.random.default_rng(11)
pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["yr"] = led["Exit Date"].dt.year
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV   # daily returns on NAV per strategy
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().shift(1)

# split OVS into its two paths (Risk bps 60 = P1, 12 = P2) so P2 dilution doesn't pollute the estimate
led["Strat2"] = led["Strategy"]
ovs = led["Strategy"] == "Overbot Vol Spike"
led.loc[ovs & (led["Risk bps"] >= 30), "Strat2"] = "OVS path 1"
led.loc[ovs & (led["Risk bps"] < 30), "Strat2"] = "OVS path 2"

# ---------------------------------------------------------------- A. bootstrap log-growth optimum per strategy
print("=== A. per-trade growth-optimal f (fraction of equity per 1R), bootstrap over trades, 2010+ ===")
def f_star(r, grid=None):
    lo = -r.min()
    fmax = 0.95 / lo if lo > 0 else 5.0
    grid = np.linspace(0.002, min(fmax, 3.0), 300)
    g = np.array([np.mean(np.log1p(f * r)) for f in grid])
    return grid[int(np.argmax(g))], g.max()
rows = []
D10 = led[led["yr"] >= 2010]
for s, g in D10.groupby("Strat2"):
    r = g["R_Multiple"].values.astype(float)
    if len(r) < 20:
        continue
    fs = []
    for _ in range(600):
        b = RNG.choice(r, len(r), replace=True)
        fs.append(f_star(b)[0])
    fs = np.array(fs)
    fq = r.mean() / (r**2).mean()
    rows.append(dict(strategy=s, N=len(r), avgR=r.mean(), sdR=r.std(), win=(r > 0).mean(), skew=pd.Series(r).skew(),
                     minR=r.min(), f_quad=fq, f_star=f_star(r)[0], f_p10=np.percentile(fs, 10), f_p50=np.median(fs), f_p90=np.percentile(fs, 90),
                     cur_f=g["Risk_flat_750k"].mean() / NAV, trades_yr=len(r) / g["yr"].nunique()))
A = pd.DataFrame(rows).set_index("strategy")
print(A.to_string())

# ---------------------------------------------------------------- B. concurrency term
print("\n=== B. concurrency: mean simultaneous legs on active days and implied pairwise rho -> adjusted f ===")
idx = pd.bdate_range("2010-01-01", "2026-09-01")
def legs_by_day(df):
    out = pd.Series(0.0, index=idx)
    for a, b in zip(df["Entry Date"], df["Exit Date"]):
        out[(out.index >= a) & (out.index <= b)] += 1
    return out
conc = {}
for s, g in D10.groupby("Strategy"):
    n = legs_by_day(g); p = strat[s].reindex(idx).fillna(0) * NAV
    act = n[n >= 1]
    one = p[n == 1].std()
    multi = n[n >= 2]
    if len(multi) > 30 and one > 0:
        nm = multi.mean(); v = p[n >= 2].var()
        rho = (v / (nm * one**2) - 1) / (nm - 1)
    else:
        rho = 0.0
    conc[s] = dict(mean_conc=act.mean(), p90_conc=act.quantile(.9), rho=float(np.clip(rho, 0, 1)), active_days=len(act))
C = pd.DataFrame(conc).T
# map onto Strat2 (OVS paths share OVS concurrency)
A["mean_conc"] = [conc.get(s.replace("OVS path 1", "Overbot Vol Spike").replace("OVS path 2", "Overbot Vol Spike"), {}).get("mean_conc", 1) for s in A.index]
A["rho"] = [conc.get(s.replace("OVS path 1", "Overbot Vol Spike").replace("OVS path 2", "Overbot Vol Spike"), {}).get("rho", 0) for s in A.index]
A["conc_factor"] = 1 / (1 + (A["mean_conc"] - 1) * A["rho"])
A["f_adj"] = A["f_p50"] * A["conc_factor"]
A["cur_over_half_adj"] = A["cur_f"] / (A["f_adj"] / 2)
print(A[["N", "f_p50", "mean_conc", "rho", "conc_factor", "f_adj", "cur_f", "cur_over_half_adj"]].to_string())
# relative allocation from f_adj with shrinkage toward the book, normalised to current total risk/yr
book_f = A["f_p50"].median()
N0 = 100
A["f_shr"] = (A["N"] * A["f_adj"] + N0 * book_f * A["conc_factor"]) / (A["N"] + N0)
c = (A["cur_f"] * A["trades_yr"]).sum() / (A["f_shr"] * A["trades_yr"]).sum()
A["kelly_bps"] = c * A["f_shr"] * 1e4
A["cur_bps"] = A["cur_f"] * 1e4
A["ratio_A"] = A["kelly_bps"] / A["cur_bps"]
print("\nrelative allocation (per-trade route, concurrency-adjusted, shrunk, total risk fixed):")
print(A[["N", "trades_yr", "cur_bps", "kelly_bps", "ratio_A"]].sort_values("ratio_A").to_string())
OUT["per_trade"] = A.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------- C. independent estimate: daily-series Sigma^-1 mu with shrinkage, and ERC
print("\n=== C. daily-series estimate: shrunk Sigma^-1 mu and equal-risk-contribution, 2010+ ===")
W = strat[(strat.index >= "2010-01-01")]
W = W.loc[:, (W != 0).mean() > 0.02]
def shrink_cov(X, delta=None):
    Sig = np.cov(X.T); n, p = X.shape
    F = np.diag(np.diag(Sig))
    if delta is None:  # Ledoit-Wolf (simplified constant-correlation-free target: diagonal)
        Xc = X - X.mean(0)
        pi = sum(np.outer(Xc[i] ** 2, Xc[i] ** 2).sum() for i in range(0, n, max(1, n // 400))) / max(1, len(range(0, n, max(1, n // 400))))
        pi_hat = ((Xc[:, :, None] * Xc[:, None, :]) ** 2).mean(0).sum() - (Sig ** 2).sum() if n < 3000 else pi
        gamma = ((Sig - F) ** 2).sum()
        delta = float(np.clip((pi_hat / n) / gamma if gamma > 0 else 1, 0, 1))
    return delta * F + (1 - delta) * Sig, delta
def weights_from(X, mu_shrink=0.5):
    Sig, delta = shrink_cov(X.values)
    mu = X.mean(0).values; sig = np.sqrt(np.diag(Sig))
    S_bar = np.mean(mu / sig)
    mu_t = mu_shrink * mu + (1 - mu_shrink) * S_bar * sig      # shrink toward equal-Sharpe prior
    w = np.linalg.solve(Sig, mu_t)
    w = w / np.abs(w).sum() * len(w)                              # mean |w| = 1, comparable to current (=1 each)
    # ERC: iterative
    e = np.ones(len(w)) / len(w)
    for _ in range(500):
        rc = e * (Sig @ e); e = e * (rc.mean() / rc); e = e / e.sum()
    e = e / e.mean()
    return pd.Series(w, index=X.columns), pd.Series(e, index=X.columns), delta
w_kel, w_erc, delta = weights_from(W)
Cc = pd.DataFrame({"kelly_w": w_kel, "erc_w": w_erc, "sharpe_daily": (W.mean() / W.std() * np.sqrt(252)), "ann_ret_pct": W.mean() * 252 * 100})
print(f"cov shrinkage delta={delta:.2f}")
print(Cc.sort_values("kelly_w").to_string())
OUT["daily_weights"] = Cc.round(4).reset_index().rename(columns={"index": "strategy"}).to_dict("records")
# agreement between the two routes
A2 = A.copy(); A2.index = [s.replace("OVS path 1", "Overbot Vol Spike").replace("OVS path 2", "Overbot Vol Spike") for s in A2.index]
agree = pd.DataFrame({"per_trade_ratio": A2.groupby(level=0)["ratio_A"].mean(), "daily_kelly_w": w_kel}).dropna()
print("\nrank agreement between per-trade ratio and daily Sigma^-1 mu weights: spearman %.2f" % agree.per_trade_ratio.rank().corr(agree.daily_kelly_w.rank()))
OUT["route_agreement_spearman"] = float(agree.per_trade_ratio.rank().corr(agree.daily_kelly_w.rank()))

# ---------------------------------------------------------------- D. walk-forward: weights from data through Y-1, applied to Y
print("\n=== D. walk-forward (2014-2026): equal (current) vs shrunk-Kelly vs ERC vs clipped-Kelly weights, re-estimated yearly ===")
def port_stats(r):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd * 100, calmar=(r.mean() * 252) / abs(dd))
full = strat[strat.index >= "2006-01-01"]
res = {k: [] for k in ["equal", "kelly", "erc", "kelly_clip", "kelly_half"]}
yearly = []
for Y in range(2014, 2027):
    tr = full[(full.index < f"{Y}-01-01")]; te = full[(full.index >= f"{Y}-01-01") & (full.index < f"{Y+1}-01-01")]
    cols = tr.columns[(tr != 0).mean() > 0.02]
    tr = tr[cols]; te = te.reindex(columns=cols).fillna(0)
    wk, we, _ = weights_from(tr)
    # the risk budget: scale each weight set so its in-sample vol equals equal-weight in-sample vol
    def scale(w):
        return w * (tr.sum(1).std() / (tr @ w).std())
    wk_s, we_s = scale(wk), scale(we)
    wc = scale(wk.clip(0.5, 1.5)); wh = scale(0.5 * wk + 0.5 * pd.Series(1.0, index=wk.index))
    for k, w in [("equal", pd.Series(1.0, index=cols)), ("kelly", wk_s), ("erc", we_s), ("kelly_clip", wc), ("kelly_half", wh)]:
        res[k].append(te @ w)
    yearly.append(dict(year=Y, equal=port_stats(te.sum(1))["sharpe"], kelly=port_stats(te @ wk_s)["sharpe"], erc=port_stats(te @ we_s)["sharpe"],
                       kelly_clip=port_stats(te @ wc)["sharpe"], kelly_half=port_stats(te @ wh)["sharpe"]))
Yr = pd.DataFrame(yearly); print(Yr.round(2).to_string(index=False))
summ = {k: port_stats(pd.concat(v)) for k, v in res.items()}
Sm = pd.DataFrame(summ).T; print(Sm.round(3).to_string())
print("years each beats equal:", {k: int((Yr[k] > Yr["equal"]).sum()) for k in ["kelly", "erc", "kelly_clip", "kelly_half"]}, "of", len(Yr))
OUT["walk_forward"] = dict(yearly=Yr.round(3).to_dict("records"), summary=Sm.round(4).to_dict())
# last-fit weights (through 2025) for the record
tr = full[full.index < "2026-01-01"]; cols = tr.columns[(tr != 0).mean() > 0.02]
wk, we, _ = weights_from(tr[cols])
print("\nweights fit through 2025 (mean |w| = 1): kelly / erc")
print(pd.DataFrame({"kelly": wk, "erc": we}).sort_values("kelly").round(2).to_string())
OUT["weights_2025"] = pd.DataFrame({"kelly": wk, "erc": we}).round(3).reset_index().rename(columns={"index": "strategy"}).to_dict("records")

# ---------------------------------------------------------------- E. tail ownership
print("\n=== E. tail ownership: component CVaR (share of book loss on worst-5% days) and worst-21d windows, 2010+ ===")
book = W.sum(1)
tail = book <= book.quantile(0.05)
comp = (W[tail].mean() / book[tail].mean()).sort_values(ascending=False)
share_var = ((W.cov() @ np.ones(len(W.columns))) / book.var()).sort_values(ascending=False)
share_pnl = (W.sum() / W.sum().sum()).reindex(comp.index)
E = pd.DataFrame({"cvar5_share": comp, "variance_share": share_var.reindex(comp.index), "pnl_share": share_pnl})
E["pnl_per_tail"] = E["pnl_share"] / E["cvar5_share"]
print(E.round(3).to_string())
OUT["tail"] = E.round(4).reset_index().rename(columns={"index": "strategy"}).to_dict("records")
r21 = book.rolling(21).sum()
worst = r21.nsmallest(40)
seen = []; rows = []
for d, v in worst.items():
    if any(abs((d - s).days) < 30 for s in seen):
        continue
    seen.append(d)
    win = W.loc[d - pd.Timedelta(days=30):d]
    contrib = (win.sum() / v).sort_values(ascending=False)
    rows.append(dict(end=d.date().isoformat(), book_21d_pct=v * 100, top=contrib.index[0], top_share=contrib.iloc[0], second=contrib.index[1], second_share=contrib.iloc[1]))
    if len(rows) >= 8:
        break
Wd = pd.DataFrame(rows); print(Wd.round(2).to_string(index=False))
OUT["worst_windows"] = Wd.round(3).to_dict("records")

# ---------------------------------------------------------------- F. era stability
print("\n=== F. era stability: trade avgR (N) by era ===")
led["era"] = pd.cut(led["yr"], [2002, 2009, 2016, 2021, 2026], labels=["2003-09", "2010-16", "2017-21", "2022-26"])
F = led.pivot_table(index="Strategy", columns="era", values="R_Multiple", aggfunc=["mean", "size"], observed=True)
F.columns = [f"{a}_{b}" for a, b in F.columns]
print(F.round(2).to_string())
OUT["era"] = F.round(3).reset_index().to_dict("records")

# ---------------------------------------------------------------- G. per-strategy daily Sharpe by dial bucket
print("\n=== G. per-strategy daily Sharpe (active days) by lagged dial bucket, 2016-07+ ===")
Wd2 = strat[strat.index >= "2016-07-20"]; dl = dial.reindex(Wd2.index)
b = pd.cut(dl, [0, 50, 65, 101], labels=["<50", "50-65", "65+"], right=False)
rows = []
for s in Wd2.columns:
    for lab in ["<50", "50-65", "65+"]:
        x = Wd2.loc[(b == lab).values, s]; x = x[x != 0]
        if len(x) >= 20:
            rows.append(dict(strategy=s, bucket=lab, days=len(x), sharpe=x.mean() / x.std() * np.sqrt(252), mean_bps=x.mean() * 1e4))
G = pd.DataFrame(rows).pivot(index="strategy", columns="bucket", values=["sharpe", "days"])
print(G.round(2).to_string())
OUT["by_dial"] = rows

json.dump(OUT, open(ROOT / "scratch/dynamic_sizing_results3_2026-09-02.json", "w"), indent=1, default=float)
print("\nwrote scratch/dynamic_sizing_results3_2026-09-02.json")
