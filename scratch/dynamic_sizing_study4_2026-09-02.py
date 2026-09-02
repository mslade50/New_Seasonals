"""Dynamic sizing study, part 4 (2026-09-02): walk-forward variants of the
strategy allocation (half-blend expanding, 5y/3y half-life, tail-aware),
per-year PnL gate, weights fit through 2025, per-strategy beta to SPY.
Companion to study3; prints only."""
import json, numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)
sd = json.load(open(ROOT / "dist/data/strategy_daily.json")); dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY"])]).to_pandas().set_index("date")["Close"].pct_change()
full = strat[strat.index >= "2006-01-01"]

def shrink_cov(X, hl=None):
    w = np.ones(len(X)) if hl is None else 0.5 ** ((len(X) - 1 - np.arange(len(X))) / hl)
    w = w / w.sum(); mu = (X * w[:, None]).sum(0); Xc = X - mu; Sig = (Xc * w[:, None]).T @ Xc
    F = np.diag(np.diag(Sig)); delta = 0.3
    return delta * F + (1 - delta) * Sig, mu

def weights(X, hl=None, mu_shrink=0.5):
    Sig, mu = shrink_cov(X.values, hl); sig = np.sqrt(np.diag(Sig)); Sbar = np.mean(mu / sig)
    mt = mu_shrink * mu + (1 - mu_shrink) * Sbar * sig; w = np.linalg.solve(Sig, mt); w = w / np.abs(w).sum() * len(w)
    return pd.Series(w, index=X.columns)

def stats(r):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd * 100, calmar=r.mean() * 252 / abs(dd))

variants = {"equal": None, "half_expanding": ("exp", 0.5), "half_hl5y": ("hl", 0.5), "half_hl3y": ("hl3", 0.5), "tail_aware": ("tail", None)}
res = {k: [] for k in variants}; yr = []
for Y in range(2014, 2027):
    tr = full[full.index < f"{Y}-01-01"]; te = full[(full.index >= f"{Y}-01-01") & (full.index < f"{Y+1}-01-01")]
    cols = tr.columns[(tr != 0).mean() > 0.02]; tr = tr[cols]; te = te.reindex(columns=cols).fillna(0)
    base_vol = tr.sum(1).std()
    sc = lambda w: w * (base_vol / (tr @ w).std())
    row = {"year": Y}
    for k, v in variants.items():
        if v is None: w = pd.Series(1.0, index=cols)
        elif v[0] == "exp": w = sc(0.5 * weights(tr) + 0.5)
        elif v[0] == "hl": w = sc(0.5 * weights(tr, hl=252 * 5) + 0.5)
        elif v[0] == "hl3": w = sc(0.5 * weights(tr, hl=252 * 3) + 0.5)
        else:
            book = tr.sum(1); tail = book <= book.quantile(.05); comp = (tr[tail].mean() / book[tail].mean()).clip(lower=0.005)
            share = tr.sum() / tr.sum().sum(); ppt = (share / comp).clip(0.3, 3); w = sc((ppt / ppt.mean()).clip(0.5, 1.5) * 0.5 + 0.5)
        r = te @ w; res[k].append(r); row[k + "_pnl"] = r.sum() * 100
    yr.append(row)
Y = pd.DataFrame(yr); print(Y.round(1).to_string(index=False))
print(pd.DataFrame({k: stats(pd.concat(v)) for k, v in res.items()}).T.round(3).to_string())
for k in variants:
    if k == "equal": continue
    d = (Y[k + "_pnl"] - Y["equal_pnl"]); loss = (d / Y["equal_pnl"].abs()).min()
    print(f"{k}: years better on PnL {(d > 0).sum()}/13, worst held-out year vs equal {loss*100:+.0f}% of equal PnL, total PnL diff {d.sum():+.0f} pct-NAV")
tr = full[full.index < "2026-01-01"]; cols = tr.columns[(tr != 0).mean() > 0.02]
w5 = weights(tr[cols], hl=252 * 5); wE = weights(tr[cols])
print("\nweights through 2025: expanding vs 5y-half-life; shipping mult = clip(0.5*wE+0.5, 0.6, 1.4)")
print(pd.DataFrame({"expanding": wE, "hl5y": w5, "ship_mult": (0.5 * wE + 0.5).clip(0.6, 1.4)}).sort_values("expanding").round(2).to_string())
W = strat[strat.index >= "2016-01-01"]; s = px.reindex(W.index).fillna(0)
b = {c: np.polyfit(s, W[c], 1)[0] for c in W.columns}; act = {c: (W[c] != 0).mean() for c in W.columns}
print("\nbeta to SPY on NAV (2016+), per active day:")
print(pd.Series({c: b[c] / max(act[c], 1e-6) for c in W.columns}).sort_values(ascending=False).round(3).to_string())
