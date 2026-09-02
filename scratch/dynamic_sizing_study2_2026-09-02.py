"""Dynamic sizing study, part 2 (2026-09-02): replays of the concrete controls
implied by part 1 -- OLV sleeve open-risk budget, same-index clone halving,
SPY beta hedge by dial regime, mu-haircut drawdown frontier, and a
hierarchical (HRP-style) cluster read of the strategy set by regime.
Writes scratch/dynamic_sizing_results2_2026-09-02.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
RNG = np.random.default_rng(7)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().shift(1)
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
spy = px["SPY"].pct_change()
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()

# ------------------------------------------------ A. SPY beta hedge of the book by regime
print("=== A. book PnL with a rolling-beta SPY hedge, by lagged dial bucket (2016-07+) ===")
D = pd.DataFrame({"ret": tot / NAV, "spy": spy, "dial": dial}).dropna()
D = D[D.index >= "2016-07-20"]
# ex-ante beta: trailing 63d OLS of book ret on spy, lagged one day
cov = D["ret"].rolling(63).cov(D["spy"]).shift(1); var = D["spy"].rolling(63).var().shift(1)
D["beta_hat"] = (cov / var).clip(-1, 2)
D["hedged"] = D["ret"] - D["beta_hat"].fillna(0) * D["spy"]
D["bucket"] = pd.cut(D["dial"], [0, 30, 50, 65, 80, 101], labels=["<30", "30-50", "50-65", "65-80", "80+"], right=False)
rows = []
for b, g in D.groupby("bucket", observed=True):
    r, h = g["ret"], g["hedged"]
    rows.append(dict(bucket=b, days=len(g), beta_hat=g["beta_hat"].mean(), spy_ann=g["spy"].mean() * 252 * 100,
                     book_sharpe=r.mean() / r.std() * np.sqrt(252), hedged_sharpe=h.mean() / h.std() * np.sqrt(252),
                     book_mean_bps=r.mean() * 1e4, hedged_mean_bps=h.mean() * 1e4, book_sd=r.std() * 1e4, hedged_sd=h.std() * 1e4,
                     hedge_cost_bps=(r.mean() - h.mean()) * 1e4))
RA = pd.DataFrame(rows); print(RA.to_string(index=False))
OUT["hedge_by_regime"] = RA.round(4).to_dict("records")
# hedge only when dial >= 65 vs never
for lab, m in [("never", np.zeros(len(D), bool)), ("dial>=65", (D["dial"] >= 65).values), ("dial>=50", (D["dial"] >= 50).values)]:
    r = np.where(m, D["hedged"], D["ret"]); r = pd.Series(r, index=D.index)
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    print(f"policy hedge {lab:9s}: ann {r.mean()*252*100:5.1f}%  vol {r.std()*np.sqrt(252)*100:5.1f}%  sharpe {r.mean()/r.std()*np.sqrt(252):.2f}  maxDD {dd*100:.1f}%  worst {r.min()*100:.2f}%")
    OUT.setdefault("hedge_policy", {})[lab] = dict(ann=r.mean() * 252, vol=r.std() * np.sqrt(252), sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd, worst=r.min())

# ------------------------------------------------ B. OLV sleeve open-risk budget replay (ledger, realized-at-exit)
print("\n=== B. OLV sleeve open-risk budget: scale new legs into remaining room (2010+, flat) ===")
olv = led[(led.Strategy == "Oversold Low Volume") & (led["Entry Date"] >= "2010-01-01")].sort_values(["Entry Date", "trade_id"]).copy()
def replay(cap_bps=None, rule=None):
    open_ = []  # (exit_date, risk)
    pnl, risk_used, scaled = [], [], 0
    for _, t in olv.iterrows():
        open_ = [(x, r) for x, r in open_ if x >= t["Entry Date"]]
        used = sum(r for _, r in open_)
        leg_risk = t["Risk_flat_750k"]
        if cap_bps is not None:
            room = cap_bps / 1e4 * NAV - used
            m = float(np.clip(room / leg_risk, 0, 1)) if leg_risk > 0 else 0
        elif rule == "sqrt":
            n = len(open_)
            m = 1.0 if n == 0 else min(1.0, (np.sqrt(n + 1) - np.sqrt(n)))  # marginal sqrt budget
        else:
            m = 1.0
        if m < 0.999:
            scaled += 1
        pnl.append(t["PnL_flat_750k"] * m); risk_used.append(leg_risk * m)
        open_.append((t["Exit Date"], leg_risk * m))
    p = pd.Series(pnl, index=olv["Exit Date"].values).groupby(level=0).sum()
    daily = p.reindex(pd.bdate_range(p.index.min(), p.index.max())).fillna(0)
    roll21 = daily.rolling(21).sum()
    eq = daily.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(total=float(p.sum()), worst21=float(roll21.min()), maxdd=float(dd), scaled_legs=scaled,
                risk_deployed=float(sum(risk_used)), pnl_per_risk=float(p.sum() / sum(risk_used)))
rows = [dict(policy="none", **replay())]
for cap in [150, 200, 250, 300, 400, 500]:
    rows.append(dict(policy=f"cap {cap} bps open", **replay(cap_bps=cap)))
rows.append(dict(policy="sqrt(n) marginal", **replay(rule="sqrt")))
RB = pd.DataFrame(rows); print(RB.to_string(index=False))
OUT["olv_budget"] = RB.round(2).to_dict("records")
# distribution of OLV open risk (bps) at entry
open_risk_at_entry = []
open_ = []
for _, t in olv.iterrows():
    open_ = [(x, r) for x, r in open_ if x >= t["Entry Date"]]
    open_risk_at_entry.append(sum(r for _, r in open_) / NAV * 1e4)
    open_.append((t["Exit Date"], t["Risk_flat_750k"]))
s = pd.Series(open_risk_at_entry)
print("OLV open risk (bps) already deployed at each new entry: p50 %.0f p75 %.0f p90 %.0f p95 %.0f max %.0f" % tuple(s.quantile([.5, .75, .9, .95, 1.0])))
OUT["olv_open_risk_quantiles"] = s.quantile([.5, .75, .9, .95, 1.0]).round(0).to_dict()

# ------------------------------------------------ C. same-day index clones (IOB, MonFri, Monthly Weak Close): half-size when SPY and QQQ both fire
print("\n=== C. same-day SPY+QQQ clones: variance vs mean effect of sizing each at 0.5x on both-fire days ===")
for s in ["Indices Oversold Bounce", "SPY QQQ MonFri Reversion", "Monthly Weak Close"]:
    d = led[(led.Strategy == s) & (led["Signal Date"] >= "2010-01-01")]
    cnt = d.groupby("Signal Date").Ticker.nunique()
    both = cnt[cnt >= 2].index
    x = d[d["Signal Date"].isin(both)]; y = d[~d["Signal Date"].isin(both)]
    day = x.groupby("Signal Date")["PnL_flat_750k"].sum()
    print(f"{s:26s}: both-fire days {len(both):3d} (of {cnt.size}), day pnl mean ${day.mean():7,.0f} sd ${day.std():7,.0f} | "
          f"single days mean ${y.groupby('Signal Date')['PnL_flat_750k'].sum().mean():7,.0f} sd ${y.groupby('Signal Date')['PnL_flat_750k'].sum().std():7,.0f} | "
          f"leg R corr on both days {x.pivot_table(index='Signal Date', columns='Ticker', values='R_Multiple').corr().iloc[0,1]:.2f}")
    OUT.setdefault("clones", {})[s] = dict(both_days=int(len(both)), day_mean=float(day.mean()), day_sd=float(day.std()),
                                          single_mean=float(y.groupby('Signal Date')['PnL_flat_750k'].sum().mean()),
                                          single_sd=float(y.groupby('Signal Date')['PnL_flat_750k'].sum().std()))

# ------------------------------------------------ D. mu-haircut drawdown frontier
print("\n=== D. drawdown frontier with the daily MEAN haircut (vol untouched), 2016+ block bootstrap ===")
r = (tot / NAV)[tot.index >= "2016-01-01"]
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
rows = []
for hc in [1.0, 0.5, 0.25]:
    rr = r - r.mean() * (1 - hc)
    paths = block_boot(rr)
    for mult in [0.75, 1.0, 1.25, 1.5, 2.0]:
        p = paths * mult; eq = p.cumsum(axis=1); dd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
        rows.append(dict(mu_haircut=hc, mult=mult, grm=1.5 * mult, median_ann=np.median(p.sum(axis=1)) * 100,
                         p_dd10=(dd < -.10).mean(), p_dd15=(dd < -.15).mean(), p_dd20=(dd < -.20).mean(), p5_dd=np.percentile(dd, 5) * 100))
RD = pd.DataFrame(rows); print(RD.to_string(index=False))
OUT["haircut_frontier"] = RD.round(4).to_dict("records")

# ------------------------------------------------ E. hierarchical clustering of strategies by regime (theme structure)
print("\n=== E. strategy cluster structure (single-linkage on 1-corr), calm vs high dial ===")
W = strat[strat.index >= "2016-07-20"].copy(); dl = dial.reindex(W.index)
active = W.columns[(W != 0).mean() > 0.05]
for lab, m in [("dial<50", dl < 50), ("dial>=50", dl >= 50)]:
    sub = W.loc[m.fillna(False), active]
    c = sub.corr().fillna(0); np.fill_diagonal(c.values, 1)
    dist = np.sqrt(0.5 * (1 - c)); Z = linkage(squareform(dist.values, checks=False), "average")
    cl = fcluster(Z, t=0.85, criterion="distance")  # corr > ~0.28 join
    groups = {}
    for name, k in zip(c.columns, cl):
        groups.setdefault(int(k), []).append(name)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"{lab}: {len(active)} strategies -> {len(groups)} clusters; multi-member: {multi}")
    OUT.setdefault("clusters", {})[lab] = {str(k): v for k, v in groups.items()}
    # risk share vs 'equal risk contribution' -- current share of variance contribution
    w = sub.std() / sub.std().sum()
    Sig = sub.cov(); port_var = float(w.values @ Sig.values @ w.values)
    mcr = (Sig.values @ w.values) * w.values / port_var
    rc = pd.Series(mcr, index=sub.columns).sort_values(ascending=False)
    print("  variance contribution share (top 6):", rc.head(6).round(3).to_dict())
    OUT["clusters"][lab + "_risk_contrib"] = rc.round(4).to_dict()

# ------------------------------------------------ F. 52wh deep-stack cell at trade level
print("\n=== F. 52wh Breakout legs by # open 52wh legs at entry (trade level, 2010+) ===")
b = led[(led.Strategy == "52wh Breakout") & (led["Entry Date"] >= "2010-01-01")].sort_values("Entry Date")
n_open = []
open_ = []
for _, t in b.iterrows():
    open_ = [x for x in open_ if x >= t["Entry Date"]]
    n_open.append(len(open_)); open_.append(t["Exit Date"])
b = b.assign(n_open=n_open)
b["nb"] = pd.cut(b.n_open, [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"])
RF = b.groupby("nb", observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), win=("R_Multiple", lambda s: (s > 0).mean()), pnl=("PnL_flat_750k", "sum"))
print(RF.to_string()); OUT["b52_stack"] = RF.round(3).reset_index().to_dict("records")

json.dump(OUT, open(ROOT / "scratch/dynamic_sizing_results2_2026-09-02.json", "w"), indent=1, default=float)
print("\nwrote scratch/dynamic_sizing_results2_2026-09-02.json")
