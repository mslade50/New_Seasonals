"""seasonality_flow_4_flow.py (2026-09-02): SIGNAL FLOW as a sizing state.

A. Is signal flow itself seasonal (per strategy monthly counts, year-paired test, FDR)?
B. Does recent flow predict the next signal's quality? Per strategy: trailing-21-session
   signal count (ex same day) relative to the strategy's expanding-window norm, and
   same-day signal count, vs avgR / sdR / R-per-risk of the signals that follow.
   Episode-clustered tests. Walk-forward test of a flow-conditioned multiplier.
C. Mechanics of the seasonal book variance: open legs and open risk (bps of NAV) by
   month; share of extreme days (|pnl| top 2%) by month.
D. Reconciliation: ledger pnl vs trade_mtm sum.
Writes seasonality_flow_flow.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps
from seasonality_flow_common import (HERE, ROOT, NAV, MONTHS, load_ledger, load_trade_mtm, load_spy, load_strategy_daily,
                                     trading_calendar, episodes, cluster_diff_t, bh_fdr, summarize, jdump, perf)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
led = load_ledger()
key = ["Strategy", "Tier", "Ticker", "sig", "Entry Date"]
g = led.groupby(key, as_index=False).agg(pnl=("pnl", "sum"), risk=("risk", "sum"), yr=("yr", "first"),
                                          exit=("Exit Date", "first"), trade_ids=("trade_id", list))
g["R"] = g["pnl"] / g["risk"]
led = g
cal = trading_calendar(load_spy().index)
cal = cal[cal.index >= "2003-01-01"]
led = led[led["sig"].isin(cal.index)].copy()
led["month"] = cal.loc[led["sig"], "month"].values
led["pos"] = cal.index.get_indexer(led["sig"])
OUT = {}

# ---------------------------------------------------------------- A. flow seasonality
rows = []
for s, df in led.groupby("Strategy"):
    yrs = sorted(df["yr"].unique())
    cnt = df.groupby(["yr", "month"]).size().unstack().reindex(index=yrs, columns=range(1, 13)).fillna(0)
    cnt = cnt[cnt.sum(axis=1) >= 6]
    if len(cnt) < 4:
        continue
    norm = cnt.div(cnt.sum(axis=1), axis=0) * 12   # 1.0 = even flow
    for m in range(1, 13):
        a = norm[m]
        b = norm.drop(columns=m).mean(axis=1)
        d = a - b
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.nan
        p = 2 * sps.t.sf(abs(t), len(d) - 1) if np.isfinite(t) else np.nan
        rows.append(dict(strategy=s, month=MONTHS[m - 1], N=int(cnt[m].sum()), years=int(len(cnt)),
                         flow_norm=float(a.mean()), years_above_even=int((a > 1).sum()), t_year=t, p_year=p))
# book
cnt = led.groupby(["yr", "month"]).size().unstack().reindex(columns=range(1, 13)).fillna(0)
norm = cnt.div(cnt.sum(axis=1), axis=0) * 12
for m in range(1, 13):
    d = norm[m] - norm.drop(columns=m).mean(axis=1)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    rows.append(dict(strategy="BOOK", month=MONTHS[m - 1], N=int(cnt[m].sum()), years=int(len(cnt)), flow_norm=float(norm[m].mean()),
                     years_above_even=int((norm[m] > 1).sum()), t_year=float(t), p_year=float(2 * sps.t.sf(abs(t), len(d) - 1))))
FS = pd.DataFrame(rows)
FS["q_fdr"] = bh_fdr(FS["p_year"].values)
FS["bonf"] = np.minimum(FS["p_year"] * FS["p_year"].notna().sum(), 1)
print("=== A. FLOW seasonality: cells with q<0.10 (year-paired, normalized monthly share) ===")
fs = FS[FS.q_fdr < 0.10].sort_values("q_fdr")
print(fs.to_string(index=False))
print("\nBOOK flow by month:")
print(FS[FS.strategy == "BOOK"][["month", "N", "flow_norm", "years_above_even", "t_year", "p_year", "q_fdr"]].to_string(index=False))
OUT["flow_seasonality"] = FS.round(4).to_dict("records")
OUT["flow_seasonality_fdr10"] = fs.round(4).to_dict("records")

# ---------------------------------------------------------------- B. recent flow -> next signal quality
def trailing_counts(df: pd.DataFrame, win: int) -> np.ndarray:
    pos = df["pos"].values
    order = np.argsort(pos, kind="stable")
    sp = pos[order]
    out = np.zeros(len(pos))
    # count signals with pos in [p-win, p-1]
    lo = np.searchsorted(sp, sp - win, side="left")
    hi = np.searchsorted(sp, sp, side="left")   # strictly before same day
    out[order] = hi - lo
    return out


led = led.sort_values(["Strategy", "sig"]).reset_index(drop=True)
led["t21"] = 0.0
led["t5"] = 0.0
led["sameday"] = 0.0
led["t21_rel"] = np.nan
for s, ix in led.groupby("Strategy").indices.items():
    df = led.iloc[ix]
    led.loc[df.index, "t21"] = trailing_counts(df, 21)
    led.loc[df.index, "t5"] = trailing_counts(df, 5)
    led.loc[df.index, "sameday"] = df.groupby("sig")["sig"].transform("size").values
    # expanding norm: mean trailing-21 count over signals so far (shifted)
    t21 = pd.Series(led.loc[df.index, "t21"].values, index=df.index)
    norm = t21.expanding().mean().shift(1)
    led.loc[df.index, "t21_rel"] = (t21 / norm.replace(0, np.nan)).values
led["ep"] = 0
off = 0
for s, ix in led.groupby("Strategy").indices.items():
    e = episodes(led.iloc[ix]["sig"], 5, cal.index)
    led.loc[led.index[ix], "ep"] = e + off
    off += e.max() + 1

flow_rows = []
def flow_table(df, sname):
    # trailing-21 buckets: 0 prior, 1-2, 3-5, 6+
    b = pd.cut(df["t21"], [-1, 0, 2, 5, 1e9], labels=["0", "1-2", "3-5", "6+"])
    for cell in b.cat.categories:
        m = (b == cell).values
        if m.sum() < 5:
            continue
        st = summarize(df[m])
        t, p, gc = cluster_diff_t(df["R"].values, m, df["ep"].values)
        flow_rows.append(dict(strategy=sname, dim="t21", cell=str(cell), N=st["N"], avgR=st["avgR"], sdR=st["sdR"], R_per_risk=st["R_per_risk"],
                              sum_pnl=st["sum_pnl"], avgR_rest=float(df.loc[~m, "R"].mean()), t_ep=t, p_ep=p, n_ep=gc, share=float(m.mean())))
    b = pd.cut(df["sameday"], [0, 1, 2, 4, 1e9], labels=["1", "2", "3-4", "5+"])
    for cell in b.cat.categories:
        m = (b == cell).values
        if m.sum() < 5:
            continue
        st = summarize(df[m])
        t, p, gc = cluster_diff_t(df["R"].values, m, df["ep"].values)
        flow_rows.append(dict(strategy=sname, dim="sameday", cell=str(cell), N=st["N"], avgR=st["avgR"], sdR=st["sdR"], R_per_risk=st["R_per_risk"],
                              sum_pnl=st["sum_pnl"], avgR_rest=float(df.loc[~m, "R"].mean()), t_ep=t, p_ep=p, n_ep=gc, share=float(m.mean())))
    # relative flow terciles (expanding norm)
    r = df["t21_rel"].dropna()
    if len(r) >= 30:
        q = r.quantile([1 / 3, 2 / 3]).values
        b = pd.Series(np.where(df["t21_rel"] <= q[0], "low", np.where(df["t21_rel"] <= q[1], "mid", "high")), index=df.index)
        b[df["t21_rel"].isna()] = "na"
        for cell in ["low", "mid", "high"]:
            m = (b == cell).values
            st = summarize(df[m])
            t, p, gc = cluster_diff_t(df["R"].values, m, df["ep"].values)
            flow_rows.append(dict(strategy=sname, dim="t21_rel_tercile", cell=cell, N=st["N"], avgR=st["avgR"], sdR=st["sdR"], R_per_risk=st["R_per_risk"],
                                  sum_pnl=st["sum_pnl"], avgR_rest=float(df.loc[~m, "R"].mean()), t_ep=t, p_ep=p, n_ep=gc, share=float(m.mean())))
for s, df in led.groupby("Strategy"):
    flow_table(df, s)
flow_table(led, "BOOK")
FL = pd.DataFrame(flow_rows)
for dim, ix in FL.groupby("dim").indices.items():
    FL.loc[FL.index[ix], "q_fdr"] = bh_fdr(FL.iloc[ix]["p_ep"].values)
print("\n=== B. recent-flow buckets: BOOK ===")
print(FL[FL.strategy == "BOOK"].to_string(index=False))
print("\n=== B. per-strategy trailing-21 count buckets (avgR by bucket) ===")
piv = FL[FL.dim == "t21"].pivot(index="strategy", columns="cell", values="avgR")[["0", "1-2", "3-5", "6+"]]
pivn = FL[FL.dim == "t21"].pivot(index="strategy", columns="cell", values="N")[["0", "1-2", "3-5", "6+"]]
print(piv.round(2).to_string()); print(pivn.to_string())
print("\n=== B. per-strategy same-day count buckets (avgR) ===")
print(FL[FL.dim == "sameday"].pivot(index="strategy", columns="cell", values="avgR")[["1", "2", "3-4", "5+"]].round(2).to_string())
print(FL[FL.dim == "sameday"].pivot(index="strategy", columns="cell", values="N")[["1", "2", "3-4", "5+"]].to_string())
print("\n=== B. flow cells with q<0.10 ===")
print(FL[FL.q_fdr < 0.10].sort_values("q_fdr").to_string(index=False))
OUT["flow_quality"] = FL.round(4).to_dict("records")

# walk-forward: flow-conditioned multiplier per strategy on t21 bucket (fit years < Y), via trade_mtm
dates, mtm = load_trade_mtm()
N_DAYS = len(dates)
def book_from(df, mults):
    out = np.zeros(N_DAYS)
    for tids, m in zip(df["trade_ids"], mults):
        for t in tids:
            if t in mtm:
                s, v = mtm[t]
                out[s:s + len(v)] += v * m
    return pd.Series(out, index=dates)
base = book_from(led, np.ones(len(led)))
led["t21b"] = pd.cut(led["t21"], [-1, 0, 2, 5, 1e9], labels=["0", "1-2", "3-5", "6+"]).astype(str)
def fit_mults(fit, N0=30, lo=0.5, hi=1.5):
    out = {}
    for s, df in fit.groupby("Strategy"):
        mu0 = df["R"].mean()
        for c, gg in df.groupby("t21b"):
            n = len(gg); w = n / (n + N0)
            out[(s, c)] = float(np.clip((mu0 + w * (gg["R"].mean() - mu0)) / mu0, lo, hi)) if mu0 > 0 else 1.0
    return out
wf = {}
for design in ["wf", "loyo"]:
    mults = np.ones(len(led))
    for Y in range(2010, 2027):
        fit = led[led["yr"] < Y] if design == "wf" else led[led["yr"] != Y]
        cm = fit_mults(fit)
        m = (led["yr"] == Y).values
        for i in np.where(m)[0]:
            mults[i] = cm.get((led["Strategy"].iat[i], led["t21b"].iat[i]), 1.0)
    alt = book_from(led, mults)
    b = base[base.index >= "2010-01-01"]; a = alt[alt.index >= "2010-01-01"]
    pb, pa = perf(b), perf(a)
    yb, ya = b.groupby(b.index.year).sum(), a.groupby(a.index.year).sum()
    wf[design] = dict(base=pb, alt=pa, d_pnl_pct=(pa["total_pnl"] / pb["total_pnl"] - 1) * 100, d_sharpe=pa["sharpe"] - pb["sharpe"],
                      d_maxdd_pts=pa["maxdd_pct"] - pb["maxdd_pct"], years_better=int((ya > yb).sum()), years=int(len(yb)),
                      mean_mult=float(np.mean(mults[(led['yr'] >= 2010).values])))
    print(f"\nflow-conditioned strat x t21 multiplier [{design}]: dPnL {wf[design]['d_pnl_pct']:.1f}%  dSharpe {wf[design]['d_sharpe']:.3f}  dMaxDD {wf[design]['d_maxdd_pts']:.2f}  yrs+ {wf[design]['years_better']}/{wf[design]['years']}")
OUT["flow_multiplier_walkforward"] = wf
OUT["reconciliation"] = dict(ledger_pnl_sum=float(led["pnl"].sum()), mtm_book_sum=float(base.sum()),
                             n_trade_ids_in_ledger=int(sum(len(t) for t in led["trade_ids"])), n_in_mtm=int(sum(1 for t in led["trade_ids"] for x in t if x in mtm)))
print("reconciliation:", OUT["reconciliation"])

# ---------------------------------------------------------------- C. mechanics of seasonal variance
strat, tot = load_strategy_daily()
idx = tot.index
open_legs = pd.Series(0.0, index=idx)
open_risk = pd.Series(0.0, index=idx)
for a, b, r in zip(led["sig"], led["exit"], led["risk"]):
    m = (idx > a) & (idx <= b)
    open_legs[m] += 1
    open_risk[m] += r
mech = pd.DataFrame({"pnl": tot, "legs": open_legs, "risk_bps": open_risk / NAV * 1e4})
mech["month"] = mech.index.month
mech["abs"] = mech["pnl"].abs()
thr = mech["abs"].quantile(0.98)
mech["extreme"] = mech["abs"] >= thr
per_leg_sd = mech.groupby("month").apply(lambda g: g["pnl"].std() / max(g["legs"].mean(), 0.1))
tab = mech.groupby("month").agg(sd_bps=("pnl", lambda s: s.std() / NAV * 1e4), mean_bps=("pnl", lambda s: s.mean() / NAV * 1e4),
                                legs=("legs", "mean"), risk_bps=("risk_bps", "mean"), extreme_share=("extreme", "mean"), days=("pnl", "size"))
tab["sd_per_open_risk"] = tab["sd_bps"] / tab["risk_bps"]
tab["mean_per_open_risk"] = tab["mean_bps"] / tab["risk_bps"]
tab.index = MONTHS
print("\n=== C. mechanics: book daily sd/mean vs mean open legs and open risk (bps) by month ===")
print(tab.round(3).to_string())
r = sps.spearmanr(tab["sd_bps"], tab["risk_bps"])
print(f"Spearman(sd_bps, open risk) across months = {r.correlation:.3f} p={r.pvalue:.3f}")
OUT["variance_mechanics"] = dict(table=tab.round(4).reset_index().rename(columns={"index": "month"}).to_dict("records"),
                                 spearman_sd_vs_open_risk=float(r.correlation), p=float(r.pvalue))
# day-level regression: sd of pnl given open risk, does month add? (year-month cells)
ym = mech.groupby([mech.index.year, mech.index.month]).agg(sd=("pnl", "std"), risk=("risk_bps", "mean"), m=("month", "first"))
ym = ym[ym["risk"] > 0]
X = np.column_stack([np.ones(len(ym)), np.log(ym["risk"])])
beta, *_ = np.linalg.lstsq(X, np.log(ym["sd"]), rcond=None)
resid = np.log(ym["sd"]) - X @ beta
res_by_m = pd.Series(resid.values, index=ym["m"].values).groupby(level=0).agg(["mean", "std", "size"])
res_by_m.index = MONTHS
res_by_m["t"] = res_by_m["mean"] / (res_by_m["std"] / np.sqrt(res_by_m["size"]))
print("\nresidual log(month sd) after controlling for log(open risk), by month (t across years):")
print(res_by_m.round(3).to_string())
print(f"elasticity of monthly sd to open risk: {beta[1]:.3f}")
OUT["variance_residual_after_open_risk"] = dict(elasticity=float(beta[1]), by_month=res_by_m.round(4).reset_index().rename(columns={"index": "month"}).to_dict("records"))
jdump(OUT, HERE / "seasonality_flow_flow.json")
print("wrote", HERE / "seasonality_flow_flow.json")
